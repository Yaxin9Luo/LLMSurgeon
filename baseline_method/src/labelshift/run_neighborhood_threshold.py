"""
Neighborhood (Perturbation Sensitivity) with Binary Thresholding for Category Composition Estimation

This script adapts the "Neighbourhood Comparison" style MIA signal (Mattern et al., ACL 2023)
to predict category composition percentages of LLM pretraining data.

We implement a lightweight variant of the perturbation-based score:
  score(x) = ll(x) - mean_j ll(perturb_j(x))

where ll(.) is the (average) log-likelihood (negative HF loss), and perturb_j(x) is created
by masking a fraction of words and filling them with random words sampled from a reference
word pool. This captures the core intuition from the neighborhood attack:
members are more "locally stable" under perturbations, so ll drop is typically larger.

Adaptation pattern (mirrors run_mink_threshold.py / run_recall_threshold.py / run_dcpdd_threshold.py):
1) Load category samples from local_samples_dir
2) Score each sample with neighborhood sensitivity
3) Threshold: score >= threshold -> member (default), else non-member
4) Normalize member counts across categories -> predicted composition

Notes:
- This script avoids the heavy T5 mask-filling dependency used in the original repo and
  uses random word fills by default, which is fast and self-contained.
"""

import argparse
import json
import os
import random
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from tqdm import tqdm

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from data_utils import detect_available_categories, _read_jsonl


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Estimate category mixture via Neighborhood perturbation sensitivity with binary thresholding"
    )
    # Data
    p.add_argument(
        "--local_samples_dir",
        type=str,
        default="/data/yaxin/data_anatomy/data_samples",
        help="Directory with category JSONL files (e.g., wikipedia.jsonl, books.jsonl, ...)",
    )
    p.add_argument("--merge_web", action="store_true", help="Merge CommonCrawl and C4 into Web (6-way)")
    p.add_argument("--max_per_class", type=int, default=None, help="Max samples per category (None = use all)")
    p.add_argument("--seed", type=int, default=0)

    # Target model
    p.add_argument("--target_model", type=str, required=True, help="HF model id for scoring (causal LM)")
    p.add_argument("--hf_revision", type=str, default=None, help="HF model revision/checkpoint")
    p.add_argument("--half", action="store_true", help="Load model in bfloat16")
    p.add_argument("--int8", action="store_true", help="Load model in 8-bit (bitsandbytes)")
    p.add_argument("--max_tokens", type=int, default=512, help="Max tokens per sample for scoring")
    p.add_argument("--trust_remote_code", action="store_true", help="Trust remote code for models like OLMo")

    # Neighborhood parameters (random-fill perturbations)
    p.add_argument("--n_perturbations", type=int, default=10, help="# perturbed variants per sample")
    p.add_argument(
        "--pct_words_masked",
        type=float,
        default=0.3,
        help="Fraction of words to replace with random words (0< pct <=1).",
    )
    p.add_argument("--span_length", type=int, default=2, help="Length (in words) of each replaced span")
    p.add_argument(
        "--fill_strategy",
        type=str,
        choices=["global", "leave_one_out"],
        default="global",
        help="How to build the random fill word pool: global pooled corpus, or leave-one-out per category.",
    )
    p.add_argument(
        "--max_fill_texts",
        type=int,
        default=5000,
        help="Cap #texts used to build fill word pool (after pooling).",
    )
    p.add_argument(
        "--min_fill_vocab",
        type=int,
        default=5000,
        help="If the fill vocab is smaller than this, falls back to using all words seen (no cap).",
    )

    # Thresholding
    p.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="Threshold for binary classification. If not specified, uses median score (adaptive).",
    )
    p.add_argument(
        "--member_if",
        type=str,
        default="gt",
        choices=["gt", "lt"],
        help="Membership rule: gt means score >= threshold => member; lt means score <= threshold => member.",
    )

    # Output
    p.add_argument("--output_dir", type=str, default=str(Path(__file__).resolve().parents[3] / "out"))
    p.add_argument("--run_name", type=str, default="neighborhood_threshold")
    return p.parse_args()


def load_model(
    name: str,
    revision: str = None,
    half: bool = False,
    int8: bool = False,
    trust_remote_code: bool = False,
):
    """Load a HuggingFace causal LM."""
    int8_kwargs = {}
    half_kwargs = {}
    if int8:
        int8_kwargs = dict(load_in_8bit=True, torch_dtype=torch.bfloat16)
    elif half:
        half_kwargs = dict(torch_dtype=torch.bfloat16)

    revision_kwargs = {"revision": revision} if revision else {}

    model = AutoModelForCausalLM.from_pretrained(
        name,
        return_dict=True,
        device_map="auto",
        trust_remote_code=trust_remote_code,
        **int8_kwargs,
        **half_kwargs,
        **revision_kwargs,
    )
    model.eval()
    tok = AutoTokenizer.from_pretrained(name, trust_remote_code=trust_remote_code, **revision_kwargs)
    return model, tok


def load_category_samples(
    local_dir: str,
    merge_web: bool,
    max_per_class: int = None,
    seed: int = 0,
) -> Tuple[List[str], Dict[str, List[str]]]:
    """Load text samples per category from JSONL files."""
    rng = random.Random(seed)
    categories, file_to_cat = detect_available_categories(local_dir, merge_web=merge_web)
    if not categories:
        raise FileNotFoundError(f"No category files detected under {local_dir}")

    cat_to_texts: Dict[str, List[str]] = {c: [] for c in categories}
    for fname, cat in file_to_cat.items():
        fpath = os.path.join(local_dir, fname)
        if not os.path.exists(fpath):
            continue
        texts = _read_jsonl(fpath)
        rng.shuffle(texts)
        if max_per_class is not None:
            texts = texts[:max_per_class]
        cat_to_texts[cat].extend(texts)

    categories = [c for c in categories if cat_to_texts.get(c)]
    cat_to_texts = {c: cat_to_texts[c] for c in categories}
    return categories, cat_to_texts


def _build_fill_vocab(texts: List[str], max_fill_texts: int, seed: int) -> List[str]:
    """Build a word-level fill vocabulary from texts."""
    rng = random.Random(seed)
    if max_fill_texts is not None and max_fill_texts > 0 and len(texts) > max_fill_texts:
        texts = rng.sample(texts, k=max_fill_texts)
    vocab: set[str] = set()
    for t in texts:
        for w in t.split():
            w = w.strip()
            if w:
                vocab.add(w)
    return sorted(vocab)


def _random_perturb_words(words: List[str], pct: float, span_length: int, fill_vocab: List[str], rng: random.Random) -> List[str]:
    """
    Replace ~pct fraction of words with random spans of random words.
    This is a simplified version of the masking/perturbation in the original repo.
    """
    if not words:
        return words
    if not (0.0 < pct <= 1.0):
        return words
    if span_length <= 0:
        span_length = 1
    if not fill_vocab:
        return words

    n_words = len(words)
    n_replace = max(1, int(round(pct * n_words)))
    # convert replacement count to span count
    n_spans = max(1, int(np.ceil(n_replace / span_length)))

    out = words[:]
    for _ in range(n_spans):
        start = rng.randrange(0, n_words)
        end = min(n_words, start + span_length)
        # Replace each word in the span with a random vocab word
        for i in range(start, end):
            out[i] = fill_vocab[rng.randrange(len(fill_vocab))]
    return out


@torch.no_grad()
def avg_ll(text: str, model, tok, max_tokens: int) -> float:
    """Average log-likelihood (negative HF loss)."""
    ids = tok.encode(text, truncation=True, max_length=max_tokens)
    if len(ids) < 2:
        return float("nan")
    input_ids = torch.tensor(ids, dtype=torch.long).unsqueeze(0).to(model.device)
    out = model(input_ids, labels=input_ids)
    # out.loss is float32 even in bf16 models, but keep robust.
    loss = out.loss.float()
    return float((-loss).item())


@torch.no_grad()
def neighborhood_score(
    text: str,
    model,
    tok,
    max_tokens: int,
    fill_vocab: List[str],
    n_perturbations: int,
    pct_words_masked: float,
    span_length: int,
    rng: random.Random,
) -> float:
    """
    score = ll(x) - mean_j ll(perturb_j(x))
    Higher score typically indicates more member-like (less robust to perturbations).
    """
    ll0 = avg_ll(text, model, tok, max_tokens=max_tokens)
    if not np.isfinite(ll0):
        return float("nan")
    if n_perturbations <= 0:
        return float(ll0)

    words = text.split()
    if len(words) < 2:
        return float("nan")

    lls: List[float] = []
    for _ in range(n_perturbations):
        pert_words = _random_perturb_words(words, pct=pct_words_masked, span_length=span_length, fill_vocab=fill_vocab, rng=rng)
        pert_text = " ".join(pert_words)
        llp = avg_ll(pert_text, model, tok, max_tokens=max_tokens)
        if np.isfinite(llp):
            lls.append(llp)
    if not lls:
        return float("nan")
    return float(ll0 - float(np.mean(lls)))


def main() -> None:
    args = parse_args()
    rng = random.Random(args.seed)

    print(f"Loading samples from: {args.local_samples_dir}")
    categories, cat_to_texts = load_category_samples(
        args.local_samples_dir,
        merge_web=args.merge_web,
        max_per_class=args.max_per_class,
        seed=args.seed,
    )
    total_samples = sum(len(v) for v in cat_to_texts.values())
    print(f"Categories: {categories}")
    print(f"Samples per category: {[len(cat_to_texts[c]) for c in categories]}")
    print(f"Total samples: {total_samples}")

    print(f"Loading model: {args.target_model}" + (f" @ {args.hf_revision}" if args.hf_revision else ""))
    model, tok = load_model(
        args.target_model,
        revision=args.hf_revision,
        half=args.half,
        int8=args.int8,
        trust_remote_code=args.trust_remote_code,
    )

    pooled_all: List[str] = []
    for c in categories:
        pooled_all.extend(cat_to_texts[c])

    # Build global fill vocab once if requested
    global_fill_vocab: List[str] | None = None
    if args.fill_strategy == "global":
        fill_vocab = _build_fill_vocab(pooled_all, max_fill_texts=args.max_fill_texts, seed=args.seed)
        if len(fill_vocab) < args.min_fill_vocab:
            fill_vocab = _build_fill_vocab(pooled_all, max_fill_texts=None, seed=args.seed)
        global_fill_vocab = fill_vocab
        print(f"Fill strategy: global | vocab_size={len(global_fill_vocab)} | max_fill_texts={args.max_fill_texts}")
    else:
        print(f"Fill strategy: leave_one_out | max_fill_texts={args.max_fill_texts}")

    print(
        f"\nScoring samples with Neighborhood perturbation sensitivity "
        f"(n_perturbations={args.n_perturbations}, pct_words_masked={args.pct_words_masked}, span_length={args.span_length})"
    )
    cat_scores: Dict[str, List[float]] = {c: [] for c in categories}
    all_scores: List[float] = []

    pbar = tqdm(total=total_samples, desc="Scoring (Neighborhood)")
    for cat in categories:
        if args.fill_strategy == "leave_one_out":
            loo_pool: List[str] = []
            for c2 in categories:
                if c2 != cat:
                    loo_pool.extend(cat_to_texts[c2])
            fill_vocab = _build_fill_vocab(loo_pool, max_fill_texts=args.max_fill_texts, seed=args.seed)
            if len(fill_vocab) < args.min_fill_vocab:
                fill_vocab = _build_fill_vocab(loo_pool, max_fill_texts=None, seed=args.seed)
        else:
            fill_vocab = global_fill_vocab or []

        for text in cat_to_texts[cat]:
            score = neighborhood_score(
                text=text,
                model=model,
                tok=tok,
                max_tokens=args.max_tokens,
                fill_vocab=fill_vocab,
                n_perturbations=args.n_perturbations,
                pct_words_masked=args.pct_words_masked,
                span_length=args.span_length,
                rng=rng,
            )
            cat_scores[cat].append(score)
            if np.isfinite(score):
                all_scores.append(score)
            pbar.update(1)
    pbar.close()

    # Determine threshold
    if args.threshold is not None:
        threshold = float(args.threshold)
        threshold_type = "specified"
        print(f"\nUsing specified threshold: {threshold}")
    else:
        threshold = float(np.median(all_scores)) if all_scores else 0.0
        threshold_type = "adaptive_median"
        print(f"\nUsing adaptive threshold (median of all scores): {threshold:.6f}")

    def _is_member(score: float) -> bool:
        if not np.isfinite(score):
            return False
        if args.member_if == "gt":
            return score >= threshold
        return score <= threshold

    # Apply threshold
    cat_members: Dict[str, int] = {c: 0 for c in categories}
    for cat in categories:
        for score in cat_scores[cat]:
            if _is_member(score):
                cat_members[cat] += 1

    # Per-category stats
    per_cat_results = []
    for cat in categories:
        scores = np.array(cat_scores[cat], dtype=float)
        n_total = int(scores.size)
        n_valid = int(np.isfinite(scores).sum())
        n_members = int(cat_members[cat])
        proportion = (n_members / n_valid) if n_valid > 0 else 0.0
        per_cat_results.append(
            {
                "category": cat,
                "n_total": n_total,
                "n_valid": n_valid,
                "n_members": n_members,
                "member_proportion": float(proportion),
                "score_mean": float(np.nanmean(scores)) if n_valid > 0 else None,
                "score_std": float(np.nanstd(scores)) if n_valid > 1 else None,
                "score_median": float(np.nanmedian(scores)) if n_valid > 0 else None,
            }
        )

    total_members = int(sum(cat_members.values()))
    if total_members > 0:
        global_mixture = {cat: float(cat_members[cat] / total_members) for cat in categories}
    else:
        global_mixture = {cat: 0.0 for cat in categories}

    print("\n" + "=" * 60)
    print("Per-Category Results:")
    print("=" * 60)
    for r in per_cat_results:
        score_str = f", score μ={r['score_mean']:.4f}" if r["score_mean"] is not None else ""
        print(
            f"  {r['category']:15s}: {r['n_members']:5d}/{r['n_valid']:5d} members "
            f"({r['member_proportion']:.2%}){score_str}"
        )

    print("\n" + "=" * 60)
    print("Predicted Category Composition (Global Mixture):")
    print("=" * 60)
    for cat in categories:
        print(f"  {cat:15s}: {global_mixture[cat]:.4f} ({global_mixture[cat]:.2%})")

    # Write outputs
    out_dir = Path(args.output_dir) / args.run_name
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "config": {
            "local_samples_dir": args.local_samples_dir,
            "merge_web": args.merge_web,
            "max_per_class": args.max_per_class,
            "seed": args.seed,
            "target_model": args.target_model,
            "hf_revision": args.hf_revision,
            "half": args.half,
            "int8": args.int8,
            "max_tokens": args.max_tokens,
            "n_perturbations": args.n_perturbations,
            "pct_words_masked": args.pct_words_masked,
            "span_length": args.span_length,
            "fill_strategy": args.fill_strategy,
            "max_fill_texts": args.max_fill_texts,
            "threshold": threshold,
            "threshold_type": threshold_type,
            "member_if": args.member_if,
        },
        "categories": categories,
        "per_category": per_cat_results,
        "global_mixture": global_mixture,
        "summary_stats": {
            "total_samples": int(total_samples),
            "total_members": int(total_members),
            "overall_member_rate": float(total_members / total_samples) if total_samples > 0 else 0.0,
            "threshold_used": float(threshold),
        },
    }

    with open(out_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    with open(out_dir / "per_category.csv", "w", encoding="utf-8") as f:
        f.write("category,n_total,n_valid,n_members,member_proportion,score_mean,score_std,score_median\n")
        for r in per_cat_results:
            f.write(
                f"{r['category']},{r['n_total']},{r['n_valid']},{r['n_members']},"
                f"{r['member_proportion']:.6f},"
                f"{r['score_mean'] if r['score_mean'] is not None else ''},"
                f"{r['score_std'] if r['score_std'] is not None else ''},"
                f"{r['score_median'] if r['score_median'] is not None else ''}\n"
            )

    with open(out_dir / "global_mixture.csv", "w", encoding="utf-8") as f:
        f.write("category,predicted_proportion\n")
        for cat in categories:
            f.write(f"{cat},{global_mixture[cat]:.6f}\n")

    print(f"\nWrote outputs to {out_dir}")


if __name__ == "__main__":
    main()


