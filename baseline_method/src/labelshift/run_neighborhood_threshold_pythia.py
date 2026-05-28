"""
Neighborhood (Perturbation Sensitivity) with Binary Thresholding for Category Composition Estimation (Pythia Taxonomy)

This script adapts the Neighborhood MIA signal (Mattern et al., ACL 2023) to predict
category composition percentages of LLM pretraining data using the Pythia/Pile taxonomy.

Score: score(x) = ll(x) - mean_j ll(perturb_j(x))
Higher score typically indicates more member-like (less robust to perturbations).
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

from data_utils_pythia import detect_available_pythia_categories, PILE_FILE_TO_PYTHIA_CAT
from data_utils import _read_jsonl


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Estimate category mixture via Neighborhood perturbation sensitivity (Pythia taxonomy)"
    )
    # Data
    p.add_argument("--local_samples_dir", type=str,
                   default=str(Path(__file__).resolve().parents[2] / "data_samples" / "pile"))
    p.add_argument("--max_per_class", type=int, default=None)
    p.add_argument("--seed", type=int, default=0)

    # Target model
    p.add_argument("--target_model", type=str, required=True)
    p.add_argument("--hf_revision", type=str, default=None)
    p.add_argument("--half", action="store_true")
    p.add_argument("--int8", action="store_true")
    p.add_argument("--max_tokens", type=int, default=512)
    p.add_argument("--trust_remote_code", action="store_true")

    # Neighborhood parameters
    p.add_argument("--n_perturbations", type=int, default=10)
    p.add_argument("--pct_words_masked", type=float, default=0.3)
    p.add_argument("--span_length", type=int, default=2)
    p.add_argument("--fill_strategy", type=str, choices=["global", "leave_one_out"], default="global")
    p.add_argument("--max_fill_texts", type=int, default=5000)
    p.add_argument("--min_fill_vocab", type=int, default=5000)

    # Thresholding
    p.add_argument("--threshold", type=float, default=None)
    p.add_argument("--member_if", type=str, default="gt", choices=["gt", "lt"])

    # Output
    p.add_argument("--output_dir", type=str, default=str(Path(__file__).resolve().parents[3] / "out"))
    p.add_argument("--run_name", type=str, default="neighborhood_threshold_pythia")
    return p.parse_args()


def load_model(name: str, revision: str = None, half: bool = False, int8: bool = False, trust_remote_code: bool = False):
    int8_kwargs = dict(load_in_8bit=True, torch_dtype=torch.bfloat16) if int8 else {}
    half_kwargs = dict(torch_dtype=torch.bfloat16) if half and not int8 else {}
    revision_kwargs = {"revision": revision} if revision else {}
    model = AutoModelForCausalLM.from_pretrained(
        name, return_dict=True, device_map="auto", trust_remote_code=trust_remote_code,
        **int8_kwargs, **half_kwargs, **revision_kwargs
    )
    model.eval()
    tok = AutoTokenizer.from_pretrained(name, trust_remote_code=trust_remote_code, **revision_kwargs)
    return model, tok


def load_category_samples_pythia(local_dir: str, max_per_class: int = None, seed: int = 0) -> Tuple[List[str], Dict[str, List[str]]]:
    rng = random.Random(seed)
    categories, file_to_cat = detect_available_pythia_categories(local_dir)
    if not categories:
        raise FileNotFoundError(f"No Pythia/Pile files under {local_dir}")
    cat_to_texts: Dict[str, List[str]] = {c: [] for c in categories}
    for fname, cat in file_to_cat.items():
        fpath = os.path.join(local_dir, fname)
        if not os.path.exists(fpath):
            continue
        texts = _read_jsonl(fpath)
        rng.shuffle(texts)
        if max_per_class:
            texts = texts[:max_per_class]
        cat_to_texts[cat].extend(texts)
    categories = [c for c in categories if cat_to_texts.get(c)]
    cat_to_texts = {c: cat_to_texts[c] for c in categories}
    return categories, cat_to_texts


def _build_fill_vocab(texts: List[str], max_fill_texts: int, seed: int) -> List[str]:
    rng = random.Random(seed)
    if max_fill_texts and len(texts) > max_fill_texts:
        texts = rng.sample(texts, k=max_fill_texts)
    vocab: set = set()
    for t in texts:
        for w in t.split():
            w = w.strip()
            if w:
                vocab.add(w)
    return sorted(vocab)


def _random_perturb_words(words: List[str], pct: float, span_length: int, fill_vocab: List[str], rng: random.Random) -> List[str]:
    if not words or not (0.0 < pct <= 1.0) or span_length <= 0 or not fill_vocab:
        return words
    n_words = len(words)
    n_replace = max(1, int(round(pct * n_words)))
    n_spans = max(1, int(np.ceil(n_replace / span_length)))
    out = words[:]
    for _ in range(n_spans):
        start = rng.randrange(0, n_words)
        end = min(n_words, start + span_length)
        for i in range(start, end):
            out[i] = fill_vocab[rng.randrange(len(fill_vocab))]
    return out


@torch.no_grad()
def avg_ll(text: str, model, tok, max_tokens: int) -> float:
    ids = tok.encode(text, truncation=True, max_length=max_tokens)
    if len(ids) < 2:
        return float("nan")
    input_ids = torch.tensor(ids, dtype=torch.long).unsqueeze(0).to(model.device)
    out = model(input_ids, labels=input_ids)
    return float((-out.loss.float()).item())


@torch.no_grad()
def neighborhood_score(text: str, model, tok, max_tokens: int, fill_vocab: List[str],
                       n_perturbations: int, pct_words_masked: float, span_length: int, rng: random.Random) -> float:
    ll0 = avg_ll(text, model, tok, max_tokens)
    if not np.isfinite(ll0):
        return float("nan")
    if n_perturbations <= 0:
        return float(ll0)
    words = text.split()
    if len(words) < 2:
        return float("nan")
    lls: List[float] = []
    for _ in range(n_perturbations):
        pert_words = _random_perturb_words(words, pct_words_masked, span_length, fill_vocab, rng)
        pert_text = " ".join(pert_words)
        llp = avg_ll(pert_text, model, tok, max_tokens)
        if np.isfinite(llp):
            lls.append(llp)
    if not lls:
        return float("nan")
    return float(ll0 - float(np.mean(lls)))


def main() -> None:
    args = parse_args()
    rng = random.Random(args.seed)

    print(f"Loading Pythia/Pile samples from: {args.local_samples_dir}")
    categories, cat_to_texts = load_category_samples_pythia(
        args.local_samples_dir, args.max_per_class, args.seed
    )
    total_samples = sum(len(v) for v in cat_to_texts.values())
    print(f"Categories ({len(categories)}): {categories}")
    print(f"Total samples: {total_samples}")

    print(f"Loading model: {args.target_model}")
    model, tok = load_model(args.target_model, args.hf_revision, args.half, args.int8, args.trust_remote_code)

    pooled_all: List[str] = [t for c in categories for t in cat_to_texts[c]]

    global_fill_vocab: List[str] | None = None
    if args.fill_strategy == "global":
        fill_vocab = _build_fill_vocab(pooled_all, args.max_fill_texts, args.seed)
        if len(fill_vocab) < args.min_fill_vocab:
            fill_vocab = _build_fill_vocab(pooled_all, None, args.seed)
        global_fill_vocab = fill_vocab
        print(f"Fill strategy: global | vocab_size={len(global_fill_vocab)}")
    else:
        print(f"Fill strategy: leave_one_out")

    print(f"\nScoring with Neighborhood (n_pert={args.n_perturbations}, pct={args.pct_words_masked})")
    cat_scores: Dict[str, List[float]] = {c: [] for c in categories}
    all_scores: List[float] = []

    pbar = tqdm(total=total_samples, desc="Scoring (Neighborhood)")
    for cat in categories:
        if args.fill_strategy == "leave_one_out":
            loo_pool = [t for c2 in categories if c2 != cat for t in cat_to_texts[c2]]
            fill_vocab = _build_fill_vocab(loo_pool, args.max_fill_texts, args.seed)
            if len(fill_vocab) < args.min_fill_vocab:
                fill_vocab = _build_fill_vocab(loo_pool, None, args.seed)
        else:
            fill_vocab = global_fill_vocab or []

        for text in cat_to_texts[cat]:
            score = neighborhood_score(text, model, tok, args.max_tokens, fill_vocab,
                                        args.n_perturbations, args.pct_words_masked, args.span_length, rng)
            cat_scores[cat].append(score)
            if np.isfinite(score):
                all_scores.append(score)
            pbar.update(1)
    pbar.close()

    if args.threshold is not None:
        threshold = float(args.threshold)
        threshold_type = "specified"
    else:
        threshold = float(np.median(all_scores)) if all_scores else 0.0
        threshold_type = "adaptive_median"
    print(f"\nUsing threshold: {threshold:.6f} ({threshold_type})")

    def _is_member(score: float) -> bool:
        if not np.isfinite(score):
            return False
        return score >= threshold if args.member_if == "gt" else score <= threshold

    cat_members = {c: sum(1 for s in cat_scores[c] if _is_member(s)) for c in categories}

    per_cat_results = []
    for cat in categories:
        scores = np.array(cat_scores[cat], dtype=float)
        n_valid = int(np.isfinite(scores).sum())
        n_members = cat_members[cat]
        per_cat_results.append({
            "category": cat, "n_total": int(scores.size), "n_valid": n_valid, "n_members": n_members,
            "member_proportion": float(n_members / n_valid) if n_valid > 0 else 0.0,
            "score_mean": float(np.nanmean(scores)) if n_valid > 0 else None,
            "score_std": float(np.nanstd(scores)) if n_valid > 1 else None,
            "score_median": float(np.nanmedian(scores)) if n_valid > 0 else None,
        })

    total_members = sum(cat_members.values())
    global_mixture = {cat: float(cat_members[cat] / total_members) for cat in categories} if total_members > 0 else {cat: 0.0 for cat in categories}

    print("\n" + "=" * 70)
    print("Per-Category Results (Pythia Taxonomy):")
    print("=" * 70)
    for r in per_cat_results:
        score_str = f", score μ={r['score_mean']:.4f}" if r["score_mean"] is not None else ""
        print(f"  {r['category']:20s}: {r['n_members']:5d}/{r['n_valid']:5d} members ({r['member_proportion']:.2%}){score_str}")

    print("\n" + "=" * 70)
    print("Predicted Category Composition (Global Mixture):")
    print("=" * 70)
    for cat in categories:
        print(f"  {cat:20s}: {global_mixture[cat]:.4f} ({global_mixture[cat]:.2%})")

    out_dir = Path(args.output_dir) / args.run_name
    out_dir.mkdir(parents=True, exist_ok=True)

    payload = {
        "config": {
            "local_samples_dir": args.local_samples_dir, "max_per_class": args.max_per_class, "seed": args.seed,
            "target_model": args.target_model, "hf_revision": args.hf_revision,
            "n_perturbations": args.n_perturbations, "pct_words_masked": args.pct_words_masked,
            "span_length": args.span_length, "fill_strategy": args.fill_strategy,
            "threshold": threshold, "threshold_type": threshold_type, "member_if": args.member_if,
            "taxonomy": "pythia",
        },
        "categories": categories, "per_category": per_cat_results, "global_mixture": global_mixture,
        "summary_stats": {"total_samples": total_samples, "total_members": total_members, "threshold_used": threshold},
    }

    with open(out_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    with open(out_dir / "per_category.csv", "w", encoding="utf-8") as f:
        f.write("category,n_total,n_valid,n_members,member_proportion,score_mean,score_std,score_median\n")
        for r in per_cat_results:
            f.write(f"{r['category']},{r['n_total']},{r['n_valid']},{r['n_members']},{r['member_proportion']:.6f},"
                    f"{r['score_mean'] if r['score_mean'] is not None else ''},"
                    f"{r['score_std'] if r['score_std'] is not None else ''},"
                    f"{r['score_median'] if r['score_median'] is not None else ''}\n")
    with open(out_dir / "global_mixture.csv", "w", encoding="utf-8") as f:
        f.write("category,predicted_proportion\n")
        for cat in categories:
            f.write(f"{cat},{global_mixture[cat]:.6f}\n")

    print(f"\nWrote outputs to {out_dir}")


if __name__ == "__main__":
    main()

