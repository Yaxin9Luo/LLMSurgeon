"""
DC-PDD with Binary Thresholding for Category Composition Estimation

This script adapts DC-PDD (Zhang et al., 2024) — a membership inference signal —
to predict category composition percentages of LLM pretraining data.

We follow the same adaptation pattern as run_mink_threshold.py / run_recall_threshold.py:
1) Load category samples from local_samples_dir (e.g., wikipedia.jsonl, books.jsonl, ...)
2) Build a token-frequency distribution f(t) from a reference corpus.
   - In the original paper/code, this is computed from a large public corpus.
   - Here (for mixture estimation), we use the pooled category samples as the reference
     corpus (configurable as global vs leave-one-out).
3) For each sample x, compute DC-PDD score:
     score(x) = - mean_i [ p_i * log(1 / f(tok_i)) ]  (with clipping at a)
   where p_i is the target model probability assigned to the observed token tok_i at
   position i (next-token probability), and the mean is taken over the *first occurrence*
   positions of each token id in the sample (as in the original DC-PDD code).
4) Threshold: score <= threshold -> member (1), else non-member (0)
5) Normalize member counts across categories to get predicted composition.

Important:
- In the original DC-PDD evaluation, they use roc_curve(labels, -score), i.e. members
  typically correspond to *lower* (more negative) score. Therefore default member rule
  here is: score <= threshold => member (member_if=lt).
"""

import argparse
import json
import os
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from tqdm import tqdm

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

from data_utils import detect_available_categories, _read_jsonl


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Estimate category mixture via DC-PDD with binary thresholding"
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

    # DC-PDD parameters
    p.add_argument("--a", type=float, default=0.01, help="DC-PDD clipping hyperparameter (a in the paper/code)")
    p.add_argument(
        "--freq_strategy",
        type=str,
        choices=["global", "leave_one_out"],
        default="global",
        help="How to build token frequency distribution: global pooled corpus, or leave-one-out per category.",
    )
    p.add_argument(
        "--max_ref_texts",
        type=int,
        default=None,
        help="Optional cap on #reference texts used to estimate token frequency (after pooling).",
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
        default="lt",
        choices=["gt", "lt"],
        help="Membership rule: lt means score <= threshold => member; gt means score >= threshold => member.",
    )

    # Output
    p.add_argument("--output_dir", type=str, default=str(Path(__file__).resolve().parents[3] / "out"))
    p.add_argument("--run_name", type=str, default="dcpdd_threshold")
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
    import random

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


def _vocab_size(tok) -> int:
    # len(tokenizer) includes added tokens; safest for allocating count arrays.
    try:
        return len(tok)
    except Exception:
        vs = getattr(tok, "vocab_size", None)
        return int(vs) if vs is not None else 0


def _sample_pool(texts: List[str], max_n: int, seed: int) -> List[str]:
    if max_n is None or max_n <= 0 or len(texts) <= max_n:
        return texts
    import random

    rng = random.Random(seed)
    return rng.sample(texts, k=max_n)


def _freq_distribution_from_texts(
    tok,
    texts: List[str],
    max_tokens: int,
    seed: int,
    max_ref_texts: int = None,
) -> np.ndarray:
    """
    Estimate token frequency distribution f over the tokenizer vocabulary.
    Uses add-one smoothing: (count+1)/(total+V).
    """
    vs = _vocab_size(tok)
    if vs <= 0:
        raise ValueError("Could not determine tokenizer vocab size for frequency distribution.")
    counts = np.zeros((vs,), dtype=np.int64)

    texts_use = _sample_pool(texts, max_ref_texts, seed=seed)
    for t in texts_use:
        ids = tok.encode(t, truncation=True, max_length=max_tokens)
        for tid in ids:
            if 0 <= tid < vs:
                counts[tid] += 1

    total = int(counts.sum())
    # Add-one smoothing to avoid log(0)
    fre = (counts.astype(np.float64) + 1.0) / (float(total) + float(vs))
    return fre


@torch.no_grad()
def dcpdd_score(
    text: str,
    model,
    tok,
    fre: np.ndarray,
    max_tokens: int,
    a: float,
) -> float:
    """
    Compute DC-PDD score for a single text (higher/lower depends on member_if).

    We follow the reference implementation:
    - Use next-token probabilities for the observed tokens (input_ids[1:])
    - Select only the first occurrence positions of each token id
    - Compute ce_i = p_i * log(1 / f(token_id))
    - Clip ce_i at a
    - Return score = -mean(ce_i)
    """
    ids = tok.encode(text, truncation=True, max_length=max_tokens)
    if len(ids) < 2:
        return float("nan")

    input_ids = torch.tensor(ids, dtype=torch.long).unsqueeze(0).to(model.device)
    out = model(input_ids, labels=input_ids)
    logits = out.logits[0, :-1]  # [T-1, V]
    target_ids = input_ids[0, 1:]  # [T-1]

    log_probs = F.log_softmax(logits, dim=-1)
    token_log_probs = log_probs[torch.arange(logits.size(0), device=logits.device), target_ids]
    # If model runs in bf16/fp16, NumPy conversion may fail; cast to float32 first.
    probs = token_log_probs.float().exp().detach().cpu().numpy()  # p_i
    ids_shift = target_ids.detach().cpu().numpy().astype(np.int64)  # token ids aligned with probs

    # first-occurrence positions
    seen = set()
    idxs: List[int] = []
    for i, tid in enumerate(ids_shift.tolist()):
        if tid not in seen:
            idxs.append(i)
            seen.add(tid)
    if not idxs:
        return float("nan")

    x_pro = probs[idxs]
    # Guard: if tokenizer ids exceed fre length, treat as tiny frequency via smoothing floor
    if ids_shift.max(initial=0) >= fre.shape[0]:
        # Expand fre minimally (rare for HF tokenizers) with a small epsilon mass.
        vs_new = int(ids_shift.max()) + 1
        fre2 = np.full((vs_new,), 1.0 / float(vs_new), dtype=np.float64)
        fre2[: fre.shape[0]] = fre
        fre_use = fre2
    else:
        fre_use = fre
    x_fre = fre_use[ids_shift[idxs]]

    # ce = p * log(1/f)
    ce = x_pro * np.log(1.0 / np.clip(x_fre, 1e-300, 1.0))
    ce = np.minimum(ce, float(a))
    return float(-np.mean(ce))


def main() -> None:
    args = parse_args()

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

    # Build reference corpus pools
    pooled_all: List[str] = []
    for c in categories:
        pooled_all.extend(cat_to_texts[c])

    global_fre: np.ndarray | None = None
    if args.freq_strategy == "global":
        print(
            f"Estimating token frequency (global): max_ref_texts={args.max_ref_texts}, "
            f"max_tokens={args.max_tokens}, pooled_texts={len(pooled_all)}"
        )
        global_fre = _freq_distribution_from_texts(
            tok,
            pooled_all,
            max_tokens=args.max_tokens,
            seed=args.seed,
            max_ref_texts=args.max_ref_texts,
        )
    else:
        print(
            f"Estimating token frequency (leave_one_out): max_ref_texts={args.max_ref_texts}, "
            f"max_tokens={args.max_tokens}, pooled_texts={len(pooled_all)}"
        )

    print(f"\nScoring samples with DC-PDD (a={args.a})")
    cat_scores: Dict[str, List[float]] = {c: [] for c in categories}
    all_scores: List[float] = []

    pbar = tqdm(total=total_samples, desc="Scoring (DC-PDD)")
    for cat in categories:
        if args.freq_strategy == "leave_one_out":
            loo_pool: List[str] = []
            for c2 in categories:
                if c2 != cat:
                    loo_pool.extend(cat_to_texts[c2])
            fre = _freq_distribution_from_texts(
                tok,
                loo_pool,
                max_tokens=args.max_tokens,
                seed=args.seed,
                max_ref_texts=args.max_ref_texts,
            )
        else:
            fre = global_fre
        assert fre is not None

        for text in cat_to_texts[cat]:
            score = dcpdd_score(
                text=text,
                model=model,
                tok=tok,
                fre=fre,
                max_tokens=args.max_tokens,
                a=args.a,
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

    # Global mixture = normalized member counts
    total_members = int(sum(cat_members.values()))
    if total_members > 0:
        global_mixture = {cat: float(cat_members[cat] / total_members) for cat in categories}
    else:
        global_mixture = {cat: 0.0 for cat in categories}

    # Print
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
            "a": args.a,
            "freq_strategy": args.freq_strategy,
            "max_ref_texts": args.max_ref_texts,
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


