"""
Zlib with Binary Thresholding for Category Composition Estimation

This script adapts a zlib-based MIA signal to predict category composition percentages
of LLM pretraining data, following the same pipeline used for Mink/ReCaLL/DC-PDD:

1) Load category samples from local_samples_dir
2) Score each sample with a zlib-based signal
3) Threshold -> member/non-member
4) Normalize member counts across categories -> global mixture

Supported modes:
- zlib_len (default): score = len(zlib.compress(text_bytes))
  Typically, "more compressible" => smaller score; you can choose member_if=lt or gt.
- ll_over_zlib: score = ll(text) / zlib_len(text)
  This matches the baseline used in `recall/src/run.py` (log-likelihood divided by zlib length).
  Requires --target_model.
"""

import argparse
import json
import os
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from tqdm import tqdm

import zlib

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from data_utils import detect_available_categories, _read_jsonl


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Estimate category mixture via zlib with binary thresholding"
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

    # Zlib mode
    p.add_argument(
        "--mode",
        type=str,
        choices=["zlib_len", "ll_over_zlib"],
        default="zlib_len",
        help="Which zlib-based signal to use.",
    )
    p.add_argument(
        "--zlib_level",
        type=int,
        default=6,
        help="zlib compression level (0-9). Only affects --mode zlib_len / ll_over_zlib denominator.",
    )

    # Target model (only required for ll_over_zlib)
    p.add_argument("--target_model", type=str, default=None, help="HF model id for scoring (causal LM)")
    p.add_argument("--hf_revision", type=str, default=None, help="HF model revision/checkpoint")
    p.add_argument("--half", action="store_true", help="Load model in bfloat16")
    p.add_argument("--int8", action="store_true", help="Load model in 8-bit (bitsandbytes)")
    p.add_argument("--max_tokens", type=int, default=512, help="Max tokens per sample for scoring")
    p.add_argument("--trust_remote_code", action="store_true", help="Trust remote code for models like OLMo")

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
        default=None,
        choices=["gt", "lt"],
        help="Membership rule: gt means score >= threshold => member; lt means score <= threshold => member. "
             "If omitted, defaults to lt for zlib_len and gt for ll_over_zlib.",
    )

    # Output
    p.add_argument("--output_dir", type=str, default=str(Path(__file__).resolve().parents[3] / "out"))
    p.add_argument("--run_name", type=str, default="zlib_threshold")
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


def zlib_len(text: str, level: int = 6) -> float:
    b = text.encode("utf-8", errors="ignore")
    return float(len(zlib.compress(b, level=level)))


@torch.no_grad()
def avg_ll(text: str, model, tok, max_tokens: int) -> float:
    """Average log-likelihood (negative HF loss)."""
    ids = tok.encode(text, truncation=True, max_length=max_tokens)
    if len(ids) < 2:
        return float("nan")
    input_ids = torch.tensor(ids, dtype=torch.long).unsqueeze(0).to(model.device)
    out = model(input_ids, labels=input_ids)
    return float((-out.loss.float()).item())


@torch.no_grad()
def ll_over_zlib(text: str, model, tok, max_tokens: int, level: int) -> float:
    ll = avg_ll(text, model, tok, max_tokens=max_tokens)
    if not np.isfinite(ll):
        return float("nan")
    zl = zlib_len(text, level=level)
    if zl <= 0:
        return float("nan")
    return float(ll / zl)


def main() -> None:
    args = parse_args()

    # Default direction if unspecified
    if args.member_if is None:
        member_if = "lt" if args.mode == "zlib_len" else "gt"
    else:
        member_if = args.member_if

    # Load category samples
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

    model = tok = None
    if args.mode == "ll_over_zlib":
        if not args.target_model:
            raise ValueError("--target_model is required when --mode ll_over_zlib")
        print(f"Loading model: {args.target_model}" + (f" @ {args.hf_revision}" if args.hf_revision else ""))
        model, tok = load_model(
            args.target_model,
            revision=args.hf_revision,
            half=args.half,
            int8=args.int8,
            trust_remote_code=args.trust_remote_code,
        )

    # Score all samples
    print(f"\nScoring samples with zlib mode={args.mode} (member_if={member_if})")
    cat_scores: Dict[str, List[float]] = {c: [] for c in categories}
    all_scores: List[float] = []

    pbar = tqdm(total=total_samples, desc=f"Scoring (zlib:{args.mode})")
    for cat in categories:
        for text in cat_to_texts[cat]:
            if args.mode == "zlib_len":
                score = zlib_len(text, level=args.zlib_level)
            else:
                score = ll_over_zlib(text, model, tok, max_tokens=args.max_tokens, level=args.zlib_level)
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
        if member_if == "gt":
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
            "mode": args.mode,
            "zlib_level": args.zlib_level,
            "member_if": member_if,
            "threshold": threshold,
            "threshold_type": threshold_type,
            "target_model": args.target_model,
            "hf_revision": args.hf_revision,
            "half": args.half,
            "int8": args.int8,
            "max_tokens": args.max_tokens,
            "trust_remote_code": args.trust_remote_code,
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


