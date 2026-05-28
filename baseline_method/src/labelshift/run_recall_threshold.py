"""
ReCaLL with Binary Thresholding for Category Composition Estimation

This script adapts ReCaLL (Xie et al., EMNLP 2024) — a membership inference attack —
to predict category composition percentages of LLM pretraining data.

High-level approach (mirrors run_mink_threshold.py):
1) Load category samples from local_samples_dir (e.g., wikipedia.jsonl, books.jsonl, ...)
2) Score each sample with ReCaLL:
     score = ll_cond(prefix, x) / ll(x)
   where ll is the (average) unconditional log-likelihood and ll_cond is the (average)
   conditional log-likelihood of x given a fixed prefix made from non-member-like texts.
3) Threshold: score > threshold -> member (1), else non-member (0)
4) For each category: member_proportion = #members / #valid
5) Normalize member counts across categories to get predicted composition.

Notes:
- This does NOT require a reference model (unlike some other MIA baselines).
- Prefix construction is configurable; by default we use a global prefix sampled from the
  pooled category texts. You can also use leave-one-out (exclude the category being scored).
"""

import argparse
import json
import os
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from tqdm import tqdm

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from data_utils import detect_available_categories, _read_jsonl


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Estimate category mixture via ReCaLL with binary thresholding"
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
    p.add_argument("--max_tokens", type=int, default=512, help="Max total tokens for prefix+sample scoring")
    p.add_argument("--trust_remote_code", action="store_true", help="Trust remote code for models like OLMo")

    # ReCaLL parameters
    p.add_argument("--num_shots", type=int, default=12, help="Number of prefix shots used for conditional LL")
    p.add_argument(
        "--prefix_strategy",
        type=str,
        choices=["global", "leave_one_out"],
        default="global",
        help="How to build the prefix: global (one shared prefix), or leave_one_out (exclude category being scored).",
    )
    p.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="Threshold for binary classification. Score > threshold -> member. "
             "If not specified, uses the median score as threshold (adaptive).",
    )
    p.add_argument(
        "--member_if",
        type=str,
        default="gt",
        choices=["gt", "lt"],
        help="Membership rule: gt means score > threshold => member; lt means score < threshold => member.",
    )

    # Output
    p.add_argument("--output_dir", type=str, default=str(Path(__file__).resolve().parents[3] / "out"))
    p.add_argument("--run_name", type=str, default="recall_threshold")
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


def _tok_encode(tok, text: str, max_len: int) -> List[int]:
    # Use encode() to stay consistent with other scripts in this repo.
    return tok.encode(text, truncation=True, max_length=max_len)


def _build_prefix_text(prefix_shots: List[str]) -> str:
    # A light separator reduces accidental token-merge between shots.
    return "\n\n".join(prefix_shots)


@torch.no_grad()
def _avg_ll_unconditional(text: str, model, tok, max_tokens: int) -> float:
    """Average log-likelihood (negative of HF loss) for the sample."""
    ids = _tok_encode(tok, text, max_tokens)
    if len(ids) < 2:
        return float("nan")
    input_ids = torch.tensor(ids, dtype=torch.long).unsqueeze(0).to(model.device)
    out = model(input_ids, labels=input_ids)
    loss = out.loss
    return float((-loss).item())


@torch.no_grad()
def _avg_ll_conditional(prefix_text: str, target_text: str, model, tok, max_tokens: int) -> float:
    """
    Average conditional log-likelihood of target_text given prefix_text.
    We mask prefix tokens in the loss, as in the original ReCaLL code.
    """
    # Tokenize target first (so we can guarantee it fits).
    tgt_ids = _tok_encode(tok, target_text, max_tokens)
    if len(tgt_ids) < 2:
        return float("nan")

    # Keep some room for prefix; if target itself is too long, truncate it.
    # Note: HF causal LM loss ignores the first token effectively; requiring len>=2 above.
    max_prefix_len = max(0, max_tokens - len(tgt_ids))
    pre_ids_full = _tok_encode(tok, prefix_text, max_prefix_len) if max_prefix_len > 0 else []

    # If still over (shouldn't be), trim prefix from the left (keep suffix).
    if len(pre_ids_full) + len(tgt_ids) > max_tokens:
        keep = max(0, max_tokens - len(tgt_ids))
        pre_ids = pre_ids_full[-keep:] if keep > 0 else []
    else:
        pre_ids = pre_ids_full

    concat_ids = pre_ids + tgt_ids
    labels = ([-100] * len(pre_ids)) + tgt_ids

    input_ids = torch.tensor(concat_ids, dtype=torch.long).unsqueeze(0).to(model.device)
    label_ids = torch.tensor(labels, dtype=torch.long).unsqueeze(0).to(model.device)
    out = model(input_ids, labels=label_ids)
    return float((-out.loss).item())


@torch.no_grad()
def recall_score(text: str, prefix_text: str, model, tok, max_tokens: int) -> float:
    """Compute ReCaLL score = ll_cond(prefix, x) / ll(x) using average log-likelihoods."""
    ll = _avg_ll_unconditional(text, model, tok, max_tokens=max_tokens)
    if not np.isfinite(ll) or ll == 0.0:
        return float("nan")
    llc = _avg_ll_conditional(prefix_text, text, model, tok, max_tokens=max_tokens)
    if not np.isfinite(llc):
        return float("nan")
    return float(llc / ll)


def _sample_prefix_shots(pool: List[str], num_shots: int, seed: int) -> List[str]:
    import random

    if num_shots <= 0:
        return []
    rng = random.Random(seed)
    if len(pool) == 0:
        return []
    if len(pool) >= num_shots:
        return rng.sample(pool, k=num_shots)
    # If pool smaller than requested shots, sample with replacement.
    return [pool[rng.randrange(len(pool))] for _ in range(num_shots)]


def main() -> None:
    args = parse_args()

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

    # Load target model
    print(f"Loading model: {args.target_model}" + (f" @ {args.hf_revision}" if args.hf_revision else ""))
    model, tok = load_model(
        args.target_model,
        revision=args.hf_revision,
        half=args.half,
        int8=args.int8,
        trust_remote_code=args.trust_remote_code,
    )

    # Build prefix pools
    pooled_all: List[str] = []
    for c in categories:
        pooled_all.extend(cat_to_texts[c])

    global_prefix_text = ""
    if args.prefix_strategy == "global":
        prefix_shots = _sample_prefix_shots(pooled_all, args.num_shots, seed=args.seed)
        global_prefix_text = _build_prefix_text(prefix_shots)
        print(f"Prefix strategy: global | num_shots={args.num_shots} | pooled={len(pooled_all)}")
    else:
        print(f"Prefix strategy: leave_one_out | num_shots={args.num_shots} | pooled={len(pooled_all)}")

    # Score all samples
    print("\nScoring samples with ReCaLL")
    cat_scores: Dict[str, List[float]] = {c: [] for c in categories}
    all_scores: List[float] = []

    pbar = tqdm(total=total_samples, desc="Scoring (ReCaLL)")
    for cat in categories:
        if args.prefix_strategy == "leave_one_out":
            loo_pool: List[str] = []
            for c2 in categories:
                if c2 != cat:
                    loo_pool.extend(cat_to_texts[c2])
            prefix_shots = _sample_prefix_shots(loo_pool, args.num_shots, seed=args.seed)
            prefix_text = _build_prefix_text(prefix_shots)
        else:
            prefix_text = global_prefix_text

        for text in cat_to_texts[cat]:
            score = recall_score(text, prefix_text, model, tok, max_tokens=args.max_tokens)
            cat_scores[cat].append(score)
            if np.isfinite(score):
                all_scores.append(score)
            pbar.update(1)
    pbar.close()

    # Determine threshold
    if args.threshold is not None:
        threshold = float(args.threshold)
        print(f"\nUsing specified threshold: {threshold}")
        threshold_type = "specified"
    else:
        threshold = float(np.median(all_scores)) if all_scores else 0.0
        print(f"\nUsing adaptive threshold (median of all scores): {threshold:.6f}")
        threshold_type = "adaptive_median"

    def _is_member(score: float) -> bool:
        if not np.isfinite(score):
            return False
        if args.member_if == "gt":
            return score >= threshold
        return score <= threshold

    # Apply threshold for binary classification
    cat_members: Dict[str, int] = {c: 0 for c in categories}
    for cat in categories:
        for score in cat_scores[cat]:
            if _is_member(score):
                cat_members[cat] += 1

    # Compute per-category statistics
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

    # Compute global mixture (normalized member counts)
    total_members = int(sum(cat_members.values()))
    if total_members > 0:
        global_mixture = {cat: float(cat_members[cat] / total_members) for cat in categories}
    else:
        global_mixture = {cat: 0.0 for cat in categories}

    # Print results
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

    # Write outputs (match benchmark_evaluation.py expectations)
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
            "num_shots": args.num_shots,
            "prefix_strategy": args.prefix_strategy,
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

    # CSV: per-category results
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

    # CSV: global mixture
    with open(out_dir / "global_mixture.csv", "w", encoding="utf-8") as f:
        f.write("category,predicted_proportion\n")
        for cat in categories:
            f.write(f"{cat},{global_mixture[cat]:.6f}\n")

    print(f"\nWrote outputs to {out_dir}")


if __name__ == "__main__":
    main()


