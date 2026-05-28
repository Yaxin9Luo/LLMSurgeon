"""
Min-K%++ with Binary Thresholding for Category Composition Estimation

This script adapts Min-K%++ (a membership inference attack) to predict 
category composition percentages of LLM pretraining data.

Approach:
1. Load category samples from local_samples_dir (e.g., wikipedia.jsonl, books.jsonl, ...)
2. Score each sample with Min-K%++ 
3. Apply a threshold: score > threshold → member (1), else non-member (0)
4. For each category: proportion = count(members) / total_samples_in_category
5. Normalize across categories to get the predicted composition

The key assumption: the sampling data distribution represents the full 
pretraining data distribution.
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
from transformers import AutoTokenizer, AutoModelForCausalLM

from data_utils import detect_available_categories, _read_jsonl


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Estimate category mixture via Min-K%++ with binary thresholding"
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

    # Min-K++ parameters
    p.add_argument("--mink_ratio", type=float, default=0.2, help="k%% for Min-K++ (0<r<=1)")
    p.add_argument(
        "--threshold", 
        type=float, 
        default=None, 
        help="Threshold for binary classification. Score > threshold → member. "
             "If not specified, uses the median score as threshold (adaptive)."
    )

    # Output
    p.add_argument("--output_dir", type=str, default=str(Path(__file__).resolve().parents[3] / "out"))
    p.add_argument("--run_name", type=str, default="minkpp_threshold")
    
    return p.parse_args()


def load_model(name: str, revision: str = None, half: bool = False, int8: bool = False, trust_remote_code: bool = False):
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
        device_map='auto', 
        trust_remote_code=trust_remote_code,
        **int8_kwargs, 
        **half_kwargs,
        **revision_kwargs
    )
    model.eval()
    tok = AutoTokenizer.from_pretrained(name, trust_remote_code=trust_remote_code, **revision_kwargs)
    return model, tok


@torch.no_grad()
def minkpp_score(text: str, model, tok, max_tokens: int, ratio: float) -> float:
    """
    Compute Min-K%++ score for a text sample.
    
    Min-K%++ standardizes token log-probs by the model's expected distribution:
        z = (log_prob - μ) / σ
    where μ, σ are computed from the full vocab softmax.
    
    Then returns the mean of the smallest k% of z-scores.
    Higher score → more likely to be a member of training data.
    """
    # Tokenize with truncation
    ids = tok.encode(text, truncation=True, max_length=max_tokens)
    if len(ids) < 2:
        return float("nan")
    
    input_ids = torch.tensor(ids, dtype=torch.long).unsqueeze(0).to(model.device)
    out = model(input_ids, labels=input_ids)
    logits = out.logits[0, :-1]  # [T-1, V]
    target_ids = input_ids[0, 1:].unsqueeze(-1)  # [T-1, 1]

    probs = F.softmax(logits, dim=-1)
    log_probs = F.log_softmax(logits, dim=-1)
    token_log_probs = log_probs.gather(dim=-1, index=target_ids).squeeze(-1)  # [T-1]
    
    # Compute expected log-prob (μ) and variance (σ²)
    mu = (probs * log_probs).sum(-1)
    sigma = (probs * torch.square(log_probs)).sum(-1) - torch.square(mu)
    sigma = torch.clamp(sigma, min=1e-8)

    # Standardized z-scores
    z = (token_log_probs - mu) / torch.sqrt(sigma)
    
    # Take mean of smallest k% (Min-K%++)
    k = max(1, int(z.numel() * ratio))
    vals, _ = torch.sort(z)
    return float(vals[:k].mean().item())


def load_category_samples(
    local_dir: str, 
    merge_web: bool, 
    max_per_class: int = None,
    seed: int = 0
) -> Tuple[List[str], Dict[str, List[str]]]:
    """
    Load samples from each category JSONL file.
    
    Returns:
        categories: list of category names
        cat_to_texts: dict mapping category name to list of text samples
    """
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
    
    # Remove empty categories
    categories = [c for c in categories if cat_to_texts[c]]
    cat_to_texts = {c: cat_to_texts[c] for c in categories}
    
    return categories, cat_to_texts


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
    
    total_samples = sum(len(texts) for texts in cat_to_texts.values())
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
        trust_remote_code=args.trust_remote_code
    )
    
    # Score all samples
    print(f"\nScoring samples with Min-K%++ (ratio={args.mink_ratio})")
    
    cat_scores: Dict[str, List[float]] = {c: [] for c in categories}
    all_scores: List[float] = []
    
    pbar = tqdm(total=total_samples, desc="Scoring (Min-K%++)")
    for cat in categories:
        for text in cat_to_texts[cat]:
            score = minkpp_score(text, model, tok, args.max_tokens, args.mink_ratio)
            cat_scores[cat].append(score)
            if np.isfinite(score):
                all_scores.append(score)
            pbar.update(1)
    pbar.close()
    
    # Determine threshold
    if args.threshold is not None:
        threshold = args.threshold
        print(f"\nUsing specified threshold: {threshold}")
    else:
        # Use median as adaptive threshold
        threshold = float(np.median(all_scores))
        print(f"\nUsing adaptive threshold (median of all scores): {threshold:.6f}")
    
    # Apply threshold for binary classification
    cat_members: Dict[str, int] = {c: 0 for c in categories}
    for cat in categories:
        for score in cat_scores[cat]:
            if np.isfinite(score) and score > threshold:
                cat_members[cat] += 1
    
    # Compute per-category statistics
    per_cat_results = []
    for cat in categories:
        scores = np.array(cat_scores[cat])
        n_total = len(scores)
        n_valid = int(np.isfinite(scores).sum())
        n_members = cat_members[cat]
        
        # Proportion of members in this category
        proportion = n_members / n_valid if n_valid > 0 else 0.0
        
        per_cat_results.append({
            "category": cat,
            "n_total": n_total,
            "n_valid": n_valid,
            "n_members": n_members,
            "member_proportion": proportion,
            "score_mean": float(np.nanmean(scores)) if n_valid > 0 else None,
            "score_std": float(np.nanstd(scores)) if n_valid > 1 else None,
            "score_median": float(np.nanmedian(scores)) if n_valid > 0 else None,
        })
    
    # Compute global mixture (normalized proportions)
    # Method 1: Weight by member count (your proposed method)
    total_members = sum(cat_members.values())
    if total_members > 0:
        global_mixture_by_members = {
            cat: cat_members[cat] / total_members for cat in categories
        }
    else:
        global_mixture_by_members = {cat: 0.0 for cat in categories}
    
    # Method 2: Weight by (samples * proportion) to account for sampling distribution
    weighted_counts = [len(cat_to_texts[cat]) * (cat_members[cat] / len(cat_to_texts[cat]) if len(cat_to_texts[cat]) > 0 else 0) 
                       for cat in categories]
    total_weighted = sum(weighted_counts)
    if total_weighted > 0:
        global_mixture_weighted = {
            cat: w / total_weighted for cat, w in zip(categories, weighted_counts)
        }
    else:
        global_mixture_weighted = {cat: 0.0 for cat in categories}
    
    # Print results
    print("\n" + "=" * 60)
    print("Per-Category Results:")
    print("=" * 60)
    for r in per_cat_results:
        print(f"  {r['category']:15s}: {r['n_members']:5d}/{r['n_valid']:5d} members "
              f"({r['member_proportion']:.2%}), score μ={r['score_mean']:.4f}" if r['score_mean'] else "")
    
    print("\n" + "=" * 60)
    print("Predicted Category Composition (Global Mixture):")
    print("=" * 60)
    for cat in categories:
        print(f"  {cat:15s}: {global_mixture_by_members[cat]:.4f} ({global_mixture_by_members[cat]:.2%})")
    
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
            "mink_ratio": args.mink_ratio,
            "threshold": threshold,
            "threshold_type": "specified" if args.threshold is not None else "adaptive_median",
        },
        "categories": categories,
        "per_category": per_cat_results,
        "global_mixture": global_mixture_by_members,
        "global_mixture_weighted": global_mixture_weighted,
        "summary_stats": {
            "total_samples": total_samples,
            "total_members": total_members,
            "overall_member_rate": total_members / total_samples if total_samples > 0 else 0.0,
            "threshold_used": threshold,
        }
    }
    
    with open(out_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    
    # CSV: per-category results
    with open(out_dir / "per_category.csv", "w", encoding="utf-8") as f:
        f.write("category,n_total,n_valid,n_members,member_proportion,score_mean,score_std,score_median\n")
        for r in per_cat_results:
            f.write(f"{r['category']},{r['n_total']},{r['n_valid']},{r['n_members']},"
                    f"{r['member_proportion']:.6f},"
                    f"{r['score_mean'] if r['score_mean'] is not None else ''},"
                    f"{r['score_std'] if r['score_std'] is not None else ''},"
                    f"{r['score_median'] if r['score_median'] is not None else ''}\n")
    
    # CSV: global mixture
    with open(out_dir / "global_mixture.csv", "w", encoding="utf-8") as f:
        f.write("category,predicted_proportion\n")
        for cat in categories:
            f.write(f"{cat},{global_mixture_by_members[cat]:.6f}\n")
    
    print(f"\nWrote outputs to {out_dir}")


if __name__ == "__main__":
    main()
