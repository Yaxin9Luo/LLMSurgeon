"""
DUCI (Dataset Usage Cardinality Inference) for Category Composition Estimation

This script applies the DUCI methodology to estimate the proportion of each category
(e.g., CommonCrawl, GitHub, Wikipedia, Books, Arxiv, StackExchange, C4) used in 
training an LLM.

Key Steps:
1. Isolate Target Datasets - Treat each category as an independent inference problem
2. Run MIA for each category - Using loss-based or RMIA-style signals
3. Estimate TPR/FPR per category - Using reference model signals or cross-validation
4. Apply Debiasing - Correct MIA predictions using category-specific TPR/FPR
5. Aggregate - Average debiased probabilities to get per-category proportions

Formula for debiasing:
    p̂_i = (m̂_i - FPR) / (TPR - FPR)
    
Where m̂_i is the binary MIA prediction (1 if score > threshold, 0 otherwise).

Reference: "How Much of My Dataset Did You Use?" (Tong et al., ICLR 2025)
"""

import argparse
import json
import os
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass

import numpy as np
from tqdm import tqdm
from sklearn.metrics import roc_curve, auc

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM

from data_utils import detect_available_categories, _read_jsonl


@dataclass
class CategoryStats:
    """Statistics for a single category."""
    category: str
    n_samples: int
    n_valid: int
    scores: np.ndarray
    threshold: float
    tpr: float
    fpr: float
    raw_member_rate: float  # Before debiasing
    debiased_proportion: float  # After debiasing (DUCI estimate)
    score_mean: float
    score_std: float


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="DUCI-based category composition estimation for LLM pretraining data"
    )
    # Data
    p.add_argument(
        "--local_samples_dir",
        type=str,
        default=str(Path(__file__).resolve().parents[3] / "data_samples"),
        help="Directory with category JSONL files",
    )
    p.add_argument("--merge_web", action="store_true", help="Merge CommonCrawl and C4 into Web")
    p.add_argument("--max_per_class", type=int, default=5000, help="Max samples per category")
    p.add_argument("--seed", type=int, default=0)

    # Target model (the model being audited)
    p.add_argument("--target_model", type=str, required=True, help="HF model to audit")
    p.add_argument("--hf_revision", type=str, default=None)
    p.add_argument("--half", action="store_true", help="Load in bfloat16")
    p.add_argument("--int8", action="store_true", help="Load in 8-bit")
    p.add_argument("--max_tokens", type=int, default=512)
    p.add_argument("--trust_remote_code", action="store_true")

    # Reference model (for TPR/FPR estimation - can be same architecture, different weights)
    p.add_argument(
        "--reference_model", 
        type=str, 
        default=None,
        help="Reference model for TPR/FPR calibration. If None, uses cross-validation on target model."
    )
    p.add_argument("--ref_revision", type=str, default=None)

    # MIA parameters
    p.add_argument("--mia_method", type=str, choices=["loss", "mink", "minkpp"], default="loss",
                   help="MIA signal type: loss (perplexity), mink (Min-K%%), minkpp (Min-K%%++)")
    p.add_argument("--mink_ratio", type=float, default=0.2, help="k%% for Min-K methods")
    
    # DUCI parameters
    p.add_argument("--calibration_split", type=float, default=0.3,
                   help="Fraction of samples per category used for TPR/FPR calibration")
    p.add_argument("--threshold_method", type=str, choices=["median", "optimal", "fixed"], default="optimal",
                   help="How to determine the MIA threshold")
    p.add_argument("--fixed_threshold", type=float, default=None,
                   help="Fixed threshold value (only used if threshold_method=fixed)")

    # Output
    p.add_argument("--output_dir", type=str, default=str(Path(__file__).resolve().parents[3] / "out"))
    p.add_argument("--run_name", type=str, default="duci_categories")
    
    return p.parse_args()


def load_model(
    name: str, 
    revision: str = None, 
    half: bool = False, 
    int8: bool = False, 
    trust_remote_code: bool = False
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
def compute_mia_signal(
    text: str, 
    model, 
    tok, 
    max_tokens: int, 
    method: str = "loss",
    mink_ratio: float = 0.2
) -> float:
    """
    Compute MIA signal for a text sample.
    
    Higher scores indicate higher likelihood of being a training member.
    
    Args:
        text: Input text
        model: HuggingFace model
        tok: Tokenizer
        max_tokens: Max tokens for truncation
        method: "loss" (negative loss), "mink" (Min-K%), or "minkpp" (Min-K%++)
        mink_ratio: k% for Min-K methods
    
    Returns:
        MIA score (higher = more likely member)
    """
    ids = tok.encode(text, truncation=True, max_length=max_tokens)
    if len(ids) < 2:
        return float("nan")
    
    input_ids = torch.tensor(ids, dtype=torch.long).unsqueeze(0).to(model.device)
    out = model(input_ids, labels=input_ids)
    
    if method == "loss":
        # Use negative loss (higher = lower perplexity = more likely member)
        return -out.loss.item()
    
    logits = out.logits[0, :-1]  # [T-1, V]
    target_ids = input_ids[0, 1:].unsqueeze(-1)  # [T-1, 1]
    
    log_probs = F.log_softmax(logits, dim=-1)
    token_log_probs = log_probs.gather(dim=-1, index=target_ids).squeeze(-1)
    
    if method == "mink":
        # Min-K%: mean of smallest k% log-probs (higher = more likely member)
        k = max(1, int(token_log_probs.numel() * mink_ratio))
        vals, _ = torch.sort(token_log_probs)
        return float(vals[:k].mean().item())
    
    elif method == "minkpp":
        # Min-K%++: standardized log-probs
        probs = F.softmax(logits, dim=-1)
        mu = (probs * log_probs).sum(-1)
        sigma = (probs * torch.square(log_probs)).sum(-1) - torch.square(mu)
        sigma = torch.clamp(sigma, min=1e-8)
        z = (token_log_probs - mu) / torch.sqrt(sigma)
        
        k = max(1, int(z.numel() * mink_ratio))
        vals, _ = torch.sort(z)
        return float(vals[:k].mean().item())
    
    else:
        raise ValueError(f"Unknown MIA method: {method}")


def load_category_samples(
    local_dir: str, 
    merge_web: bool, 
    max_per_class: int = None,
    seed: int = 0
) -> Tuple[List[str], Dict[str, List[str]]]:
    """Load samples from each category JSONL file."""
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


def estimate_tpr_fpr_crossval(
    scores: np.ndarray,
    calibration_fraction: float,
    seed: int = 0,
    threshold_method: str = "optimal"
) -> Tuple[float, float, float]:
    """
    Estimate TPR and FPR using cross-validation on the score distribution.
    
    For the DUCI setting without a separate reference model, we simulate
    "members" and "non-members" by splitting the score distribution.
    
    This is a simplification - ideally you'd have a reference model where
    you know ground-truth membership.
    
    Args:
        scores: Array of MIA scores for this category
        calibration_fraction: Fraction of scores to use for calibration
        seed: Random seed
        threshold_method: How to determine threshold
    
    Returns:
        (threshold, tpr, fpr)
    """
    rng = np.random.default_rng(seed)
    valid_scores = scores[np.isfinite(scores)]
    
    if len(valid_scores) < 10:
        # Not enough samples for reliable estimation
        return float(np.median(valid_scores)), 0.5, 0.5
    
    # Split scores into calibration set
    n_calib = max(10, int(len(valid_scores) * calibration_fraction))
    perm = rng.permutation(len(valid_scores))
    calib_scores = valid_scores[perm[:n_calib]]
    
    # For calibration, we assume the top half of scores are "likely members"
    # and bottom half are "likely non-members" (this is a heuristic)
    sorted_calib = np.sort(calib_scores)
    half = len(sorted_calib) // 2
    pseudo_nonmembers = sorted_calib[:half]
    pseudo_members = sorted_calib[half:]
    
    # Create pseudo-labels
    pseudo_labels = np.concatenate([np.zeros(len(pseudo_nonmembers)), np.ones(len(pseudo_members))])
    pseudo_scores = np.concatenate([pseudo_nonmembers, pseudo_members])
    
    # Compute ROC curve
    fpr_list, tpr_list, thresholds = roc_curve(pseudo_labels, pseudo_scores)
    
    if threshold_method == "optimal":
        # Find threshold that maximizes TPR - FPR (Youden's J)
        best_idx = np.argmax(tpr_list - fpr_list)
        threshold = thresholds[best_idx]
        tpr = tpr_list[best_idx]
        fpr = fpr_list[best_idx]
    elif threshold_method == "median":
        threshold = float(np.median(valid_scores))
        # Estimate TPR/FPR at median threshold
        tpr = np.mean(pseudo_members > threshold)
        fpr = np.mean(pseudo_nonmembers > threshold)
    else:
        raise ValueError(f"Unknown threshold method: {threshold_method}")
    
    # Ensure TPR > FPR to avoid division issues
    if tpr <= fpr:
        tpr = max(tpr, fpr + 0.01)
    
    return threshold, tpr, fpr


def estimate_tpr_fpr_reference(
    target_scores: np.ndarray,
    ref_scores: np.ndarray,
    threshold_method: str = "optimal"
) -> Tuple[float, float, float]:
    """
    Estimate TPR and FPR using a reference model.
    
    Assumption: Data that the target model "memorizes" (high scores) but the
    reference model doesn't (low scores) indicates membership in target's training.
    
    This is a more principled approach when you have a reference model.
    """
    valid_target = target_scores[np.isfinite(target_scores)]
    valid_ref = ref_scores[np.isfinite(ref_scores)]
    
    if len(valid_target) < 10 or len(valid_ref) < 10:
        return float(np.median(valid_target)), 0.5, 0.5
    
    # Use reference model scores to determine "non-membership" baseline
    # Points with high target score and low ref score are likely members
    # Points with similar scores in both are likely non-members
    
    # Simple heuristic: use reference model's score distribution as non-member proxy
    # and target model's top scores as member proxy
    ref_median = np.median(valid_ref)
    
    # Pseudo non-members: samples where target score is close to reference
    score_diff = valid_target - np.median(valid_ref)
    pseudo_nonmembers = valid_target[score_diff < np.percentile(score_diff, 50)]
    pseudo_members = valid_target[score_diff >= np.percentile(score_diff, 50)]
    
    pseudo_labels = np.concatenate([np.zeros(len(pseudo_nonmembers)), np.ones(len(pseudo_members))])
    pseudo_scores = np.concatenate([pseudo_nonmembers, pseudo_members])
    
    fpr_list, tpr_list, thresholds = roc_curve(pseudo_labels, pseudo_scores)
    
    if threshold_method == "optimal":
        best_idx = np.argmax(tpr_list - fpr_list)
        threshold = thresholds[best_idx]
        tpr = tpr_list[best_idx]
        fpr = fpr_list[best_idx]
    else:
        threshold = float(np.median(valid_target))
        tpr = np.mean(pseudo_members > threshold)
        fpr = np.mean(pseudo_nonmembers > threshold)
    
    if tpr <= fpr:
        tpr = max(tpr, fpr + 0.01)
    
    return threshold, tpr, fpr


def apply_duci_debiasing(
    scores: np.ndarray,
    threshold: float,
    tpr: float,
    fpr: float
) -> Tuple[float, float]:
    """
    Apply DUCI debiasing to get unbiased proportion estimate.
    
    Formula: p̂_i = (m̂_i - FPR) / (TPR - FPR)
    
    Returns:
        (raw_member_rate, debiased_proportion)
    """
    valid_scores = scores[np.isfinite(scores)]
    if len(valid_scores) == 0:
        return 0.0, 0.0
    
    # Binary MIA predictions
    m_hat = (valid_scores > threshold).astype(float)
    raw_member_rate = np.mean(m_hat)
    
    # Apply debiasing formula
    denominator = tpr - fpr
    if abs(denominator) < 1e-6:
        # TPR ≈ FPR, debiasing not meaningful
        debiased_proportion = raw_member_rate
    else:
        # Debias each prediction
        p_hat_i = (m_hat - fpr) / denominator
        # Clip to [0, 1] range
        p_hat_i = np.clip(p_hat_i, 0.0, 1.0)
        debiased_proportion = np.mean(p_hat_i)
    
    return raw_member_rate, debiased_proportion


def main() -> None:
    args = parse_args()
    
    print("=" * 70)
    print("DUCI: Dataset Usage Cardinality Inference for Category Composition")
    print("=" * 70)
    
    # Load category samples
    print(f"\nLoading samples from: {args.local_samples_dir}")
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
    print(f"\nLoading target model: {args.target_model}")
    target_model, tok = load_model(
        args.target_model, 
        revision=args.hf_revision,
        half=args.half, 
        int8=args.int8,
        trust_remote_code=args.trust_remote_code
    )
    
    # Load reference model if specified
    ref_model = None
    if args.reference_model:
        print(f"Loading reference model: {args.reference_model}")
        ref_model, _ = load_model(
            args.reference_model,
            revision=args.ref_revision,
            half=args.half,
            int8=args.int8,
            trust_remote_code=args.trust_remote_code
        )
    
    # Score all samples per category
    print(f"\nComputing MIA signals using method: {args.mia_method}")
    
    cat_target_scores: Dict[str, np.ndarray] = {}
    cat_ref_scores: Dict[str, np.ndarray] = {}
    
    for cat in categories:
        texts = cat_to_texts[cat]
        target_scores = []
        ref_scores = []
        
        for text in tqdm(texts, desc=f"Scoring {cat}"):
            # Target model score
            score = compute_mia_signal(
                text, target_model, tok, 
                args.max_tokens, args.mia_method, args.mink_ratio
            )
            target_scores.append(score)
            
            # Reference model score (if available)
            if ref_model is not None:
                ref_score = compute_mia_signal(
                    text, ref_model, tok,
                    args.max_tokens, args.mia_method, args.mink_ratio
                )
                ref_scores.append(ref_score)
        
        cat_target_scores[cat] = np.array(target_scores)
        if ref_model is not None:
            cat_ref_scores[cat] = np.array(ref_scores)
    
    # Estimate TPR/FPR and apply DUCI for each category
    print("\n" + "=" * 70)
    print("Estimating TPR/FPR and applying DUCI debiasing per category")
    print("=" * 70)
    
    category_stats: List[CategoryStats] = []
    
    for cat in categories:
        scores = cat_target_scores[cat]
        valid_mask = np.isfinite(scores)
        n_valid = int(valid_mask.sum())
        
        # Estimate TPR/FPR
        if ref_model is not None and cat in cat_ref_scores:
            threshold, tpr, fpr = estimate_tpr_fpr_reference(
                scores, cat_ref_scores[cat], args.threshold_method
            )
        else:
            if args.threshold_method == "fixed" and args.fixed_threshold is not None:
                threshold = args.fixed_threshold
                # Estimate TPR/FPR at fixed threshold using cross-val
                _, tpr, fpr = estimate_tpr_fpr_crossval(
                    scores, args.calibration_split, args.seed, "median"
                )
            else:
                threshold, tpr, fpr = estimate_tpr_fpr_crossval(
                    scores, args.calibration_split, args.seed, args.threshold_method
                )
        
        # Apply DUCI debiasing
        raw_rate, debiased_prop = apply_duci_debiasing(scores, threshold, tpr, fpr)
        
        stats = CategoryStats(
            category=cat,
            n_samples=len(scores),
            n_valid=n_valid,
            scores=scores,
            threshold=threshold,
            tpr=tpr,
            fpr=fpr,
            raw_member_rate=raw_rate,
            debiased_proportion=debiased_prop,
            score_mean=float(np.nanmean(scores)) if n_valid > 0 else 0.0,
            score_std=float(np.nanstd(scores)) if n_valid > 1 else 0.0,
        )
        category_stats.append(stats)
        
        print(f"\n{cat}:")
        print(f"  Samples: {stats.n_valid}/{stats.n_samples}")
        print(f"  Threshold: {stats.threshold:.4f}")
        print(f"  TPR: {stats.tpr:.3f}, FPR: {stats.fpr:.3f}")
        print(f"  Raw member rate: {stats.raw_member_rate:.2%}")
        print(f"  DUCI proportion: {stats.debiased_proportion:.2%}")
    
    # Normalize to get mixture (optional - proportions should already sum to meaningful values)
    total_debiased = sum(s.debiased_proportion for s in category_stats)
    
    print("\n" + "=" * 70)
    print("FINAL DUCI RESULTS: Estimated Category Proportions")
    print("=" * 70)
    
    for stats in category_stats:
        normalized = stats.debiased_proportion / total_debiased if total_debiased > 0 else 0.0
        print(f"  {stats.category:15s}: {stats.debiased_proportion:.2%} "
              f"(normalized: {normalized:.2%})")
    
    # Save results
    out_dir = Path(args.output_dir) / args.run_name
    out_dir.mkdir(parents=True, exist_ok=True)
    
    payload = {
        "config": {
            "local_samples_dir": args.local_samples_dir,
            "merge_web": args.merge_web,
            "max_per_class": args.max_per_class,
            "seed": args.seed,
            "target_model": args.target_model,
            "reference_model": args.reference_model,
            "mia_method": args.mia_method,
            "mink_ratio": args.mink_ratio,
            "threshold_method": args.threshold_method,
            "calibration_split": args.calibration_split,
        },
        "categories": categories,
        "per_category": [
            {
                "category": s.category,
                "n_samples": s.n_samples,
                "n_valid": s.n_valid,
                "threshold": s.threshold,
                "tpr": s.tpr,
                "fpr": s.fpr,
                "raw_member_rate": s.raw_member_rate,
                "duci_proportion": s.debiased_proportion,
                "normalized_proportion": s.debiased_proportion / total_debiased if total_debiased > 0 else 0.0,
                "score_mean": s.score_mean,
                "score_std": s.score_std,
            }
            for s in category_stats
        ],
        "summary": {
            "total_samples": total_samples,
            "total_debiased_sum": total_debiased,
        }
    }
    
    with open(out_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    
    # CSV output
    with open(out_dir / "duci_proportions.csv", "w", encoding="utf-8") as f:
        f.write("category,n_samples,threshold,tpr,fpr,raw_member_rate,duci_proportion,normalized_proportion\n")
        for s in category_stats:
            norm = s.debiased_proportion / total_debiased if total_debiased > 0 else 0.0
            f.write(f"{s.category},{s.n_samples},{s.threshold:.6f},{s.tpr:.4f},{s.fpr:.4f},"
                    f"{s.raw_member_rate:.6f},{s.debiased_proportion:.6f},{norm:.6f}\n")
    
    print(f"\nWrote DUCI results to {out_dir}")


if __name__ == "__main__":
    main()

