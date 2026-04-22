"""
DUCI (Dataset Usage Cardinality Inference) for language composition (StarCoder taxonomy)

Applies DUCI to StarCoder's language set. Supports MIA signals: loss, mink, minkpp.
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple
from dataclasses import dataclass

import numpy as np
from tqdm import tqdm
from sklearn.metrics import roc_curve

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM

from data_utils_starcoder import load_starcoder_categories, detect_language_files, _read_jsonl


@dataclass
class CategoryStats:
    category: str
    n_samples: int
    n_valid: int
    scores: np.ndarray
    threshold: float
    tpr: float
    fpr: float
    raw_member_rate: float
    debiased_proportion: float
    score_mean: float
    score_std: float


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="DUCI-based language composition estimation (StarCoder taxonomy)"
    )
    # Data
    p.add_argument(
        "--local_samples_dir",
        type=str,
        default=str(Path(__file__).resolve().parents[2] / "data_samples" / "starcoder"),
        help="Directory with <language>.jsonl files",
    )
    p.add_argument("--spec_path", type=str, default=None, help="YAML spec for StarCoder languages")
    p.add_argument("--max_per_class", type=int, default=5000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--allow_missing", action="store_true")

    # Target model
    p.add_argument("--target_model", type=str, required=True)
    p.add_argument("--hf_revision", type=str, default=None)
    p.add_argument("--half", action="store_true")
    p.add_argument("--int8", action="store_true")
    p.add_argument("--max_tokens", type=int, default=512)
    p.add_argument("--trust_remote_code", action="store_true")

    # Reference model (optional)
    p.add_argument("--reference_model", type=str, default=None)
    p.add_argument("--ref_revision", type=str, default=None)

    # MIA parameters
    p.add_argument("--mia_method", type=str, choices=["loss", "mink", "minkpp"], default="loss")
    p.add_argument("--mink_ratio", type=float, default=0.2)

    # DUCI parameters
    p.add_argument("--calibration_split", type=float, default=0.3)
    p.add_argument("--threshold_method", type=str, choices=["median", "optimal", "fixed"], default="optimal")
    p.add_argument("--fixed_threshold", type=float, default=None)

    # Output
    p.add_argument("--output_dir", type=str, default=str(Path(__file__).resolve().parents[3] / "out"))
    p.add_argument("--run_name", type=str, default="duci_categories_starcoder")
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


@torch.no_grad()
def compute_mia_signal(text: str, model, tok, max_tokens: int, method: str = "loss", mink_ratio: float = 0.2) -> float:
    ids = tok.encode(text, truncation=True, max_length=max_tokens)
    if len(ids) < 2:
        return float("nan")
    input_ids = torch.tensor(ids, dtype=torch.long).unsqueeze(0).to(model.device)
    out = model(input_ids, labels=input_ids)
    if method == "loss":
        return -out.loss.item()

    logits = out.logits[0, :-1]
    target_ids = input_ids[0, 1:].unsqueeze(-1)
    log_probs = F.log_softmax(logits, dim=-1)
    token_log_probs = log_probs.gather(dim=-1, index=target_ids).squeeze(-1)

    if method == "mink":
        k = max(1, int(token_log_probs.numel() * mink_ratio))
        vals, _ = torch.sort(token_log_probs)
        return float(vals[:k].mean().item())

    probs = F.softmax(logits, dim=-1)
    mu = (probs * log_probs).sum(-1)
    sigma = (probs * torch.square(log_probs)).sum(-1) - torch.square(mu)
    sigma = torch.clamp(sigma, min=1e-8)
    z = (token_log_probs - mu) / torch.sqrt(sigma)
    k = max(1, int(z.numel() * mink_ratio))
    vals, _ = torch.sort(z)
    return float(vals[:k].mean().item())


def load_language_samples(
    samples_dir: str,
    spec_path: str | None,
    max_per_class: int | None,
    seed: int,
    require_all: bool,
) -> Tuple[List[str], Dict[str, List[str]]]:
    import random
    rng = random.Random(seed)
    languages = load_starcoder_categories(spec_path)
    lang_to_file = detect_language_files(samples_dir, languages)
    if require_all and len(lang_to_file) != len(languages):
        missing = [c for c in languages if c not in lang_to_file]
        raise FileNotFoundError(f"Missing samples for {len(missing)} languages under {samples_dir}. e.g., {missing[:5]}")
    categories = [c for c in languages if c in lang_to_file]
    if not categories:
        raise FileNotFoundError(f"No language files detected under {samples_dir}")
    cat_to_texts: Dict[str, List[str]] = {c: [] for c in categories}
    for c in categories:
        path = lang_to_file[c]
        texts = _read_jsonl(path)
        rng.shuffle(texts)
        if max_per_class:
            texts = texts[:max_per_class]
        cat_to_texts[c].extend(texts)
    return categories, cat_to_texts


def estimate_tpr_fpr_crossval(scores: np.ndarray, calibration_fraction: float, seed: int = 0,
                               threshold_method: str = "optimal") -> Tuple[float, float, float]:
    rng = np.random.default_rng(seed)
    valid_scores = scores[np.isfinite(scores)]
    if len(valid_scores) < 10:
        return float(np.median(valid_scores)), 0.5, 0.5
    n_calib = max(10, int(len(valid_scores) * calibration_fraction))
    perm = rng.permutation(len(valid_scores))
    calib_scores = valid_scores[perm[:n_calib]]
    sorted_calib = np.sort(calib_scores)
    half = len(sorted_calib) // 2
    pseudo_nonmembers = sorted_calib[:half]
    pseudo_members = sorted_calib[half:]
    pseudo_labels = np.concatenate([np.zeros(len(pseudo_nonmembers)), np.ones(len(pseudo_members))])
    pseudo_scores = np.concatenate([pseudo_nonmembers, pseudo_members])
    fpr_list, tpr_list, thresholds = roc_curve(pseudo_labels, pseudo_scores)
    if threshold_method == "optimal":
        best_idx = np.argmax(tpr_list - fpr_list)
        threshold = thresholds[best_idx]
        tpr = tpr_list[best_idx]
        fpr = fpr_list[best_idx]
    elif threshold_method == "median":
        threshold = float(np.median(valid_scores))
        tpr = np.mean(pseudo_members > threshold)
        fpr = np.mean(pseudo_nonmembers > threshold)
    else:
        raise ValueError(f"Unknown threshold method: {threshold_method}")
    if tpr <= fpr:
        tpr = max(tpr, fpr + 0.01)
    return threshold, tpr, fpr


def estimate_tpr_fpr_reference(target_scores: np.ndarray, ref_scores: np.ndarray,
                                threshold_method: str = "optimal") -> Tuple[float, float, float]:
    valid_target = target_scores[np.isfinite(target_scores)]
    valid_ref = ref_scores[np.isfinite(ref_scores)]
    if len(valid_target) < 10 or len(valid_ref) < 10:
        return float(np.median(valid_target)), 0.5, 0.5
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


def apply_duci_debiasing(scores: np.ndarray, threshold: float, tpr: float, fpr: float) -> Tuple[float, float]:
    valid_scores = scores[np.isfinite(scores)]
    if len(valid_scores) == 0:
        return 0.0, 0.0
    m_hat = (valid_scores > threshold).astype(float)
    raw_member_rate = np.mean(m_hat)
    denom = tpr - fpr
    if abs(denom) < 1e-6:
        debiased = raw_member_rate
    else:
        p_hat_i = (m_hat - fpr) / denom
        p_hat_i = np.clip(p_hat_i, 0.0, 1.0)
        debiased = np.mean(p_hat_i)
    return raw_member_rate, debiased


def main() -> None:
    args = parse_args()

    print("=" * 70)
    print("DUCI: Dataset Usage Cardinality Inference (StarCoder languages)")
    print("=" * 70)

    print(f"\nLoading StarCoder language samples from: {args.local_samples_dir}")
    categories, cat_to_texts = load_language_samples(
        samples_dir=args.local_samples_dir,
        spec_path=args.spec_path,
        max_per_class=args.max_per_class,
        seed=args.seed,
        require_all=not args.allow_missing,
    )
    total_samples = sum(len(v) for v in cat_to_texts.values())
    print(f"Languages ({len(categories)}): {categories}")
    print(f"Total samples: {total_samples}")

    print(f"\nLoading target model: {args.target_model}")
    target_model, tok = load_model(
        args.target_model, revision=args.hf_revision, half=args.half, int8=args.int8, trust_remote_code=args.trust_remote_code
    )

    ref_model = None
    if args.reference_model:
        print(f"Loading reference model: {args.reference_model}")
        ref_model, _ = load_model(
            args.reference_model, revision=args.ref_revision, half=args.half, int8=args.int8, trust_remote_code=args.trust_remote_code
        )

    print(f"\nComputing MIA signals using: {args.mia_method}")
    cat_target_scores: Dict[str, np.ndarray] = {}
    cat_ref_scores: Dict[str, np.ndarray] = {}

    for cat in categories:
        texts = cat_to_texts[cat]
        target_scores = []
        ref_scores = []
        for text in tqdm(texts, desc=f"Scoring {cat}"):
            score = compute_mia_signal(text, target_model, tok, args.max_tokens, args.mia_method, args.mink_ratio)
            target_scores.append(score)
            if ref_model is not None:
                ref_score = compute_mia_signal(text, ref_model, tok, args.max_tokens, args.mia_method, args.mink_ratio)
                ref_scores.append(ref_score)
        cat_target_scores[cat] = np.array(target_scores)
        if ref_model is not None:
            cat_ref_scores[cat] = np.array(ref_scores)

    print("\n" + "=" * 70)
    print("Estimating TPR/FPR and applying DUCI")
    print("=" * 70)

    category_stats: List[CategoryStats] = []
    for cat in categories:
        scores = cat_target_scores[cat]
        n_valid = int(np.isfinite(scores).sum())

        if ref_model is not None and cat in cat_ref_scores:
            threshold, tpr, fpr = estimate_tpr_fpr_reference(scores, cat_ref_scores[cat], args.threshold_method)
        else:
            if args.threshold_method == "fixed" and args.fixed_threshold is not None:
                threshold = args.fixed_threshold
                _, tpr, fpr = estimate_tpr_fpr_crossval(scores, args.calibration_split, args.seed, "median")
            else:
                threshold, tpr, fpr = estimate_tpr_fpr_crossval(scores, args.calibration_split, args.seed, args.threshold_method)

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

    total_debiased = sum(s.debiased_proportion for s in category_stats)
    global_mixture = {s.category: (s.debiased_proportion / total_debiased if total_debiased > 0 else 0.0) for s in category_stats}

    print("\n" + "=" * 70)
    print("FINAL DUCI RESULTS: Estimated Language Proportions (StarCoder)")
    print("=" * 70)
    for cat, val in global_mixture.items():
        print(f"  {cat:20s}: {val:.2%}")

    out_dir = Path(args.output_dir) / args.run_name
    out_dir.mkdir(parents=True, exist_ok=True)

    payload = {
        "config": {
            "local_samples_dir": args.local_samples_dir,
            "spec_path": args.spec_path,
            "max_per_class": args.max_per_class,
            "seed": args.seed,
            "target_model": args.target_model,
            "reference_model": args.reference_model,
            "mia_method": args.mia_method,
            "mink_ratio": args.mink_ratio,
            "threshold_method": args.threshold_method,
            "calibration_split": args.calibration_split,
            "allow_missing": args.allow_missing,
            "taxonomy": "starcoder",
        },
        "categories": [s.category for s in category_stats],
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
                "normalized_proportion": global_mixture.get(s.category, 0.0),
                "score_mean": s.score_mean,
                "score_std": s.score_std,
            }
            for s in category_stats
        ],
        "global_mixture": global_mixture,
        "summary": {
            "total_samples": total_samples,
            "total_debiased_sum": total_debiased,
        },
    }

    with open(out_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    with open(out_dir / "duci_proportions.csv", "w", encoding="utf-8") as f:
        f.write("category,n_samples,threshold,tpr,fpr,raw_member_rate,duci_proportion,normalized_proportion\n")
        for s in category_stats:
            norm = global_mixture.get(s.category, 0.0)
            f.write(
                f"{s.category},{s.n_samples},{s.threshold:.6f},{s.tpr:.4f},{s.fpr:.4f},"
                f"{s.raw_member_rate:.6f},{s.debiased_proportion:.6f},{norm:.6f}\n"
            )

    with open(out_dir / "global_mixture.csv", "w", encoding="utf-8") as f:
        f.write("category,predicted_proportion\n")
        for cat, val in global_mixture.items():
            f.write(f"{cat},{val:.6f}\n")

    print(f"\nWrote DUCI results to {out_dir}")


if __name__ == "__main__":
    main()

