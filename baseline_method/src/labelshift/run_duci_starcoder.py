"""
DUCI (Dataset Usage Cardinality Inference) for StarCoder Language Composition

This script applies the DUCI methodology to estimate the proportion of each 
programming language (86 languages) used in training the StarCoder model.

Key Steps:
1. Isolate Target Datasets - Treat each language as an independent inference problem
2. Run MIA for each language - Using loss-based or RMIA-style signals
3. Estimate TPR/FPR per language - Using cross-validation or reference model
4. Apply Debiasing - Correct MIA predictions using language-specific TPR/FPR
5. Aggregate - Average debiased probabilities to get per-language proportions

Formula for debiasing:
    p̂_i = (m̂_i - FPR) / (TPR - FPR)

Reference: "How Much of My Dataset Did You Use?" (Tong et al., ICLR 2025)
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
class LanguageStats:
    """Statistics for a single programming language."""
    language: str
    n_samples: int
    n_valid: int
    threshold: float
    tpr: float
    fpr: float
    raw_member_rate: float
    duci_proportion: float
    score_mean: float
    score_std: float


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="DUCI-based language composition estimation for StarCoder"
    )
    # Data
    p.add_argument(
        "--samples_dir",
        type=str,
        default=str(Path(__file__).resolve().parents[3] / "data_samples" / "starcoder"),
        help="Directory containing <language>.jsonl files",
    )
    p.add_argument("--spec_path", type=str, default=None)
    p.add_argument("--max_per_class", type=int, default=200, help="Max samples per language")
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
    p.add_argument("--threshold_method", type=str, choices=["median", "optimal"], default="optimal")

    # Output
    p.add_argument("--output_dir", type=str, default=str(Path(__file__).resolve().parents[3] / "out"))
    p.add_argument("--run_name", type=str, default="duci_starcoder")
    
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
def compute_mia_signal(text: str, model, tok, max_tokens: int, method: str = "loss", mink_ratio: float = 0.2) -> float:
    """Compute MIA signal for a text sample."""
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
    
    elif method == "minkpp":
        probs = F.softmax(logits, dim=-1)
        mu = (probs * log_probs).sum(-1)
        sigma = (probs * torch.square(log_probs)).sum(-1) - torch.square(mu)
        sigma = torch.clamp(sigma, min=1e-8)
        z = (token_log_probs - mu) / torch.sqrt(sigma)
        k = max(1, int(z.numel() * mink_ratio))
        vals, _ = torch.sort(z)
        return float(vals[:k].mean().item())
    
    raise ValueError(f"Unknown method: {method}")


def load_language_samples(
    samples_dir: str,
    spec_path: str = None,
    max_per_class: int = None,
    seed: int = 0,
    require_all: bool = True
) -> Tuple[List[str], Dict[str, List[str]]]:
    """Load samples from each language JSONL file."""
    import random
    rng = random.Random(seed)
    
    languages = load_starcoder_categories(spec_path)
    lang_to_file = detect_language_files(samples_dir, languages)
    
    if require_all and len(lang_to_file) != len(languages):
        missing = [c for c in languages if c not in lang_to_file]
        raise FileNotFoundError(f"Missing samples for {len(missing)} languages: {missing[:5]}")
    
    languages = [c for c in languages if c in lang_to_file]
    if not languages:
        raise FileNotFoundError(f"No language files detected under {samples_dir}")
    
    lang_to_texts: Dict[str, List[str]] = {}
    for lang in languages:
        path = lang_to_file[lang]
        texts = _read_jsonl(path)
        rng.shuffle(texts)
        if max_per_class is not None:
            texts = texts[:max_per_class]
        lang_to_texts[lang] = texts
    
    languages = [lang for lang in languages if lang_to_texts.get(lang)]
    lang_to_texts = {lang: lang_to_texts[lang] for lang in languages}
    
    return languages, lang_to_texts


def estimate_tpr_fpr(scores: np.ndarray, calibration_fraction: float, seed: int, method: str) -> Tuple[float, float, float]:
    """Estimate TPR and FPR using cross-validation on the score distribution."""
    rng = np.random.default_rng(seed)
    valid_scores = scores[np.isfinite(scores)]
    
    if len(valid_scores) < 10:
        return float(np.median(valid_scores)) if len(valid_scores) > 0 else 0.0, 0.5, 0.5
    
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
    
    if method == "optimal":
        best_idx = np.argmax(tpr_list - fpr_list)
        threshold = thresholds[best_idx]
        tpr = tpr_list[best_idx]
        fpr = fpr_list[best_idx]
    else:  # median
        threshold = float(np.median(valid_scores))
        tpr = np.mean(pseudo_members > threshold) if len(pseudo_members) > 0 else 0.5
        fpr = np.mean(pseudo_nonmembers > threshold) if len(pseudo_nonmembers) > 0 else 0.5
    
    if tpr <= fpr:
        tpr = max(tpr, fpr + 0.01)
    
    return threshold, tpr, fpr


def apply_duci_debiasing(scores: np.ndarray, threshold: float, tpr: float, fpr: float) -> Tuple[float, float]:
    """Apply DUCI debiasing."""
    valid_scores = scores[np.isfinite(scores)]
    if len(valid_scores) == 0:
        return 0.0, 0.0
    
    m_hat = (valid_scores > threshold).astype(float)
    raw_member_rate = np.mean(m_hat)
    
    denominator = tpr - fpr
    if abs(denominator) < 1e-6:
        debiased_proportion = raw_member_rate
    else:
        p_hat_i = (m_hat - fpr) / denominator
        p_hat_i = np.clip(p_hat_i, 0.0, 1.0)
        debiased_proportion = np.mean(p_hat_i)
    
    return raw_member_rate, debiased_proportion


def main() -> None:
    args = parse_args()
    
    print("=" * 70)
    print("DUCI: Language Composition Estimation for StarCoder")
    print("=" * 70)
    
    # Load language samples
    print(f"\nLoading samples from: {args.samples_dir}")
    languages, lang_to_texts = load_language_samples(
        args.samples_dir,
        spec_path=args.spec_path,
        max_per_class=args.max_per_class,
        seed=args.seed,
        require_all=not args.allow_missing,
    )
    
    total_samples = sum(len(texts) for texts in lang_to_texts.values())
    print(f"Languages: {len(languages)}")
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
    
    # Score all samples per language
    print(f"\nComputing MIA signals using method: {args.mia_method}")
    
    lang_scores: Dict[str, np.ndarray] = {}
    
    for lang in tqdm(languages, desc="Languages"):
        texts = lang_to_texts[lang]
        scores = []
        for text in texts:
            score = compute_mia_signal(text, target_model, tok, args.max_tokens, args.mia_method, args.mink_ratio)
            scores.append(score)
        lang_scores[lang] = np.array(scores)
    
    # Apply DUCI for each language
    print("\n" + "=" * 70)
    print("Applying DUCI debiasing per language")
    print("=" * 70)
    
    language_stats: List[LanguageStats] = []
    
    for lang in languages:
        scores = lang_scores[lang]
        valid_mask = np.isfinite(scores)
        n_valid = int(valid_mask.sum())
        
        threshold, tpr, fpr = estimate_tpr_fpr(scores, args.calibration_split, args.seed, args.threshold_method)
        raw_rate, debiased_prop = apply_duci_debiasing(scores, threshold, tpr, fpr)
        
        stats = LanguageStats(
            language=lang,
            n_samples=len(scores),
            n_valid=n_valid,
            threshold=threshold,
            tpr=tpr,
            fpr=fpr,
            raw_member_rate=raw_rate,
            duci_proportion=debiased_prop,
            score_mean=float(np.nanmean(scores)) if n_valid > 0 else 0.0,
            score_std=float(np.nanstd(scores)) if n_valid > 1 else 0.0,
        )
        language_stats.append(stats)
    
    # Sort by DUCI proportion
    language_stats.sort(key=lambda x: x.duci_proportion, reverse=True)
    total_debiased = sum(s.duci_proportion for s in language_stats)
    
    # Print top 20
    print("\n" + "=" * 70)
    print("Top 20 Languages by DUCI Proportion:")
    print("=" * 70)
    for stats in language_stats[:20]:
        norm = stats.duci_proportion / total_debiased if total_debiased > 0 else 0.0
        print(f"  {stats.language:20s}: {stats.duci_proportion:.2%} (norm: {norm:.2%}), "
              f"TPR={stats.tpr:.2f}, FPR={stats.fpr:.2f}")
    
    # Save results
    out_dir = Path(args.output_dir) / args.run_name
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # Re-sort by original order for output
    language_stats_ordered = sorted(language_stats, key=lambda x: languages.index(x.language))
    
    payload = {
        "config": {
            "samples_dir": args.samples_dir,
            "max_per_class": args.max_per_class,
            "target_model": args.target_model,
            "mia_method": args.mia_method,
            "mink_ratio": args.mink_ratio,
            "threshold_method": args.threshold_method,
            "calibration_split": args.calibration_split,
        },
        "languages": languages,
        "per_language": [
            {
                "language": s.language,
                "n_samples": s.n_samples,
                "n_valid": s.n_valid,
                "threshold": s.threshold,
                "tpr": s.tpr,
                "fpr": s.fpr,
                "raw_member_rate": s.raw_member_rate,
                "duci_proportion": s.duci_proportion,
                "normalized_proportion": s.duci_proportion / total_debiased if total_debiased > 0 else 0.0,
                "score_mean": s.score_mean,
                "score_std": s.score_std,
            }
            for s in language_stats_ordered
        ],
        "summary": {
            "total_languages": len(languages),
            "total_samples": total_samples,
            "total_debiased_sum": total_debiased,
        }
    }
    
    with open(out_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    
    with open(out_dir / "duci_proportions.csv", "w", encoding="utf-8") as f:
        f.write("language,n_samples,threshold,tpr,fpr,raw_member_rate,duci_proportion,normalized_proportion\n")
        for s in language_stats_ordered:
            norm = s.duci_proportion / total_debiased if total_debiased > 0 else 0.0
            f.write(f"{s.language},{s.n_samples},{s.threshold:.6f},{s.tpr:.4f},{s.fpr:.4f},"
                    f"{s.raw_member_rate:.6f},{s.duci_proportion:.6f},{norm:.6f}\n")
    
    print(f"\nWrote DUCI results to {out_dir}")


if __name__ == "__main__":
    main()

