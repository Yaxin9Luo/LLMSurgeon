"""
Min-K% with Binary Thresholding for StarCoder Category Composition Estimation

This script adapts Min-K% (a membership inference attack) to predict 
category composition percentages of StarCoder pretraining data across 86 programming languages.

Min-K% is the original method from "Detecting Pretraining Data from Large Language Models"
(Shi et al., 2023). It uses the mean of the smallest k% of token log-probabilities.

Approach:
1. Load per-language samples from samples_dir (e.g., python.jsonl, java.jsonl, ...)
2. Score each sample with Min-K% 
3. Apply a threshold: score > threshold → member (1), else non-member (0)
4. For each language: proportion = count(members) / total_samples_in_language
5. Normalize across languages to get the predicted composition
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from tqdm import tqdm

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM

from data_utils_starcoder import load_starcoder_categories, detect_language_files, _read_jsonl


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Estimate StarCoder category mixture via Min-K%% with binary thresholding (86 languages)"
    )
    # Data
    p.add_argument(
        "--samples_dir",
        type=str,
        default=str(Path(__file__).resolve().parents[3] / "data_samples" / "starcoder"),
        help="Directory containing <language>.jsonl files (one per StarCoder language)",
    )
    p.add_argument(
        "--spec_path",
        type=str,
        default=None,
        help="Path to starcoder.yaml spec file (defaults to bench/specs/starcoder.yaml)",
    )
    p.add_argument("--max_per_class", type=int, default=None, help="Max samples per language (None = use all)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--allow_missing", action="store_true", help="Proceed with subset of languages if some are missing")

    # Target model
    p.add_argument("--target_model", type=str, required=True, help="HF model id for scoring (causal LM, e.g., bigcode/starcoder)")
    p.add_argument("--hf_revision", type=str, default=None, help="HF model revision/checkpoint")
    p.add_argument("--half", action="store_true", help="Load model in bfloat16")
    p.add_argument("--int8", action="store_true", help="Load model in 8-bit (bitsandbytes)")
    p.add_argument("--max_tokens", type=int, default=512, help="Max tokens per sample for scoring")
    p.add_argument("--trust_remote_code", action="store_true", help="Trust remote code for models")

    # Min-K parameters
    p.add_argument("--mink_ratio", type=float, default=0.2, help="k%% for Min-K (0<r<=1)")
    p.add_argument(
        "--threshold", 
        type=float, 
        default=None, 
        help="Threshold for binary classification. Score > threshold → member. "
             "If not specified, uses the median score as threshold (adaptive)."
    )

    # Output
    p.add_argument("--output_dir", type=str, default=str(Path(__file__).resolve().parents[3] / "out"))
    p.add_argument("--run_name", type=str, default="mink_threshold_starcoder")
    
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
def mink_score(text: str, model, tok, max_tokens: int, ratio: float) -> float:
    """
    Compute Min-K% score for a text sample.
    
    Min-K% takes the mean of the smallest k% of token log-probabilities.
    Higher score (less negative) → more likely to be a member of training data.
    
    This is the original method from Shi et al. (2023).
    """
    # Tokenize with truncation
    ids = tok.encode(text, truncation=True, max_length=max_tokens)
    if len(ids) < 2:
        return float("nan")
    
    input_ids = torch.tensor(ids, dtype=torch.long).unsqueeze(0).to(model.device)
    out = model(input_ids, labels=input_ids)
    logits = out.logits[0, :-1]  # [T-1, V]
    target_ids = input_ids[0, 1:].unsqueeze(-1)  # [T-1, 1]

    # Get log-probabilities
    log_probs = F.log_softmax(logits, dim=-1)
    token_log_probs = log_probs.gather(dim=-1, index=target_ids).squeeze(-1)  # [T-1]
    
    # Take mean of smallest k% (Min-K%)
    k = max(1, int(token_log_probs.numel() * ratio))
    vals, _ = torch.sort(token_log_probs)
    return float(vals[:k].mean().item())


def load_language_samples(
    samples_dir: str,
    spec_path: str = None,
    max_per_class: int = None,
    seed: int = 0,
    require_all: bool = True
) -> Tuple[List[str], Dict[str, List[str]]]:
    """
    Load samples from each language JSONL file.
    
    Returns:
        languages: list of language names
        lang_to_texts: dict mapping language name to list of text samples
    """
    import random
    rng = random.Random(seed)
    
    languages = load_starcoder_categories(spec_path)
    lang_to_file = detect_language_files(samples_dir, languages)
    
    if require_all and len(lang_to_file) != len(languages):
        missing = [c for c in languages if c not in lang_to_file]
        raise FileNotFoundError(
            f"Missing samples for {len(missing)} languages under {samples_dir}. E.g., {missing[:5]}"
        )
    
    # Use only languages with files
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
    
    # Remove empty languages
    languages = [lang for lang in languages if lang_to_texts.get(lang)]
    lang_to_texts = {lang: lang_to_texts[lang] for lang in languages}
    
    return languages, lang_to_texts


def main() -> None:
    args = parse_args()
    
    # Load language samples
    print(f"Loading StarCoder samples from: {args.samples_dir}")
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
    print(f"Samples per language: min={min(len(lang_to_texts[l]) for l in languages)}, max={max(len(lang_to_texts[l]) for l in languages)}")
    
    # Load target model
    print(f"\nLoading model: {args.target_model}" + (f" @ {args.hf_revision}" if args.hf_revision else ""))
    model, tok = load_model(
        args.target_model, 
        revision=args.hf_revision,
        half=args.half, 
        int8=args.int8,
        trust_remote_code=args.trust_remote_code
    )
    
    # Score all samples
    print(f"\nScoring samples with Min-K% (ratio={args.mink_ratio})")
    
    lang_scores: Dict[str, List[float]] = {lang: [] for lang in languages}
    all_scores: List[float] = []
    
    pbar = tqdm(total=total_samples, desc="Scoring (Min-K%)")
    for lang in languages:
        for text in lang_to_texts[lang]:
            score = mink_score(text, model, tok, args.max_tokens, args.mink_ratio)
            lang_scores[lang].append(score)
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
    lang_members: Dict[str, int] = {lang: 0 for lang in languages}
    for lang in languages:
        for score in lang_scores[lang]:
            if np.isfinite(score) and score > threshold:
                lang_members[lang] += 1
    
    # Compute per-language statistics
    per_lang_results = []
    for lang in languages:
        scores = np.array(lang_scores[lang])
        n_total = len(scores)
        n_valid = int(np.isfinite(scores).sum())
        n_members = lang_members[lang]
        
        # Proportion of members in this language
        proportion = n_members / n_valid if n_valid > 0 else 0.0
        
        per_lang_results.append({
            "language": lang,
            "n_total": n_total,
            "n_valid": n_valid,
            "n_members": n_members,
            "member_proportion": proportion,
            "score_mean": float(np.nanmean(scores)) if n_valid > 0 else None,
            "score_std": float(np.nanstd(scores)) if n_valid > 1 else None,
            "score_median": float(np.nanmedian(scores)) if n_valid > 0 else None,
        })
    
    # Compute global mixture (normalized proportions)
    # Weight by member count
    total_members = sum(lang_members.values())
    if total_members > 0:
        global_mixture = {
            lang: lang_members[lang] / total_members for lang in languages
        }
    else:
        global_mixture = {lang: 0.0 for lang in languages}
    
    # Print results (top 15 by predicted proportion)
    print("\n" + "=" * 70)
    print("Top 15 Languages by Predicted Proportion:")
    print("=" * 70)
    sorted_langs = sorted(languages, key=lambda l: global_mixture[l], reverse=True)
    for lang in sorted_langs[:15]:
        r = next(x for x in per_lang_results if x["language"] == lang)
        score_str = f", score μ={r['score_mean']:.4f}" if r['score_mean'] is not None else ""
        print(f"  {lang:20s}: {r['n_members']:5d}/{r['n_valid']:5d} members "
              f"({r['member_proportion']:.2%}){score_str} → {global_mixture[lang]:.4f}")
    
    print("\n" + "=" * 70)
    print(f"Total languages: {len(languages)}, Total samples: {total_samples}")
    print(f"Total members: {total_members}, Overall member rate: {total_members/total_samples:.2%}")
    print("=" * 70)
    
    # Write outputs
    out_dir = Path(args.output_dir) / args.run_name
    out_dir.mkdir(parents=True, exist_ok=True)
    
    payload = {
        "config": {
            "samples_dir": args.samples_dir,
            "spec_path": args.spec_path,
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
        "languages": languages,
        "per_language": per_lang_results,
        "global_mixture": global_mixture,
        "summary_stats": {
            "total_languages": len(languages),
            "total_samples": total_samples,
            "total_members": total_members,
            "overall_member_rate": total_members / total_samples if total_samples > 0 else 0.0,
            "threshold_used": threshold,
        }
    }
    
    with open(out_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    
    # CSV: per-language results
    with open(out_dir / "per_language.csv", "w", encoding="utf-8") as f:
        f.write("language,n_total,n_valid,n_members,member_proportion,score_mean,score_std,score_median\n")
        for r in per_lang_results:
            f.write(f"{r['language']},{r['n_total']},{r['n_valid']},{r['n_members']},"
                    f"{r['member_proportion']:.6f},"
                    f"{r['score_mean'] if r['score_mean'] is not None else ''},"
                    f"{r['score_std'] if r['score_std'] is not None else ''},"
                    f"{r['score_median'] if r['score_median'] is not None else ''}\n")
    
    # CSV: global mixture
    with open(out_dir / "global_mixture.csv", "w", encoding="utf-8") as f:
        f.write("language,predicted_proportion\n")
        for lang in languages:
            f.write(f"{lang},{global_mixture[lang]:.6f}\n")
    
    print(f"\nWrote StarCoder Min-K% outputs to {out_dir}")


if __name__ == "__main__":
    main()

