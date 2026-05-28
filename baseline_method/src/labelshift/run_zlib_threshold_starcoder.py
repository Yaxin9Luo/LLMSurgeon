"""
Zlib with Binary Thresholding for Category Composition Estimation (StarCoder languages)

This adapts the zlib-based MIA signal to StarCoder's language taxonomy (from bench/specs/starcoder.yaml).
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

from data_utils_starcoder import load_starcoder_categories, detect_language_files, _read_jsonl


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Estimate language mixture via zlib with binary thresholding (StarCoder taxonomy)"
    )
    # Data
    p.add_argument(
        "--local_samples_dir",
        type=str,
        default=str(Path(__file__).resolve().parents[3] / "data_samples" / "starcoder"),
        help="Directory with <language>.jsonl files",
    )
    p.add_argument("--spec_path", type=str, default=None, help="YAML spec for StarCoder languages")
    p.add_argument("--max_per_class", type=int, default=None, help="Max samples per language (None = all)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--allow_missing", action="store_true", help="Proceed even if some languages are missing")

    # Zlib mode
    p.add_argument("--mode", type=str, choices=["zlib_len", "ll_over_zlib"], default="zlib_len")
    p.add_argument("--zlib_level", type=int, default=6, help="zlib compression level (0-9)")

    # Target model (only required for ll_over_zlib)
    p.add_argument("--target_model", type=str, default=None, help="HF model id for scoring (causal LM)")
    p.add_argument("--hf_revision", type=str, default=None)
    p.add_argument("--half", action="store_true")
    p.add_argument("--int8", action="store_true")
    p.add_argument("--max_tokens", type=int, default=512)
    p.add_argument("--trust_remote_code", action="store_true")

    # Thresholding
    p.add_argument("--threshold", type=float, default=None)
    p.add_argument("--member_if", type=str, default=None, choices=["gt", "lt"])

    # Output
    p.add_argument("--output_dir", type=str, default=str(Path(__file__).resolve().parents[3] / "out"))
    p.add_argument("--run_name", type=str, default="zlib_threshold_starcoder")
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
        if max_per_class is not None:
            texts = texts[:max_per_class]
        cat_to_texts[c].extend(texts)

    return categories, cat_to_texts


def zlib_len(text: str, level: int = 6) -> float:
    b = text.encode("utf-8", errors="ignore")
    return float(len(zlib.compress(b, level=level)))


@torch.no_grad()
def avg_ll(text: str, model, tok, max_tokens: int) -> float:
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

    member_if = args.member_if if args.member_if is not None else ("lt" if args.mode == "zlib_len" else "gt")

    print(f"Loading StarCoder language samples from: {args.local_samples_dir}")
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

    model = tok = None
    if args.mode == "ll_over_zlib":
        if not args.target_model:
            raise ValueError("--target_model is required when --mode ll_over_zlib")
        print(f"Loading model: {args.target_model}")
        model, tok = load_model(args.target_model, args.hf_revision, args.half, args.int8, args.trust_remote_code)

    print(f"\nScoring samples with zlib mode={args.mode} (member_if={member_if})")
    cat_scores: Dict[str, List[float]] = {c: [] for c in categories}
    all_scores: List[float] = []

    pbar = tqdm(total=total_samples, desc=f"Scoring (zlib:{args.mode})")
    for cat in categories:
        for text in cat_to_texts[cat]:
            if args.mode == "zlib_len":
                score = zlib_len(text, level=args.zlib_level)
            else:
                score = ll_over_zlib(text, model, tok, args.max_tokens, args.zlib_level)
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
        return score >= threshold if member_if == "gt" else score <= threshold

    cat_members: Dict[str, int] = {c: 0 for c in categories}
    for cat in categories:
        for score in cat_scores[cat]:
            if _is_member(score):
                cat_members[cat] += 1

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
    global_mixture = {cat: float(cat_members[cat] / total_members) for cat in categories} if total_members > 0 else {cat: 0.0 for cat in categories}

    print("\n" + "=" * 70)
    print("Per-Language Results (StarCoder):")
    print("=" * 70)
    for r in per_cat_results:
        score_str = f", score μ={r['score_mean']:.4f}" if r["score_mean"] is not None else ""
        print(
            f"  {r['category']:20s}: {r['n_members']:5d}/{r['n_valid']:5d} members "
            f"({r['member_proportion']:.2%}){score_str}"
        )

    print("\n" + "=" * 70)
    print("Predicted Language Composition (Global Mixture):")
    print("=" * 70)
    for cat in categories:
        print(f"  {cat:20s}: {global_mixture[cat]:.4f} ({global_mixture[cat]:.2%})")

    out_dir = Path(args.output_dir) / args.run_name
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "config": {
            "local_samples_dir": args.local_samples_dir,
            "spec_path": args.spec_path,
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
            "allow_missing": args.allow_missing,
            "taxonomy": "starcoder",
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

    print(f"\nWrote StarCoder zlib outputs to {out_dir}")


if __name__ == "__main__":
    main()

