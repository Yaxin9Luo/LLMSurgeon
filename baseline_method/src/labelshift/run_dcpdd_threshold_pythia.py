"""
DC-PDD with Binary Thresholding for Category Composition Estimation (Pythia Taxonomy)

This script adapts DC-PDD (Zhang et al., 2024) to predict category composition
percentages of LLM pretraining data using the Pythia/Pile taxonomy.

DC-PDD score:
    score(x) = - mean_i [ p_i * log(1 / f(tok_i)) ]  (with clipping at a)

Lower (more negative) score → member. Default: member_if=lt.
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

from data_utils_pythia import detect_available_pythia_categories, PILE_FILE_TO_PYTHIA_CAT
from data_utils import _read_jsonl


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Estimate category mixture via DC-PDD with binary thresholding (Pythia taxonomy)"
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

    # DC-PDD parameters
    p.add_argument("--a", type=float, default=0.01, help="DC-PDD clipping hyperparameter")
    p.add_argument("--freq_strategy", type=str, choices=["global", "leave_one_out"], default="global")
    p.add_argument("--max_ref_texts", type=int, default=None)

    # Thresholding
    p.add_argument("--threshold", type=float, default=None)
    p.add_argument("--member_if", type=str, default="lt", choices=["gt", "lt"])

    # Output
    p.add_argument("--output_dir", type=str, default=str(Path(__file__).resolve().parents[3] / "out"))
    p.add_argument("--run_name", type=str, default="dcpdd_threshold_pythia")
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
    import random
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


def _vocab_size(tok) -> int:
    try:
        return len(tok)
    except Exception:
        return int(getattr(tok, "vocab_size", 0))


def _sample_pool(texts: List[str], max_n: int, seed: int) -> List[str]:
    if max_n is None or max_n <= 0 or len(texts) <= max_n:
        return texts
    import random
    rng = random.Random(seed)
    return rng.sample(texts, k=max_n)


def _freq_distribution_from_texts(tok, texts: List[str], max_tokens: int, seed: int, max_ref_texts: int = None) -> np.ndarray:
    vs = _vocab_size(tok)
    if vs <= 0:
        raise ValueError("Could not determine tokenizer vocab size")
    counts = np.zeros((vs,), dtype=np.int64)
    texts_use = _sample_pool(texts, max_ref_texts, seed)
    for t in texts_use:
        ids = tok.encode(t, truncation=True, max_length=max_tokens)
        for tid in ids:
            if 0 <= tid < vs:
                counts[tid] += 1
    total = int(counts.sum())
    fre = (counts.astype(np.float64) + 1.0) / (float(total) + float(vs))
    return fre


@torch.no_grad()
def dcpdd_score(text: str, model, tok, fre: np.ndarray, max_tokens: int, a: float) -> float:
    ids = tok.encode(text, truncation=True, max_length=max_tokens)
    if len(ids) < 2:
        return float("nan")

    input_ids = torch.tensor(ids, dtype=torch.long).unsqueeze(0).to(model.device)
    out = model(input_ids, labels=input_ids)
    logits = out.logits[0, :-1]
    target_ids = input_ids[0, 1:]

    log_probs = F.log_softmax(logits, dim=-1)
    token_log_probs = log_probs[torch.arange(logits.size(0), device=logits.device), target_ids]
    probs = token_log_probs.float().exp().detach().cpu().numpy()
    ids_shift = target_ids.detach().cpu().numpy().astype(np.int64)

    # First-occurrence positions
    seen = set()
    idxs: List[int] = []
    for i, tid in enumerate(ids_shift.tolist()):
        if tid not in seen:
            idxs.append(i)
            seen.add(tid)
    if not idxs:
        return float("nan")

    x_pro = probs[idxs]
    if ids_shift.max(initial=0) >= fre.shape[0]:
        vs_new = int(ids_shift.max()) + 1
        fre2 = np.full((vs_new,), 1.0 / float(vs_new), dtype=np.float64)
        fre2[:fre.shape[0]] = fre
        fre_use = fre2
    else:
        fre_use = fre
    x_fre = fre_use[ids_shift[idxs]]

    ce = x_pro * np.log(1.0 / np.clip(x_fre, 1e-300, 1.0))
    ce = np.minimum(ce, float(a))
    return float(-np.mean(ce))


def main() -> None:
    args = parse_args()

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

    global_fre: np.ndarray | None = None
    if args.freq_strategy == "global":
        print(f"Estimating token frequency (global)")
        global_fre = _freq_distribution_from_texts(tok, pooled_all, args.max_tokens, args.seed, args.max_ref_texts)
    else:
        print(f"Estimating token frequency (leave_one_out)")

    print(f"\nScoring samples with DC-PDD (a={args.a})")
    cat_scores: Dict[str, List[float]] = {c: [] for c in categories}
    all_scores: List[float] = []

    pbar = tqdm(total=total_samples, desc="Scoring (DC-PDD)")
    for cat in categories:
        if args.freq_strategy == "leave_one_out":
            loo_pool = [t for c2 in categories if c2 != cat for t in cat_to_texts[c2]]
            fre = _freq_distribution_from_texts(tok, loo_pool, args.max_tokens, args.seed, args.max_ref_texts)
        else:
            fre = global_fre
        assert fre is not None

        for text in cat_to_texts[cat]:
            score = dcpdd_score(text, model, tok, fre, args.max_tokens, args.a)
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
            "a": args.a, "freq_strategy": args.freq_strategy, "max_ref_texts": args.max_ref_texts,
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

