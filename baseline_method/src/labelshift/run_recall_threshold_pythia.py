"""
ReCaLL with Binary Thresholding for Category Composition Estimation (Pythia Taxonomy)

This script adapts ReCaLL (Xie et al., EMNLP 2024) to predict category composition
percentages of LLM pretraining data using the Pythia/Pile taxonomy.

Approach:
1. Load category samples from local_samples_dir (Pile sources)
2. Score each sample with ReCaLL: score = ll_cond(prefix, x) / ll(x)
3. Threshold: score > threshold → member
4. Normalize member counts across categories
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

from data_utils_pythia import detect_available_pythia_categories, PILE_FILE_TO_PYTHIA_CAT
from data_utils import _read_jsonl


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Estimate category mixture via ReCaLL with binary thresholding (Pythia taxonomy)"
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

    # ReCaLL parameters
    p.add_argument("--num_shots", type=int, default=12)
    p.add_argument("--prefix_strategy", type=str, choices=["global", "leave_one_out"], default="global")
    p.add_argument("--threshold", type=float, default=None)
    p.add_argument("--member_if", type=str, default="gt", choices=["gt", "lt"])

    # Output
    p.add_argument("--output_dir", type=str, default=str(Path(__file__).resolve().parents[3] / "out"))
    p.add_argument("--run_name", type=str, default="recall_threshold_pythia")
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


def _tok_encode(tok, text: str, max_len: int) -> List[int]:
    return tok.encode(text, truncation=True, max_length=max_len)


def _build_prefix_text(prefix_shots: List[str]) -> str:
    return "\n\n".join(prefix_shots)


@torch.no_grad()
def _avg_ll_unconditional(text: str, model, tok, max_tokens: int) -> float:
    ids = _tok_encode(tok, text, max_tokens)
    if len(ids) < 2:
        return float("nan")
    input_ids = torch.tensor(ids, dtype=torch.long).unsqueeze(0).to(model.device)
    out = model(input_ids, labels=input_ids)
    return float((-out.loss).item())


@torch.no_grad()
def _avg_ll_conditional(prefix_text: str, target_text: str, model, tok, max_tokens: int) -> float:
    tgt_ids = _tok_encode(tok, target_text, max_tokens)
    if len(tgt_ids) < 2:
        return float("nan")
    max_prefix_len = max(0, max_tokens - len(tgt_ids))
    pre_ids = _tok_encode(tok, prefix_text, max_prefix_len) if max_prefix_len > 0 else []
    if len(pre_ids) + len(tgt_ids) > max_tokens:
        keep = max(0, max_tokens - len(tgt_ids))
        pre_ids = pre_ids[-keep:] if keep > 0 else []
    concat_ids = pre_ids + tgt_ids
    labels = ([-100] * len(pre_ids)) + tgt_ids
    input_ids = torch.tensor(concat_ids, dtype=torch.long).unsqueeze(0).to(model.device)
    label_ids = torch.tensor(labels, dtype=torch.long).unsqueeze(0).to(model.device)
    out = model(input_ids, labels=label_ids)
    return float((-out.loss).item())


@torch.no_grad()
def recall_score(text: str, prefix_text: str, model, tok, max_tokens: int) -> float:
    ll = _avg_ll_unconditional(text, model, tok, max_tokens)
    if not np.isfinite(ll) or ll == 0.0:
        return float("nan")
    llc = _avg_ll_conditional(prefix_text, text, model, tok, max_tokens)
    if not np.isfinite(llc):
        return float("nan")
    return float(llc / ll)


def _sample_prefix_shots(pool: List[str], num_shots: int, seed: int) -> List[str]:
    import random
    if num_shots <= 0 or not pool:
        return []
    rng = random.Random(seed)
    if len(pool) >= num_shots:
        return rng.sample(pool, k=num_shots)
    return [pool[rng.randrange(len(pool))] for _ in range(num_shots)]


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

    pooled_all: List[str] = []
    for c in categories:
        pooled_all.extend(cat_to_texts[c])

    global_prefix_text = ""
    if args.prefix_strategy == "global":
        prefix_shots = _sample_prefix_shots(pooled_all, args.num_shots, args.seed)
        global_prefix_text = _build_prefix_text(prefix_shots)
        print(f"Prefix strategy: global | num_shots={args.num_shots}")
    else:
        print(f"Prefix strategy: leave_one_out | num_shots={args.num_shots}")

    print("\nScoring samples with ReCaLL")
    cat_scores: Dict[str, List[float]] = {c: [] for c in categories}
    all_scores: List[float] = []

    pbar = tqdm(total=total_samples, desc="Scoring (ReCaLL)")
    for cat in categories:
        if args.prefix_strategy == "leave_one_out":
            loo_pool = [t for c2 in categories if c2 != cat for t in cat_to_texts[c2]]
            prefix_shots = _sample_prefix_shots(loo_pool, args.num_shots, args.seed)
            prefix_text = _build_prefix_text(prefix_shots)
        else:
            prefix_text = global_prefix_text

        for text in cat_to_texts[cat]:
            score = recall_score(text, prefix_text, model, tok, args.max_tokens)
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

    cat_members: Dict[str, int] = {c: sum(1 for s in cat_scores[c] if _is_member(s)) for c in categories}

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
            "num_shots": args.num_shots, "prefix_strategy": args.prefix_strategy,
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

