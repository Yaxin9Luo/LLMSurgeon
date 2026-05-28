import argparse
import json
import os
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from tqdm import tqdm
from sklearn.mixture import GaussianMixture

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM

from data_utils import build_balanced_splits


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Estimate category mixture via Min-K%++ membership inference over local samples"
    )
    # Data
    p.add_argument(
        "--local_samples_dir",
        type=str,
        default=str("/data/yaxin/data_anatomy/data_samples"),
        help="Directory with category JSONL files (e.g., wikipedia.jsonl, books.jsonl, ...)",
    )
    p.add_argument("--merge_web", action="store_true", help="Merge CommonCrawl and C4 into Web (6-way)")
    p.add_argument("--max_per_class", type=int, default=3000)
    p.add_argument("--val_fraction", type=float, default=0.2)
    p.add_argument("--seed", type=int, default=0)

    # Target model
    p.add_argument("--target_model", type=str, required=True, help="HF model id for scoring (causal LM)")
    p.add_argument("--half", action="store_true", help="Load model in bfloat16")
    p.add_argument("--int8", action="store_true", help="Load model in 8-bit (bitsandbytes)")
    p.add_argument("--max_tokens", type=int, default=128, help="Max tokens per sample for scoring")

    # Min-K++
    p.add_argument("--mink_ratio", type=float, default=0.1, help="k% for Min-K++ (0<r<=1)")

    # Bootstrap for per-category CI (optional)
    p.add_argument("--bootstrap", action="store_true")
    p.add_argument("--n_boot", type=int, default=200)

    # Output
    p.add_argument("--output_dir", type=str, default=str(Path(__file__).resolve().parents[2] / "out"))
    p.add_argument("--run_name", type=str, default="minkpp_mix")
    return p.parse_args()


def load_model(name: str, half: bool, int8: bool):
    int8_kwargs = {}
    half_kwargs = {}
    if int8:
        int8_kwargs = dict(load_in_8bit=True, torch_dtype=torch.bfloat16)
    elif half:
        half_kwargs = dict(torch_dtype=torch.bfloat16)
    model = AutoModelForCausalLM.from_pretrained(
        name, return_dict=True, device_map='auto', **int8_kwargs, **half_kwargs
    )
    model.eval()
    tok = AutoTokenizer.from_pretrained(name)
    return model, tok


@torch.no_grad()
def minkpp_score(text: str, model, tok, max_tokens: int, ratio: float) -> float:
    # Tokenize with truncation
    ids = tok.encode(text, truncation=True, max_length=max_tokens)
    if len(ids) < 2:
        return float("nan")
    input_ids = torch.tensor(ids, dtype=torch.long).unsqueeze(0).to(model.device)
    out = model(input_ids, labels=input_ids)
    loss, logits = out[:2]
    logits = logits[0, :-1]  # [T-1, V]
    target_ids = input_ids[0, 1:].unsqueeze(-1)  # [T-1, 1]

    probs = F.softmax(logits, dim=-1)
    log_probs = F.log_softmax(logits, dim=-1)
    token_log_probs = log_probs.gather(dim=-1, index=target_ids).squeeze(-1)  # [T-1]
    mu = (probs * log_probs).sum(-1)
    sigma = (probs * torch.square(log_probs)).sum(-1) - torch.square(mu)
    sigma = torch.clamp(sigma, min=1e-8)

    z = (token_log_probs - mu) / torch.sqrt(sigma)  # standardized
    k = max(1, int(z.numel() * ratio))
    vals, _ = torch.sort(z)
    return float(vals[:k].mean().item())


def fit_membership_rate(scores: np.ndarray, seed: int) -> Tuple[float, np.ndarray]:
    """
    Fit a 2-component Gaussian mixture on 1D Min-K++ scores. Return
    (pi_member, comp_means) where pi_member is the weight of the higher-mean
    component (assumed to correspond to 'member' given larger score => member).
    """
    X = scores.reshape(-1, 1).astype(np.float64)
    # Remove NaNs
    X = X[np.isfinite(X[:, 0])]
    if X.shape[0] < 2:
        return float("nan"), np.array([np.nan, np.nan])
    gm = GaussianMixture(n_components=2, random_state=seed)
    gm.fit(X)
    means = gm.means_.flatten()
    weights = gm.weights_.flatten()
    member_idx = int(np.argmax(means))
    return float(weights[member_idx]), means


def bootstrap_ci(scores: np.ndarray, seed: int, n_boot: int = 200) -> Tuple[float, float, float]:
    rng = np.random.default_rng(seed)
    S = scores[np.isfinite(scores)]
    if S.size < 2:
        return float("nan"), float("nan"), float("nan")
    ests = []
    N = S.shape[0]
    for _ in range(n_boot):
        idx = rng.integers(0, N, size=N)
        pi_b, _ = fit_membership_rate(S[idx], seed)
        if np.isfinite(pi_b):
            ests.append(pi_b)
    if not ests:
        return float("nan"), float("nan"), float("nan")
    arr = np.array(ests)
    return float(arr.mean()), float(np.percentile(arr, 2.5)), float(np.percentile(arr, 97.5))


def main() -> None:
    args = parse_args()

    # Build balanced category dataset (train/val splits not used separately here)
    ds = build_balanced_splits(
        local_dir=args.local_samples_dir,
        merge_web=args.merge_web,
        max_per_class=args.max_per_class,
        val_fraction=args.val_fraction,
        seed=args.seed,
    )
    categories = ds.categories

    # Collect texts per category using both train and val for more coverage
    cat_texts: Dict[int, List[str]] = {i: [] for i in range(len(categories))}
    for t, y in zip(ds.train.texts, ds.train.labels):
        cat_texts[y].append(t)
    for t, y in zip(ds.val.texts, ds.val.labels):
        cat_texts[y].append(t)

    # Load target model
    print(f"Loading model: {args.target_model}")
    model, tok = load_model(args.target_model, half=args.half, int8=args.int8)

    # Score all samples
    cat_scores: Dict[int, List[float]] = {i: [] for i in range(len(categories))}
    total = sum(len(v) for v in cat_texts.values())
    pbar = tqdm(total=total, desc="Scoring (Min-K%++)")
    for ci, texts in cat_texts.items():
        for x in texts:
            s = minkpp_score(x, model, tok, args.max_tokens, args.mink_ratio)
            cat_scores[ci].append(s)
            pbar.update(1)
    pbar.close()

    # Estimate per-category membership rate via GMM
    rng = np.random.default_rng(args.seed)
    per_cat = []
    weights_detected = []  # n_c * pi_c for global mixture share
    for i, c in enumerate(categories):
        sc = np.array(cat_scores[i], dtype=float)
        n = int(np.isfinite(sc).sum())
        pi_hat, means = fit_membership_rate(sc, seed=args.seed)
        mu = float(np.nanmean(sc)) if n > 0 else float("nan")
        sd = float(np.nanstd(sc)) if n > 1 else float("nan")
        if args.bootstrap:
            m, lo, hi = bootstrap_ci(sc, seed=args.seed, n_boot=args.n_boot)
            pi_mean, ci_lo, ci_hi = m, lo, hi
        else:
            pi_mean, ci_lo, ci_hi = pi_hat, float("nan"), float("nan")
        per_cat.append({
            "category": c,
            "n": n,
            "pi_hat": float(pi_hat) if np.isfinite(pi_hat) else None,
            "pi_mean": float(pi_mean) if np.isfinite(pi_mean) else None,
            "ci_lo": float(ci_lo) if np.isfinite(ci_lo) else None,
            "ci_hi": float(ci_hi) if np.isfinite(ci_hi) else None,
            "score_mean": mu,
            "score_std": sd,
            "gmm_means": [float(m) if np.isfinite(m) else None for m in means],
        })
        w = max(0, n) * (pi_hat if np.isfinite(pi_hat) else 0.0)
        weights_detected.append(w)

    # Global mixture over sampled pool: normalize detected members per category
    W = np.array(weights_detected, dtype=float)
    if np.sum(W) > 0:
        global_mix = (W / np.sum(W)).tolist()
        overall_coverage = float(np.sum(W) / sum(len(v) for v in cat_texts.values()))
    else:
        global_mix = [0.0 for _ in categories]
        overall_coverage = 0.0

    # Write outputs
    out_dir = Path(args.output_dir) / args.run_name
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "config": {
            "local_samples_dir": args.local_samples_dir,
            "merge_web": args.merge_web,
            "max_per_class": args.max_per_class,
            "val_fraction": args.val_fraction,
            "seed": args.seed,
            "target_model": args.target_model,
            "half": args.half,
            "int8": args.int8,
            "max_tokens": args.max_tokens,
            "mink_ratio": args.mink_ratio,
            "bootstrap": args.bootstrap,
            "n_boot": args.n_boot,
        },
        "categories": categories,
        "per_category": per_cat,
        "global_mixture_over_sampled_pool": global_mix,
        "overall_coverage_over_sampled_pool": overall_coverage,
    }
    with open(out_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    # CSV with per-category pi and global mix
    with open(out_dir / "per_category.csv", "w", encoding="utf-8") as f:
        f.write("category,n,pi_hat,pi_mean,ci_lo,ci_hi,score_mean,score_std\n")
        for r in per_cat:
            f.write(
                f"{r['category']},{r['n']},{r['pi_hat'] if r['pi_hat'] is not None else ''},{r['pi_mean'] if r['pi_mean'] is not None else ''},"
                f"{r['ci_lo'] if r['ci_lo'] is not None else ''},{r['ci_hi'] if r['ci_hi'] is not None else ''},"
                f"{r['score_mean'] if np.isfinite(r['score_mean']) else ''},{r['score_std'] if np.isfinite(r['score_std']) else ''}\n"
            )
    with open(out_dir / "global_mixture.csv", "w", encoding="utf-8") as f:
        f.write("category,share_over_sampled_pool\n")
        for c, w in zip(categories, global_mix):
            f.write(f"{c},{w:.6f}\n")

    print(f"Wrote Min-K++ mixture outputs to {out_dir}")


if __name__ == "__main__":
    main()

