#!/usr/bin/env python3
"""
Plot trend curves across multiple checkpoints (training steps/tokens).

This script scans result directories (default: out/) whose names match a
pattern like 'olmo1b_step*-mergeweb', loads their summary.json files,
computes evaluation metrics against a ground-truth spec, and produces trend
plots (metrics vs step/tokens and per-category estimates vs step).

Usage examples:
  python scripts/plot_ckpt_trends.py \
      --results_root out \
      --pattern 'olmo1b_step*_mergeweb' \
      --ground_truth bench/specs/olmo1b.yaml \
      --output_dir benchmark_output/olmo1b_ckpt_trends

Notes:
  - Expects each matched run under results_root/<run_name>/summary.json.
  - Supports token units 'B' (billions) and 'T' (trillions) in run names.
  - For 6-class (merge_web) runs, merges ground-truth CommonCrawl+C4 -> Web.
"""

import argparse
import json
import os
import re
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import yaml


ORDER_6 = ["Web", "GitHub", "Wikipedia", "Books", "Arxiv", "StackExchange"]
ORDER_7 = ["CommonCrawl", "C4", "GitHub", "Wikipedia", "Books", "Arxiv", "StackExchange"]


def load_ground_truth(path: str, *, merge_web: bool = True) -> Tuple[List[str], np.ndarray]:
    with open(path, "r", encoding="utf-8") as f:
        spec = yaml.safe_load(f)
    weights: Dict[str, float] = spec["category_weights"]
    if merge_web:
        cc = float(weights.get("CommonCrawl", 0.0))
        c4 = float(weights.get("C4", 0.0))
        gt = {k: float(v) for k, v in weights.items() if k not in ("CommonCrawl", "C4")}
        gt["Web"] = gt.get("Web", 0.0) + cc + c4
        cats = ORDER_6
        arr = np.array([gt.get(c, 0.0) for c in cats], dtype=float)
    else:
        cats = ORDER_7
        arr = np.array([float(weights.get(c, 0.0)) for c in cats], dtype=float)
    s = arr.sum()
    if s > 0:
        arr = arr / s
    return cats, arr


def parse_step_tokens_from_name(name: str) -> Tuple[int, float, str]:
    """Return (step_int, tokens_in_billions, raw_tokens_str) from run dir name.

    Supports patterns like:
      - ...step330000-tokens1384B...
      - ...ckpt150...
      - ...step30000_tokens126B...
    Tokens part is optional. Units: B=billions, T=trillions.
    """
    step = -1
    tokens_B = float("nan")
    raw = ""
    # Step/ckpt number
    m_step = re.search(r"(?:step|ckpt)[-_]?(\d+)", name, flags=re.IGNORECASE)
    if m_step:
        step = int(m_step.group(1))
    # Tokens (optional)
    m_tok = re.search(r"tokens[_-]?(\d+)([BT])", name, flags=re.IGNORECASE)
    if m_tok:
        num = float(m_tok.group(1))
        unit = m_tok.group(2).upper()
        tokens_B = num * (1000.0 if unit == "T" else 1.0)
        raw = f"{int(num) if num.is_integer() else num}{unit}"
    return step, tokens_B, raw


def load_run_summary(path: Path) -> Dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def align_estimates_6(summary: Dict) -> Tuple[List[str], np.ndarray]:
    cats = summary.get("categories", [])
    est = summary.get("priors", {}).get("mean") or summary.get("priors", {}).get("point")
    if not cats or est is None:
        raise ValueError("Invalid summary.json: missing categories/priors")
    mapping = {c: float(v) for c, v in zip(cats, est)}
    arr = np.array([mapping.get(c, 0.0) for c in ORDER_6], dtype=float)
    s = arr.sum()
    if s > 0:
        arr = arr / s
    return ORDER_6, arr


def compute_metrics(est: np.ndarray, gt: np.ndarray) -> Dict[str, float]:
    eps = 1e-10
    l1 = float(np.sum(np.abs(gt - est)))
    l2 = float(np.sqrt(np.sum((gt - est) ** 2)))
    kl = float(np.sum(gt * np.log((gt + eps) / (est + eps))))
    abs_err = np.abs(gt - est)
    return {
        "l1": l1,
        "l2": l2,
        "kl": kl,
        "mae": float(np.mean(abs_err)),
        "max_abs": float(np.max(abs_err)),
    }


def collect_runs(results_root: str, pattern: str) -> List[Path]:
    root = Path(results_root)
    runs = [p for p in root.iterdir() if p.is_dir()]
    # Glob-like simple filter: convert pattern '*' to regex '.*'
    regex = re.compile("^" + re.escape(pattern).replace("\\*", ".*") + "$")
    out = [p for p in runs if regex.match(p.name)]
    return sorted(out, key=lambda p: parse_step_tokens_from_name(p.name)[0])


def make_plots(df: pd.DataFrame, cats: List[str], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    sns.set_context("talk")

    # Metrics vs step
    plt.figure(figsize=(9, 6))
    for key, label in [("l1", "L1"), ("l2", "L2"), ("kl", "KL"), ("mae", "MAE"), ("max_abs", "MaxAbs")]:
        plt.plot(df["step"], df[key], marker="o", label=label)
    plt.xlabel("Training step")
    plt.ylabel("Metric value")
    plt.title("Evaluation metrics vs training step")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "metrics_vs_step.png", dpi=200)
    plt.close()

    # Metrics vs tokens (B)
    plt.figure(figsize=(9, 6))
    for key, label in [("l1", "L1"), ("l2", "L2"), ("kl", "KL"), ("mae", "MAE"), ("max_abs", "MaxAbs")]:
        plt.plot(df["tokens_B"], df[key], marker="o", label=label)
    plt.xlabel("Tokens seen (Billions)")
    plt.ylabel("Metric value")
    plt.title("Evaluation metrics vs tokens")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "metrics_vs_tokens.png", dpi=200)
    plt.close()

    # Category estimates vs step
    plt.figure(figsize=(10, 6))
    for c in cats:
        plt.plot(df["step"], df[f"est_{c}"], marker="o", label=c)
    # Ground truth overlay as dashed horizontal lines
    for c in cats:
        v = float(df[f"gt_{c}"].iloc[0])
        plt.axhline(v, color="gray", linestyle="--", linewidth=1)
    plt.xlabel("Training step")
    plt.ylabel("Estimated proportion")
    plt.title("Per-category estimates vs training step (dashed=GT)")
    plt.grid(True, alpha=0.3)
    plt.legend(ncol=3)
    plt.tight_layout()
    plt.savefig(output_dir / "category_estimates_vs_step.png", dpi=200)
    plt.close()

    # Category absolute error vs step
    plt.figure(figsize=(10, 6))
    for c in cats:
        plt.plot(df["step"], df[f"abs_err_{c}"], marker="o", label=c)
    plt.xlabel("Training step")
    plt.ylabel("Absolute error")
    plt.title("Per-category absolute error vs training step")
    plt.grid(True, alpha=0.3)
    plt.legend(ncol=3)
    plt.tight_layout()
    plt.savefig(output_dir / "category_abs_error_vs_step.png", dpi=200)
    plt.close()


def main():
    ap = argparse.ArgumentParser(description="Plot trends across checkpoints")
    ap.add_argument("--results_root", type=str, default="out", help="Root with run directories (each has summary.json)")
    ap.add_argument("--pattern", type=str, default="olmo1b_step*_mergeweb", help="Glob-like pattern for run directory names")
    ap.add_argument("--ground_truth", type=str, default="bench/specs/olmo1b.yaml", help="Ground-truth YAML file")
    ap.add_argument("--output_dir", type=str, default="benchmark_output/olmo1b_ckpt_trends", help="Where to write plots/CSV")
    args = ap.parse_args()

    runs = collect_runs(args.results_root, args.pattern)
    if not runs:
        print(f"No runs found under {args.results_root} matching '{args.pattern}'")
        return

    cats, gt = load_ground_truth(args.ground_truth, merge_web=True)
    rows = []
    for run in runs:
        summary_path = run / "summary.json"
        if not summary_path.exists():
            print(f"Warning: missing summary.json in {run}")
            continue
        step, tokens_B, raw_tokens = parse_step_tokens_from_name(run.name)
        # Optional ckpt label (string like 010, 050, 100)
        m_ckpt = re.search(r"ckpt[_-]?(\d+)", run.name, flags=re.IGNORECASE)
        ckpt_label = m_ckpt.group(1) if m_ckpt else ""
        try:
            summary = load_run_summary(summary_path)
            _, est = align_estimates_6(summary)
        except Exception as e:
            print(f"Warning: failed to read {summary_path}: {e}")
            continue
        metrics = compute_metrics(est, gt)
        row = {
            "run": run.name,
            "step": step,
            "tokens_B": tokens_B,
            "tokens_raw": raw_tokens,
            "ckpt": ckpt_label,
            **metrics,
        }
        # Per-category values
        for i, c in enumerate(cats):
            row[f"est_{c}"] = float(est[i])
            row[f"gt_{c}"] = float(gt[i])
            row[f"abs_err_{c}"] = abs(float(est[i] - gt[i]))
        rows.append(row)

    if not rows:
        print("No valid runs to plot.")
        return

    df = pd.DataFrame(rows)
    # Sort by ckpt (if present), else step (if available), else tokens_B
    if (df.get("ckpt") is not None) and (df["ckpt"] != "").any():
        # numeric sort on ckpt string
        df["ckpt_num"] = pd.to_numeric(df["ckpt"], errors="coerce")
        df = df.sort_values(by=["ckpt_num", "tokens_B", "step"]).reset_index(drop=True)
    elif (df["step"] >= 0).any():
        df = df.sort_values(by=["step", "tokens_B"]).reset_index(drop=True)
    else:
        df = df.sort_values(by=["tokens_B"]).reset_index(drop=True)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_dir / "summary_trends.csv", index=False)
    make_plots(df, cats, out_dir)

    # Additional plots with ckpt labels on x-axis if available
    if (df.get("ckpt") is not None) and (df["ckpt"] != "").any():
        # Use categorical axis with provided labels
        order = df.sort_values(by=["ckpt_num"]) if "ckpt_num" in df.columns else df
        x = order["ckpt"].astype(str).tolist()

        plt.figure(figsize=(9, 6))
        for key, label in [("l1", "L1"), ("l2", "L2"), ("kl", "KL"), ("mae", "MAE"), ("max_abs", "MaxAbs")]:
            plt.plot(x, order[key], marker="o", label=label)
        plt.xlabel("Checkpoint (ckpt)")
        plt.ylabel("Metric value")
        plt.title("Evaluation metrics vs checkpoint")
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig(out_dir / "metrics_vs_ckpt.png", dpi=200)
        plt.close()

        plt.figure(figsize=(10, 6))
        for c in cats:
            plt.plot(x, order[f"est_{c}"], marker="o", label=c)
        for c in cats:
            v = float(df[f"gt_{c}"].iloc[0])
            plt.axhline(v, color="gray", linestyle="--", linewidth=1)
        plt.xlabel("Checkpoint (ckpt)")
        plt.ylabel("Estimated proportion")
        plt.title("Per-category estimates vs checkpoint (dashed=GT)")
        plt.grid(True, alpha=0.3)
        plt.legend(ncol=3)
        plt.tight_layout()
        plt.savefig(out_dir / "category_estimates_vs_ckpt.png", dpi=200)
        plt.close()

        plt.figure(figsize=(10, 6))
        for c in cats:
            plt.plot(x, order[f"abs_err_{c}"], marker="o", label=c)
        plt.xlabel("Checkpoint (ckpt)")
        plt.ylabel("Absolute error")
        plt.title("Per-category absolute error vs checkpoint")
        plt.grid(True, alpha=0.3)
        plt.legend(ncol=3)
        plt.tight_layout()
        plt.savefig(out_dir / "category_abs_error_vs_ckpt.png", dpi=200)
        plt.close()

    print(f"Wrote trend CSV and plots to {out_dir}")


if __name__ == "__main__":
    main()
