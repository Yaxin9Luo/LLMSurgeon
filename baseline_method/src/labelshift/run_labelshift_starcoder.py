import argparse
import json
from pathlib import Path
from typing import Optional, List

import numpy as np
from tqdm import tqdm

from classifier import train_tfidf_classifier, train_distilbert_classifier
from generate import generate_texts, NEUTRAL_PROMPTS
from prior import estimate_priors_least_squares
from viz import plot_confusion_matrix, plot_priors_with_ci, plot_pbar_vs_ctpi
from data_utils_starcoder import build_balanced_splits_starcoder


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="StarCoder label-shift: language classifier + prior correction over 86 languages"
    )
    # Data
    p.add_argument(
        "--local_samples_dir",
        type=str,
        default=str(Path(__file__).resolve().parents[3] / "data_samples" / "starcoder"),
        help="Directory containing <language>.jsonl files (one per StarCoder language)",
    )
    p.add_argument("--max_per_class", type=int, default=2000)
    p.add_argument("--val_fraction", type=float, default=0.2)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--allow_missing", action="store_true", help="Proceed with subset of languages if some are missing")

    # Classifier
    p.add_argument("--classifier", type=str, choices=["tfidf", "distilbert"], default="tfidf")
    p.add_argument("--n_jobs", type=int, default=4)
    p.add_argument("--predict_batch_size", type=int, default=128)
    p.add_argument("--clf_verbose", type=int, default=0)

    # HF classifier (DistilBERT) options (works on code too, but TF-IDF is default for speed)
    p.add_argument("--hf_model_name", type=str, default="distilbert/distilbert-base-uncased")
    p.add_argument("--hf_epochs", type=int, default=20)
    p.add_argument("--hf_batch_size", type=int, default=64)
    p.add_argument("--hf_lr", type=float, default=2e-5)
    p.add_argument("--hf_weight_decay", type=float, default=0.01)
    p.add_argument("--hf_max_length", type=int, default=512)

    # Generations
    p.add_argument("--target_model", type=str, required=False, help="HF model name to sample from (e.g., bigcode/starcoder)")
    p.add_argument("--hf_revision", type=str, default=None, help="HF revision/commit/tag for --target_model")
    p.add_argument("--use_cached_generations", type=str, default=None, help="Path to JSONL with {text} per line to skip generation")
    p.add_argument("--num_prompts", type=int, default=300)
    p.add_argument("--prompts_file", type=str, default=None)
    p.add_argument("--max_new_tokens", type=int, default=512)
    p.add_argument("--gen_temperature", type=float, default=0.8)
    p.add_argument("--top_p", type=float, default=0.9)
    p.add_argument("--gen_batch_size", type=int, default=8)

    # Bootstrap (optional); disabled by default for 86 classes to keep runtime reasonable
    p.add_argument("--bootstrap", action="store_true")
    p.add_argument("--n_boot", type=int, default=200)

    # Output
    # Default outputs to repo-root 'out' to match evaluator convention
    p.add_argument("--output_dir", type=str, default=str(Path(__file__).resolve().parents[3] / "out"))
    p.add_argument("--run_name", type=str, default="starcoder_labelshift")
    return p.parse_args()


def load_prompts_from_file(path: str) -> List[str]:
    arr: List[str] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if s:
                arr.append(s)
    return arr


def read_jsonl_texts(path: str) -> List[str]:
    arr: List[str] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            t = obj.get("text")
            if isinstance(t, str) and t.strip():
                arr.append(t)
    return arr


def main() -> None:
    args = parse_args()

    # 1) Build balanced dataset across languages
    ds = build_balanced_splits_starcoder(
        samples_dir=args.local_samples_dir,
        max_per_class=args.max_per_class,
        val_fraction=args.val_fraction,
        seed=args.seed,
        require_all=not args.allow_missing,
    )
    K = len(ds.categories)

    # Prepare output directory
    out_dir = Path(args.output_dir) / args.run_name
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Training classifier over {K} languages using: {args.classifier}")

    # 2) Train classifier
    if args.classifier == "tfidf":
        model, clf_metrics, C = train_tfidf_classifier(
            ds.train.texts,
            ds.train.labels,
            ds.val.texts,
            ds.val.labels,
            seed=args.seed,
            n_jobs=args.n_jobs,
            verbose=args.clf_verbose,
        )
    else:
        model, clf_metrics, C = train_distilbert_classifier(
            ds.train.texts,
            ds.train.labels,
            ds.val.texts,
            ds.val.labels,
            model_name=args.hf_model_name,
            epochs=args.hf_epochs,
            batch_size=args.hf_batch_size,
            lr=args.hf_lr,
            weight_decay=args.hf_weight_decay,
            max_length=args.hf_max_length,
            seed=args.seed,
        )
    print(f"Classifier trained! Validation accuracy: {clf_metrics['val_acc']:.3f}")

    # 3) Gather model generations or load cached
    if args.use_cached_generations:
        gen_texts = read_jsonl_texts(args.use_cached_generations)
    else:
        if not args.target_model:
            raise ValueError("--target_model required unless --use_cached_generations is provided")
        if args.prompts_file:
            prompts = load_prompts_from_file(args.prompts_file)
            prompts = prompts[: args.num_prompts]
        else:
            # Neutral prompts suffice; they should elicit generic code/text from StarCoder
            prompts = NEUTRAL_PROMPTS[: args.num_prompts]
        gen_texts = generate_texts(
            model_name=args.target_model,
            prompts=prompts,
            max_new_tokens=args.max_new_tokens,
            temperature=args.gen_temperature,
            top_p=args.top_p,
            batch_size=args.gen_batch_size,
            revision=args.hf_revision,
        )

    # 4) Predict class probabilities on generations and average
    probs_chunks: List[np.ndarray] = []
    bs = max(1, int(args.predict_batch_size))
    for i in tqdm(range(0, len(gen_texts), bs), total=(len(gen_texts)+bs-1)//bs, desc="Predicting probs"):
        batch = gen_texts[i:i+bs]
        probs_chunks.append(model.predict_proba(batch))
    probs = np.concatenate(probs_chunks, axis=0) if probs_chunks else np.zeros((0, K))
    if probs.shape[0] == 0:
        raise RuntimeError("No generations were scored. Ensure prompts produced text and predict_proba returned outputs.")
    # Guard against numerical issues: enforce finite and row-stochastic probabilities
    if not np.all(np.isfinite(probs)):
        probs = np.nan_to_num(probs, nan=0.0, posinf=0.0, neginf=0.0)
    row_sums = probs.sum(axis=1, keepdims=True)
    # If any row is zero (all zeros), set to uniform
    bad = (row_sums <= 0)
    if np.any(bad):
        probs[bad[:, 0], :] = 1.0 / float(K)
        row_sums = probs.sum(axis=1, keepdims=True)
    probs = probs / row_sums
    pbar = probs.mean(axis=0)
    # Final sanity
    if not np.all(np.isfinite(pbar)):
        raise RuntimeError("Non-finite values in averaged probabilities pbar; check classifier outputs.")

    # 5) Prior correction: solve for mixture pi
    if C.shape != (K, K):
        raise RuntimeError(f"Confusion matrix shape {C.shape} does not match K={K} languages")
    if pbar.shape[0] != K:
        raise RuntimeError(f"pbar length {pbar.shape[0]} does not match K={K}")
    pi = estimate_priors_least_squares(C, pbar)

    # 6) Optional bootstrap for CIs
    pi_mean = pi
    lo = np.zeros_like(pi)
    hi = np.zeros_like(pi)
    if args.bootstrap:
        rng = np.random.default_rng(args.seed)
        N = probs.shape[0]
        pis = []
        for _ in tqdm(range(args.n_boot), desc="Bootstrapping"):
            idx = rng.integers(0, N, size=N)
            pbar_b = probs[idx].mean(axis=0)
            pi_b = estimate_priors_least_squares(C, pbar_b)
            pis.append(pi_b)
        P = np.stack(pis, axis=0)
        pi_mean = P.mean(axis=0)
        lo = np.percentile(P, 2.5, axis=0)
        hi = np.percentile(P, 97.5, axis=0)

    # 7) Write outputs
    payload = {
        "config": {
            "local_samples_dir": args.local_samples_dir,
            "classifier": args.classifier,
            "seed": args.seed,
            "val_fraction": args.val_fraction,
            "target_model": args.target_model,
            "num_prompts": args.num_prompts,
            "max_new_tokens": args.max_new_tokens,
            "gen_temperature": args.gen_temperature,
            "top_p": args.top_p,
            "hf_revision": args.hf_revision,
            "hf_model_name": args.hf_model_name,
            "hf_epochs": args.hf_epochs,
            "hf_batch_size": args.hf_batch_size,
            "hf_lr": args.hf_lr,
            "hf_weight_decay": args.hf_weight_decay,
            "hf_max_length": args.hf_max_length,
            "bootstrap": args.bootstrap,
            "n_boot": args.n_boot,
        },
        "categories": ds.categories,
        "val_metrics": clf_metrics,
        "confusion_matrix": C.tolist(),
        "pbar": pbar.tolist(),
        "priors": {
            "point": pi.tolist(),
            "mean": pi_mean.tolist(),
            "ci_lo": lo.tolist(),
            "ci_hi": hi.tolist(),
        },
    }
    with open(out_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    with open(out_dir / "summary.csv", "w", encoding="utf-8") as f:
        f.write("category,pi,ci_lo,ci_hi\n")
        for c, p, a, b in zip(ds.categories, pi_mean, lo, hi):
            f.write(f"{c},{p:.6f},{a:.6f},{b:.6f}\n")

    print(f"Wrote StarCoder label-shift outputs to {out_dir}")

    # 8) Visualizations
    try:
        plot_confusion_matrix(C, ds.categories, str(out_dir / "confusion_matrix.png"))
        plot_priors_with_ci(ds.categories, pi_mean, lo, hi, str(out_dir / "priors.png"))
        Ctpi = (C.T @ pi_mean)
        plot_pbar_vs_ctpi(ds.categories, pbar, Ctpi, str(out_dir / "pbar_vs_ctpi.png"))
    except Exception as e:
        print(f"Warning: plotting failed: {e}")


if __name__ == "__main__":
    main()
