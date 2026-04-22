import argparse
import json
from pathlib import Path
from typing import List, Tuple
from itertools import cycle, islice

import numpy as np
from tqdm import tqdm

from data_utils_pythia import build_balanced_splits_pythia
from classifier import train_tfidf_classifier, train_distilbert_classifier
from generate import (
    generate_texts,
    NEUTRAL_PROMPTS,
    INSTRUCTIONAL_PROMPTS,
    EXPOSITORY_PROMPTS,
    CONVERSATIONAL_PROMPTS,
    CODING_PROMPTS,
)
from prior import estimate_priors_least_squares
from viz import plot_confusion_matrix, plot_priors_with_ci, plot_pbar_vs_ctpi


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Label-shift mixture estimation (Pythia taxonomy) via domain classifier + prior correction"
    )
    # Data (Pile-style)
    p.add_argument("--local_samples_dir", type=str, default=str(Path(__file__).resolve().parents[2] / "data_samples" / "pile"))
    p.add_argument("--max_per_class", type=int, default=5000)
    p.add_argument("--val_fraction", type=float, default=0.2)
    p.add_argument("--seed", type=int, default=0)

    # Classifier
    p.add_argument("--classifier", type=str, choices=["tfidf", "distilbert"], default="distilbert")
    p.add_argument("--n_jobs", type=int, default=4)
    p.add_argument("--predict_batch_size", type=int, default=256)
    p.add_argument("--clf_verbose", type=int, default=0)

    # HF classifier (DistilBERT) options
    p.add_argument("--hf_model_name", type=str, default="distilbert/distilbert-base-uncased")
    p.add_argument("--hf_epochs", type=int, default=3)
    p.add_argument("--hf_batch_size", type=int, default=64)
    p.add_argument("--hf_lr", type=float, default=2e-5)
    p.add_argument("--hf_weight_decay", type=float, default=0.01)
    p.add_argument("--hf_max_length", type=int, default=256)
    p.add_argument(
        "--hf_pretrained_dir",
        type=str,
        default=None,
        help="Path to a fine-tuned HF classifier directory to load instead of training from scratch",
    )
    p.add_argument(
        "--hf_train_from_scratch",
        action="store_true",
        help="If set, train DistilBERT classifier from scratch instead of fine-tuning a pre-trained model",
    )

    # Generations
    p.add_argument("--generator", type=str, choices=["hf"], default="hf", help="Generation engine to use.")
    p.add_argument("--target_model", type=str, required=False, help="HF model name (for --generator hf)")
    p.add_argument("--hf_revision", type=str, default=None, help="HF revision/commit/tag for --target_model")
    p.add_argument("--use_cached_generations", type=str, default=None, help="Path to JSONL with {text} per line to skip generation")
    p.add_argument("--num_prompts", type=int, default=400)
    p.add_argument("--prompts_style", type=str, default="neutral", help="Prompt style to use")
    p.add_argument("--max_new_tokens", type=int, default=512)
    p.add_argument("--gen_temperature", type=float, default=0.8)
    p.add_argument("--top_p", type=float, default=0.9)
    p.add_argument("--gen_batch_size", type=int, default=8)

    # Bootstrap
    p.add_argument("--bootstrap", action="store_true")
    p.add_argument("--n_boot", type=int, default=300)

    # Output
    p.add_argument("--output_dir", type=str, default=str(Path(__file__).resolve().parents[1] / "out"))
    p.add_argument("--run_name", type=str, default="labelshift_pythia")
    p.add_argument("--naive", action="store_true", help="Use naive Probabilistic Classify and Count (PCC) baseline (skip confusion matrix correction)")
    p.add_argument("--simulate_label_shift", action="store_true", help="Run a synthetic label shift experiment on validation data to measure estimation accuracy")
    p.add_argument("--acc_log_file", type=str, default=None, help="Path to CSV file to append (max_per_class, val_acc, estimation_acc) results")
    return p.parse_args()


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

    # 1) Build balanced dataset (Pythia taxonomy)
    ds = build_balanced_splits_pythia(
        local_dir=args.local_samples_dir,
        max_per_class=args.max_per_class,
        val_fraction=args.val_fraction,
        seed=args.seed,
    )
    K = len(ds.categories)

    out_dir = Path(args.output_dir) / args.run_name
    out_dir.mkdir(parents=True, exist_ok=True)

    # 2) Train classifier
    print(f"Training classifier: {args.classifier}...")
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
            pretrained_dir=args.hf_pretrained_dir,
            train_from_scratch=args.hf_train_from_scratch,
            epochs=args.hf_epochs,
            batch_size=args.hf_batch_size,
            lr=args.hf_lr,
            weight_decay=args.hf_weight_decay,
            max_length=args.hf_max_length,
            seed=args.seed,
        )
    print(f"Classifier trained! Validation accuracy: {clf_metrics['val_acc']:.3f}")

    # 2.5) Optional: Simulate label shift on validation data
    if args.simulate_label_shift:
        print("Running synthetic label shift simulation on validation set...")
        from sklearn.metrics import confusion_matrix as sklearn_confusion_matrix
        
        # Split val into calib (50%) and test (50%)
        rng_sim = np.random.default_rng(args.seed + 123)
        val_indices = np.arange(len(ds.val.texts))
        rng_sim.shuffle(val_indices)
        half = len(val_indices) // 2
        idx_calib = val_indices[:half]
        idx_test = val_indices[half:]

        # 1. Get predictions for all validation data
        val_probs_all = model.predict_proba(ds.val.texts)
        val_preds_all = np.argmax(val_probs_all, axis=1)
        
        # Calibration part
        y_calib = np.array(ds.val.labels)[idx_calib]
        preds_calib = val_preds_all[idx_calib]
        
        # Re-compute C from calib split
        cm_calib = sklearn_confusion_matrix(y_calib, preds_calib, labels=range(K))
        row_sums = cm_calib.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1
        C_calib = cm_calib / row_sums
        
        # Test part: construct a random target mixture
        # Sample a random Dirichlet distribution
        target_pi = rng_sim.dirichlet(alpha=np.ones(K))
        
        # Resample test indices to match target_pi
        # We'll pick a fixed size target set, e.g. same size as idx_test
        n_target = len(idx_test)
        target_counts = (target_pi * n_target).astype(int)
        # Fix rounding
        target_counts[-1] = n_target - target_counts[:-1].sum()
        
        # Group test indices by class
        y_test_all = np.array(ds.val.labels)[idx_test]
        test_by_class = {k: idx_test[y_test_all == k] for k in range(K)}
        
        chosen_indices = []
        for k in range(K):
            needed = target_counts[k]
            available = test_by_class[k]
            if len(available) == 0:
                continue
            # Sample with replacement if needed, or without
            if len(available) >= needed:
                chosen = rng_sim.choice(available, size=needed, replace=False)
            else:
                chosen = rng_sim.choice(available, size=needed, replace=True)
            chosen_indices.extend(chosen)
            
        chosen_indices = np.array(chosen_indices)
        rng_sim.shuffle(chosen_indices)
        
        # Predict on synthetic target
        # We already have probs in val_probs_all
        # We just map chosen_indices (which are indices into ds.val.texts)
        probs_target = val_probs_all[chosen_indices]
        
        # Estimate
        pbar_target = probs_target.mean(axis=0)
        
        if args.naive:
            pi_est = pbar_target
        else:
            pi_est = estimate_priors_least_squares(C_calib, pbar_target)
            
        # Compute accuracy (1 - TVD)
        # TVD = 0.5 * sum(|pi_est - pi_true|)
        tvd = 0.5 * np.sum(np.abs(pi_est - target_pi))
        est_acc = 1.0 - tvd
        
        print(f"[Simulation] Val Acc (Full): {clf_metrics['val_acc']:.3f}")
        print(f"[Simulation] Target Pi: {np.round(target_pi, 3)}")
        print(f"[Simulation] Est Pi:    {np.round(pi_est, 3)}")
        print(f"[Simulation] Est Acc (1-TVD): {est_acc:.3f}")
        
        if args.acc_log_file:
            log_path = Path(args.acc_log_file)
            write_header = not log_path.exists()
            with open(log_path, "a", encoding="utf-8") as f:
                if write_header:
                    f.write("max_per_class,val_acc,est_acc,naive\n")
                f.write(f"{args.max_per_class},{clf_metrics['val_acc']:.5f},{est_acc:.5f},{args.naive}\n")
            print(f"[Simulation] Appended results to {log_path}")

    # 3) Gather model generations or load cached
    if args.use_cached_generations:
        gen_texts = read_jsonl_texts(args.use_cached_generations)
    else:
        if args.prompts_style == "neutral":
            prompts = NEUTRAL_PROMPTS
        elif args.prompts_style == "instructional":
            prompts = INSTRUCTIONAL_PROMPTS
        elif args.prompts_style == "expository":
            prompts = EXPOSITORY_PROMPTS
        elif args.prompts_style == "conversational":
            prompts = CONVERSATIONAL_PROMPTS
        elif args.prompts_style == "coding":
            prompts = CODING_PROMPTS
        else:
            raise NotImplementedError(f"Unknown prompts_style: {args.prompts_style}")

        if args.num_prompts is not None and args.num_prompts > 0:
            if len(prompts) >= args.num_prompts:
                prompts = prompts[: args.num_prompts]
            else:
                prompts = list(islice(cycle(prompts), args.num_prompts))

        if args.generator != "hf":
            raise ValueError(f"Unknown generator: {args.generator}")
        if not args.target_model:
            raise ValueError("--target_model is required for --generator 'hf'")

        gen_texts = generate_texts(
            model_name=args.target_model,
            prompts=prompts,
            max_new_tokens=args.max_new_tokens,
            temperature=args.gen_temperature,
            top_p=args.top_p,
            batch_size=args.gen_batch_size,
            revision=args.hf_revision,
            seed=args.seed,
        )

    # 4) Predict class probabilities on generations and average
    probs_chunks: List[np.ndarray] = []
    bs = max(1, int(args.predict_batch_size))
    for i in tqdm(range(0, len(gen_texts), bs), total=(len(gen_texts) + bs - 1) // bs, desc="Predicting probs"):
        batch = gen_texts[i : i + bs]
        probs_chunks.append(model.predict_proba(batch))
    probs = np.concatenate(probs_chunks, axis=0) if probs_chunks else np.zeros((0, K))
    pbar = probs.mean(axis=0) if probs.size > 0 else np.zeros((K,))

    # 5) Prior correction: solve for mixture pi
    if args.naive:
        print("Using naive baseline (PCC) - skipping inverse estimation.")
        pi = pbar
    else:
        pi = estimate_priors_least_squares(C, pbar)

    # 6) Optional bootstrap for CIs
    pi_mean = pi
    lo = np.zeros_like(pi)
    hi = np.zeros_like(pi)
    if args.bootstrap and probs.shape[0] > 0:
        rng = np.random.default_rng(args.seed)
        N = probs.shape[0]
        pis = []
        for _ in tqdm(range(args.n_boot), desc="Bootstrapping"):
            idx = rng.integers(0, N, size=N)
            pbar_b = probs[idx].mean(axis=0)
            if args.naive:
                pi_b = pbar_b
            else:
                pi_b = estimate_priors_least_squares(C, pbar_b)
            pis.append(pi_b)
        if pis:
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
            "generator": args.generator,
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
            "hf_pretrained_dir": args.hf_pretrained_dir,
            "bootstrap": args.bootstrap,
            "n_boot": args.n_boot,
            "prompts_style": args.prompts_style,
            "taxonomy": "pythia",
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

    print(f"Wrote label-shift outputs to {out_dir}")

    # 8) Visualizations
    try:
        plot_confusion_matrix(C, ds.categories, str(out_dir / "confusion_matrix.png"))
        Ctpi_known = (C.T @ pi_mean)
        plot_priors_with_ci(ds.categories, pi_mean, lo, hi, str(out_dir / "priors.png"))
        plot_pbar_vs_ctpi(ds.categories, pbar, Ctpi_known, str(out_dir / "pbar_vs_ctpi.png"))
    except Exception as e:
        print(f"Warning: plotting failed: {e}")


if __name__ == "__main__":
    main()


