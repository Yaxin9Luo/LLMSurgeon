import argparse
import json
from pathlib import Path
from typing import Optional, List, Tuple
from itertools import cycle, islice

import numpy as np
from tqdm import tqdm

from data_utils import build_balanced_splits
from classifier import train_tfidf_classifier, train_distilbert_classifier
from generate import generate_texts, NEUTRAL_PROMPTS, INSTRUCTIONAL_PROMPTS , EXPOSITORY_PROMPTS, CONVERSATIONAL_PROMPTS, CODING_PROMPTS
from prior import estimate_priors_least_squares
from viz import plot_confusion_matrix, plot_priors_with_ci, plot_pbar_vs_ctpi
from inspect_viz import (
    plot_assignment_sankey,
    write_assignment_gallery_html_train_val,
    write_assignment_gallery_html_generated,
    nn_composition_and_diagnostics,
    distance_diagnostics,
    plot_embeddings_map,
    class_prototypes_and_medoids,
)
from sklearn.metrics import confusion_matrix
from sklearn.neighbors import NearestNeighbors
import csv


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Label-shift mixture estimation via domain classifier + prior correction"
    )
    # Data
    p.add_argument("--local_samples_dir", type=str, default=str(Path(__file__).resolve().parents[2] / "data_samples"))
    p.add_argument("--merge_web", action="store_true", help="Merge CommonCrawl and C4 into a single Web class (6-way)")
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
        help="If set, train DistilBERT classifier from scratch instead of fine-tuninging a pre-trained model",
    )

    # Generations
    p.add_argument("--generator", type=str, choices=["hf"], default="hf", help="Generation engine to use.")
    p.add_argument("--target_model", type=str, required=False, help="HF model name (for --generator hf)")
    p.add_argument("--hf_revision", type=str, default=None, help="HF revision/commit/tag for --target_model (e.g., specific training step)")
    p.add_argument("--use_cached_generations", type=str, default=None, help="Path to JSONL with {text} per line to skip generation")
    p.add_argument("--num_prompts", type=int, default=400)
    # p.add_argument("--prompts_file", type=str, default=None)
    p.add_argument("--prompts_style", type=str, default="neutral", help="Prompt style to use")
    p.add_argument("--max_new_tokens", type=int, default=512)
    p.add_argument("--gen_temperature", type=float, default=0.8)
    p.add_argument("--top_p", type=float, default=0.9)
    p.add_argument("--gen_batch_size", type=int, default=8)

    # Unknown thresholding (confidence-based)
    p.add_argument(
        "--unknown_threshold",
        type=float,
        default=0.9,
        help="If set (0<τ≤1), scale known-class probabilities by confidence and assign residual to Unknown",
    )
    p.add_argument(
        "--unknown_metric",
        type=str,
        default="maxprob",
        choices=["maxprob"],
        help="Confidence score used for Unknown thresholding (currently only maxprob is supported)",
    )

    # Bootstrap
    p.add_argument("--bootstrap", action="store_true")
    p.add_argument("--n_boot", type=int, default=300)

    # Inspection options
    p.add_argument("--inspect", action="store_true", help="Dump per-sample predictions and NN neighbors (DistilBERT only)")
    p.add_argument("--nn_k", type=int, default=5, help="#neighbors per generated sample for inspection")
    p.add_argument("--nn_max_gens", type=int, default=20, help="Max generations to analyze for NN neighbors")
    p.add_argument("--snippet_len", type=int, default=200, help="Max chars for text snippets in CSV/JSONL")

    # Output
    p.add_argument("--output_dir", type=str, default=str(Path(__file__).resolve().parents[1] / "out"))
    p.add_argument("--run_name", type=str, default="labelshift")
    p.add_argument("--naive", action="store_true", help="Use naive Probabilistic Classify and Count (PCC) baseline (skip confusion matrix correction)")
    p.add_argument("--simulate_label_shift", action="store_true", help="Run a synthetic label shift experiment on validation data to measure estimation accuracy")
    p.add_argument("--acc_log_file", type=str, default=None, help="Path to CSV file to append (max_per_class, val_acc, estimation_acc) results")
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


def apply_unknown_threshold(
    probs: np.ndarray,
    threshold: float,
    metric: str = "maxprob",
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    if probs.ndim != 2:
        raise ValueError("probs must be 2D")
    if metric != "maxprob":
        raise ValueError(f"Unsupported unknown_metric: {metric}")
    if not (0.0 < threshold <= 1.0):
        raise ValueError("--unknown_threshold must satisfy 0 < threshold ≤ 1")
    if probs.shape[0] == 0:
        return probs.copy(), np.zeros((0,), dtype=float), np.zeros((0,), dtype=float)

    max_probs = probs.max(axis=1)
    denom = max(threshold, 1e-8)
    unknown_mass = np.clip((threshold - max_probs) / denom, 0.0, 1.0)
    scaled = probs * (1.0 - unknown_mass)[:, None]
    return scaled, unknown_mass, max_probs


def main() -> None:
    args = parse_args()

    # 1) Build balanced dataset
    ds = build_balanced_splits(
        local_dir=args.local_samples_dir,
        merge_web=args.merge_web,
        max_per_class=args.max_per_class,
        val_fraction=args.val_fraction,
        seed=args.seed,
    )
    K = len(ds.categories)
    # Prepare output directory early to store training logs/curves
    out_dir = Path(args.output_dir) / args.run_name
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Training classifier: {args.classifier}...")
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
        # Split val into calib (50%) and test (50%)
        rng_sim = np.random.default_rng(args.seed + 123)
        val_indices = np.arange(len(ds.val.texts))
        rng_sim.shuffle(val_indices)
        half = len(val_indices) // 2
        idx_calib = val_indices[:half]
        idx_test = val_indices[half:]

        # 1. Get predictions for all validation data
        # We need to act carefully to avoid re-predicting if expensive, but here we just do it.
        # Note: model.predict_proba might be expensive for DistilBERT.
        # Optimization: define a helper or just run it.
        # Since we already have C computed on FULL val in 'train_*_classifier',
        # we strictly should re-compute C on just the calib split to be rigorous.
        
        val_probs_all = model.predict_proba(ds.val.texts)
        val_preds_all = np.argmax(val_probs_all, axis=1)
        
        # Calibration part
        y_calib = np.array(ds.val.labels)[idx_calib]
        preds_calib = val_preds_all[idx_calib]
        
        # Re-compute C from calib split
        cm_calib = confusion_matrix(y_calib, preds_calib, labels=range(K))
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
        # Get prompts
        if args.prompts_style == "neutral":
            print("Using neutral prompts !!!!!!!!!!!!!!!!!")
            prompts = NEUTRAL_PROMPTS
        elif args.prompts_style == "instructional":
            print("Using instructional prompts !!!!!!!!!!!!!!!!!")
            prompts = INSTRUCTIONAL_PROMPTS 
        elif args.prompts_style == "expository":
            print("Using expository prompts !!!!!!!!!!!!!!!!!")
            prompts = EXPOSITORY_PROMPTS
        elif args.prompts_style == "conversational":
            print("Using conversational prompts !!!!!!!!!!!!!!!!!")
            prompts = CONVERSATIONAL_PROMPTS
        elif args.prompts_style == "coding":
            print("Using coding prompts !!!!!!!!!!!!!!!!!")
            prompts = CODING_PROMPTS
        else:
            raise NotImplementedError(f"Unknown prompts_style: {args.prompts_style}")

        if not prompts:
            raise ValueError("No prompts available for generation")
        if args.num_prompts is not None and args.num_prompts > 0:
            if len(prompts) >= args.num_prompts:
                prompts = prompts[: args.num_prompts]
            else:
                prompts = list(islice(cycle(prompts), args.num_prompts))

        # Generate texts using the selected engine
        if args.generator == "hf":
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
            
            ############################################
            # Save LLM generations to ./cache/llm_generation/
            ############################################
            llm_name = args.target_model.replace("/", "_")
            import time

            timestamp = time.strftime("%Y%m%d_%H%M%S")
            save_dir = Path("./cache/llm_generation") / f"{llm_name}_{args.prompts_style}_{timestamp}"
            save_dir.mkdir(parents=True, exist_ok=True)

            save_path = save_dir / "generated_texts.jsonl"

            with open(save_path, "w", encoding="utf-8") as f:
                for txt in gen_texts:
                    f.write(json.dumps({"text": txt}, ensure_ascii=False) + "\n")

            print(f"[LLM Generation] Saved generations to: {save_path}")
            
            meta = {
                "model_name": args.target_model,
                "revision": args.hf_revision,
                "prompts_style": args.prompts_style,
                "num_prompts": args.num_prompts,
                "max_new_tokens": args.max_new_tokens,
                "temperature": args.gen_temperature,
                "top_p": args.top_p,
                "batch_size": args.gen_batch_size,
                "seed": args.seed,
                "timestamp": timestamp,
                "num_generations": len(gen_texts),
            }

            # 保存 meta.json
            with open(save_dir / "meta.json", "w", encoding="utf-8") as f:
                json.dump(meta, f, indent=2, ensure_ascii=False)

            ############################################


        else:
            raise ValueError(f"Unknown generator: {args.generator}")

    # 4) Predict class probabilities on generations and average
    probs_chunks: List[np.ndarray] = []
    bs = max(1, int(args.predict_batch_size))
    for i in tqdm(range(0, len(gen_texts), bs), total=(len(gen_texts)+bs-1)//bs, desc="Predicting probs"):
        batch = gen_texts[i:i+bs]
        probs_chunks.append(model.predict_proba(batch))
    probs = np.concatenate(probs_chunks, axis=0) if probs_chunks else np.zeros((0, K))

    unknown_threshold = args.unknown_threshold
    if unknown_threshold is not None and probs.shape[0] > 0:
        scaled_probs, unknown_mass, max_probs = apply_unknown_threshold(
            probs,
            threshold=float(unknown_threshold),
            metric=args.unknown_metric,
        )
    else:
        scaled_probs = probs
        unknown_mass = np.zeros((probs.shape[0],), dtype=float)
        max_probs = probs.max(axis=1) if probs.shape[0] > 0 else np.zeros((0,), dtype=float)

    unknown_mean = float(unknown_mass.mean()) if unknown_mass.size > 0 else 0.0
    known_mass = max(1e-8, 1.0 - unknown_mean)
    pbar_raw = scaled_probs.mean(axis=0) if scaled_probs.size > 0 else np.zeros((K,))
    pbar = pbar_raw / known_mass if pbar_raw.size > 0 else np.zeros((K,))

    # 5) Prior correction: solve for mixture pi
    if args.naive:
        print("Using naive baseline (PCC) - skipping inverse estimation.")
        pi = pbar
    else:
        pi = estimate_priors_least_squares(C, pbar)

    # 6) Optional bootstrap for CIs (known categories + unknown mass)
    pi_mean = pi
    lo = np.zeros_like(pi)
    hi = np.zeros_like(pi)
    unknown_ci_lo = unknown_mean
    unknown_ci_hi = unknown_mean
    unknown_bootstrap_mean = unknown_mean

    pi_ext_point = np.concatenate([pi * (1.0 - unknown_mean), np.array([unknown_mean])])
    pi_mean_ext = pi_ext_point.copy()
    lo_ext = np.concatenate([lo * (1.0 - unknown_mean), np.array([unknown_mean])])
    hi_ext = np.concatenate([hi * (1.0 - unknown_mean), np.array([unknown_mean])])

    if args.bootstrap and probs.shape[0] > 0:
        # Manual bootstrap loop to expose progress bar
        rng = np.random.default_rng(args.seed)
        N = probs.shape[0]
        pis = []
        pis_ext = []
        unknown_records: List[float] = []
        for _ in tqdm(range(args.n_boot), desc="Bootstrapping"):
            idx = rng.integers(0, N, size=N)
            um_b = float(unknown_mass[idx].mean())
            km_b = max(1e-8, 1.0 - um_b)
            scaled_mean_b = scaled_probs[idx].mean(axis=0)
            if scaled_mean_b.size == 0:
                continue
            pbar_b = scaled_mean_b / km_b
            if args.naive:
                pi_b = pbar_b
            else:
                pi_b = estimate_priors_least_squares(C, pbar_b)
            pis.append(pi_b)
            pis_ext.append(np.concatenate([pi_b * (1.0 - um_b), np.array([um_b])]))
            unknown_records.append(um_b)
        if pis:
            P = np.stack(pis, axis=0)
            pi_mean = P.mean(axis=0)
            lo = np.percentile(P, 2.5, axis=0)
            hi = np.percentile(P, 97.5, axis=0)
        if pis_ext:
            P_ext = np.stack(pis_ext, axis=0)
            pi_mean_ext = P_ext.mean(axis=0)
            lo_ext = np.percentile(P_ext, 2.5, axis=0)
            hi_ext = np.percentile(P_ext, 97.5, axis=0)
        if unknown_records:
            unknown_bootstrap = np.array(unknown_records, dtype=float)
            unknown_bootstrap_mean = float(unknown_bootstrap.mean())
            unknown_ci_lo = float(np.percentile(unknown_bootstrap, 2.5))
            unknown_ci_hi = float(np.percentile(unknown_bootstrap, 97.5))

    unknown_mass_for_plot = unknown_bootstrap_mean
    pbar_ext = None
    if args.unknown_threshold is not None:
        if pbar.size > 0:
            p_known_scaled = pbar * max(0.0, 1.0 - unknown_mass_for_plot)
        else:
            p_known_scaled = np.zeros((K,), dtype=float)
        pbar_ext = np.concatenate([p_known_scaled, np.array([unknown_mass_for_plot])])

    if args.unknown_threshold is not None:
        print(
            f"Average unknown probability (threshold {args.unknown_threshold:.3f}): {unknown_bootstrap_mean:.3f}"
        )

    # 7) Write outputs (out_dir already created)

    payload = {
        "config": {
            "local_samples_dir": args.local_samples_dir,
            "merge_web": args.merge_web,
            "classifier": args.classifier,
            "seed": args.seed,
            "val_fraction": args.val_fraction,
            "target_model": args.target_model,
            "generator": args.generator,
            "num_prompts": args.num_prompts,
            "max_new_tokens": args.max_new_tokens,
            "gen_temperature": args.gen_temperature,
            "top_p": args.top_p,
            "gen_seed": args.seed,
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
        "unknown": {
            "mode": "threshold" if args.unknown_threshold is not None else "disabled",
            "metric": args.unknown_metric,
            "threshold": args.unknown_threshold,
            "mean_probability": unknown_bootstrap_mean,
            "ci_lo": unknown_ci_lo,
            "ci_hi": unknown_ci_hi,
        },
        "categories_with_unknown": ds.categories + ["Unknown"],
        "priors_with_unknown": {
            "point": pi_ext_point.tolist(),
            "mean": pi_mean_ext.tolist(),
            "ci_lo": lo_ext.tolist(),
            "ci_hi": hi_ext.tolist(),
        },
    }
    if pbar_ext is not None:
        payload["pbar_with_unknown"] = pbar_ext.tolist()
    with open(out_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    with open(out_dir / "summary.csv", "w", encoding="utf-8") as f:
        f.write("category,pi,ci_lo,ci_hi\n")
        for c, p, a, b in zip(ds.categories, pi_mean, lo, hi):
            f.write(f"{c},{p:.6f},{a:.6f},{b:.6f}\n")
        f.write(f"Unknown,{unknown_bootstrap_mean:.6f},{unknown_ci_lo:.6f},{unknown_ci_hi:.6f}\n")

    print(f"Wrote label-shift outputs to {out_dir}")

    # 8) Optional inspection: dump per-sample predictions and nearest neighbors
    if args.inspect:
        try:
            # Dump predictions for train/val
            def _topk_info(probs_row: np.ndarray, cats: List[str], k: int = 3) -> List[str]:
                idx = np.argsort(-probs_row)[:k]
                return [f"{cats[j]}:{probs_row[j]:.3f}" for j in idx]

            # Train predictions
            train_probs = model.predict_proba(ds.train.texts)
            train_pred = np.argmax(train_probs, axis=1)
            # Val predictions
            val_probs = model.predict_proba(ds.val.texts)
            val_pred = np.argmax(val_probs, axis=1)

            tv_path = out_dir / "train_val_predictions.csv"
            with open(tv_path, "w", encoding="utf-8", newline="") as fcsv:
                writer = csv.writer(fcsv)
                writer.writerow(["split", "index", "true_class", "pred_class", "top1_conf", "top3", "text_snippet"])
                for i, (y, pp, txt) in enumerate(zip(ds.train.labels, train_probs, ds.train.texts)):
                    writer.writerow([
                        "train",
                        i,
                        ds.categories[y],
                        ds.categories[int(np.argmax(pp))],
                        f"{float(np.max(pp)):.3f}",
                        "|".join(_topk_info(pp, ds.categories)),
                        txt[: args.snippet_len].replace("\n", " ")
                    ])
                for i, (y, pp, txt) in enumerate(zip(ds.val.labels, val_probs, ds.val.texts)):
                    writer.writerow([
                        "val",
                        i,
                        ds.categories[y],
                        ds.categories[int(np.argmax(pp))],
                        f"{float(np.max(pp)):.3f}",
                        "|".join(_topk_info(pp, ds.categories)),
                        txt[: args.snippet_len].replace("\n", " ")
                    ])

            # Sankey diagram from true (val) -> predicted (val)
            try:
                cm_counts = confusion_matrix(ds.val.labels, val_pred)
                plot_assignment_sankey(cm_counts, ds.categories, str(out_dir / "sankey_val_true_pred.png"))
            except Exception as e:
                print(f"Inspector: failed to plot Sankey: {e}")

            # HTML gallery for train/val assignments
            try:
                write_assignment_gallery_html_train_val(str(tv_path), str(out_dir / "gallery_train_val.html"))
            except Exception as e:
                print(f"Inspector: failed to write train/val gallery: {e}")

            # Generated predictions
            gen_path = out_dir / "generated_predictions.csv"
            with open(gen_path, "w", encoding="utf-8", newline="") as fcsv:
                writer = csv.writer(fcsv)
                # If prompts variable exists in scope, we will include prompt metadata; else leave blank
                have_prompts = 'prompts' in locals()
                writer.writerow([
                    "gen_index",
                    "prompt_id",
                    "pred_class",
                    "top1_conf",
                    "top3",
                    "prompt",
                    "text_snippet",
                    "unknown_prob",
                    "in_known_pool",
                ])
                thr = float(args.unknown_threshold) if args.unknown_threshold is not None else None
                for i, (pp, txt) in enumerate(zip(probs, gen_texts)):
                    pr_idx = i if have_prompts else ""
                    pr_text = (prompts[i] if have_prompts and i < len(prompts) else "")
                    top_conf = float(max_probs[i]) if max_probs.size else (float(np.max(pp)) if pp.size else 0.0)
                    pred_idx = int(np.argmax(pp)) if pp.size else 0
                    is_unknown = bool(thr is not None and top_conf < thr)
                    pred_label = "Unknown" if is_unknown else ds.categories[pred_idx]
                    unk_prob = float(unknown_mass[i]) if unknown_mass.size else 0.0
                    writer.writerow([
                        i,
                        pr_idx,
                        pred_label,
                        f"{top_conf:.3f}",
                        "|".join(_topk_info(pp, ds.categories)),
                        str(pr_text).replace("\n", " "),
                        txt[: args.snippet_len].replace("\n", " "),
                        f"{unk_prob:.3f}",
                        "yes" if not is_unknown else "no",
                    ])

            # HTML gallery for generated assignments
            try:
                write_assignment_gallery_html_generated(str(gen_path), str(out_dir / "gallery_generated.html"))
            except Exception as e:
                print(f"Inspector: failed to write generated gallery: {e}")

            # Nearest neighbors (DistilBERT only)
            # Guard: embeddings available only for HFSequenceClassifier
            if hasattr(model, "embeddings"):
                # Train embeddings index
                train_emb = model.embeddings(ds.train.texts, batch_size=max(8, bs))
                nn = NearestNeighbors(n_neighbors=max(1, args.nn_k), metric="cosine", algorithm="brute")
                nn.fit(train_emb)
                # Validation embeddings
                try:
                    val_emb = model.embeddings(ds.val.texts, batch_size=max(8, bs))
                except Exception:
                    val_emb = None
                # Gen subset
                m = min(max(1, args.nn_max_gens), len(gen_texts))
                sub_gen_texts = gen_texts[:m]
                gen_emb = model.embeddings(sub_gen_texts, batch_size=max(8, bs))
                dists, nbrs = nn.kneighbors(gen_emb, return_distance=True)

                # Write JSONL with neighbors
                import json as _json
                neigh_path = out_dir / "gen_neighbors.jsonl"
                with open(neigh_path, "w", encoding="utf-8") as fj:
                    thr_local = float(args.unknown_threshold) if args.unknown_threshold is not None else None
                    for gi in range(m):
                        pp = probs[gi]
                        top_conf = float(max_probs[gi]) if max_probs.size else (float(np.max(pp)) if pp.size else 0.0)
                        pred_idx = int(np.argmax(pp)) if pp.size else 0
                        is_unknown = bool(thr_local is not None and top_conf < thr_local)
                        pred_label = "Unknown" if is_unknown else ds.categories[pred_idx]
                        rec = {
                            "gen_index": gi,
                            "pred_class": pred_label,
                            "top1_conf": top_conf,
                            "text_snippet": sub_gen_texts[gi][: args.snippet_len],
                            "neighbors": []
                        }
                        for rk, (ti, dd) in enumerate(zip(nbrs[gi].tolist(), dists[gi].tolist())):
                            ti = int(ti)
                            rec["neighbors"].append({
                                "rank": rk + 1,
                                "train_index": ti,
                                "true_class": ds.categories[int(ds.train.labels[ti])],
                                "distance": float(dd),
                                "text_snippet": ds.train.texts[ti][: args.snippet_len]
                            })
                        fj.write(_json.dumps(rec, ensure_ascii=False) + "\n")

                # NN composition (per-sample + aggregated), hubness
                try:
                    nn_composition_and_diagnostics(
                        str(neigh_path), out_dir=str(out_dir), categories=ds.categories,
                    )
                except Exception as e:
                    print(f"Inspector: NN composition failed: {e}")

                # Distance diagnostics plot (uses kNN set)
                try:
                    gen_pred_sub = np.argmax(probs[:m], axis=1).tolist()
                    distance_diagnostics(gen_pred_sub, (dists, nbrs), ds.train.labels, ds.categories, str(out_dir / "nn_distance_diagnostics.png"))
                except Exception as e:
                    print(f"Inspector: distance diagnostics failed: {e}")

                # Embedding map (t-SNE) across splits
                try:
                    plot_embeddings_map(
                        train_emb=train_emb,
                        train_labels=ds.train.labels,
                        val_emb=(val_emb if val_emb is not None else np.zeros((0, train_emb.shape[1]))),
                        val_labels=(ds.val.labels if val_emb is not None else []),
                        gen_emb=gen_emb,
                        gen_pred=gen_pred_sub,
                        categories=ds.categories,
                        out_path=str(out_dir / "embeddings_tsne.png"),
                    )
                except Exception as e:
                    print(f"Inspector: embedding map failed: {e}")

                # Class prototypes & medoids (from training set)
                try:
                    class_prototypes_and_medoids(
                        texts=ds.train.texts,
                        labels=ds.train.labels,
                        emb=train_emb,
                        categories=ds.categories,
                        out_json=str(out_dir / "class_prototypes_medoids.json"),
                    )
                except Exception as e:
                    print(f"Inspector: prototypes/medoids failed: {e}")
            else:
                print("Inspector: skipping NN neighbors because embeddings() not available for this classifier.")

            print(f"Inspector outputs written to {out_dir}")
        except Exception as e:
            print(f"Warning: inspection failed: {e}")

    # 9) Visualizations
    try:
        plot_confusion_matrix(C, ds.categories, str(out_dir / "confusion_matrix.png"))
        Ctpi_known = (C.T @ pi_mean)
        if args.unknown_threshold is not None:
            cats_plot = ds.categories + ["Unknown"]
            plot_priors_with_ci(cats_plot, pi_mean_ext, lo_ext, hi_ext, str(out_dir / "priors.png"))
            unknown_mass_plot = float(unknown_mass_for_plot)
            if pbar_ext is not None:
                pbar_plot = pbar_ext
            else:
                if pbar.size > 0:
                    p_known_scaled = pbar * max(0.0, 1.0 - unknown_mass_plot)
                else:
                    p_known_scaled = np.zeros((K,), dtype=float)
                pbar_plot = np.concatenate([p_known_scaled, np.array([unknown_mass_plot])])
            Ctpi_ext = np.concatenate([Ctpi_known * max(0.0, 1.0 - unknown_mass_plot), np.array([unknown_mass_plot])])
            plot_pbar_vs_ctpi(cats_plot, pbar_plot, Ctpi_ext, str(out_dir / "pbar_vs_ctpi.png"))
        else:
            plot_priors_with_ci(ds.categories, pi_mean, lo, hi, str(out_dir / "priors.png"))
            plot_pbar_vs_ctpi(ds.categories, pbar, Ctpi_known, str(out_dir / "pbar_vs_ctpi.png"))
    except Exception as e:
        print(f"Warning: plotting failed: {e}")


if __name__ == "__main__":
    main()
