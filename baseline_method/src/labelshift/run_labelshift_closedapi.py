"""
Label-shift mixture estimation for closed-source LLMs (OpenAI GPT, Google Gemini).

Since these models are accessed via API only, we generate texts through their
respective APIs and then apply the same domain-classifier + prior-correction
pipeline used for open-source models.  There is **no ground truth** for the
training data mixture of these proprietary models -- the script simply produces
predicted mixture estimates.
"""

import argparse
import json
import os
import time
from pathlib import Path
from typing import List, Tuple, Optional
from itertools import cycle, islice

import numpy as np
from tqdm import tqdm

from data_utils import build_balanced_splits
from classifier import train_tfidf_classifier, train_distilbert_classifier
from generate import (
    NEUTRAL_PROMPTS,
    INSTRUCTIONAL_PROMPTS,
    EXPOSITORY_PROMPTS,
    CONVERSATIONAL_PROMPTS,
    CODING_PROMPTS,
)
from prior import estimate_priors_least_squares
from viz import plot_confusion_matrix, plot_priors_with_ci, plot_pbar_vs_ctpi


# ---------------------------------------------------------------------------
# Prompt templates designed for API-based chat models
# ---------------------------------------------------------------------------
API_GENERATION_SYSTEM_PROMPT = (
    "You are a helpful assistant. When given a topic or prompt, generate a "
    "natural, detailed passage of text. Do not add any meta-commentary -- "
    "just produce the passage directly."
)


# ---------------------------------------------------------------------------
# API generation helpers
# ---------------------------------------------------------------------------

def generate_texts_openai(
    model: str,
    prompts: List[str],
    max_tokens: int = 512,
    temperature: float = 0.8,
    top_p: float = 0.9,
    seed: Optional[int] = None,
    batch_size: int = 8,
) -> List[str]:
    """Generate texts using the OpenAI Chat Completions API."""
    try:
        from openai import OpenAI
    except ImportError:
        raise ImportError(
            "openai package not installed. Install with: pip install openai"
        )

    client = OpenAI()  # reads OPENAI_API_KEY from env
    results: List[str] = []

    for i in tqdm(range(0, len(prompts), batch_size), desc=f"OpenAI ({model})"):
        batch = prompts[i : i + batch_size]
        for prompt in batch:
            for attempt in range(5):
                try:
                    kwargs = dict(
                        model=model,
                        messages=[
                            {"role": "system", "content": API_GENERATION_SYSTEM_PROMPT},
                            {"role": "user", "content": prompt},
                        ],
                        max_tokens=max_tokens,
                        temperature=temperature,
                        top_p=top_p,
                    )
                    if seed is not None:
                        kwargs["seed"] = seed
                    resp = client.chat.completions.create(**kwargs)
                    text = resp.choices[0].message.content or ""
                    results.append(text.strip())
                    break
                except Exception as e:
                    wait = 2 ** attempt
                    print(f"  OpenAI API error (attempt {attempt+1}/5): {e}. Retrying in {wait}s...")
                    time.sleep(wait)
            else:
                print(f"  WARNING: Skipping prompt after 5 failures: {prompt[:80]}...")
                results.append("")

    return results


def generate_texts_google(
    model: str,
    prompts: List[str],
    max_tokens: int = 512,
    temperature: float = 0.8,
    top_p: float = 0.9,
    seed: Optional[int] = None,
    batch_size: int = 8,
) -> List[str]:
    """Generate texts using the Google Generative AI (Gemini) API."""
    try:
        from google import genai
        from google.genai import types
    except ImportError:
        raise ImportError(
            "google-genai package not installed. Install with: pip install google-genai"
        )

    client = genai.Client()  # reads GOOGLE_API_KEY or GEMINI_API_KEY from env
    results: List[str] = []

    for i in tqdm(range(0, len(prompts), batch_size), desc=f"Gemini ({model})"):
        batch = prompts[i : i + batch_size]
        for prompt in batch:
            for attempt in range(5):
                try:
                    config = types.GenerateContentConfig(
                        max_output_tokens=max_tokens,
                        temperature=temperature,
                        top_p=top_p,
                        system_instruction=API_GENERATION_SYSTEM_PROMPT,
                    )
                    if seed is not None:
                        config.seed = seed

                    resp = client.models.generate_content(
                        model=model,
                        contents=prompt,
                        config=config,
                    )
                    text = resp.text or ""
                    results.append(text.strip())
                    break
                except Exception as e:
                    wait = 2 ** attempt
                    print(f"  Gemini API error (attempt {attempt+1}/5): {e}. Retrying in {wait}s...")
                    time.sleep(wait)
            else:
                print(f"  WARNING: Skipping prompt after 5 failures: {prompt[:80]}...")
                results.append("")

    return results


# ---------------------------------------------------------------------------
# Threshold helpers (same as run_labelshift.py)
# ---------------------------------------------------------------------------

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
        raise ValueError("--unknown_threshold must satisfy 0 < threshold <= 1")
    if probs.shape[0] == 0:
        return probs.copy(), np.zeros((0,), dtype=float), np.zeros((0,), dtype=float)

    max_probs = probs.max(axis=1)
    denom = max(threshold, 1e-8)
    unknown_mass = np.clip((threshold - max_probs) / denom, 0.0, 1.0)
    scaled = probs * (1.0 - unknown_mass)[:, None]
    return scaled, unknown_mass, max_probs


# ---------------------------------------------------------------------------
# JSONL helpers
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Predict training data mixture of closed-source LLMs (GPT / Gemini)"
    )
    # Data
    p.add_argument("--local_samples_dir", type=str,
                    default=str(Path(__file__).resolve().parents[2] / "data_samples"))
    p.add_argument("--merge_web", action="store_true",
                    help="Merge CommonCrawl and C4 into a single Web class (6-way)")
    p.add_argument("--max_per_class", type=int, default=5000)
    p.add_argument("--val_fraction", type=float, default=0.2)
    p.add_argument("--seed", type=int, default=0)

    # Classifier
    p.add_argument("--classifier", type=str, choices=["tfidf", "distilbert"], default="distilbert")
    p.add_argument("--n_jobs", type=int, default=4)
    p.add_argument("--predict_batch_size", type=int, default=256)
    p.add_argument("--clf_verbose", type=int, default=0)

    # HF classifier options
    p.add_argument("--hf_model_name", type=str, default="distilbert/distilbert-base-uncased")
    p.add_argument("--hf_epochs", type=int, default=3)
    p.add_argument("--hf_batch_size", type=int, default=64)
    p.add_argument("--hf_lr", type=float, default=2e-5)
    p.add_argument("--hf_weight_decay", type=float, default=0.01)
    p.add_argument("--hf_max_length", type=int, default=256)
    p.add_argument("--hf_pretrained_dir", type=str, default=None,
                    help="Path to a fine-tuned HF classifier directory")
    p.add_argument("--hf_train_from_scratch", action="store_true")

    # API Generation
    p.add_argument("--generator", type=str, choices=["openai", "google"],
                    required=True, help="API provider: openai or google")
    p.add_argument("--target_model", type=str, required=True,
                    help="Model identifier (e.g. gpt-4o, gpt-4o-mini, gemini-2.0-flash)")
    p.add_argument("--use_cached_generations", type=str, default=None,
                    help="Path to JSONL with {text} per line to skip generation")
    p.add_argument("--num_prompts", type=int, default=400)
    p.add_argument("--prompts_style", type=str, default="neutral",
                    choices=["neutral", "instructional", "expository", "conversational", "coding"],
                    help="Prompt style to use for generation")
    p.add_argument("--max_new_tokens", type=int, default=512)
    p.add_argument("--gen_temperature", type=float, default=0.8)
    p.add_argument("--top_p", type=float, default=0.9)
    p.add_argument("--gen_batch_size", type=int, default=8)

    # Unknown thresholding
    p.add_argument("--unknown_threshold", type=float, default=0.9,
                    help="Confidence threshold for Unknown class")
    p.add_argument("--unknown_metric", type=str, default="maxprob", choices=["maxprob"])

    # Bootstrap
    p.add_argument("--bootstrap", action="store_true")
    p.add_argument("--n_boot", type=int, default=300)

    # Output
    p.add_argument("--output_dir", type=str,
                    default=str(Path(__file__).resolve().parents[1] / "out"))
    p.add_argument("--run_name", type=str, default="closedapi_labelshift")
    p.add_argument("--naive", action="store_true",
                    help="Use naive PCC baseline (skip confusion matrix correction)")

    return p.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()

    # 1) Build balanced dataset (domain classifier training data)
    print("=" * 70)
    print(f"  Closed-API Label-Shift Estimation")
    print(f"  Provider : {args.generator}")
    print(f"  Model    : {args.target_model}")
    print(f"  Prompts  : {args.num_prompts} x {args.prompts_style}")
    print("=" * 70)

    ds = build_balanced_splits(
        local_dir=args.local_samples_dir,
        merge_web=args.merge_web,
        max_per_class=args.max_per_class,
        val_fraction=args.val_fraction,
        seed=args.seed,
    )
    K = len(ds.categories)
    print(f"Domain categories ({K}): {ds.categories}")

    out_dir = Path(args.output_dir) / args.run_name
    out_dir.mkdir(parents=True, exist_ok=True)

    # 2) Train domain classifier
    print(f"\nTraining domain classifier: {args.classifier} ...")
    if args.classifier == "tfidf":
        model, clf_metrics, C = train_tfidf_classifier(
            ds.train.texts, ds.train.labels,
            ds.val.texts, ds.val.labels,
            seed=args.seed, n_jobs=args.n_jobs, verbose=args.clf_verbose,
        )
    else:
        model, clf_metrics, C = train_distilbert_classifier(
            ds.train.texts, ds.train.labels,
            ds.val.texts, ds.val.labels,
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

    # 3) Generate texts from the closed-source model (or load cached)
    if args.use_cached_generations:
        print(f"\nLoading cached generations from: {args.use_cached_generations}")
        gen_texts = read_jsonl_texts(args.use_cached_generations)
        print(f"  Loaded {len(gen_texts)} texts")
    else:
        # Select prompts
        prompt_map = {
            "neutral": NEUTRAL_PROMPTS,
            "instructional": INSTRUCTIONAL_PROMPTS,
            "expository": EXPOSITORY_PROMPTS,
            "conversational": CONVERSATIONAL_PROMPTS,
            "coding": CODING_PROMPTS,
        }
        prompts = prompt_map[args.prompts_style]
        if not prompts:
            raise ValueError("No prompts available for generation")
        if args.num_prompts > 0:
            if len(prompts) >= args.num_prompts:
                prompts = prompts[: args.num_prompts]
            else:
                prompts = list(islice(cycle(prompts), args.num_prompts))

        print(f"\nGenerating {len(prompts)} texts via {args.generator} API ({args.target_model}) ...")

        if args.generator == "openai":
            gen_texts = generate_texts_openai(
                model=args.target_model,
                prompts=prompts,
                max_tokens=args.max_new_tokens,
                temperature=args.gen_temperature,
                top_p=args.top_p,
                seed=args.seed,
                batch_size=args.gen_batch_size,
            )
        elif args.generator == "google":
            gen_texts = generate_texts_google(
                model=args.target_model,
                prompts=prompts,
                max_tokens=args.max_new_tokens,
                temperature=args.gen_temperature,
                top_p=args.top_p,
                seed=args.seed,
                batch_size=args.gen_batch_size,
            )
        else:
            raise ValueError(f"Unknown generator: {args.generator}")

        # Filter empty generations
        gen_texts = [t for t in gen_texts if t.strip()]
        print(f"  Generated {len(gen_texts)} non-empty texts")

        # Cache generations
        model_slug = args.target_model.replace("/", "_").replace(":", "_")
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        cache_dir = Path("./cache/llm_generation") / f"{model_slug}_{args.prompts_style}_{timestamp}"
        cache_dir.mkdir(parents=True, exist_ok=True)

        with open(cache_dir / "generated_texts.jsonl", "w", encoding="utf-8") as f:
            for txt in gen_texts:
                f.write(json.dumps({"text": txt}, ensure_ascii=False) + "\n")

        meta = {
            "provider": args.generator,
            "model_name": args.target_model,
            "prompts_style": args.prompts_style,
            "num_prompts": args.num_prompts,
            "max_new_tokens": args.max_new_tokens,
            "temperature": args.gen_temperature,
            "top_p": args.top_p,
            "seed": args.seed,
            "timestamp": timestamp,
            "num_generations": len(gen_texts),
        }
        with open(cache_dir / "meta.json", "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2, ensure_ascii=False)
        print(f"  Cached generations to: {cache_dir}")

    if len(gen_texts) == 0:
        raise RuntimeError("No generated texts available. Check API connectivity / cached file.")

    # 4) Predict domain class probabilities on generated texts
    print(f"\nClassifying {len(gen_texts)} generated texts ...")
    probs_chunks: List[np.ndarray] = []
    bs = max(1, int(args.predict_batch_size))
    for i in tqdm(range(0, len(gen_texts), bs), total=(len(gen_texts)+bs-1)//bs, desc="Predicting probs"):
        batch = gen_texts[i : i + bs]
        probs_chunks.append(model.predict_proba(batch))
    probs = np.concatenate(probs_chunks, axis=0) if probs_chunks else np.zeros((0, K))

    # 5) Apply unknown threshold
    unknown_threshold = args.unknown_threshold
    if unknown_threshold is not None and probs.shape[0] > 0:
        scaled_probs, unknown_mass, max_probs = apply_unknown_threshold(
            probs, threshold=float(unknown_threshold), metric=args.unknown_metric,
        )
    else:
        scaled_probs = probs
        unknown_mass = np.zeros((probs.shape[0],), dtype=float)
        max_probs = probs.max(axis=1) if probs.shape[0] > 0 else np.zeros((0,), dtype=float)

    unknown_mean = float(unknown_mass.mean()) if unknown_mass.size > 0 else 0.0
    known_mass = max(1e-8, 1.0 - unknown_mean)
    pbar_raw = scaled_probs.mean(axis=0) if scaled_probs.size > 0 else np.zeros((K,))
    pbar = pbar_raw / known_mass if pbar_raw.size > 0 else np.zeros((K,))

    # 6) Prior correction (label-shift estimation)
    if args.naive:
        print("Using naive PCC baseline (no confusion matrix correction).")
        pi = pbar
    else:
        pi = estimate_priors_least_squares(C, pbar)

    # 7) Bootstrap CIs
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

    # 8) Print results
    print("\n" + "=" * 70)
    print(f"  PREDICTED TRAINING DATA MIXTURE: {args.target_model}")
    print("=" * 70)
    print(f"  {'Category':<25s}  {'Estimated %':>12s}  {'95% CI':>20s}")
    print("-" * 65)
    for i, cat in enumerate(ds.categories):
        ci_str = f"[{lo[i]*100:.1f}%, {hi[i]*100:.1f}%]" if args.bootstrap else "N/A"
        print(f"  {cat:<25s}  {pi_mean[i]*100:>11.1f}%  {ci_str:>20s}")
    if args.unknown_threshold is not None:
        ci_str_unk = (
            f"[{unknown_ci_lo*100:.1f}%, {unknown_ci_hi*100:.1f}%]"
            if args.bootstrap else "N/A"
        )
        print(f"  {'Unknown':<25s}  {unknown_bootstrap_mean*100:>11.1f}%  {ci_str_unk:>20s}")
    print("=" * 70)
    print(f"  Note: No ground truth available for {args.target_model}.")
    print(f"        These are *predicted* mixture proportions.")
    print("=" * 70)

    # 9) Save outputs
    payload = {
        "config": {
            "provider": args.generator,
            "target_model": args.target_model,
            "local_samples_dir": args.local_samples_dir,
            "merge_web": args.merge_web,
            "classifier": args.classifier,
            "seed": args.seed,
            "val_fraction": args.val_fraction,
            "num_prompts": args.num_prompts,
            "max_new_tokens": args.max_new_tokens,
            "gen_temperature": args.gen_temperature,
            "top_p": args.top_p,
            "prompts_style": args.prompts_style,
            "hf_model_name": args.hf_model_name,
            "hf_epochs": args.hf_epochs,
            "hf_batch_size": args.hf_batch_size,
            "hf_lr": args.hf_lr,
            "hf_weight_decay": args.hf_weight_decay,
            "hf_max_length": args.hf_max_length,
            "hf_pretrained_dir": args.hf_pretrained_dir,
            "bootstrap": args.bootstrap,
            "n_boot": args.n_boot,
            "naive": args.naive,
        },
        "ground_truth_available": False,
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

    print(f"\nOutputs written to: {out_dir}")

    # 10) Visualizations
    try:
        plot_confusion_matrix(C, ds.categories, str(out_dir / "confusion_matrix.png"))
        Ctpi_known = C.T @ pi_mean
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
        print("Plots saved.")
    except Exception as e:
        print(f"Warning: plotting failed: {e}")

    # 11) Comparison bar chart across models (if multiple summaries exist in output_dir)
    try:
        _plot_multi_model_comparison(Path(args.output_dir), out_dir)
    except Exception as e:
        print(f"Note: multi-model comparison plot skipped: {e}")


def _plot_multi_model_comparison(output_root: Path, current_out: Path) -> None:
    """
    Scan output_root for summary.json files from closed-API runs and produce a
    grouped bar chart comparing predicted mixtures across models.
    """
    import matplotlib.pyplot as plt

    summaries = {}
    for sub in sorted(output_root.iterdir()):
        sj = sub / "summary.json"
        if not sj.exists():
            continue
        with open(sj, "r") as f:
            data = json.load(f)
        cfg = data.get("config", {})
        if cfg.get("provider") not in ("openai", "google"):
            continue
        model_name = cfg.get("target_model", sub.name)
        summaries[model_name] = data

    if len(summaries) < 2:
        return

    all_cats = None
    for s in summaries.values():
        cats = s.get("categories_with_unknown", s.get("categories", []))
        if all_cats is None:
            all_cats = cats
        elif cats != all_cats:
            return

    if all_cats is None:
        return

    n_models = len(summaries)
    n_cats = len(all_cats)
    x = np.arange(n_cats)
    width = 0.8 / n_models

    fig, ax = plt.subplots(figsize=(max(10, n_cats * 1.2), 6))
    for i, (mname, sdata) in enumerate(summaries.items()):
        priors_key = "priors_with_unknown" if "priors_with_unknown" in sdata else "priors"
        vals = np.array(sdata[priors_key]["mean"])
        ax.bar(x + i * width, vals, width, label=mname, alpha=0.85)

    ax.set_xticks(x + width * (n_models - 1) / 2)
    ax.set_xticklabels(all_cats, rotation=45, ha="right")
    ax.set_ylabel("Estimated Proportion")
    ax.set_title("Predicted Training Data Mixture Comparison (Closed-Source Models)")
    ax.legend(fontsize=8)
    ax.set_ylim(bottom=0)
    plt.tight_layout()
    plt.savefig(str(current_out / "multi_model_comparison.png"), dpi=150)
    plt.close()
    print(f"Multi-model comparison chart saved to: {current_out / 'multi_model_comparison.png'}")


if __name__ == "__main__":
    main()
