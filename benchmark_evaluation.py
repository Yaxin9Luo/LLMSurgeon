#!/usr/bin/env python3
"""
Benchmark Evaluation Tool for Label-Shift Estimation Methods

This tool provides standardized evaluation metrics and comparison capabilities
for different approaches to estimating training data mixture proportions from
language model generations.

Usage:
    python benchmark_evaluation.py --results_dir out/labelshift_llama7b --ground_truth bench/specs/llama-1.yaml [--tol 0.02]
    python benchmark_evaluation.py --compare out/method1 out/method2 --ground_truth bench/specs/llama-1.yaml [--tol 0.02]
"""

import argparse
import json
import yaml
import numpy as np
# Matplotlib is only needed for plotting; import lazily in plot methods
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union
from dataclasses import dataclass
import pandas as pd


@dataclass
class MethodResult:
    """Container for a method's results"""
    name: str
    estimates: Dict[str, float]
    confidence_intervals: Optional[Dict[str, Tuple[float, float]]] = None
    metadata: Optional[Dict] = None


@dataclass
class BenchmarkMetrics:
    """Container for benchmark evaluation metrics"""
    l1_distance: float
    l2_distance: float
    kl_divergence: float
    max_absolute_error: float
    mean_absolute_error: float
    relative_errors: Dict[str, float]
    coverage_rate: Optional[float] = None  # For methods with confidence intervals
    interval_widths: Optional[Dict[str, float]] = None
    # Derived accuracy-style metrics
    # overlap_accuracy = 1 - 0.5 * L1, in [0,1]
    overlap_accuracy: float = 0.0
    # within_tolerance_rate = fraction of categories with |err| <= tol (if tol provided)
    within_tolerance_rate: Optional[float] = None
    # dual-threshold accuracy: I(|err|/y <= tau_rel or |err| <= tau_floor|)
    dual_threshold_accuracy: Optional[float] = None

class LabelShiftBenchmark:
    """Main benchmark evaluation class"""
    
    def __init__(
        self,
        ground_truth_file: str,
        tol: Optional[float] = None,
        rel_tol: Optional[float] = None,
    ):
        """
        Initialize benchmark with ground truth mixture proportions.
        
        Args:
            ground_truth_file: Path to YAML file with true category weights
            tol: Optional absolute tolerance for per-category accuracy (τ_floor)
            rel_tol: Optional relative tolerance for dual-threshold accuracy (τ_rel)
        """
        self.ground_truth_file = ground_truth_file
        self.ground_truth = self._load_ground_truth(ground_truth_file)
        # Canonical 7-class order from ground truth spec
        self.gt_categories_7 = list(self.ground_truth.keys())
        # Back-compat: some methods still reference self.categories
        self.categories = list(self.ground_truth.keys())
        # Optional absolute tolerance for per-category accuracy
        self.tolerance: Optional[float] = tol          # τ_floor
        # Optional relative tolerance for dual-threshold accuracy
        self.rel_tolerance: Optional[float] = rel_tol  # τ_rel

        
    def _load_ground_truth(self, filepath: str) -> Dict[str, float]:
        """Load ground truth proportions from YAML specification"""
        with open(filepath, 'r') as f:
            spec = yaml.safe_load(f)
        
        weights = spec['category_weights']
        # Normalize to probabilities
        total = sum(weights.values())
        return {cat: weight/total for cat, weight in weights.items()}
    
    def _load_method_results(self, results_dir: str) -> MethodResult:
        """
        Load results from a method's output directory.
        
        Expected format: method should provide results for all 7 categories:
        - CommonCrawl, C4, GitHub, Wikipedia, Books, Arxiv, StackExchange
        
        Alternative category names are mapped automatically:
        - Code -> GitHub
        - Papers -> Arxiv
        """
        results_path = Path(results_dir)
        summary_file = results_path / "summary.json"
        
        if not summary_file.exists():
            raise FileNotFoundError(f"No summary.json found in {results_dir}")
            
        with open(summary_file, 'r') as f:
            data = json.load(f)
        
        # Extract estimates (prefer priors; fall back to known alternative fields)
        if 'priors' in data:
            if 'mean' in data['priors']:
                estimates_list = data['priors']['mean']
                ci_lo = data['priors'].get('ci_lo')
                ci_hi = data['priors'].get('ci_hi')
            else:
                estimates_list = data['priors']['point']
                ci_lo = ci_hi = None
        elif 'global_mixture_over_sampled_pool' in data:
            # Fallback format used by some methods (e.g., minkpp_mix). No CI provided.
            estimates_list = data['global_mixture_over_sampled_pool']
            ci_lo = ci_hi = None
        elif 'global_mixture' in data:
            # Format used by mink/minkpp threshold methods (dict format)
            gm = data['global_mixture']
            if isinstance(gm, dict):
                # Dict format: {"Web": 0.12, "GitHub": 0.27, ...}
                categories = data.get('categories', list(gm.keys()))
                estimates_list = [gm[cat] for cat in categories]
            else:
                # List format
                estimates_list = gm
            ci_lo = ci_hi = None
        elif 'per_category' in data:
            # Format used by DUCI methods (list of category dicts with normalized_proportion)
            per_cat = data['per_category']
            categories = data.get('categories', [pc.get('category') for pc in per_cat])
            # Prefer normalized_proportion, fall back to duci_proportion
            estimates_list = []
            for pc in per_cat:
                if 'normalized_proportion' in pc:
                    estimates_list.append(pc['normalized_proportion'])
                elif 'duci_proportion' in pc:
                    estimates_list.append(pc['duci_proportion'])
                else:
                    estimates_list.append(0.0)
            ci_lo = ci_hi = None
        else:
            raise ValueError(f"No prior estimates found in {summary_file}")
        
        categories = data.get('categories', [])
        estimates = dict(zip(categories, estimates_list))
        
        # Handle confidence intervals
        confidence_intervals = None
        if ci_lo is not None and ci_hi is not None:
            confidence_intervals = {
                cat: (lo, hi) for cat, lo, hi in zip(categories, ci_lo, ci_hi)
            }
        
        # Do NOT map to ground truth yet. We'll align adaptively (6 vs 7 classes) later.
        method_name = results_path.name
        meta = data.get('config', {}) or {}
        meta['_original_categories'] = categories
        return MethodResult(
            name=method_name,
            estimates=estimates,
            confidence_intervals=confidence_intervals,
            metadata=meta,
        )
    
    def _map_categories(self, estimates: Dict[str, float]) -> Dict[str, float]:
        """Deprecated: legacy direct mapping. Kept for backward-compat but unused now."""
        mapped = {}
        for gt_cat in self.ground_truth:
            if gt_cat in estimates:
                mapped[gt_cat] = estimates[gt_cat]
            elif gt_cat == "GitHub" and "Code" in estimates:
                mapped[gt_cat] = estimates["Code"]
            elif gt_cat == "Arxiv" and "Papers" in estimates:
                mapped[gt_cat] = estimates["Papers"]
            else:
                for est_cat, val in estimates.items():
                    if est_cat.lower() == gt_cat.lower():
                        mapped[gt_cat] = val
                        break
                else:
                    mapped[gt_cat] = 0.0
        return mapped

    def _map_categories_ci(self, cis: Dict[str, Tuple[float, float]]) -> Dict[str, Tuple[float, float]]:
        """Deprecated: legacy CI mapping. Kept for backward-compat but unused now."""
        mapped = {}
        for gt_cat in self.ground_truth:
            if gt_cat in cis:
                mapped[gt_cat] = cis[gt_cat]
            elif gt_cat == "GitHub" and "Code" in cis:
                mapped[gt_cat] = cis["Code"]
            elif gt_cat == "Arxiv" and "Papers" in cis:
                mapped[gt_cat] = cis["Papers"]
            else:
                for est_cat, val in cis.items():
                    if est_cat.lower() == gt_cat.lower():
                        mapped[gt_cat] = val
                        break
                else:
                    mapped[gt_cat] = (0.0, 0.0)
        return mapped

    # ------------------------------
    # Category alignment helpers
    # ------------------------------
    @staticmethod
    def _norm_name(name: str) -> str:
        return name.strip().lower()

    def _scheme_categories(self, scheme: str) -> List[str]:
        if scheme == '6-class':
            return ['Web', 'GitHub', 'Wikipedia', 'Books', 'Arxiv', 'StackExchange']
        return self.gt_categories_7

    def _merge_web_estimates(self, estimates: Dict[str, float]) -> Dict[str, float]:
        out = dict(estimates)
        cc_key = next((k for k in list(out.keys()) if self._norm_name(k) == 'commoncrawl'), None)
        c4_key = next((k for k in list(out.keys()) if self._norm_name(k) == 'c4'), None)
        web_val = 0.0
        if cc_key is not None:
            web_val += out.pop(cc_key)
        if c4_key is not None:
            web_val += out.pop(c4_key)
        if web_val > 0.0:
            out['Web'] = out.get('Web', 0.0) + web_val
        return out

    def _merge_web_cis(self, cis: Optional[Dict[str, Tuple[float, float]]]) -> Optional[Dict[str, Tuple[float, float]]]:
        if cis is None:
            return None
        out = dict(cis)
        cc_key = next((k for k in list(out.keys()) if self._norm_name(k) == 'commoncrawl'), None)
        c4_key = next((k for k in list(out.keys()) if self._norm_name(k) == 'c4'), None)
        lo_sum = hi_sum = 0.0
        changed = False
        if cc_key is not None:
            lo, hi = out.pop(cc_key)
            lo_sum += lo
            hi_sum += hi
            changed = True
        if c4_key is not None:
            lo, hi = out.pop(c4_key)
            lo_sum += lo
            hi_sum += hi
            changed = True
        if changed:
            lo_web, hi_web = out.get('Web', (0.0, 0.0))
            out['Web'] = (lo_web + lo_sum, hi_web + hi_sum)
        return out

    def _align(self, estimates: Dict[str, float], cis: Optional[Dict[str, Tuple[float, float]]], scheme: str) -> Tuple[List[str], np.ndarray, Optional[Dict[str, Tuple[float, float]]]]:
        cats = self._scheme_categories(scheme)
        est_local = estimates
        cis_local = cis
        if scheme == '6-class':
            est_local = self._merge_web_estimates(est_local)
            cis_local = self._merge_web_cis(cis_local)
        out = {}
        cis_out: Optional[Dict[str, Tuple[float, float]]] = {} if cis_local is not None else None
        for cat in cats:
            key = None
            if cat in est_local:
                key = cat
            elif cat == 'GitHub' and 'Code' in est_local:
                key = 'Code'
            elif cat == 'Arxiv' and 'Papers' in est_local:
                key = 'Papers'
            else:
                for k in est_local.keys():
                    if self._norm_name(k) == self._norm_name(cat):
                        key = k
                        break
            out[cat] = est_local.get(key, 0.0)
            if cis_out is not None:
                if cis_local and key in (cis_local.keys() if cis_local else []):
                    cis_out[cat] = cis_local[key]  # type: ignore[index]
                else:
                    cis_out[cat] = (0.0, 0.0)
        est_arr = np.array([out[c] for c in cats], dtype=float)
        return cats, est_arr, cis_out

    def _align_ground_truth(self, scheme: str) -> Tuple[List[str], np.ndarray]:
        if scheme == '6-class':
            gt = dict(self.ground_truth)
            cc = gt.pop('CommonCrawl', 0.0)
            c4 = gt.pop('C4', 0.0)
            gt['Web'] = gt.get('Web', 0.0) + cc + c4
            cats = self._scheme_categories('6-class')
            arr = np.array([gt.get(c, 0.0) for c in cats], dtype=float)
            s = arr.sum()
            if s > 0:
                arr = arr / s
            return cats, arr
        cats = self._scheme_categories('7-class')
        arr = np.array([self.ground_truth[c] for c in cats], dtype=float)
        return cats, arr

    def _decide_scheme_for_methods(self, methods: List[MethodResult]) -> str:
        # Check if ground truth has both CommonCrawl and C4 (Llama-1 style)
        gt_has_cc = any(self._norm_name(c) == 'commoncrawl' for c in self.ground_truth.keys())
        gt_has_c4 = any(self._norm_name(c) == 'c4' for c in self.ground_truth.keys())
        gt_is_7class_llama_style = gt_has_cc and gt_has_c4
        
        for m in methods:
            cats = m.metadata.get('_original_categories') if m.metadata else None
            if cats and any(self._norm_name(c) == 'web' for c in cats):
                # Method merged web sources into 'Web' - only use 6-class if GT has both CC and C4
                if gt_is_7class_llama_style:
                    return '6-class'
            if cats and gt_is_7class_llama_style:
                # Only check CC/C4 XOR logic when GT actually has both
                has_cc = any(self._norm_name(c) == 'commoncrawl' for c in cats)
                has_c4 = any(self._norm_name(c) == 'c4' for c in cats)
                if has_cc ^ has_c4:
                    return '6-class'
        return '7-class'
    
    def evaluate_method(self, results_dir: str) -> Tuple[MethodResult, BenchmarkMetrics]:
        """Evaluate a single method against ground truth (adaptive 6/7-class)."""
        method_result = self._load_method_results(results_dir)
        scheme = self._decide_scheme_for_methods([method_result])
        metrics = self._compute_metrics(method_result, scheme)
        return method_result, metrics
    
    def _compute_metrics(self, method_result: MethodResult, scheme: str) -> BenchmarkMetrics:
        """Compute evaluation metrics for a method under a chosen category scheme."""
        cats_gt, true_probs = self._align_ground_truth(scheme)
        cats_est, est_probs, _ = self._align(method_result.estimates, method_result.confidence_intervals, scheme)
        # Sanity: ensure same order
        if cats_gt != cats_est:
            raise RuntimeError("Internal mismatch in category alignment")
        
        # Basic distance metrics
        l1_distance = np.sum(np.abs(true_probs - est_probs))
        l2_distance = np.sqrt(np.sum((true_probs - est_probs) ** 2))
        
        # KL divergence (add small epsilon to avoid log(0))
        epsilon = 1e-10
        kl_div = np.sum(true_probs * np.log((true_probs + epsilon) / (est_probs + epsilon)))
        
        # Error metrics
        abs_errors = np.abs(true_probs - est_probs)
        max_abs_error = np.max(abs_errors)
        mean_abs_error = np.mean(abs_errors)

        # Accuracy-style metrics
        # Overlap accuracy: 1 - TV distance = 1 - 0.5 * L1, bounded in [0,1]
        overlap_accuracy = float(1.0 - 0.5 * l1_distance)
        # Within-tolerance rate if a tolerance is provided
        within_tol_rate: Optional[float] = None
        if getattr(self, 'tolerance', None) is not None:
            tol = float(self.tolerance)  # type: ignore[arg-type]
            within_tol_rate = float(np.mean(abs_errors <= tol))
            
        # Dual-threshold accuracy:
        # 1/N * sum I( |err_i| / y_i <= tau_rel  OR  |err_i| <= tau_floor )
        dual_threshold_accuracy: Optional[float] = None
        rel_tol = getattr(self, 'rel_tolerance', None)
        floor_tol = getattr(self, 'tolerance', None)

        if rel_tol is not None or floor_tol is not None:
            indicators = []
            for i in range(len(true_probs)):
                err = abs_errors[i]
                y = true_probs[i]

                cond_rel = False
                if rel_tol is not None and y > 0:
                    cond_rel = (err / y) <= rel_tol

                cond_abs = False
                if floor_tol is not None:
                    cond_abs = err <= floor_tol

                indicators.append(1.0 if (cond_rel or cond_abs) else 0.0)

            dual_threshold_accuracy = float(np.mean(indicators))

        
        # Relative errors (avoid division by zero)
        rel_errors = {}
        for i, cat in enumerate(cats_gt):
            if true_probs[i] > 0:
                rel_errors[cat] = abs_errors[i] / true_probs[i]
            else:
                rel_errors[cat] = float('inf') if abs_errors[i] > 0 else 0.0
        
        # Confidence interval metrics (if available)
        coverage_rate = None
        interval_widths = None
        if method_result.confidence_intervals:
            coverage_count = 0
            widths = {}
            cats_ci, _, cis_aligned = self._align(method_result.estimates, method_result.confidence_intervals, scheme)
            if cats_ci != cats_gt:
                raise RuntimeError("Internal mismatch in CI alignment")
            for i, cat in enumerate(cats_gt):
                if cis_aligned and cat in cis_aligned:
                    ci_lo, ci_hi = cis_aligned[cat]
                    lo = float(min(ci_lo, ci_hi))
                    hi = float(max(ci_lo, ci_hi))
                    if lo <= true_probs[i] <= hi:
                        coverage_count += 1
                    widths[cat] = max(0.0, hi - lo)
                else:
                    widths[cat] = 0.0
            coverage_rate = coverage_count / len(cats_gt)
            interval_widths = widths
        
        return BenchmarkMetrics(
            l1_distance=l1_distance,
            l2_distance=l2_distance,
            kl_divergence=kl_div,
            max_absolute_error=max_abs_error,
            mean_absolute_error=mean_abs_error,
            relative_errors=rel_errors,
            coverage_rate=coverage_rate,
            interval_widths=interval_widths,
            overlap_accuracy=overlap_accuracy,
            within_tolerance_rate=within_tol_rate,
            dual_threshold_accuracy=dual_threshold_accuracy,
        )

    
    def compare_methods(self, results_dirs: List[str]) -> pd.DataFrame:
        """Compare multiple methods and return summary table (adaptive 6/7-class)."""
        loaded: List[MethodResult] = []
        for d in results_dirs:
            try:
                loaded.append(self._load_method_results(d))
            except Exception as e:
                print(f"Warning: Failed to load {d}: {e}")
        scheme = self._decide_scheme_for_methods(loaded)

        results = []
        for mr in loaded:
            try:
                metrics = self._compute_metrics(mr, scheme)
                row = {
                    'Method': mr.name,
                    'L1 Distance': f"{metrics.l1_distance:.4f}",
                    'L2 Distance': f"{metrics.l2_distance:.4f}",
                    'KL Divergence': f"{metrics.kl_divergence:.4f}",
                    'Max Abs Error': f"{metrics.max_absolute_error:.4f}",
                    'Mean Abs Error': f"{metrics.mean_absolute_error:.4f}",
                }
                # Add accuracy-style summaries
                row['Accuracy (1 - L1/2)'] = f"{metrics.overlap_accuracy:.4f}"
                if metrics.within_tolerance_rate is not None:
                    row['Within Tol'] = f"{metrics.within_tolerance_rate:.2%}"
                if metrics.dual_threshold_accuracy is not None:
                    row['DualThresh Acc'] = f"{metrics.dual_threshold_accuracy:.4f}"
                if metrics.coverage_rate is not None:
                    row['CI Coverage'] = f"{metrics.coverage_rate:.2%}"
                    avg_width = np.mean(list(metrics.interval_widths.values()))
                    row['Avg CI Width'] = f"{avg_width:.4f}"
                results.append(row)
            except Exception as e:
                print(f"Warning: Failed to evaluate {mr.name}: {e}")
                continue
        return pd.DataFrame(results)
    
    def plot_comparison(self, results_dirs: List[str], output_path: str = "benchmark_comparison.png"):
        """Create comprehensive comparison plots with adaptive 6/7-class alignment."""
        import matplotlib.pyplot as plt
        methods: List[MethodResult] = []
        for d in results_dirs:
            try:
                methods.append(self._load_method_results(d))
            except Exception as e:
                print(f"Warning: Failed to load {d}: {e}")
        if not methods:
            print("No valid methods to plot")
            return

        scheme = self._decide_scheme_for_methods(methods)
        cats, gt_arr = self._align_ground_truth(scheme)

        aligned = []
        for m in methods:
            cats_m, est_arr, cis_m = self._align(m.estimates, m.confidence_intervals, scheme)
            if cats_m != cats:
                print(f"Warning: Category mismatch for {m.name}; skipping")
                continue
            aligned.append((m, est_arr, cis_m))
        if not aligned:
            print("No aligned methods to plot")
            return

        # Figure 1: Estimated vs Ground Truth (bar chart)
        fig_width = max(16, min(48, 0.35 * len(cats)))
        plt.figure(figsize=(fig_width, 6))
        x = np.arange(len(cats))
        width = 0.8 / (len(aligned) + 1)
        plt.bar(x - width * len(aligned)/2, gt_arr.tolist(), width,
                label='Ground Truth', color='black', alpha=0.7)
        colors = plt.cm.Set3(np.linspace(0, 1, len(aligned)))
        for i, (m, est_arr, cis_m) in enumerate(aligned):
            vals = est_arr.tolist()
            plt.bar(x - width * len(aligned)/2 + width * (i+1), vals, width,
                    label=m.name, color=colors[i], alpha=0.8)
            if cis_m:
                cis = [cis_m.get(cat, (0.0, 0.0)) for cat in cats]
                lower = []
                upper = []
                for v, ci in zip(vals, cis):
                    lo, hi = float(min(ci[0], ci[1])), float(max(ci[0], ci[1]))
                    lower.append(max(0.0, v - lo))
                    upper.append(max(0.0, hi - v))
                yerr = np.vstack([lower, upper])
                plt.errorbar(x - width * len(aligned)/2 + width * (i+1), vals,
                             yerr=yerr, fmt='none', color='black', capsize=3, alpha=0.6)
        plt.xlabel('Categories')
        plt.ylabel('Proportion')
        plt.title('Estimated vs Ground Truth Proportions')
        # Show all categories on the x-axis
        plt.xticks(x, cats, rotation=45, ha='right')
        plt.tick_params(axis='x', labelsize=7)
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Comparison plot saved to {output_path}")

        # Figure 2: Absolute Error by Category (line plot)
        plt.figure(figsize=(fig_width, 5))
        for i, (m, est_arr, _) in enumerate(aligned):
            errors = np.abs(est_arr - gt_arr)
            plt.plot(range(len(cats)), errors, 'o-', label=m.name, color=colors[i])
        plt.xlabel('Categories')
        plt.ylabel('Absolute Error')
        plt.title('Absolute Error by Category')
        # Show all categories on the x-axis
        plt.xticks(range(len(cats)), cats, rotation=45, ha='right')
        plt.tick_params(axis='x', labelsize=7)
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        out_errors = output_path.replace('.png', '_errors.png')
        plt.savefig(out_errors, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Absolute error plot saved to {out_errors}")

        # Additionally, save a zoomed-in plot on top-K categories by ground truth proportion
        try:
            top_k = min(20, len(cats))
            idx_sorted = np.argsort(gt_arr)[-top_k:][::-1]
            cats_top = [cats[i] for i in idx_sorted]
            gt_top = gt_arr[idx_sorted]
            xt = np.arange(len(cats_top))
            width = 0.8 / (len(aligned) + 1)
            plt.figure(figsize=(max(12, 0.6 * top_k), 5))
            plt.bar(xt - width * len(aligned)/2, gt_top.tolist(), width, label='Ground Truth', color='black', alpha=0.7)
            colors = plt.cm.Set3(np.linspace(0, 1, len(aligned)))
            for i, (m, est_arr, cis_m) in enumerate(aligned):
                vals = est_arr[idx_sorted].tolist()
                plt.bar(xt - width * len(aligned)/2 + width * (i+1), vals, width, label=m.name, color=colors[i], alpha=0.8)
            plt.xticks(xt, cats_top, rotation=45, ha='right')
            plt.ylabel('Proportion')
            plt.title(f'Estimated vs Ground Truth (Top {top_k} Categories)')
            plt.legend()
            plt.tight_layout()
            out_top = output_path.replace('.png', f'_top{top_k}.png')
            plt.savefig(out_top, dpi=300, bbox_inches='tight')
            plt.close()
            print(f"Top-{top_k} comparison plot saved to {out_top}")
        except Exception as e:
            print(f"Warning: failed to save top-K zoomed plot: {e}")
    
    def generate_report(self, results_dirs: List[str], output_file: str = "benchmark_report.md"):
        """Generate a comprehensive markdown report (adaptive 6/7-class)."""
        report_lines = [
            "# Label-Shift Estimation Benchmark Report",
            "",
            f"**Ground Truth Model**: {Path(self.ground_truth_file).stem}",
            "",
            "## Ground Truth Proportions",
            "",
        ]

        # Load methods to choose scheme
        methods: List[MethodResult] = []
        for d in results_dirs:
            try:
                methods.append(self._load_method_results(d))
            except Exception as e:
                report_lines.append(f"- Failed to load {Path(d).name}: {e}")
        scheme = self._decide_scheme_for_methods(methods) if methods else '7-class'
        cats, gt_arr = self._align_ground_truth(scheme)

        # Ground truth table
        report_lines.append("| Category | Proportion |")
        report_lines.append("|----------|------------|")
        for c, v in zip(cats, gt_arr):
            report_lines.append(f"| {c} | {v:.4f} |")
        report_lines.append("")

        # Method comparison table
        comparison_df = self.compare_methods(results_dirs)
        if not comparison_df.empty:
            report_lines.append("## Method Comparison")
            report_lines.append("")
            report_lines.append(comparison_df.to_markdown(index=False))
            report_lines.append("")

        # Detailed results
        report_lines.append("## Detailed Results")
        report_lines.append("")
        for d in results_dirs:
            try:
                m = self._load_method_results(d)
                metrics = self._compute_metrics(m, scheme)
                cats_eval, est_arr, cis_aligned = self._align(m.estimates, m.confidence_intervals, scheme)
                if cats_eval != cats:
                    raise RuntimeError("Category mismatch in report alignment")
                report_lines.extend([
                    f"### {m.name}",
                    "",
                    "**Estimates vs Ground Truth:**",
                    "",
                    "| Category | Estimate | Ground Truth | Absolute Error | Relative Error |",
                    "|----------|----------|--------------|----------------|----------------|"
                ])
                for i, c in enumerate(cats):
                    est = float(est_arr[i])
                    true_val = float(gt_arr[i])
                    abs_err = abs(est - true_val)
                    rel_err = metrics.relative_errors[c]
                    rel_err_str = f"{rel_err:.2%}" if rel_err != float('inf') else "∞"
                    ci_str = ""
                    if cis_aligned and c in cis_aligned:
                        lo, hi = cis_aligned[c]
                        ci_str = f" [{lo:.4f}, {hi:.4f}]"
                    report_lines.append(f"| {c} | {est:.4f}{ci_str} | {true_val:.4f} | {abs_err:.4f} | {rel_err_str} |")
                report_lines.extend([
                    "",
                    "**Summary Metrics:**",
                    f"- L1 Distance: {metrics.l1_distance:.4f}",
                    f"- L2 Distance: {metrics.l2_distance:.4f}",
                    f"- KL Divergence: {metrics.kl_divergence:.4f}",
                    f"- Mean Absolute Error: {metrics.mean_absolute_error:.4f}",
                    f"- Max Absolute Error: {metrics.max_absolute_error:.4f}",
                    f"- Accuracy (1 - L1/2): {metrics.overlap_accuracy:.4f}",
                ])
                if metrics.within_tolerance_rate is not None:
                    report_lines.append(f"- Within-Tolerance Accuracy: {metrics.within_tolerance_rate:.2%}")
                if metrics.dual_threshold_accuracy is not None:
                    report_lines.append(f"- Dual-Threshold Accuracy: {metrics.dual_threshold_accuracy:.4f}")
                if metrics.coverage_rate is not None:
                    report_lines.append(f"- Confidence Interval Coverage: {metrics.coverage_rate:.2%}")
                    avg_width = np.mean(list(metrics.interval_widths.values()))
                    report_lines.append(f"- Average CI Width: {avg_width:.4f}")
                report_lines.append("")
            except Exception as e:
                report_lines.extend([
                    f"### {Path(d).name} (Failed)",
                    f"Error: {e}",
                    ""
                ])

        with open(output_file, 'w') as f:
            f.write('\n'.join(report_lines))
        print(f"Benchmark report saved to {output_file}")


def main():
    parser = argparse.ArgumentParser(description="Benchmark evaluation for label-shift estimation methods")
    parser.add_argument("--results_dir", type=str, help="Single method results directory to evaluate")
    parser.add_argument("--compare", nargs='+', help="Multiple method directories to compare")
    parser.add_argument("--ground_truth", type=str, required=True, help="Ground truth YAML file")
    parser.add_argument("--output_dir", type=str, default="benchmark_output", help="Output directory for plots and reports")
    parser.add_argument("--no_plots", action="store_true", help="Skip generating plots")
    parser.add_argument("--no_report", action="store_true", help="Skip generating markdown report")
    parser.add_argument("--tol", type=float, default=None, help="Absolute tolerance for per-category accuracy (e.g., 0.02)")
    parser.add_argument("--rel_tol", type=float, default=None, help="Relative tolerance for dual-threshold accuracy (e.g., 0.10)")

    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    
    # Initialize benchmark
    benchmark = LabelShiftBenchmark(args.ground_truth, tol=args.tol, rel_tol=args.rel_tol)

    
    # Determine methods to evaluate
    if args.results_dir:
        methods = [args.results_dir]
    elif args.compare:
        methods = args.compare
    else:
        parser.error("Must specify either --results_dir or --compare")
    
    # Run evaluation
    if len(methods) == 1:
        # Single method evaluation
        method_result, metrics = benchmark.evaluate_method(methods[0])
        print(f"\n=== Evaluation Results for {method_result.name} ===")
        # print(f"L1 Distance: {metrics.l1_distance:.4f}")
        # print(f"L2 Distance: {metrics.l2_distance:.4f}")
        # print(f"KL Divergence: {metrics.kl_divergence:.4f}")
        # print(f"Mean Absolute Error: {metrics.mean_absolute_error:.4f}")
        # print(f"Max Absolute Error: {metrics.max_absolute_error:.4f}")
        print(f"Accuracy (1 - L1/2): {metrics.overlap_accuracy:.4f}")
        if metrics.within_tolerance_rate is not None:
            print(f"Within-Tolerance Accuracy: {metrics.within_tolerance_rate:.2%} (tol={args.tol})")
        if metrics.dual_threshold_accuracy is not None:
            print(f"Dual-Threshold Accuracy: {metrics.dual_threshold_accuracy:.2%} (rel_tol={args.rel_tol}, floor_tol={args.tol})")

        if metrics.coverage_rate is not None:
            print(f"CI Coverage Rate: {metrics.coverage_rate:.2%}")
            avg_width = np.mean(list(metrics.interval_widths.values()))
            print(f"Average CI Width: {avg_width:.4f}")
    else:
        # Multiple method comparison
        comparison_df = benchmark.compare_methods(methods)
        print("\n=== Method Comparison ===")
        print(comparison_df.to_string(index=False))
    
    # Generate plots
    if not args.no_plots:
        plot_path = output_dir / "comparison.png"
        benchmark.plot_comparison(methods, str(plot_path))
    
    # Generate report
    if not args.no_report:
        report_path = output_dir / "report.md"
        benchmark.generate_report(methods, str(report_path))


if __name__ == "__main__":
    main()
