from typing import List, Optional

import numpy as np
import matplotlib.pyplot as plt


def plot_confusion_matrix(C: np.ndarray, categories: List[str], path: str) -> None:
    plt.figure(figsize=(6, 5))
    im = plt.imshow(C, interpolation='nearest', cmap='Blues', vmin=0.0, vmax=1.0)
    plt.title('Confusion Matrix (row-normalized)')
    plt.colorbar(im, fraction=0.046, pad=0.04)
    tick_marks = np.arange(len(categories))
    plt.xticks(tick_marks, categories, rotation=45, ha='right')
    plt.yticks(tick_marks, categories)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def plot_priors_with_ci(categories: List[str], pi_mean: np.ndarray, lo: np.ndarray, hi: np.ndarray, path: str) -> None:
    x = np.arange(len(categories))
    y = pi_mean
    # Ensure non-negative error extents for Matplotlib
    lo_arr = np.asarray(lo, dtype=float)
    hi_arr = np.asarray(hi, dtype=float)
    # If any intervals are reversed, swap locally
    lo_fixed = np.minimum(lo_arr, hi_arr)
    hi_fixed = np.maximum(lo_arr, hi_arr)
    lower = np.maximum(0.0, y - lo_fixed)
    upper = np.maximum(0.0, hi_fixed - y)
    yerr = np.vstack([lower, upper])
    plt.figure(figsize=(7, 4))
    plt.bar(x, y, color='#4C72B0')
    plt.errorbar(x, y, yerr=yerr, fmt='none', ecolor='black', elinewidth=1, capsize=3)
    plt.xticks(x, categories, rotation=45, ha='right')
    plt.ylabel('Estimated Prior')
    plt.title('Estimated Class Priors (with 95% CI)')
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def plot_pbar_vs_ctpi(categories: List[str], pbar: np.ndarray, Ctpi: np.ndarray, path: str) -> None:
    x = np.arange(len(categories))
    width = 0.38
    plt.figure(figsize=(7, 4))
    plt.bar(x - width/2, pbar, width=width, label='p̄')
    plt.bar(x + width/2, Ctpi, width=width, label='Cᵀπ')
    plt.xticks(x, categories, rotation=45, ha='right')
    plt.ylabel('Probability')
    plt.title('Moment Matching Check: p̄ vs Cᵀπ')
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()
