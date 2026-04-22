from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict

import numpy as np


def project_to_simplex(v: np.ndarray) -> np.ndarray:
    """
    Euclidean projection of a vector onto the probability simplex {x: x>=0, sum x = 1}.
    """
    if v.ndim != 1:
        raise ValueError("v must be 1D")
    # Sanitize non-finite values to avoid NaN propagation
    v = np.asarray(v, dtype=float)
    if not np.all(np.isfinite(v)):
        v = np.nan_to_num(v, nan=0.0, posinf=0.0, neginf=0.0)
    n = v.shape[0]
    u = np.sort(v)[::-1]
    cssv = np.cumsum(u)
    mask = u * np.arange(1, n + 1) > (cssv - 1)
    idxs = np.nonzero(mask)[0]
    if idxs.size == 0:
        # Degenerate case (e.g., all zeros): return uniform
        return np.ones(n, dtype=float) / float(n)
    rho = idxs[-1]
    theta = (cssv[rho] - 1) / (rho + 1)
    w = np.maximum(v - theta, 0)
    # Handle degenerate all-zeros
    s = w.sum()
    if s <= 0:
        return np.ones_like(w) / n
    return w / s


def estimate_priors_least_squares(C: np.ndarray, pbar: np.ndarray) -> np.ndarray:
    """
    Solve pi = argmin || C^T pi - pbar ||_2^2, then project onto simplex.
    """
    A = C.T  # shape [K, K]
    b = pbar  # shape [K]
    # Unconstrained least squares
    pi, *_ = np.linalg.lstsq(A, b, rcond=None)
    pi = pi.astype(np.float64)
    # Project onto simplex
    return project_to_simplex(pi)


def bootstrap_priors(
    C: np.ndarray,
    probs: np.ndarray,
    n_boot: int = 200,
    seed: int = 0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Bootstrap over per-sample predicted probabilities to get CIs for priors.
    probs: [N, K] matrix of p^(y|x) per generated sample x.
    Returns (pi_mean, pi_lo, pi_hi) at 95% CI.
    """
    rng = np.random.default_rng(seed)
    N, K = probs.shape
    pis = []
    for _ in range(n_boot):
        idx = rng.integers(0, N, size=N)
        pbar = probs[idx].mean(axis=0)
        pi = estimate_priors_least_squares(C, pbar)
        pis.append(pi)
    P = np.stack(pis, axis=0)  # [B, K]
    pi_mean = P.mean(axis=0)
    lo = np.percentile(P, 2.5, axis=0)
    hi = np.percentile(P, 97.5, axis=0)
    return pi_mean, lo, hi
