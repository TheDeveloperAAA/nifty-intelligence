"""Volatility-scaled split-conformal prediction intervals.

Nonconformity score: s_i = |y_i - yhat_i| / (sigma_i * sqrt(h)), with sigma_i
the EWMA daily vol at prediction time. The (1-alpha) empirical quantile q of
calibration scores gives the interval yhat +/- q * sigma_t * sqrt(h). Scores
are calibrated only on data that precedes the period being predicted.
"""
from __future__ import annotations

import numpy as np


def conformal_quantile(scores: np.ndarray, alpha: float) -> float:
    """Finite-sample-valid quantile: ceil((1-alpha)(n+1))/n empirical quantile."""
    s = np.asarray(scores, dtype=float)
    s = s[np.isfinite(s)]
    n = len(s)
    if n == 0:
        return float("nan")
    k = int(np.ceil((1 - alpha) * (n + 1)))
    k = min(k, n)
    return float(np.sort(s)[k - 1])


def nonconformity(y: np.ndarray, yhat: np.ndarray, sigma: np.ndarray, h: int) -> np.ndarray:
    scale = np.clip(sigma * np.sqrt(h), 1e-8, None)
    return np.abs(y - yhat) / scale


def interval(yhat: np.ndarray, sigma: np.ndarray, h: int, q: float) -> tuple[np.ndarray, np.ndarray]:
    half = q * np.clip(sigma * np.sqrt(h), 1e-8, None)
    return yhat - half, yhat + half


def coverage(y: np.ndarray, lo: np.ndarray, hi: np.ndarray) -> float:
    mask = np.isfinite(y) & np.isfinite(lo) & np.isfinite(hi)
    if mask.sum() == 0:
        return float("nan")
    return float(((y[mask] >= lo[mask]) & (y[mask] <= hi[mask])).mean())
