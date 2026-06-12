"""Evaluation metrics: return-space, price-space, direction, calibration.

Honesty conventions:
- return-space metrics measure true skill (persistence is the bar);
- price-space metrics are reported alongside the SAME metrics for the naive
  forecast because price-level R^2 is dominated by persistence;
- direction CIs use effective sample size N/h (overlapping h-day labels).
"""
from __future__ import annotations

import numpy as np


def _clean(*arrays: np.ndarray) -> tuple[np.ndarray, ...]:
    mask = np.ones(len(arrays[0]), dtype=bool)
    for a in arrays:
        mask &= np.isfinite(a)
    return tuple(a[mask] for a in arrays)


def return_space(y: np.ndarray, yhat: np.ndarray) -> dict:
    y, yhat = _clean(np.asarray(y, float), np.asarray(yhat, float))
    if len(y) == 0:
        return {"n": 0}
    err = y - yhat
    ss_tot = float(((y - y.mean()) ** 2).sum())
    return {
        "n": int(len(y)),
        "mae": float(np.abs(err).mean()),
        "rmse": float(np.sqrt((err**2).mean())),
        "r2": float(1 - (err @ err) / ss_tot) if ss_tot > 0 else float("nan"),
    }


def price_space(close_t: np.ndarray, y: np.ndarray, yhat: np.ndarray) -> dict:
    close_t, y, yhat = _clean(np.asarray(close_t, float), np.asarray(y, float),
                              np.asarray(yhat, float))
    if len(y) == 0:
        return {"n": 0}
    p_true = close_t * np.exp(y)
    p_hat = close_t * np.exp(yhat)
    err = p_true - p_hat
    ss_tot = float(((p_true - p_true.mean()) ** 2).sum())
    return {
        "n": int(len(y)),
        "mae_rupees": float(np.abs(err).mean()),
        "rmse_rupees": float(np.sqrt((err**2).mean())),
        "r2": float(1 - (err @ err) / ss_tot) if ss_tot > 0 else float("nan"),
        "median_ape_pct": float(np.median(np.abs(err) / p_true) * 100),
    }


def wilson_ci(p: float, n_eff: float, z: float = 1.96) -> tuple[float, float]:
    if n_eff <= 0:
        return (float("nan"), float("nan"))
    denom = 1 + z**2 / n_eff
    center = (p + z**2 / (2 * n_eff)) / denom
    half = z * np.sqrt(p * (1 - p) / n_eff + z**2 / (4 * n_eff**2)) / denom
    return (float(center - half), float(center + half))


def _auc(y: np.ndarray, score: np.ndarray) -> float:
    """Rank-based AUC (Mann-Whitney), ties handled by average ranks."""
    pos = score[y == 1]
    neg = score[y == 0]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    from scipy.stats import rankdata

    ranks = rankdata(np.concatenate([pos, neg]))
    r_pos = ranks[: len(pos)].sum()
    return float((r_pos - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def direction(dir_true: np.ndarray, p_up: np.ndarray, h: int, n_bins: int = 10) -> dict:
    dir_true, p_up = _clean(np.asarray(dir_true, float), np.asarray(p_up, float))
    if len(dir_true) == 0:
        return {"n": 0}
    pred = (p_up > 0.5).astype(float)
    acc = float((pred == dir_true).mean())
    n_eff = len(dir_true) / h
    lo, hi = wilson_ci(acc, n_eff)
    base_rate = float(dir_true.mean())
    always_up_acc = max(base_rate, 1 - base_rate)
    brier = float(((p_up - dir_true) ** 2).mean())
    # expected calibration error, 10 equal-width bins
    bins = np.clip((p_up * n_bins).astype(int), 0, n_bins - 1)
    ece, rel = 0.0, []
    for b in range(n_bins):
        m = bins == b
        if m.sum() == 0:
            continue
        conf, obs = float(p_up[m].mean()), float(dir_true[m].mean())
        ece += (m.sum() / len(p_up)) * abs(conf - obs)
        rel.append({"bin": b, "n": int(m.sum()), "confidence": conf, "observed": obs})
    # conviction deciles: hit rate among top-10% |p - 0.5|
    conviction = np.abs(p_up - 0.5)
    top = conviction >= np.quantile(conviction, 0.9)
    top_acc = float((pred[top] == dir_true[top]).mean()) if top.sum() else float("nan")
    return {
        "n": int(len(dir_true)),
        "n_eff": float(n_eff),
        "accuracy": acc,
        "acc_wilson_lo": lo,
        "acc_wilson_hi": hi,
        "base_rate_up": base_rate,
        "always_up_accuracy": always_up_acc,
        "auc": _auc(dir_true, p_up),
        "brier": brier,
        "ece": float(ece),
        "reliability": rel,
        "top_decile_accuracy": top_acc,
    }
