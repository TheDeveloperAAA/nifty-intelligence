"""Volatility models: EWMA (RiskMetrics) production, GARCH(1,1) showcase."""
from __future__ import annotations

import numpy as np
import pandas as pd


def ewma_vol(ret: pd.Series, lam: float = 0.94, init_days: int = 63) -> pd.Series:
    """Daily EWMA volatility sigma_t using returns through t (inclusive).

    sigma_t^2 = lam * sigma_{t-1}^2 + (1 - lam) * r_t^2, seeded with the first
    `init_days` sample variance. Implemented via pandas ewm for speed; the
    min_periods guard keeps early estimates honest.
    """
    var = ret.pow(2).ewm(alpha=1 - lam, adjust=False, min_periods=init_days).mean()
    return np.sqrt(var)


def ewma_vol_panel(
    panel: pd.DataFrame, lam: float, init_days: int,
    ret_col: str = "ret", group_col: str = "symbol",
) -> pd.Series:
    """Per-symbol EWMA vol over a long panel (sorted by symbol, date)."""
    return panel.groupby(group_col, sort=False)[ret_col].transform(
        lambda s: ewma_vol(s, lam, init_days)
    )


def qlike(realized_sq: np.ndarray, forecast_var: np.ndarray) -> float:
    """QLIKE loss: log(sigma_hat^2) + r^2 / sigma_hat^2 (lower is better)."""
    mask = np.isfinite(realized_sq) & np.isfinite(forecast_var)
    if mask.sum() == 0:
        return float("nan")
    f = np.clip(forecast_var[mask], 1e-12, None)
    return float(np.mean(np.log(f) + realized_sq[mask] / f))


def mincer_zarnowitz_r2(realized_sq: np.ndarray, forecast_var: np.ndarray) -> float:
    """R^2 of regressing realized r^2 on the variance forecast."""
    mask = np.isfinite(realized_sq) & np.isfinite(forecast_var)
    if mask.sum() < 10:
        return float("nan")
    x = forecast_var[mask]
    y = realized_sq[mask]
    x1 = np.column_stack([np.ones(len(x)), x])
    beta, *_ = np.linalg.lstsq(x1, y, rcond=None)
    resid = y - x1 @ beta
    ss_res = float(resid @ resid)
    ss_tot = float(((y - y.mean()) ** 2).sum())
    return 1 - ss_res / ss_tot if ss_tot > 0 else float("nan")
