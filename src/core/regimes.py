"""Volatility-regime detection: Calm / Normal / Turbulent.

Market 21-day realized volatility compared against *expanding-window* terciles
of its own history, refreshed at each month start. Fully causal: the thresholds
applied during month M use data through the end of month M-1 only.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

REGIME_LABELS = {0: "Calm", 1: "Normal", 2: "Turbulent"}


def market_realized_vol(mkt_ret: pd.Series, window: int, trading_days: int = 252) -> pd.Series:
    """Annualized rolling realized volatility of daily market returns."""
    return mkt_ret.rolling(window).std(ddof=1) * np.sqrt(trading_days)


def regime_states(
    mkt_ret: pd.Series,
    vol_window: int = 21,
    min_history_days: int = 756,
    trading_days: int = 252,
) -> pd.DataFrame:
    """Per-date regime state (0=Calm, 1=Normal, 2=Turbulent) plus thresholds.

    mkt_ret must be indexed by DatetimeIndex (daily, sorted).
    """
    rv = market_realized_vol(mkt_ret, vol_window, trading_days)
    dates = rv.index
    month_key = dates.to_period("M")

    state = np.ones(len(rv), dtype=np.int8)  # default Normal until enough history
    lo_thr = np.full(len(rv), np.nan)
    hi_thr = np.full(len(rv), np.nan)

    # threshold lookup computed once per month from history strictly before it
    valid = rv.dropna()
    for month in month_key.unique():
        in_month = month_key == month
        hist = valid[valid.index < month.to_timestamp()]
        if len(hist) < min_history_days - vol_window:
            continue
        lo, hi = np.nanpercentile(hist, [100 / 3, 200 / 3])
        cur = rv[in_month]
        s = np.where(cur < lo, 0, np.where(cur > hi, 2, 1)).astype(np.int8)
        s[np.isnan(cur)] = 1
        state[in_month] = s
        lo_thr[in_month] = lo
        hi_thr[in_month] = hi

    return pd.DataFrame(
        {"mkt_rv": rv, "regime": state, "regime_lo_thr": lo_thr, "regime_hi_thr": hi_thr},
        index=dates,
    )
