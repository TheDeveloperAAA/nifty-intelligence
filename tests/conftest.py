"""Synthetic OHLCV fixtures for unit tests."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


def make_ohlcv(
    symbol: str = "TEST",
    n: int = 400,
    start: str = "2018-01-01",
    seed: int = 7,
    daily_vol: float = 0.015,
) -> pd.DataFrame:
    """Geometric random walk with consistent OHLC around the close path."""
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range(start, periods=n)
    rets = rng.normal(0.0004, daily_vol, n)
    close = 100 * np.exp(np.cumsum(rets))
    open_ = np.empty(n)
    open_[0] = close[0]
    open_[1:] = close[:-1] * np.exp(rng.normal(0, 0.003, n - 1))
    spread = np.abs(rng.normal(0.006, 0.002, n))
    high = np.maximum(open_, close) * (1 + spread)
    low = np.minimum(open_, close) * (1 - spread)
    volume = rng.integers(1e5, 5e5, n).astype(float)
    prev_close = np.empty(n)
    prev_close[0] = close[0]
    prev_close[1:] = close[:-1]
    df = pd.DataFrame(
        {
            "Date": dates,
            "Symbol": symbol,
            "symbol": symbol,
            "symbol_raw": symbol,
            "Series": "EQ",
            "Prev Close": prev_close,
            "Open": open_,
            "High": high,
            "Low": low,
            "Last": close,
            "Close": close,
            "VWAP": (high + low + close) / 3,
            "Volume": volume,
            "Turnover": close * volume,
            "Trades": np.nan,
            "Deliverable Volume": volume * 0.4,
            "%Deliverble": 0.4,
        }
    )
    return df


def plant_action(df: pd.DataFrame, t: int, factor: float, calm: bool = True) -> pd.DataFrame:
    """Plant a split/bonus at index t: scale all prices from t onward by factor.

    On the ex-date the open gaps exactly to the adjusted level and (if calm)
    the day trades in a tight range, mimicking a true corporate action.
    """
    df = df.copy()
    cols = ["Prev Close", "Open", "High", "Low", "Last", "Close", "VWAP"]
    df.loc[df.index >= t, cols] = df.loc[df.index >= t, cols] * factor
    df.loc[df.index >= t, "Volume"] /= factor
    df.loc[df.index >= t, "Deliverable Volume"] /= factor
    # Prev Close on the ex-date is the *unadjusted* prior close in this dataset
    df.loc[t, "Prev Close"] = df.loc[t - 1, "Close"]
    base = df.loc[t - 1, "Close"] * factor
    if calm:
        df.loc[t, "Open"] = base * 1.002
        df.loc[t, "Close"] = base * 1.004
        df.loc[t, "Last"] = df.loc[t, "Close"]
        df.loc[t, "High"] = base * 1.012
        df.loc[t, "Low"] = base * 0.996
        df.loc[t, "VWAP"] = base * 1.004
    return df


def plant_crash(df: pd.DataFrame, t: int, drop: float = -0.25) -> pd.DataFrame:
    """Plant a genuine crash at index t: open near flat then a wide-range fall."""
    df = df.copy()
    prev = df.loc[t - 1, "Close"]
    df.loc[t, "Prev Close"] = prev
    df.loc[t, "Open"] = prev * 0.99          # opens near previous close
    df.loc[t, "Close"] = prev * (1 + drop)   # ends deeply down
    df.loc[t, "Last"] = df.loc[t, "Close"]
    df.loc[t, "High"] = prev * 1.0
    df.loc[t, "Low"] = prev * (1 + drop) * 0.99
    df.loc[t, "VWAP"] = prev * (1 + drop / 2)
    # propagate the new level forward so later returns stay normal
    ratio = (1 + drop) / (df.loc[t + 1, "Prev Close"] / prev)
    cols = ["Prev Close", "Open", "High", "Low", "Last", "Close", "VWAP"]
    df.loc[df.index > t, cols] = df.loc[df.index > t, cols] * ratio
    return df


@pytest.fixture
def ohlcv():
    return make_ohlcv()


@pytest.fixture
def adjust_params():
    return {
        "large_drop_ratio": 0.70,
        "large_snap_tol": 0.06,
        "moderate_band": [0.70, 0.90],
        "moderate_snap_tol": 0.025,
        "moderate_open_tol": 0.025,
        "moderate_range_max": 0.08,
        "moderate_market_max": 0.04,
        "prevclose_mismatch_tol": 0.02,
    }
