"""Forward-return labels. Label at t uses only prices in (t, t+h]."""
from __future__ import annotations

import numpy as np
import pandas as pd


def add_labels(df: pd.DataFrame, horizons: list[int]) -> pd.DataFrame:
    """Append y_{h} = log(C_{t+h}/C_t) and dir_{h} = 1{y_h > 0} per symbol."""
    df = df.sort_values(["symbol", "date"]).reset_index(drop=True)
    g = df.groupby("symbol", sort=False)["close"]
    for h in horizons:
        y = np.log(g.shift(-h)) - np.log(df["close"])
        df[f"y_{h}"] = y
        df[f"dir_{h}"] = np.where(y.isna(), np.nan, (y > 0).astype(float))
    return df


def winsorize_train(y: pd.Series, pct: tuple[float, float]) -> pd.Series:
    """Clip a TRAIN target at its own percentiles (never applied to eval data)."""
    lo, hi = y.quantile(pct[0]), y.quantile(pct[1])
    return y.clip(lo, hi)
