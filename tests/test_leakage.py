"""The two killer leakage properties, plus label timing.

(a) prefix invariance — features at date t computed on data truncated at t
    equal the features at t computed on the full series;
(b) future-shuffle invariance — perturbing prices after t leaves features
    at t unchanged;
(c) labels at t use only prices in (t, t+h].
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.core.indicators import ALL_FEATURES, NUMERIC_FEATURES, compute_features
from src.core.labeling import add_labels
from tests.conftest import make_ohlcv

FPARAMS = {
    "warmup_days": 252,
    "sma_windows": [10, 50],
    "vol_windows": [21, 63],
    "rsi_window": 14,
    "macd": [12, 26, 9],
    "bollinger_window": 20,
    "bollinger_k": 2.0,
    "atr_window": 14,
    "beta_window": 252,
    "beta_min_periods": 126,
    "amihud_window": 21,
    "regime_vol_window": 21,
    "regime_min_history": 252,
}


def _panel(n=520, n_symbols=3):
    frames = []
    for i in range(n_symbols):
        df = make_ohlcv(symbol=f"S{i}", n=n, seed=100 + i)
        frames.append(df)
    panel = pd.concat(frames, ignore_index=True)
    panel = panel.rename(
        columns={
            "Date": "date", "Open": "open", "High": "high", "Low": "low",
            "Close": "close", "VWAP": "vwap", "Volume": "volume",
            "Turnover": "turnover", "%Deliverble": "pct_deliv",
        }
    )
    panel["sector"] = panel["symbol"].map({"S0": "A", "S1": "A", "S2": "B"})
    return panel.sort_values(["symbol", "date"]).reset_index(drop=True)


def _proxy(panel):
    ret = panel.groupby("symbol")["close"].pct_change()
    mkt = ret.groupby(panel["date"]).mean().rename("mkt_ret").to_frame()
    sec = (
        panel.assign(r=ret)
        .groupby(["date", "sector"])["r"].mean().unstack().add_prefix("sec_ret_")
    )
    return mkt.join(sec).reset_index()


def test_prefix_invariance():
    panel = _panel()
    cutoff = panel["date"].sort_values().unique()[400]
    full = compute_features(panel, _proxy(panel), FPARAMS)
    trunc_panel = panel[panel["date"] <= cutoff].reset_index(drop=True)
    trunc = compute_features(trunc_panel, _proxy(trunc_panel), FPARAMS)

    full_cut = full[full["date"] <= cutoff].reset_index(drop=True)
    assert len(full_cut) == len(trunc)
    for col in NUMERIC_FEATURES:
        a = full_cut[col].to_numpy(float)
        b = trunc[col].to_numpy(float)
        ok = np.isclose(a, b, equal_nan=True, rtol=1e-9, atol=1e-12)
        assert ok.all(), f"prefix invariance violated for {col} ({(~ok).sum()} rows)"


def test_future_shuffle_invariance():
    panel = _panel()
    dates = panel["date"].sort_values().unique()
    cutoff = dates[400]
    base = compute_features(panel, _proxy(panel), FPARAMS)

    rng = np.random.default_rng(0)
    perturbed = panel.copy()
    future = perturbed["date"] > cutoff
    for col in ["open", "high", "low", "close", "vwap", "volume", "turnover"]:
        perturbed.loc[future, col] *= rng.uniform(0.5, 1.5, future.sum())
    pert = compute_features(perturbed, _proxy(perturbed), FPARAMS)

    past = base["date"] <= cutoff
    for col in NUMERIC_FEATURES:
        a = base.loc[past, col].to_numpy(float)
        b = pert.loc[past.to_numpy(), col].to_numpy(float)
        ok = np.isclose(a, b, equal_nan=True, rtol=1e-9, atol=1e-12)
        assert ok.all(), f"future-shuffle invariance violated for {col}"


def test_labels_use_only_future_window():
    panel = _panel(n=320, n_symbols=1)
    feats = panel.copy()
    out = add_labels(feats, horizons=[5])
    s = out[out["symbol"] == "S0"].reset_index(drop=True)
    t = 100
    expected = np.log(s.loc[t + 5, "close"]) - np.log(s.loc[t, "close"])
    assert np.isclose(s.loc[t, "y_5"], expected)
    # final h rows have no label
    assert s["y_5"].tail(5).isna().all()
    assert s["dir_5"].tail(5).isna().all()


def test_feature_columns_complete():
    panel = _panel()
    feats = compute_features(panel, _proxy(panel), FPARAMS)
    missing = [c for c in ALL_FEATURES if c not in feats.columns]
    assert missing == []
