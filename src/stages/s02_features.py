"""Stage 2 — features: 42 causal features + 3-horizon labels.

Outputs
-------
artifacts/heavy/features_full.parquet   model matrix (features + labels), float32
artifacts/display_indicators.parquet    slim per-stock indicator set for the app
artifacts/regimes.parquet               daily market regime states + thresholds
"""
from __future__ import annotations

import time

import numpy as np
import pandas as pd

from src.core.indicators import ALL_FEATURES, compute_features
from src.core.labeling import add_labels
from src.core.regimes import regime_states
from src.io_utils import StageCache, read_parquet, write_parquet


def run(cfg, force: bool = False) -> bool:
    art = cfg.path_for("artifacts")
    heavy = cfg.path_for("heavy")
    inputs = [art / "prices_adjusted.parquet", art / "market_proxy.parquet"]
    outputs = [
        heavy / "features_full.parquet",
        art / "display_indicators.parquet",
        art / "regimes.parquet",
    ]
    cache = StageCache(art, "features")
    chash = cfg.section_hash("features", "labels", "run")
    if not force and cache.is_fresh(chash, inputs, outputs):
        return True
    t0 = time.time()

    panel = read_parquet(inputs[0])
    proxy = read_parquet(inputs[1])
    panel["date"] = pd.to_datetime(panel["date"])
    proxy["date"] = pd.to_datetime(proxy["date"])

    fparams = dict(cfg["features"])
    fparams["regime_vol_window"] = cfg["regimes"]["vol_window"]
    fparams["regime_min_history"] = cfg["regimes"]["min_history_days"]

    feats = compute_features(panel, proxy, fparams)
    feats = add_labels(feats, list(cfg["labels"]["horizons"]))

    label_cols = [c for c in feats.columns if c.startswith(("y_", "dir_"))]
    keep = ["date", "symbol", "close"] + ALL_FEATURES + label_cols
    out = feats[keep].copy()
    for c in out.columns:
        if out[c].dtype == np.float64:
            out[c] = out[c].astype(np.float32)
    write_parquet(out, outputs[0])

    # slim display set for the dashboard (price overlays need raw rolling stats)
    g = panel.groupby("symbol", sort=False)
    disp = panel[["date", "symbol", "close", "close_raw", "volume", "ret"]].copy()
    disp["sma10"] = g["close"].transform(lambda s: s.rolling(10).mean())
    disp["sma50"] = g["close"].transform(lambda s: s.rolling(50).mean())
    sma20 = g["close"].transform(lambda s: s.rolling(20).mean())
    sd20 = g["close"].transform(lambda s: s.rolling(20).std(ddof=1))
    disp["boll_up"] = sma20 + 2 * sd20
    disp["boll_lo"] = sma20 - 2 * sd20
    disp = disp.merge(
        feats[["date", "symbol", "rsi_14", "rv_21", "drawdown_252", "beta_252"]],
        on=["date", "symbol"],
        how="left",
    )
    disp["rv21_ann"] = disp["rv_21"] * np.sqrt(cfg["risk"]["trading_days"])
    for c in disp.columns:
        if disp[c].dtype == np.float64:
            disp[c] = disp[c].astype(np.float32)
    write_parquet(disp.drop(columns=["rv_21"]), outputs[1])

    mkt_log = np.log1p(proxy.set_index("date")["mkt_ret"])
    reg = regime_states(
        mkt_log,
        vol_window=cfg["regimes"]["vol_window"],
        min_history_days=cfg["regimes"]["min_history_days"],
    ).reset_index()
    write_parquet(reg, outputs[2])

    cache.record(chash, inputs, outputs, time.time() - t0,
                 extra={"n_rows": int(len(out)), "n_features": len(ALL_FEATURES)})
    return False
