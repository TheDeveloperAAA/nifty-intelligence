"""Stage 4 — volatility: EWMA production forecasts + GARCH(1,1) showcase.

EWMA (lambda=0.94) is the production model: zero-fit, robust across 49 names,
feeds conformal scaling, risk metrics and the aggressive portfolio. GARCH(1,1)
with Student-t errors is fitted on the train period for a handful of liquid
names as an evidence-backed justification of the simple choice (QLIKE + MZ R2
on the locked test window).
"""
from __future__ import annotations

import time
import warnings

import numpy as np
import pandas as pd

from src.core.vol_models import ewma_vol_panel, mincer_zarnowitz_r2, qlike
from src.io_utils import StageCache, read_parquet, write_json, write_parquet


def run(cfg, force: bool = False) -> bool:
    art = cfg.path_for("artifacts")
    inputs = [art / "prices_adjusted.parquet", art / "market_proxy.parquet"]
    outputs = [art / "vol_forecasts.parquet", art / "garch_params.json"]
    cache = StageCache(art, "volatility")
    chash = cfg.section_hash("volatility", "run")
    if not force and cache.is_fresh(chash, inputs, outputs):
        return True
    t0 = time.time()

    px = read_parquet(inputs[0])[["date", "symbol", "ret"]]
    px["date"] = pd.to_datetime(px["date"])
    px = px.sort_values(["symbol", "date"]).reset_index(drop=True)
    lam = cfg["volatility"]["ewma_lambda"]
    init = cfg["volatility"]["ewma_init_days"]
    px["sigma_ewma"] = ewma_vol_panel(px, lam, init).astype(np.float32)
    px["sigma_ann"] = (px["sigma_ewma"] * np.sqrt(cfg["risk"]["trading_days"])).astype(np.float32)
    write_parquet(px[["date", "symbol", "sigma_ewma", "sigma_ann"]], outputs[0])

    # ---- GARCH(1,1)-t showcase: params from train only, filtered through test
    test_start = pd.Timestamp(cfg["cv"]["test"]["val_start"])
    train_end = pd.Timestamp(cfg["cv"]["test"]["train_end"])
    showcase = list(cfg["volatility"]["garch_showcase"])
    results = {}
    mkt = read_parquet(inputs[1])[["date", "mkt_ret"]]
    mkt["date"] = pd.to_datetime(mkt["date"])
    series_map = {s: px[px["symbol"] == s].set_index("date")["ret"] for s in showcase}
    series_map["_MARKET_"] = np.log1p(mkt.set_index("date")["mkt_ret"])

    for name, ret in series_map.items():
        ret = ret.dropna()
        train = ret[ret.index <= train_end]
        test = ret[ret.index >= test_start]
        if len(train) < 500 or len(test) < 60:
            continue
        entry = {"n_train": int(len(train)), "n_test": int(len(test))}
        try:
            from arch import arch_model

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                am = arch_model(ret * 100, vol="GARCH", p=1, q=1, dist="t", mean="Constant")
                res = am.fit(disp="off", last_obs=test_start)
                fc = res.forecast(horizon=1, start=test_start, reindex=True)
                # row t holds the forecast MADE at t (for t+1): shift so the
                # forecast aligned with realized r_t^2 was made at t-1
                garch_var = fc.variance.iloc[:, 0].shift(1).reindex(test.index) / 100**2
            entry["garch_converged"] = bool(res.convergence_flag == 0)
            entry["params"] = {k: float(v) for k, v in res.params.items()}
        except Exception as exc:  # noqa: BLE001 — record and fall back
            garch_var = None
            entry["garch_converged"] = False
            entry["error"] = str(exc)[:200]

        ewma_sigma = (
            px[px["symbol"] == name].set_index("date")["sigma_ewma"]
            if name != "_MARKET_"
            else np.sqrt(
                series_map["_MARKET_"].pow(2).ewm(alpha=1 - lam, adjust=False,
                                                  min_periods=init).mean()
            )
        )
        ewma_var = (ewma_sigma.shift(1) ** 2).reindex(test.index)  # forecast for t uses t-1 info
        realized_sq = (test**2).to_numpy()
        entry["ewma"] = {
            "qlike": qlike(realized_sq, ewma_var.to_numpy()),
            "mz_r2": mincer_zarnowitz_r2(realized_sq, ewma_var.to_numpy()),
        }
        if garch_var is not None:
            entry["garch"] = {
                "qlike": qlike(realized_sq, garch_var.to_numpy()),
                "mz_r2": mincer_zarnowitz_r2(realized_sq, garch_var.to_numpy()),
            }
        results[name] = entry

    write_json({"lambda": lam, "showcase": results}, outputs[1])
    cache.record(chash, inputs, outputs, time.time() - t0)
    return False
