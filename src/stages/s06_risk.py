"""Stage 6 — risk: per-stock and per-portfolio metric suite, VaR backtest,
rolling diagnostics, sector-ordered correlation matrix.
"""
from __future__ import annotations

import time

import numpy as np
import pandas as pd

from src.core.risk_metrics import (
    drawdown_series,
    kupiec_pof,
    summary,
    var_historical,
)
from src.io_utils import StageCache, read_parquet, write_json, write_parquet


def run(cfg, force: bool = False) -> bool:
    art = cfg.path_for("artifacts")
    inputs = [
        art / "prices_adjusted.parquet",
        art / "market_proxy.parquet",
        art / "backtest_curves.parquet",
    ]
    outputs = [
        art / "risk_stock.parquet",
        art / "risk_metrics.json",
        art / "rolling_risk.parquet",
        art / "drawdowns.parquet",
        art / "correlation.parquet",
        art / "var_backtest.json",
    ]
    cache = StageCache(art, "risk")
    chash = cfg.section_hash("risk", "run")
    if not force and cache.is_fresh(chash, inputs, outputs):
        return True
    t0 = time.time()
    rf = cfg["risk"]["rf_annual"]
    var_levels = tuple(cfg["risk"]["var_levels"])
    window = cfg["risk"]["rolling_window"]

    px = read_parquet(inputs[0])
    px["date"] = pd.to_datetime(px["date"])
    proxy = read_parquet(inputs[1])
    proxy["date"] = pd.to_datetime(proxy["date"])
    mkt_ret = proxy.set_index("date")["mkt_ret"]

    # ---- per-stock metrics (full sample + last 3y) --------------------------
    rows = []
    for sym, sdf in px.groupby("symbol"):
        s = sdf.set_index("date")["simple_ret"]
        sector = sdf["sector"].iloc[0]
        full = summary(s, rf, mkt_ret.reindex(s.index), var_levels)
        recent = summary(s.tail(756), rf, mkt_ret.reindex(s.tail(756).index), var_levels)
        rows.append({
            "symbol": sym, "sector": sector,
            **{f"{k}": v for k, v in full.items()},
            **{f"recent3y_{k}": v for k, v in recent.items()},
        })
    write_parquet(pd.DataFrame(rows), outputs[0])

    # ---- portfolio metrics come from stage 5; recompute drawdowns/rolling ---
    curves = read_parquet(inputs[2])
    curves["date"] = pd.to_datetime(curves["date"])

    roll_rows, dd_rows = [], []
    mkt_wealth = (1 + mkt_ret.fillna(0)).cumprod()
    entities = {"market_ew": pd.DataFrame({"date": mkt_wealth.index, "ret": mkt_ret.values,
                                           "wealth": mkt_wealth.values})}
    for prof, d in curves.groupby("profile"):
        entities[prof] = d[["date", "ret", "wealth"]]
    for name, d in entities.items():
        d = d.set_index("date").sort_index()
        dd = drawdown_series(d["wealth"])
        dd_rows.append(pd.DataFrame({"date": d.index, "entity": name,
                                     "wealth": d["wealth"].values, "drawdown": dd.values}))
        r = d["ret"]
        roll = pd.DataFrame({
            "date": d.index,
            "entity": name,
            "roll_vol": r.rolling(window).std(ddof=1) * np.sqrt(252),
            "roll_sharpe": (
                (r.rolling(window).mean() * 252 - rf)
                / (r.rolling(window).std(ddof=1) * np.sqrt(252))
            ),
        })
        if name != "market_ew":
            aligned = pd.concat([r, mkt_ret.reindex(r.index)], axis=1)
            cov = aligned.iloc[:, 0].rolling(window).cov(aligned.iloc[:, 1])
            var = aligned.iloc[:, 1].rolling(window).var()
            roll["roll_beta"] = cov / var
        roll_rows.append(roll)
    write_parquet(pd.concat(roll_rows, ignore_index=True), outputs[2])
    write_parquet(pd.concat(dd_rows, ignore_index=True), outputs[3])

    # ---- correlation matrix (last 756d), sector-ordered ----------------------
    rets_wide = px.pivot(index="date", columns="symbol", values="simple_ret").tail(756)
    sectors = px.groupby("symbol")["sector"].first()
    order = sectors.sort_values().index.tolist()
    corr = rets_wide[order].corr()
    corr_long = corr.reset_index().melt(id_vars="symbol", var_name="symbol2", value_name="corr")
    corr_long["sector1"] = corr_long["symbol"].map(sectors)
    corr_long["sector2"] = corr_long["symbol2"].map(sectors)
    write_parquet(corr_long, outputs[4])

    # ---- VaR backtest (trailing-252d historical VaR, next-day exceptions) ----
    var_results = {}
    for prof, d in curves.groupby("profile"):
        r = d.set_index("date")["ret"].sort_index()
        var95 = r.rolling(252).apply(lambda x: var_historical(pd.Series(x), 0.95), raw=False)
        var_results[prof] = kupiec_pof(r, var95.shift(1), 0.95)
    write_json(var_results, outputs[5])

    # ---- headline risk JSON --------------------------------------------------
    mkt_summary = summary(mkt_ret, rf, None, var_levels)
    stock_df = pd.DataFrame(rows)
    risk_json = {
        "market_ew": mkt_summary,
        "stocks_top_sharpe": stock_df.nlargest(10, "sharpe")[
            ["symbol", "sector", "cagr", "ann_vol", "sharpe", "max_drawdown"]
        ].to_dict("records"),
        "stocks_lowest_vol": stock_df.nsmallest(10, "ann_vol")[
            ["symbol", "sector", "cagr", "ann_vol", "sharpe", "max_drawdown"]
        ].to_dict("records"),
        "notes": "rf=6% documented assumption; per-stock metrics on full history "
                 "(survivorship caveat applies, see model card)",
    }
    write_json(risk_json, outputs[1])
    cache.record(chash, inputs, outputs, time.time() - t0)
    return False
