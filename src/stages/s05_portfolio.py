"""Stage 5 — portfolio: three investor profiles + costed walk-forward backtest.

Weights at each monthly rebalance use information through the prior close only:
Ledoit-Wolf covariance (504d), shrunk means (756d), OOS model scores from the
walk-forward fold covering that date, and the causal regime state. Benchmark is
a monthly-rebalanced equal-weight portfolio facing identical costs.
"""
from __future__ import annotations

import time

import numpy as np
import pandas as pd

from src.core import optimizer as opt
from src.core.backtest import month_starts, run_backtest
from src.core.risk_metrics import risk_contributions, summary
from src.core.vol_models import ewma_vol_panel
from src.io_utils import StageCache, read_parquet, write_json, write_parquet

PROFILES = ("conservative", "balanced", "aggressive")


def run(cfg, force: bool = False) -> bool:  # noqa: PLR0915 — orchestration shell
    art = cfg.path_for("artifacts")
    inputs = [
        art / "prices_adjusted.parquet",
        art / "predictions.parquet",
        art / "regimes.parquet",
    ]
    outputs = [
        art / "backtest_curves.parquet",
        art / "rebalance_log.parquet",
        art / "weights_history.parquet",
        art / "portfolio_summary.json",
        art / "frontier.json",
    ]
    cache = StageCache(art, "portfolio")
    chash = cfg.section_hash("portfolio", "risk", "run")
    if not force and cache.is_fresh(chash, inputs, outputs):
        return True
    t0 = time.time()
    pcfg = cfg["portfolio"]
    rf = cfg["risk"]["rf_annual"]

    px = read_parquet(inputs[0])
    px["date"] = pd.to_datetime(px["date"])
    sectors = px.groupby("symbol")["sector"].first()
    rets = px.pivot(index="date", columns="symbol", values="simple_ret").sort_index()
    closes = px.pivot(index="date", columns="symbol", values="close").sort_index()

    px_sig = px[["date", "symbol", "ret"]].sort_values(["symbol", "date"]).reset_index(drop=True)
    px_sig["sigma"] = ewma_vol_panel(
        px_sig, cfg["volatility"]["ewma_lambda"], cfg["volatility"]["ewma_init_days"]
    )
    sigma_wide = px_sig.pivot(index="date", columns="symbol", values="sigma").sort_index()

    preds = read_parquet(inputs[1])
    preds["date"] = pd.to_datetime(preds["date"])
    h_max = max(cfg["labels"]["horizons"])
    p21 = preds[preds["horizon"] == h_max].pivot(index="date", columns="symbol", values="y_pred")
    p21 = p21.reindex(rets.index).ffill(limit=10)

    regimes = read_parquet(inputs[2]).set_index("date")["regime"]
    regimes.index = pd.to_datetime(regimes.index)

    calendar = rets.index
    start = pd.Timestamp(pcfg["backtest_start"])
    end = calendar.max()
    rebal_dates = month_starts(calendar, start, end)
    overlay = pcfg["regime_overlay"]

    def eligible(t_prev: pd.Timestamp) -> pd.Index:
        hist = rets.loc[:t_prev]
        enough = hist.notna().sum() >= pcfg["min_history_days"]
        recent = hist.tail(252).notna().mean() >= pcfg["min_traded_frac"]
        return rets.columns[enough & recent]

    def prev_day(t: pd.Timestamp) -> pd.Timestamp:
        return calendar[calendar < t][-1]

    def make_weight_fn(profile: str):
        prof = pcfg["profiles"][profile]

        def weight_fn(t: pd.Timestamp):
            tp = prev_day(t)
            uni = eligible(tp)
            window = rets.loc[:tp, uni].tail(pcfg["lookback_cov_days"])
            window = window.dropna(axis=1, how="any")
            uni = window.columns
            sec = sectors.reindex(uni)
            if profile == "conservative":
                cov = opt.lw_covariance(window)
                w = opt.min_vol_weights(cov, prof["max_weight"], sec, prof["sector_cap"])
            elif profile == "balanced":
                cov = opt.lw_covariance(window)
                mu_win = rets.loc[:tp, uni].tail(pcfg["lookback_mu_days"])
                mu = opt.shrunk_mu(mu_win)
                w = opt.max_sharpe_weights(mu, cov, rf, prof["max_weight"], sec, prof["sector_cap"])
            else:  # aggressive: momentum + model score, inverse-vol weighted
                c = closes.loc[:tp, uni]
                mom = np.log(c.shift(21).iloc[-1] / c.shift(252).iloc[-1])
                score = (mom - mom.mean()) / mom.std(ddof=1)
                model_score = p21.loc[tp, uni] if tp in p21.index else pd.Series(np.nan, index=uni)
                if model_score.notna().sum() >= 10:
                    mz = (model_score - model_score.mean()) / model_score.std(ddof=1)
                    score = 0.5 * score + 0.5 * mz.fillna(0.0)
                sig = sigma_wide.loc[tp, uni]
                w = opt.score_tilt_weights(
                    score, sig, prof["top_n"], prof["max_weight"], sec, prof["sector_cap"]
                )
            scale = 1.0
            if profile in overlay and regimes.reindex([tp]).fillna(1).iloc[0] == 2:
                scale = float(overlay[profile])
            return w * scale, scale

        return weight_fn

    def ew_weight_fn(t: pd.Timestamp):
        uni = eligible(prev_day(t))
        return pd.Series(1.0 / len(uni), index=uni), 1.0

    def hrp_weight_fn(t: pd.Timestamp):
        tp = prev_day(t)
        uni = eligible(tp)
        window = rets.loc[:tp, uni].tail(pcfg["lookback_cov_days"]).dropna(axis=1, how="any")
        return opt.hrp_weights(window), 1.0

    curves, rebals, weights = [], [], []
    strategies = {p: make_weight_fn(p) for p in PROFILES}
    strategies["benchmark_ew"] = ew_weight_fn
    strategies["hrp"] = hrp_weight_fn
    for name, fn in strategies.items():
        result = run_backtest(rets, fn, rebal_dates, pcfg["cost_bps_per_side"], rf)
        curves.append(result.daily.assign(profile=name))
        rebals.append(result.rebalances.assign(profile=name))
        weights.append(result.weights.assign(profile=name))

    curves_df = pd.concat(curves, ignore_index=True)
    rebals_df = pd.concat(rebals, ignore_index=True)
    weights_df = pd.concat(weights, ignore_index=True)

    # ---- summary block ------------------------------------------------------
    mkt_simple = read_parquet(art / "market_proxy.parquet")[["date", "mkt_ret"]]
    mkt_simple["date"] = pd.to_datetime(mkt_simple["date"])
    mkt_ret = mkt_simple.set_index("date")["mkt_ret"]

    summary_block: dict = {"profiles": {}, "assumptions": {
        "rf_annual": rf, "cost_bps_per_side": pcfg["cost_bps_per_side"],
        "rebalance": pcfg["rebalance"], "backtest_start": str(start.date()),
        "regime_overlay": overlay,
    }}
    var_levels = tuple(cfg["risk"]["var_levels"])
    for name in strategies:
        d = curves_df[curves_df["profile"] == name].set_index("date")
        s = summary(d["ret"], rf, mkt_ret.reindex(d.index), var_levels)
        rb = rebals_df[rebals_df["profile"] == name]
        years = max(len(d) / 252, 1e-9)
        s["annual_turnover"] = float(rb["turnover"].sum() / years)
        s["total_costs"] = float(rb["cost"].sum())
        for rf_alt in cfg["risk"]["rf_sensitivity"]:
            from src.core.risk_metrics import sharpe as _sharpe

            s[f"sharpe_rf{int(rf_alt*100)}"] = _sharpe(d["ret"], rf_alt)
        summary_block["profiles"][name] = s

    # final-date weights detail with risk contributions
    final_date = rebal_dates[-1]
    final_detail = {}
    for name in PROFILES:
        wdf = weights_df[(weights_df["profile"] == name) & (weights_df["date"] == final_date)]
        w = wdf.set_index("symbol")["weight"]
        uni = w.index
        window = rets.loc[:prev_day(final_date), uni].tail(pcfg["lookback_cov_days"]).dropna(
            axis=1, how="any"
        )
        cov = opt.lw_covariance(window)
        rc = risk_contributions(w.reindex(cov.index).fillna(0).to_numpy(), cov.to_numpy())
        final_detail[name] = {
            "date": str(final_date.date()),
            "weights": {k: float(v) for k, v in w.sort_values(ascending=False).items()},
            "sector_weights": {
                k: float(v) for k, v in
                w.groupby(sectors.reindex(w.index)).sum().sort_values(ascending=False).items()
            },
            "risk_contribution": {
                k: float(v) for k, v in
                pd.Series(rc, index=cov.index).sort_values(ascending=False).items()
            },
            "portfolio_vol_forecast": float(np.sqrt(
                w.reindex(cov.index).fillna(0) @ cov @ w.reindex(cov.index).fillna(0)
            )),
        }
    summary_block["final_weights"] = final_detail

    # ---- efficient frontier at the final rebalance --------------------------
    tp = prev_day(final_date)
    uni = eligible(tp)
    window = rets.loc[:tp, uni].tail(pcfg["lookback_cov_days"]).dropna(axis=1, how="any")
    cov = opt.lw_covariance(window)
    mu = opt.shrunk_mu(rets.loc[:tp, window.columns].tail(pcfg["lookback_mu_days"]))
    frontier = opt.efficient_frontier(mu, cov, pcfg["frontier_points"], 0.15)
    points = {}
    for name in list(PROFILES) + ["benchmark_ew", "hrp"]:
        wdf = weights_df[(weights_df["profile"] == name) & (weights_df["date"] == final_date)]
        w = wdf.set_index("symbol")["weight"].reindex(cov.index).fillna(0.0)
        if w.sum() > 0:
            w = w / w.sum()
        points[name] = {
            "vol": float(np.sqrt(w @ cov @ w)),
            "ret": float(mu.reindex(cov.index).fillna(0) @ w),
        }
    stocks_scatter = [
        {"symbol": s, "vol": float(np.sqrt(cov.loc[s, s])), "ret": float(mu[s])}
        for s in cov.index
    ]
    write_json(
        {"as_of": str(final_date.date()), "frontier": frontier, "portfolios": points,
         "stocks": stocks_scatter, "note": "expected returns are shrunk trailing means; "
         "illustrative positioning, not a forecast"},
        outputs[4],
    )

    write_parquet(curves_df, outputs[0])
    write_parquet(rebals_df, outputs[1])
    write_parquet(weights_df, outputs[2])
    write_json(summary_block, outputs[3])
    cache.record(chash, inputs, outputs, time.time() - t0,
                 extra={"n_rebalances": len(rebal_dates), "strategies": list(strategies)})
    return False
