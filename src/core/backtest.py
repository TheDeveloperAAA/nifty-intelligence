"""Monthly-rebalanced portfolio backtest with transaction costs and weight drift.

Convention: weights are decided at the close of the last trading day before the
rebalance date using information through that close, trades execute at the
rebalance-day close, and costs (bps per side on traded notional, i.e. on
sum |w_new - w_drifted|) are deducted from that day's return.
"""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass
class BacktestResult:
    daily: pd.DataFrame        # date, ret, wealth, cash_weight
    rebalances: pd.DataFrame   # date, turnover, cost, n_names, risky_scale
    weights: pd.DataFrame      # rebalance date x symbol weights (long format)


def month_starts(calendar: pd.DatetimeIndex, start: pd.Timestamp, end: pd.Timestamp
                 ) -> list[pd.Timestamp]:
    cal = calendar[(calendar >= start) & (calendar <= end)]
    s = pd.Series(cal, index=cal)
    return list(s.groupby(s.dt.to_period("M")).first())


def run_backtest(
    returns_wide: pd.DataFrame,          # date x symbol simple returns
    weight_fn,                           # f(rebalance_date) -> (pd.Series weights, float risky_scale)
    rebalance_dates: list[pd.Timestamp],
    cost_bps_per_side: float,
    rf_annual: float,
) -> BacktestResult:
    cal = returns_wide.index
    rf_daily = (1 + rf_annual) ** (1 / 252) - 1
    start = rebalance_dates[0]
    days = cal[cal >= start]

    w = pd.Series(dtype=float)
    cash = 1.0
    wealth = 1.0
    rebal_set = set(rebalance_dates)
    daily_rows, rebal_rows, weight_rows = [], [], []

    for day in days:
        cost = 0.0
        if day in rebal_set:
            target, risky_scale = weight_fn(day)
            target = target[target > 0]
            drifted = w.copy()
            union = target.index.union(drifted.index)
            turnover = float(
                (target.reindex(union, fill_value=0.0)
                 - drifted.reindex(union, fill_value=0.0)).abs().sum()
            )
            cost = turnover * cost_bps_per_side / 1e4
            w = target
            cash = max(0.0, 1.0 - float(w.sum()))
            rebal_rows.append({
                "date": day, "turnover": turnover, "cost": cost,
                "n_names": int((w > 0).sum()), "risky_scale": risky_scale,
            })
            for sym, wt in w.items():
                weight_rows.append({"date": day, "symbol": sym, "weight": float(wt)})

        r = returns_wide.loc[day]
        held = w.index
        r_held = r.reindex(held).fillna(0.0)
        port_ret = float((w * r_held).sum()) + cash * rf_daily - cost
        wealth *= 1 + port_ret
        daily_rows.append({"date": day, "ret": port_ret, "wealth": wealth,
                           "cash_weight": cash})

        # drift weights with realized returns
        if len(w):
            grown = w * (1 + r_held)
            total = float(grown.sum()) + cash * (1 + rf_daily)
            w = grown / total
            cash = cash * (1 + rf_daily) / total

    return BacktestResult(
        daily=pd.DataFrame(daily_rows),
        rebalances=pd.DataFrame(rebal_rows),
        weights=pd.DataFrame(weight_rows),
    )
