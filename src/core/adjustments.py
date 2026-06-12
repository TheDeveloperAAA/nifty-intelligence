"""Corporate-action (split/bonus) detection and back-adjustment.

The dataset's prices are unadjusted: 92 days carry |close-to-close return| > 30%
that are splits/bonus issues, not market moves. External corporate-action data
is not allowed, so events are inferred from the price series itself:

Pass 1 — large drops (rho = Close_t/Close_{t-1} < 0.70):
  * If the dataset's own `Prev Close` deviates from the lagged Close, NSE
    published an adjusted base price (rights issues): factor = PrevClose/lagClose.
  * Otherwise snap rho to the nearest plausible split/bonus factor (log space).
    Near-ties are arbitrated by the *open anchor*: on a true ex-date the open
    sits at the adjusted level c * Close_{t-1}.
Pass 2 — moderate drops (rho in [0.70, 0.90]) hide small bonuses (1:3 -> 0.75).
  Flagged only when ALL hold: tight ratio snap, tight open anchor, calm intraday
  range, and no market-wide stress (cross-sectional median return).
Upward moves are never adjusted: the three genuine >+30% days in this dataset
(e.g. INDUSINDBK +44.7% on 2020-03-26) are crisis rallies, not reverse splits.

A factor error would corrupt only the event-day return — log returns elsewhere
are invariant to the scaling — so the algorithm only needs those ~100 days
approximately right; winsorization of training targets is defense-in-depth.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

PRICE_COLS = ["Prev Close", "Open", "High", "Low", "Last", "Close", "VWAP"]
SHARE_COLS = ["Volume", "Deliverable Volume"]


def candidate_factors() -> np.ndarray:
    """Plausible NSE split/bonus price ratios, all < 0.85."""
    splits = {0.1, 0.2, 0.4, 0.5}  # face-value changes (10->1, 10->2, 10->4, 10->5)
    bonuses = {n / (n + m) for n in range(1, 6) for m in range(1, 6)}  # m:n bonus
    products = {s * b for s in splits for b in bonuses}
    reciprocals = {1.0 / n for n in range(2, 26)}  # compound actions (e.g. ITC ~1/14)
    cands = {round(c, 6) for c in splits | bonuses | products | reciprocals if c < 0.85}
    return np.array(sorted(cands))


@dataclass
class Event:
    symbol: str
    date: pd.Timestamp
    rho: float          # observed Close_t / Close_{t-1}
    factor: float       # applied adjustment factor
    kind: str           # split_bonus_large | exchange_prevclose | split_bonus_moderate | unsnapped
    snap_gap: float     # |log(rho/factor)|
    open_gap: float     # |log(Open_t/(factor*Close_{t-1}))|


def _snap(rho: float, open_anchor_ratio: float, cands: np.ndarray, tie_tol: float = 0.02):
    """Nearest candidate in log space; near-ties resolved by the open anchor."""
    gaps = np.abs(np.log(rho / cands))
    order = np.argsort(gaps)
    best = order[0]
    if len(order) > 1 and gaps[order[1]] - gaps[best] < tie_tol and open_anchor_ratio > 0:
        contenders = [order[0], order[1]]
        open_gaps = [abs(np.log(open_anchor_ratio / cands[i])) for i in contenders]
        best = contenders[int(np.argmin(open_gaps))]
    return float(cands[best]), float(gaps[best])


def detect_events(
    sdf: pd.DataFrame,
    market_ret: pd.Series,
    params: dict,
    cands: np.ndarray | None = None,
) -> list[Event]:
    """Detect corporate actions for one symbol (sdf sorted by Date)."""
    if cands is None:
        cands = candidate_factors()
    close = sdf["Close"].to_numpy(float)
    open_ = sdf["Open"].to_numpy(float)
    high = sdf["High"].to_numpy(float)
    low = sdf["Low"].to_numpy(float)
    prevc = sdf["Prev Close"].to_numpy(float)
    dates = sdf["Date"].to_numpy()
    symbol = sdf["symbol"].iloc[0]

    lo_band, hi_band = params["moderate_band"]
    events: list[Event] = []
    for t in range(1, len(sdf)):
        lag = close[t - 1]
        if not np.isfinite(lag) or lag <= 0 or close[t] <= 0:
            continue
        rho = close[t] / lag
        if rho >= hi_band:
            continue  # upward moves and small drops never adjusted

        open_anchor = open_[t] / lag if open_[t] > 0 else -1.0

        if rho < params["large_drop_ratio"]:
            # exchange-published base (rights issues): PrevClose != lagged Close
            if np.isfinite(prevc[t]) and abs(prevc[t] / lag - 1) > params["prevclose_mismatch_tol"]:
                f = prevc[t] / lag
                events.append(Event(symbol, dates[t], rho, f, "exchange_prevclose",
                                    abs(np.log(rho / f)), -1.0))
                continue
            f, gap = _snap(rho, open_anchor, cands)
            if gap <= params["large_snap_tol"]:
                ogap = abs(np.log(open_anchor / f)) if open_anchor > 0 else np.nan
                events.append(Event(symbol, dates[t], rho, f, "split_bonus_large", gap, ogap))
            else:
                # no plausible factor: treat the day itself as the factor so the
                # fake return is neutralized, and flag for the event register
                events.append(Event(symbol, dates[t], rho, rho, "unsnapped", gap, -1.0))
        else:
            # moderate band: require all four ex-date signatures
            f, gap = _snap(rho, open_anchor, cands)
            if gap > params["moderate_snap_tol"]:
                continue
            if open_anchor <= 0:
                continue
            ogap = abs(np.log(open_anchor / f))
            if ogap > params["moderate_open_tol"]:
                continue
            day_range = (high[t] - low[t]) / open_[t] if open_[t] > 0 else np.inf
            if day_range > params["moderate_range_max"]:
                continue
            mret = market_ret.get(pd.Timestamp(dates[t]), 0.0)
            if abs(mret) >= params["moderate_market_max"]:
                continue
            events.append(Event(symbol, dates[t], rho, f, "split_bonus_moderate", gap, ogap))
    return events


def apply_adjustments(sdf: pd.DataFrame, events: list[Event]) -> pd.DataFrame:
    """Back-adjust prices (multiply) and share counts (divide) before each event.

    Adjusted P_t = P_t * prod(f_e for events e with date_e > t).
    """
    sdf = sdf.copy()
    factor = np.ones(len(sdf))
    dates = sdf["Date"].to_numpy()
    for ev in events:
        mask = dates < np.datetime64(ev.date)
        factor[mask] *= ev.factor
    sdf["adj_factor"] = factor
    sdf["close_raw"] = sdf["Close"]
    for col in PRICE_COLS:
        if col in sdf.columns:
            sdf[col] = sdf[col] * factor
    for col in SHARE_COLS:
        if col in sdf.columns:
            sdf[col] = sdf[col] / factor
    return sdf


def adjust_universe(panel: pd.DataFrame, params: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Detect and apply adjustments for every symbol in the long panel.

    Returns (adjusted panel, events table). The market-stress filter uses the
    cross-sectional MEDIAN raw return per date — robust to other symbols'
    own split artifacts, unlike the mean.
    """
    panel = panel.sort_values(["symbol", "Date"]).reset_index(drop=True)
    raw_ret = panel.groupby("symbol")["Close"].pct_change()
    market_ret = raw_ret.groupby(panel["Date"]).median()

    cands = candidate_factors()
    adjusted_frames, all_events = [], []
    for _, sdf in panel.groupby("symbol", sort=True):
        sdf = sdf.reset_index(drop=True)
        events = detect_events(sdf, market_ret, params, cands)
        adjusted_frames.append(apply_adjustments(sdf, events))
        all_events.extend(events)

    out = pd.concat(adjusted_frames, ignore_index=True)
    events_df = pd.DataFrame(
        [
            {
                "symbol": e.symbol,
                "date": pd.Timestamp(e.date),
                "observed_ratio": round(e.rho, 6),
                "factor": round(e.factor, 6),
                "kind": e.kind,
                "snap_gap": round(e.snap_gap, 4),
                "open_gap": round(e.open_gap, 4) if np.isfinite(e.open_gap) else None,
            }
            for e in all_events
        ]
    )
    if not events_df.empty:
        events_df = events_df.sort_values(["symbol", "date"]).reset_index(drop=True)
    return out, events_df
