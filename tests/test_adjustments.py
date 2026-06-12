"""Corporate-action detector: planted splits found, planted crashes ignored."""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.core.adjustments import adjust_universe, candidate_factors, detect_events
from tests.conftest import make_ohlcv, plant_action, plant_crash

CALM_MARKET = pd.Series(dtype=float)  # empty -> .get() returns 0.0 default


def _max_abs_ret(df: pd.DataFrame) -> float:
    r = np.log(df["Close"]).diff().abs()
    return float(r.max())


def test_one_to_five_split_detected(adjust_params):
    df = plant_action(make_ohlcv(), t=200, factor=0.2)
    events = detect_events(df, CALM_MARKET, adjust_params)
    assert len(events) == 1
    ev = events[0]
    assert ev.kind == "split_bonus_large"
    assert ev.factor == 0.2
    assert ev.date == df.loc[200, "Date"]


def test_one_to_one_bonus_detected(adjust_params):
    df = plant_action(make_ohlcv(seed=11), t=150, factor=0.5)
    events = detect_events(df, CALM_MARKET, adjust_params)
    assert len(events) == 1
    assert events[0].factor == 0.5


def test_moderate_bonus_one_to_three_detected(adjust_params):
    # 1:3 bonus -> factor 0.75, inside the moderate band
    df = plant_action(make_ohlcv(seed=13), t=250, factor=0.75)
    events = detect_events(df, CALM_MARKET, adjust_params)
    assert len(events) == 1
    assert events[0].kind == "split_bonus_moderate"
    assert events[0].factor == 0.75


def test_genuine_crash_not_flagged(adjust_params):
    # -25% crash with near-flat open and wide range: must NOT be adjusted
    df = plant_crash(make_ohlcv(seed=17), t=200, drop=-0.25)
    events = detect_events(df, CALM_MARKET, adjust_params)
    assert events == []


def test_crash_on_stress_day_not_flagged(adjust_params):
    # even a calm-looking -25% is rejected when the market itself is crashing
    df = plant_action(make_ohlcv(seed=19), t=120, factor=0.75)
    stress = pd.Series(-0.06, index=pd.DatetimeIndex([df.loc[120, "Date"]]))
    events = detect_events(df, stress, adjust_params)
    assert events == []


def test_back_adjustment_continuity(adjust_params):
    df = plant_action(make_ohlcv(seed=23), t=200, factor=0.2)
    panel = df.assign(symbol="TEST")
    adjusted, events = adjust_universe(panel, adjust_params)
    assert len(events) == 1
    # post-adjustment the series has no fake cliff
    assert _max_abs_ret(adjusted) < np.log(1.20)
    # rows before the event were scaled by the factor; after, untouched
    assert np.isclose(adjusted.loc[100, "Close"], df.loc[100, "Close"] * 0.2)
    assert np.isclose(adjusted.loc[300, "Close"], df.loc[300, "Close"])
    # volume scaled inversely before the event
    assert np.isclose(adjusted.loc[100, "Volume"], df.loc[100, "Volume"] / 0.2)


def test_rights_issue_uses_prevclose_factor(adjust_params):
    # exchange-published adjusted base: PrevClose deviates from lagged close
    df = make_ohlcv(seed=29)
    t = 180
    factor = 0.63  # arbitrary non-canonical factor
    df = plant_action(df, t=t, factor=factor)
    df.loc[t, "Prev Close"] = df.loc[t - 1, "Close"] * factor  # NSE adjusted base
    events = detect_events(df, CALM_MARKET, adjust_params)
    assert len(events) == 1
    assert events[0].kind == "exchange_prevclose"
    assert np.isclose(events[0].factor, factor, rtol=1e-6)


def test_upward_moves_never_adjusted(adjust_params):
    df = make_ohlcv(seed=31)
    t = 220
    prev = df.loc[t - 1, "Close"]
    cols = ["Open", "High", "Low", "Last", "Close", "VWAP"]
    df.loc[t, cols] = prev * 1.45  # +45% rally
    df.loc[t, "Prev Close"] = prev
    ratio = 1.45 / (df.loc[t + 1, "Prev Close"] / prev)
    df.loc[df.index > t, cols + ["Prev Close"]] *= ratio
    events = detect_events(df, CALM_MARKET, adjust_params)
    assert events == []


def test_candidate_factors_sane():
    c = candidate_factors()
    assert (c > 0).all() and (c < 0.85).all()
    for expected in (0.5, 0.2, 0.1, 2 / 3, 0.75, 1 / 14):
        assert np.isclose(c, expected, atol=1e-6).any()
