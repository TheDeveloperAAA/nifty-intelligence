"""Risk metrics vs hand-computed small examples."""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.core.risk_metrics import (
    annualized_return,
    annualized_vol,
    calmar,
    cvar_historical,
    kupiec_pof,
    max_drawdown,
    risk_contributions,
    sharpe,
    sortino,
    var_historical,
)


def test_annualized_return_geometric():
    # +1% then -1% over 2 days
    r = pd.Series([0.01, -0.01])
    total = 1.01 * 0.99
    expected = total ** (252 / 2) - 1
    assert np.isclose(annualized_return(r), expected)


def test_annualized_vol():
    r = pd.Series([0.01, -0.01, 0.02, -0.02, 0.0])
    assert np.isclose(annualized_vol(r), r.std(ddof=1) * np.sqrt(252))


def test_sharpe_hand_example():
    r = pd.Series([0.01, -0.005, 0.008, -0.002])
    rf = 0.06
    cagr = float((1 + r).prod()) ** (252 / 4) - 1
    vol = r.std(ddof=1) * np.sqrt(252)
    assert np.isclose(sharpe(r, rf), (cagr - rf) / vol)


def test_sharpe_sign():
    up = pd.Series([0.002] * 252 + [0.001, -0.001])   # well above rf, tiny vol
    down = pd.Series([-0.001] * 252 + [0.001, -0.001])
    assert sharpe(up, 0.06) > 0
    assert sharpe(down, 0.06) < 0


def test_sortino_uses_downside_only():
    r = pd.Series([0.02, 0.02, -0.01, 0.02, -0.01] * 100)
    assert sortino(r, 0.0) > sharpe(r, 0.0)


def test_max_drawdown_hand_example():
    wealth = pd.Series([1.0, 1.2, 0.9, 1.1, 0.8, 1.3])
    # peak 1.2 -> trough 0.8: dd = 1 - 0.8/1.2 = 1/3
    assert np.isclose(max_drawdown(wealth), 1 / 3)


def test_var_cvar_hand_example():
    r = pd.Series(np.arange(-0.10, 0.10, 0.002))  # uniform 100 points
    v = var_historical(r, 0.95)
    assert 0.085 <= v <= 0.095
    c = cvar_historical(r, 0.95)
    assert c >= v  # expected shortfall beyond VaR


def test_calmar_consistency():
    r = pd.Series([0.001] * 504)
    assert np.isnan(calmar(r)) or calmar(r) > 0  # no drawdown -> nan


def test_kupiec_well_calibrated_var_accepted():
    rng = np.random.default_rng(0)
    r = pd.Series(rng.normal(0, 0.01, 2000))
    var_series = pd.Series(np.full(2000, var_historical(r, 0.95)))
    res = kupiec_pof(r, var_series, 0.95)
    assert res["p_value"] > 0.05  # cannot reject correct coverage


def test_risk_contributions_sum_to_one():
    cov = np.array([[0.04, 0.01], [0.01, 0.09]])
    w = np.array([0.6, 0.4])
    rc = risk_contributions(w, cov)
    assert np.isclose(rc.sum(), 1.0)
    assert (rc > 0).all()
