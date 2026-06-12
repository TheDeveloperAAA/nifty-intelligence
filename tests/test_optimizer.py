"""Optimizer constraints and HRP sanity on toy problems."""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.core.optimizer import (
    hrp_weights,
    lw_covariance,
    max_sharpe_weights,
    min_vol_weights,
    score_tilt_weights,
    shrunk_mu,
)


def _toy_returns(n=600, seed=0):
    rng = np.random.default_rng(seed)
    base = rng.normal(0.0005, 0.01, (n, 1))
    block_a = base + rng.normal(0.0002, 0.004, (n, 3))   # correlated trio
    block_b = rng.normal(0.0004, 0.02, (n, 2))           # volatile pair
    data = np.hstack([block_a, block_b])
    cols = ["A1", "A2", "A3", "B1", "B2"]
    return pd.DataFrame(data, columns=cols)


SECTORS = pd.Series({"A1": "X", "A2": "X", "A3": "X", "B1": "Y", "B2": "Y"})


def test_min_vol_respects_constraints():
    rets = _toy_returns()
    cov = lw_covariance(rets)
    w = min_vol_weights(cov, max_weight=0.30, sectors=SECTORS, sector_cap=0.70)
    assert np.isclose(w.sum(), 1.0)
    assert (w >= -1e-9).all() and (w <= 0.30 + 1e-6).all()
    assert w.groupby(SECTORS).sum().max() <= 0.70 + 1e-6


def test_max_sharpe_respects_constraints():
    rets = _toy_returns()
    cov = lw_covariance(rets)
    mu = shrunk_mu(rets)
    w = max_sharpe_weights(mu, cov, rf=0.06, max_weight=0.40, sectors=SECTORS, sector_cap=0.80)
    assert np.isclose(w.sum(), 1.0)
    assert (w <= 0.40 + 1e-6).all()


def test_min_vol_prefers_low_vol_assets():
    rets = _toy_returns()
    cov = lw_covariance(rets)
    w = min_vol_weights(cov, max_weight=1.0)
    # the volatile B-block should get less weight than the calm A-block
    assert w[["A1", "A2", "A3"]].sum() > w[["B1", "B2"]].sum()


def test_score_tilt_top_n_and_caps():
    scores = pd.Series({"A1": 2.0, "A2": 1.5, "A3": 1.0, "B1": 0.5, "B2": -1.0})
    sigma = pd.Series({"A1": 0.01, "A2": 0.02, "A3": 0.01, "B1": 0.03, "B2": 0.02})
    w = score_tilt_weights(scores, sigma, top_n=3, max_weight=0.50,
                           sectors=SECTORS, sector_cap=0.90)
    assert np.isclose(w.sum(), 1.0)
    assert (w[["B1", "B2"]] == 0).all()  # outside top-3
    assert (w <= 0.50 + 1e-9).all()


def test_hrp_weights_sane():
    rets = _toy_returns()
    w = hrp_weights(rets)
    assert np.isclose(w.sum(), 1.0)
    assert (w >= 0).all()
    # inverse-variance flavor: B2 (highest vol) must not dominate
    assert w["B2"] < w[["A1", "A2", "A3"]].max()
