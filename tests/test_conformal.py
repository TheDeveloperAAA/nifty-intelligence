"""Conformal intervals: ~90% coverage on synthetic Gaussian noise."""
from __future__ import annotations

import numpy as np

from src.core.conformal import conformal_quantile, coverage, interval, nonconformity


def test_gaussian_coverage_close_to_nominal():
    rng = np.random.default_rng(0)
    n_cal, n_test, h = 5000, 5000, 5
    sigma = rng.uniform(0.01, 0.03, n_cal + n_test)
    y = rng.normal(0, sigma * np.sqrt(h))
    yhat = np.zeros_like(y)

    s_cal = nonconformity(y[:n_cal], yhat[:n_cal], sigma[:n_cal], h)
    q = conformal_quantile(s_cal, alpha=0.10)
    lo, hi = interval(yhat[n_cal:], sigma[n_cal:], h, q)
    cov = coverage(y[n_cal:], lo, hi)
    assert 0.88 <= cov <= 0.92


def test_quantile_finite_sample_correction():
    s = np.arange(1, 101, dtype=float)  # scores 1..100
    q = conformal_quantile(s, alpha=0.10)
    # ceil(0.9 * 101) = 91 -> 91st smallest
    assert q == 91.0


def test_interval_symmetry_and_scaling():
    yhat = np.array([0.01])
    sigma = np.array([0.02])
    lo, hi = interval(yhat, sigma, h=4, q=2.0)
    assert np.isclose(hi - yhat, yhat - lo)
    assert np.isclose(hi[0] - yhat[0], 2.0 * 0.02 * 2)
