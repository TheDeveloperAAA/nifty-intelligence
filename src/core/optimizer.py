"""Portfolio construction: shrinkage MVO (min-vol / max-Sharpe), score-tilt,
HRP comparison, efficient frontier. Long-only with per-name and sector caps.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import leaves_list, linkage
from scipy.optimize import minimize
from sklearn.covariance import LedoitWolf

TRADING_DAYS = 252


def lw_covariance(returns: pd.DataFrame) -> pd.DataFrame:
    """Ledoit-Wolf shrunk covariance of daily returns, annualized."""
    clean = returns.dropna(axis=0, how="any")
    lw = LedoitWolf().fit(clean.to_numpy())
    cov = lw.covariance_ * TRADING_DAYS
    return pd.DataFrame(cov, index=returns.columns, columns=returns.columns)


def shrunk_mu(returns: pd.DataFrame, shrink: float = 0.5) -> pd.Series:
    """James-Stein-style: blend each stock's mean toward the cross-sectional mean."""
    mu = returns.mean() * TRADING_DAYS
    return shrink * mu.mean() + (1 - shrink) * mu


def _constraints(n: int, sectors: pd.Series | None, sector_cap: float | None):
    cons = [{"type": "eq", "fun": lambda w: w.sum() - 1.0}]
    if sectors is not None and sector_cap is not None:
        for sec in sectors.unique():
            idx = np.where((sectors == sec).to_numpy())[0]
            cons.append(
                {"type": "ineq", "fun": lambda w, idx=idx: sector_cap - w[idx].sum()}
            )
    return cons


def _solve(objective, n: int, max_weight: float, sectors, sector_cap, x0=None) -> np.ndarray:
    bounds = [(0.0, max_weight)] * n
    if x0 is None:
        x0 = np.full(n, 1.0 / n)
        x0 = np.minimum(x0, max_weight)
        x0 = x0 / x0.sum()
    res = minimize(
        objective, x0, method="SLSQP", bounds=bounds,
        constraints=_constraints(n, sectors, sector_cap),
        options={"maxiter": 500, "ftol": 1e-10},
    )
    w = np.clip(res.x, 0, max_weight)
    return w / w.sum()


def _tidy(weights: np.ndarray, names: pd.Index, floor: float = 0.01) -> pd.Series:
    """Zero dust positions (<1%) and renormalize."""
    w = pd.Series(weights, index=names)
    w[w < floor] = 0.0
    if w.sum() == 0:
        return w
    return w / w.sum()


def min_vol_weights(cov: pd.DataFrame, max_weight: float, sectors: pd.Series | None = None,
                    sector_cap: float | None = None) -> pd.Series:
    C = cov.to_numpy()
    w = _solve(lambda w: w @ C @ w, len(cov), max_weight, sectors, sector_cap)
    return _tidy(w, cov.index)


def max_sharpe_weights(mu: pd.Series, cov: pd.DataFrame, rf: float, max_weight: float,
                       sectors: pd.Series | None = None,
                       sector_cap: float | None = None) -> pd.Series:
    m = mu.to_numpy()
    C = cov.to_numpy()

    def neg_sharpe(w):
        vol = np.sqrt(max(w @ C @ w, 1e-12))
        return -(w @ m - rf) / vol

    w = _solve(neg_sharpe, len(cov), max_weight, sectors, sector_cap)
    return _tidy(w, cov.index)


def score_tilt_weights(scores: pd.Series, sigma: pd.Series, top_n: int, max_weight: float,
                       sectors: pd.Series | None = None,
                       sector_cap: float | None = None) -> pd.Series:
    """Top-N by score, inverse-vol weighted, then capped iteratively."""
    top = scores.dropna().nlargest(top_n).index
    inv = (1.0 / sigma.reindex(top).clip(lower=1e-6)).replace([np.inf, -np.inf], np.nan).dropna()
    w = inv / inv.sum()
    # iterative cap-and-redistribute (per-name, then sector)
    for _ in range(20):
        over = w > max_weight
        if over.any():
            excess = (w[over] - max_weight).sum()
            w[over] = max_weight
            under = ~over
            if w[under].sum() > 0:
                w[under] += excess * w[under] / w[under].sum()
            continue
        if sectors is not None and sector_cap is not None:
            sec_w = w.groupby(sectors.reindex(w.index)).sum()
            bad = sec_w[sec_w > sector_cap]
            if len(bad):
                for sec in bad.index:
                    names = w.index[sectors.reindex(w.index) == sec]
                    scale = sector_cap / w[names].sum()
                    freed = w[names].sum() - sector_cap
                    w[names] *= scale
                    others = w.index.difference(names)
                    if w[others].sum() > 0:
                        w[others] += freed * w[others] / w[others].sum()
                continue
        break
    out = pd.Series(0.0, index=scores.index)
    out[w.index] = w / w.sum()
    return out


def hrp_weights(returns: pd.DataFrame) -> pd.Series:
    """Hierarchical Risk Parity (Lopez de Prado): single-linkage cluster tree,
    quasi-diagonalization, recursive inverse-variance bisection."""
    clean = returns.dropna(axis=0, how="any")
    corr = clean.corr()
    cov = clean.cov() * TRADING_DAYS
    dist = np.sqrt(np.clip((1 - corr) / 2, 0, 1))
    link = linkage(dist.to_numpy()[np.triu_indices(len(dist), 1)], method="single")
    order = leaves_list(link)
    names = corr.index[order]

    w = pd.Series(1.0, index=names)
    clusters = [list(names)]
    while clusters:
        nxt = []
        for cl in clusters:
            if len(cl) <= 1:
                continue
            half = len(cl) // 2
            left, right = cl[:half], cl[half:]

            def cluster_var(items):
                sub = cov.loc[items, items]
                ivp = 1.0 / np.diag(sub)
                ivp /= ivp.sum()
                return float(ivp @ sub.to_numpy() @ ivp)

            vl, vr = cluster_var(left), cluster_var(right)
            alpha = 1 - vl / (vl + vr)
            w[left] *= alpha
            w[right] *= 1 - alpha
            nxt += [left, right]
        clusters = nxt
    return w.reindex(returns.columns).fillna(0.0)


def efficient_frontier(mu: pd.Series, cov: pd.DataFrame, n_points: int, max_weight: float
                       ) -> list[dict]:
    """Min-variance frontier over a grid of target returns (long-only, capped)."""
    C = cov.to_numpy()
    m = mu.to_numpy()
    w_minvol = min_vol_weights(cov, max_weight)
    r_lo = float(mu @ w_minvol)
    r_hi = float(mu.nlargest(3).mean())  # cap target near the top names' mean
    points = []
    for target in np.linspace(r_lo, r_hi, n_points):
        cons_extra = {"type": "ineq", "fun": lambda w, t=target: w @ m - t}

        def objective(w):
            return w @ C @ w

        bounds = [(0.0, max_weight)] * len(mu)
        x0 = w_minvol.to_numpy()
        res = minimize(objective, x0, method="SLSQP", bounds=bounds,
                       constraints=[{"type": "eq", "fun": lambda w: w.sum() - 1.0}, cons_extra],
                       options={"maxiter": 300, "ftol": 1e-10})
        if not res.success:
            continue
        w = np.clip(res.x, 0, max_weight)
        w = w / w.sum()
        points.append({
            "target_return": float(target),
            "ret": float(w @ m),
            "vol": float(np.sqrt(w @ C @ w)),
        })
    return points
