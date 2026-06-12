"""Risk metric suite. All formulas operate on daily returns (log or simple as
noted); annualization uses 252 trading days; rf is a documented assumption.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

TRADING_DAYS = 252


def annualized_return(ret: pd.Series, trading_days: int = TRADING_DAYS) -> float:
    """Geometric annualized return from daily simple returns."""
    ret = ret.dropna()
    if len(ret) == 0:
        return float("nan")
    total = float((1 + ret).prod())
    if total <= 0:
        return float("nan")
    return total ** (trading_days / len(ret)) - 1


def annualized_vol(ret: pd.Series, trading_days: int = TRADING_DAYS) -> float:
    return float(ret.std(ddof=1) * np.sqrt(trading_days))


def sharpe(ret: pd.Series, rf_annual: float, trading_days: int = TRADING_DAYS) -> float:
    vol = annualized_vol(ret, trading_days)
    if not np.isfinite(vol) or vol == 0:
        return float("nan")
    return (annualized_return(ret, trading_days) - rf_annual) / vol


def downside_deviation(ret: pd.Series, rf_annual: float, trading_days: int = TRADING_DAYS) -> float:
    mar_daily = rf_annual / trading_days
    short = np.minimum(ret.dropna() - mar_daily, 0.0)
    return float(np.sqrt(trading_days * np.mean(short**2)))


def sortino(ret: pd.Series, rf_annual: float, trading_days: int = TRADING_DAYS) -> float:
    dd = downside_deviation(ret, rf_annual, trading_days)
    if not np.isfinite(dd) or dd == 0:
        return float("nan")
    return (annualized_return(ret, trading_days) - rf_annual) / dd


def max_drawdown(wealth: pd.Series) -> float:
    """Max peak-to-trough loss of a wealth curve (positive number)."""
    wealth = wealth.dropna()
    if len(wealth) == 0:
        return float("nan")
    peak = wealth.cummax()
    return float((1 - wealth / peak).max())


def drawdown_series(wealth: pd.Series) -> pd.Series:
    peak = wealth.cummax()
    return wealth / peak - 1


def calmar(ret: pd.Series, trading_days: int = TRADING_DAYS) -> float:
    wealth = (1 + ret.fillna(0)).cumprod()
    mdd = max_drawdown(wealth)
    if not np.isfinite(mdd) or mdd == 0:
        return float("nan")
    return annualized_return(ret, trading_days) / mdd


def var_historical(ret: pd.Series, level: float = 0.95) -> float:
    """Daily historical VaR at `level` (positive number = loss)."""
    ret = ret.dropna()
    if len(ret) < 20:
        return float("nan")
    return float(-np.quantile(ret, 1 - level))


def cvar_historical(ret: pd.Series, level: float = 0.95) -> float:
    ret = ret.dropna()
    if len(ret) < 20:
        return float("nan")
    q = np.quantile(ret, 1 - level)
    tail = ret[ret <= q]
    return float(-tail.mean()) if len(tail) else float("nan")


def beta(ret: pd.Series, mkt_ret: pd.Series) -> float:
    df = pd.concat([ret, mkt_ret], axis=1).dropna()
    if len(df) < 60:
        return float("nan")
    cov = np.cov(df.iloc[:, 0], df.iloc[:, 1], ddof=1)
    return float(cov[0, 1] / cov[1, 1]) if cov[1, 1] > 0 else float("nan")


def kupiec_pof(ret: pd.Series, var_series: pd.Series, level: float = 0.95) -> dict:
    """Kupiec proportion-of-failures test of a VaR backtest.

    H0: the exception rate equals 1 - level. LR ~ chi2(1).
    """
    from scipy.stats import chi2

    df = pd.concat([ret, var_series], axis=1).dropna()
    if len(df) < 50:
        return {"n": int(len(df)), "lr": float("nan"), "p_value": float("nan")}
    exceptions = (df.iloc[:, 0] < -df.iloc[:, 1]).sum()
    n = len(df)
    p = 1 - level
    phat = exceptions / n
    if phat in (0.0, 1.0):
        lr = -2 * (n * np.log(1 - p) if exceptions == 0 else n * np.log(p))
    else:
        lr = -2 * (
            (n - exceptions) * np.log((1 - p) / (1 - phat))
            + exceptions * np.log(p / phat)
        )
    return {
        "n": int(n),
        "exceptions": int(exceptions),
        "expected_exceptions": float(n * p),
        "exception_rate": float(phat),
        "lr": float(lr),
        "p_value": float(chi2.sf(lr, df=1)),
    }


def summary(ret: pd.Series, rf_annual: float, mkt_ret: pd.Series | None = None,
            var_levels: tuple[float, ...] = (0.95, 0.99)) -> dict:
    """Full metric block for one return series (daily simple returns)."""
    wealth = (1 + ret.fillna(0)).cumprod()
    out = {
        "n_days": int(ret.notna().sum()),
        "cagr": annualized_return(ret),
        "ann_vol": annualized_vol(ret),
        "sharpe": sharpe(ret, rf_annual),
        "sortino": sortino(ret, rf_annual),
        "max_drawdown": max_drawdown(wealth),
        "calmar": calmar(ret),
    }
    for lv in var_levels:
        key = str(int(lv * 100))
        out[f"var_{key}"] = var_historical(ret, lv)
        out[f"cvar_{key}"] = cvar_historical(ret, lv)
    if mkt_ret is not None:
        out["beta"] = beta(ret, mkt_ret)
    return out


def risk_contributions(weights: np.ndarray, cov: np.ndarray) -> np.ndarray:
    """Percent risk contribution RC_i = w_i (Sigma w)_i / (w' Sigma w)."""
    w = np.asarray(weights, float)
    port_var = float(w @ cov @ w)
    if port_var <= 0:
        return np.full_like(w, np.nan)
    return w * (cov @ w) / port_var
