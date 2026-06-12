"""Leakage-free feature engineering: 42 features, all causal at close of day t.

Every rolling statistic is trailing (ends at t); cross-sectional features use
only date-t values; regime thresholds come from data before the current month.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.core.regimes import regime_states

CATEGORICAL_FEATURES = ["symbol_cat", "sector_cat"]

NUMERIC_FEATURES = [
    # returns / momentum
    "ret_1d", "ret_5d", "ret_21d", "ret_63d", "mom_12_1",
    # trend
    "c_sma10", "c_sma50", "sma10_sma50",
    # oscillators
    "rsi_14", "macd_norm", "macd_hist_norm", "boll_pctb", "boll_bw", "atr_norm",
    # volatility
    "rv_21", "rv_63", "rv_ratio", "parkinson_21",
    # candle / gap
    "range_norm", "overnight_gap",
    # volume / liquidity
    "vol_surprise", "vol_trend", "pct_deliv", "deliv_dev", "amihud", "c_vwap",
    # calendar
    "dow", "month", "is_month_end",
    # cross-sectional ranks
    "cs_rank_ret1", "cs_rank_ret21", "cs_rank_rv21", "cs_rank_volsurp",
    # market / sector context
    "mkt_ret_1d", "mkt_rv21", "beta_252", "rel_sector_21", "sector_ret_21",
    # regime / path
    "regime", "drawdown_252",
]

ALL_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES


def _wilder_ema(s: pd.Series, window: int) -> pd.Series:
    return s.ewm(alpha=1.0 / window, min_periods=window, adjust=False).mean()


def _rsi(close: pd.Series, window: int) -> pd.Series:
    delta = close.diff()
    gain = _wilder_ema(delta.clip(lower=0), window)
    loss = _wilder_ema((-delta).clip(lower=0), window)
    rs = gain / loss.replace(0, np.nan)
    return 100 - 100 / (1 + rs)


def _true_range(high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
    prev_close = close.shift(1)
    return pd.concat(
        [high - low, (high - prev_close).abs(), (low - prev_close).abs()], axis=1
    ).max(axis=1)


def compute_features(panel: pd.DataFrame, proxy: pd.DataFrame, params: dict) -> pd.DataFrame:
    """panel: adjusted long panel from stage 1 (sorted by symbol, date).

    Returns the panel keyed (symbol, date) with all feature columns appended.
    """
    df = panel.sort_values(["symbol", "date"]).reset_index(drop=True).copy()
    g = lambda col: df.groupby("symbol", sort=False)[col]  # noqa: E731

    log_close = np.log(df["close"])
    df["ret_1d"] = g("close").transform(lambda s: np.log(s).diff())
    for w, name in [(5, "ret_5d"), (21, "ret_21d"), (63, "ret_63d")]:
        df[name] = log_close - np.log(g("close").shift(w))
    df["mom_12_1"] = np.log(g("close").shift(21)) - np.log(g("close").shift(252))

    sma10 = g("close").transform(lambda s: s.rolling(10).mean())
    sma50 = g("close").transform(lambda s: s.rolling(50).mean())
    df["c_sma10"] = df["close"] / sma10 - 1
    df["c_sma50"] = df["close"] / sma50 - 1
    df["sma10_sma50"] = sma10 / sma50 - 1

    df["rsi_14"] = g("close").transform(lambda s: _rsi(s, params["rsi_window"]))

    fast, slow, sig = params["macd"]
    ema_f = g("close").transform(lambda s: s.ewm(span=fast, min_periods=fast, adjust=False).mean())
    ema_s = g("close").transform(lambda s: s.ewm(span=slow, min_periods=slow, adjust=False).mean())
    macd = ema_f - ema_s
    df["_macd"] = macd
    signal = g("_macd").transform(lambda s: s.ewm(span=sig, min_periods=sig, adjust=False).mean())
    df["macd_norm"] = macd / df["close"]
    df["macd_hist_norm"] = (macd - signal) / df["close"]
    df.drop(columns="_macd", inplace=True)

    bw = params["bollinger_window"]
    k = params["bollinger_k"]
    sma_b = g("close").transform(lambda s: s.rolling(bw).mean())
    sd_b = g("close").transform(lambda s: s.rolling(bw).std(ddof=1))
    df["boll_pctb"] = (df["close"] - (sma_b - k * sd_b)) / (2 * k * sd_b)
    df["boll_bw"] = (2 * k * sd_b) / sma_b

    df["_tr"] = (
        df.groupby("symbol", sort=False)
        .apply(lambda x: _true_range(x["high"], x["low"], x["close"]), include_groups=False)
        .reset_index(level=0, drop=True)
        .sort_index()
    )
    atr = g("_tr").transform(lambda s: _wilder_ema(s, params["atr_window"]))
    df["atr_norm"] = atr / df["close"]
    df.drop(columns="_tr", inplace=True)

    w1, w2 = params["vol_windows"]
    df["rv_21"] = g("ret_1d").transform(lambda s: s.rolling(w1).std(ddof=1))
    df["rv_63"] = g("ret_1d").transform(lambda s: s.rolling(w2).std(ddof=1))
    df["rv_ratio"] = df["rv_21"] / df["rv_63"]
    log_hl_sq = np.log(df["high"] / df["low"]) ** 2
    df["_hl2"] = log_hl_sq
    df["parkinson_21"] = np.sqrt(
        g("_hl2").transform(lambda s: s.rolling(w1).mean()) / (4 * np.log(2))
    )
    df.drop(columns="_hl2", inplace=True)

    df["range_norm"] = (df["high"] - df["low"]) / df["close"]
    df["overnight_gap"] = df["open"] / g("close").shift(1) - 1

    med_v21 = g("volume").transform(lambda s: s.rolling(21).median())
    df["vol_surprise"] = np.log(df["volume"] / med_v21.replace(0, np.nan))
    sma_v5 = g("volume").transform(lambda s: s.rolling(5).mean())
    sma_v63 = g("volume").transform(lambda s: s.rolling(63).mean())
    df["vol_trend"] = sma_v5 / sma_v63.replace(0, np.nan)
    df["deliv_dev"] = df["pct_deliv"] - g("pct_deliv").transform(lambda s: s.rolling(21).mean())

    df["_illiq"] = df["ret_1d"].abs() / df["turnover"].replace(0, np.nan)
    aw = params["amihud_window"]
    df["amihud"] = np.log(g("_illiq").transform(lambda s: s.rolling(aw).mean()))
    df.drop(columns="_illiq", inplace=True)
    df["c_vwap"] = (df["close"] - df["vwap"]) / df["vwap"]

    dates = pd.DatetimeIndex(df["date"])
    df["dow"] = dates.dayofweek.astype(np.int8)
    df["month"] = dates.month.astype(np.int8)
    # within 3 calendar days of month end — deterministic and known ex ante
    # (a trading-day definition would depend on data after t under truncation)
    days_to_eom = (dates + pd.offsets.MonthEnd(0)) - dates
    df["is_month_end"] = (days_to_eom.days <= 3).astype(np.int8)

    # cross-sectional same-day percentile ranks
    by_date = df.groupby("date", sort=False)
    df["cs_rank_ret1"] = by_date["ret_1d"].rank(pct=True)
    df["cs_rank_ret21"] = by_date["ret_21d"].rank(pct=True)
    df["cs_rank_rv21"] = by_date["rv_21"].rank(pct=True)
    df["cs_rank_volsurp"] = by_date["vol_surprise"].rank(pct=True)

    # market context (EW proxy from stage 1, median-robust regime thresholds)
    proxy = proxy.set_index("date")
    mkt_log = np.log1p(proxy["mkt_ret"])
    df["mkt_ret_1d"] = df["date"].map(mkt_log)
    reg = regime_states(
        mkt_log,
        vol_window=params["regime_vol_window"],
        min_history_days=params["regime_min_history"],
    )
    df["mkt_rv21"] = df["date"].map(reg["mkt_rv"])
    df["regime"] = df["date"].map(reg["regime"]).fillna(1).astype(np.int8)

    # rolling beta vs market proxy
    bw_, bmin = params["beta_window"], params["beta_min_periods"]
    mkt = df["mkt_ret_1d"]
    ret = df["ret_1d"]
    df["_xy"] = ret * mkt
    df["_x"] = mkt
    df["_x2"] = mkt**2
    df["_y"] = ret
    roll = lambda c: g(c).transform(  # noqa: E731
        lambda s: s.rolling(bw_, min_periods=bmin).mean()
    )
    cov = roll("_xy") - roll("_x") * roll("_y")
    var = roll("_x2") - roll("_x") ** 2
    df["beta_252"] = cov / var.replace(0, np.nan)
    df.drop(columns=["_xy", "_x", "_x2", "_y"], inplace=True)

    # sector relative momentum (sector EW log-return, trailing 21d sum)
    sec_cols = {c: c.removeprefix("sec_ret_") for c in proxy.columns if c.startswith("sec_ret_")}
    sec_log = np.log1p(proxy[list(sec_cols)]).rename(columns=sec_cols)
    sec_21 = sec_log.rolling(21).sum()
    sec_long = sec_21.stack().rename("sector_ret_21").reset_index()
    sec_long.columns = ["date", "sector", "sector_ret_21"]
    df = df.merge(sec_long, on=["date", "sector"], how="left")
    df["rel_sector_21"] = df["ret_21d"] - df["sector_ret_21"]

    df["drawdown_252"] = df["close"] / g("close").transform(lambda s: s.rolling(252).max()) - 1

    # categoricals
    df["symbol_cat"] = df["symbol"].astype("category")
    df["sector_cat"] = df["sector"].astype("category")

    # warm-up: drop the first `warmup_days` rows of each symbol
    df["_n"] = df.groupby("symbol", sort=False).cumcount()
    df = df[df["_n"] >= params["warmup_days"]].drop(columns="_n").reset_index(drop=True)
    return df
