"""Explainable anomaly detection: robust z-scores with reason strings.

Four detectors, each trivially explainable:
- return shock:   |r - median252| / (1.4826 * MAD252) > 5
- volume spike:   robust z of log(V / median63(V)) > 4
- drawdown event: 21-day return < -25%
- market-wide:    robust z of the EW market return > 4
"""
from __future__ import annotations

import numpy as np
import pandas as pd

MAD_K = 1.4826  # consistency constant for Gaussian data


def robust_z(s: pd.Series, window: int) -> pd.Series:
    med = s.rolling(window).median()
    mad = (s - med).abs().rolling(window).median()
    return (s - med) / (MAD_K * mad.replace(0, np.nan))


def detect_anomalies(panel: pd.DataFrame, proxy: pd.DataFrame, params: dict) -> pd.DataFrame:
    """panel: adjusted long panel with ret/volume; proxy: market returns."""
    df = panel.sort_values(["symbol", "date"]).reset_index(drop=True).copy()
    g = df.groupby("symbol", sort=False)

    zr = g["ret"].transform(lambda s: robust_z(s, params["ret_z_window"]))
    logv = np.log(df["volume"].replace(0, np.nan))
    df["_logv"] = logv
    zv = g["_logv"].transform(lambda s: robust_z(s, params["vol_z_window"]))
    ret21 = g["close"].transform(lambda s: s.pct_change(21))

    events = []

    m = zr.abs() > params["ret_z_threshold"]
    for i in np.where(m.fillna(False))[0]:
        events.append({
            "date": df.at[i, "date"], "symbol": df.at[i, "symbol"], "type": "return_shock",
            "severity": float(abs(zr.iat[i])),
            "reason": f"1-day return {df.at[i, 'ret']*100:+.1f}% is {abs(zr.iat[i]):.1f} robust SDs from its 1y median",
        })

    m = zv > params["vol_z_threshold"]
    for i in np.where(m.fillna(False))[0]:
        events.append({
            "date": df.at[i, "date"], "symbol": df.at[i, "symbol"], "type": "volume_spike",
            "severity": float(zv.iat[i]),
            "reason": f"volume z-score {zv.iat[i]:.1f} vs 63-day history",
        })

    m = ret21 < params["drawdown_21d_threshold"]
    for i in np.where(m.fillna(False))[0]:
        events.append({
            "date": df.at[i, "date"], "symbol": df.at[i, "symbol"], "type": "drawdown_event",
            "severity": float(-ret21.iat[i] * 10),
            "reason": f"21-day return {ret21.iat[i]*100:.1f}%",
        })

    mkt = proxy.set_index("date")["mkt_ret"]
    zm = robust_z(mkt, params["ret_z_window"])
    for dt in zm.index[zm.abs() > params["market_z_threshold"]]:
        events.append({
            "date": dt, "symbol": "_MARKET_", "type": "market_wide",
            "severity": float(abs(zm.loc[dt])),
            "reason": f"market return {mkt.loc[dt]*100:+.1f}% is {abs(zm.loc[dt]):.1f} robust SDs from median",
        })

    out = pd.DataFrame(events)
    if not out.empty:
        out["date"] = pd.to_datetime(out["date"])
        out = out.sort_values(["date", "symbol"]).reset_index(drop=True)
    return out
