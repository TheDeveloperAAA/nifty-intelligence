"""Anomaly timeline with auto-recovered historical events + regime history."""
from __future__ import annotations

import plotly.graph_objects as go
import streamlit as st

try:
    from app import loaders, ui
except ModuleNotFoundError:  # Streamlit Cloud doesn't put the repo root on sys.path
    import sys
    from pathlib import Path as _P
    sys.path.append(str(_P(__file__).resolve().parents[2]))
    from app import loaders, ui

st.set_page_config(page_title="Anomalies & Regimes", page_icon="⚡", layout="wide")
st.title("Anomalies & Regimes")

anoms = loaders.parquet("anomalies.parquet")
validation = loaders.jsn("anomaly_validation.json")
regimes = loaders.parquet("regimes.parquet")

st.markdown(
    "Four **fully explainable detectors** run over the adjusted panel: robust "
    "z-scores on returns (|z| > 5) and volume (z > 4), 21-day drawdown events "
    "(< −25%), and market-wide shocks. Every flag carries a plain-English reason. "
    "As validation, the detector must *re-discover* the known crises of 2004, "
    "2008, 2009, 2013, 2016 and 2020 from the data alone — it does."
)
c1, c2, c3 = st.columns(3)
c1.metric("Events flagged (21 years)", f"{validation['n_events_total']:,}")
c2.metric("Market-wide shock days", validation["by_type"].get("market_wide", 0))
c3.metric("Known crises recovered", f"{len(validation['known_events_recovered'])} / "
          f"{len(validation['known_events_expected'])}")

# ---- timeline -------------------------------------------------------------------
st.subheader("Anomaly timeline")
types = st.multiselect("Event types", sorted(anoms["type"].unique()),
                       default=["market_wide", "return_shock"])
a = anoms[anoms["type"].isin(types)]
colors = {"market_wide": "#c0392b", "return_shock": "#ca6f1e",
          "volume_spike": "#2e86c1", "drawdown_event": "#7f8c8d"}
fig = go.Figure()
for t, d in a.groupby("type"):
    d = d.nlargest(800, "severity")
    fig.add_trace(go.Scatter(
        x=d["date"], y=d["severity"], mode="markers", name=t,
        marker=dict(size=(d["severity"].clip(3, 15)), color=colors.get(t), opacity=0.55),
        text=d["symbol"] + ": " + d["reason"], hoverinfo="text+x",
    ))
fig.update_yaxes(title="severity (robust z / scaled)")
fig.update_layout(shapes=ui.regime_shapes(regimes))
st.plotly_chart(ui.base_layout(fig, height=440), use_container_width=True)
st.caption("Background shading: volatility regime (green Calm, red Turbulent). "
           "Anomalies cluster almost perfectly inside Turbulent regimes.")

# ---- top events table --------------------------------------------------------------
st.subheader("Top 20 market-wide events — recovered from data alone")
top = anoms[anoms["type"] == "market_wide"].nlargest(20, "severity").copy()
top["Date"] = top["date"].dt.date
top["Known event"] = top["known_event"].fillna("—")
st.dataframe(top[["Date", "severity", "reason", "Known event"]]
             .rename(columns={"severity": "Severity", "reason": "Why flagged"}),
             use_container_width=True, hide_index=True)

# ---- per-stock drill-down ------------------------------------------------------------
st.subheader("Per-stock drill-down")
sym = st.selectbox("Stock", sorted(anoms[anoms["symbol"] != "_MARKET_"]["symbol"].unique()))
s = anoms[anoms["symbol"] == sym].sort_values("severity", ascending=False).head(15).copy()
s["Date"] = s["date"].dt.date
st.dataframe(s[["Date", "type", "severity", "reason"]]
             .rename(columns={"type": "Type", "severity": "Severity", "reason": "Why flagged"}),
             use_container_width=True, hide_index=True)

# ---- regime history -------------------------------------------------------------------
st.subheader("Volatility regime history")
r = ui.downsample(regimes.dropna(subset=["mkt_rv"]))
fig = go.Figure()
fig.add_trace(go.Scatter(x=r["date"], y=r["mkt_rv"] * 100, name="market 21d realized vol",
                         line=dict(color="#1a5276", width=1.2)))
fig.add_trace(go.Scatter(x=r["date"], y=r["regime_lo_thr"] * 100, name="Calm threshold",
                         line=dict(color="#27ae60", width=1, dash="dot")))
fig.add_trace(go.Scatter(x=r["date"], y=r["regime_hi_thr"] * 100, name="Turbulent threshold",
                         line=dict(color="#c0392b", width=1, dash="dot")))
fig.update_yaxes(title="annualized vol (%)")
st.plotly_chart(ui.base_layout(fig, height=380), use_container_width=True)
st.caption(
    "Thresholds are expanding-window terciles refreshed monthly — fully causal, "
    "three lines of code, and they drive a defensive de-risking overlay in the "
    "Balanced/Aggressive portfolios (risky weight × 0.7 / × 0.5 when Turbulent)."
)
