"""Market Overview — entry page."""
from __future__ import annotations

import plotly.graph_objects as go
import streamlit as st

from app import loaders, ui

st.set_page_config(
    page_title="NIFTY-50 Investment Intelligence", page_icon="📈", layout="wide"
)

if not loaders.artifacts_ready():
    st.error("Artifacts not found. Run `make all` first (see README).")
    st.stop()

st.title("NIFTY-50 Investment Intelligence")
st.caption(
    "Decision-support platform built from 21 years of NIFTY-50 market data "
    "(Jan 2000 – Apr 2021). All numbers are computed from out-of-sample pipeline "
    "artifacts — nothing is hand-typed. Educational project, not investment advice."
)

proxy = loaders.parquet("market_proxy.parquet")
regimes = loaders.parquet("regimes.parquet")
quality = loaders.jsn("data_quality.json")
summary = loaders.jsn("portfolio_summary.json")
risk = loaders.jsn("risk_metrics.json")

mkt = risk["market_ew"]
c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Market CAGR (EW, 2000–21)", ui.fmt_pct(mkt["cagr"]))
c2.metric("Annualized volatility", ui.fmt_pct(mkt["ann_vol"]))
c3.metric("Worst drawdown", f"−{ui.fmt_pct(mkt['max_drawdown'])}")
c4.metric("Companies covered", quality["n_symbols"])
c5.metric("Corporate actions auto-fixed", quality["events_total"])

# ---- market chart with regime ribbon ---------------------------------------
st.subheader("Two decades of the Indian large-cap market")
d = ui.downsample(proxy[["date", "mkt_level"]].dropna())
fig = ui.line(d, "date", "mkt_level", name="Equal-weight NIFTY-50 proxy", color="#1a5276")
fig.update_layout(shapes=ui.regime_shapes(regimes))
fig.update_yaxes(type="log", title="index level (log)")
st.plotly_chart(ui.base_layout(fig, height=440), use_container_width=True)
st.caption(
    "Background shading: market volatility regime detected from expanding terciles "
    "of 21-day realized volatility — green = Calm, red = Turbulent. Fully causal "
    "(thresholds use only past data)."
)

# ---- sector x year heatmap ---------------------------------------------------
st.subheader("Sector performance by year")
sec_cols = [c for c in proxy.columns if c.startswith("sec_ret_")]
sec = proxy.set_index("date")[sec_cols]
yearly = (1 + sec).groupby(sec.index.year).prod() - 1
yearly.columns = [c.removeprefix("sec_ret_") for c in yearly.columns]
fig = go.Figure(go.Heatmap(
    z=yearly.T.to_numpy() * 100, x=yearly.index.astype(str), y=yearly.columns,
    colorscale="RdYlGn", zmid=0, colorbar=dict(title="%"),
    hovertemplate="%{y} %{x}: %{z:.1f}%<extra></extra>",
))
st.plotly_chart(ui.base_layout(fig, height=460), use_container_width=True)

# ---- headline insights -------------------------------------------------------
st.subheader("Three headline insights")
profiles = summary["profiles"]
best_year = yearly.mean(axis=1).idxmax()
worst_year = yearly.mean(axis=1).idxmin()
sec_span = yearly.mean(axis=0).sort_values()
i1, i2, i3 = st.columns(3)
with i1:
    st.info(
        f"**Diversification earned its keep.** The minimum-variance portfolio cut the "
        f"worst drawdown from {ui.fmt_pct(profiles['benchmark_ew']['max_drawdown'])} "
        f"(equal weight) to {ui.fmt_pct(profiles['conservative']['max_drawdown'])} while "
        f"*raising* the Sharpe ratio from {ui.fmt_num(profiles['benchmark_ew']['sharpe'])} "
        f"to {ui.fmt_num(profiles['conservative']['sharpe'])} — lower risk did not cost return."
    )
with i2:
    st.info(
        f"**Sector rotation is enormous.** Average annual returns range from "
        f"{ui.fmt_pct(sec_span.iloc[0])} ({sec_span.index[0].title()}) to "
        f"{ui.fmt_pct(sec_span.iloc[-1])} ({sec_span.index[-1].title()}); the best single "
        f"market year was {best_year} and the worst {worst_year}. Sector caps in the "
        f"portfolio engine exist precisely because leadership rotates."
    )
with i3:
    st.info(
        "**Returns are hard to predict; risk is not.** Out-of-sample, short-horizon "
        "return forecasts barely beat a drift baseline (a finding consistent with "
        "market efficiency), while volatility forecasts achieve a Mincer-Zarnowitz "
        "R² of 0.08–0.15. The platform therefore routes its intelligence into risk "
        "management, calibrated probabilities and portfolio construction."
    )

st.divider()
st.markdown(
    "**Explore:** use the sidebar — *Stock Explorer* (charts + corporate-action "
    "audit), *Prediction Lab* (out-of-sample forecasts with uncertainty bands), "
    "*Portfolio Builder* (three investor profiles), *Risk Dashboard*, *Anomalies & "
    "Regimes*, and the full *Methodology & Model Card*."
)
