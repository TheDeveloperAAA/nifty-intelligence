"""Three investor profiles: weights, rationale, backtest, frontier."""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

try:
    from app import loaders, ui
except ModuleNotFoundError:  # Streamlit Cloud doesn't put the repo root on sys.path
    import sys
    from pathlib import Path as _P
    sys.path.append(str(_P(__file__).resolve().parents[2]))
    from app import loaders, ui

st.set_page_config(page_title="Portfolio Builder", page_icon="💼", layout="wide")
st.title("Portfolio Builder")

summary = loaders.jsn("portfolio_summary.json")
curves = loaders.parquet("backtest_curves.parquet")
rebal = loaders.parquet("rebalance_log.parquet")
frontier = loaders.jsn("frontier.json")

profile = st.radio(
    "Investor profile", ["conservative", "balanced", "aggressive"], horizontal=True,
    format_func=str.title,
)
desc = {
    "conservative": "**Global minimum-variance** — needs no return forecasts at all, the most "
                    "estimation-robust optimizer there is. Caps: 10% per stock, 25% per sector.",
    "balanced": "**Maximum Sharpe ratio** on shrunk inputs (Ledoit-Wolf covariance, "
                "James-Stein-style means). Caps: 15% per stock, 30% per sector.",
    "aggressive": "**Momentum + model tilt** — top 15 stocks by a 50/50 blend of 12-1 momentum "
                  "and the out-of-sample 21-day model score, inverse-volatility weighted. "
                  "Scales to 50% risky weight in Turbulent regimes.",
}
st.markdown(desc[profile])

prof = summary["profiles"][profile]
bench = summary["profiles"]["benchmark_ew"]
c = st.columns(6)
c[0].metric("CAGR", ui.fmt_pct(prof["cagr"]), f"EW: {bench['cagr']*100:.1f}%", delta_color="off")
c[1].metric("Volatility", ui.fmt_pct(prof["ann_vol"]), f"EW: {bench['ann_vol']*100:.1f}%", delta_color="off")
c[2].metric("Sharpe", ui.fmt_num(prof["sharpe"]), f"EW: {bench['sharpe']:.2f}", delta_color="off")
c[3].metric("Sortino", ui.fmt_num(prof["sortino"]))
c[4].metric("Max drawdown", f"−{ui.fmt_pct(prof['max_drawdown'])}",
            f"EW: −{bench['max_drawdown']*100:.1f}%", delta_color="off")
c[5].metric("Turnover / year", f"{prof['annual_turnover']:.1f}×")

# ---- weights ------------------------------------------------------------------
final = summary["final_weights"][profile]
w = pd.Series(final["weights"]).sort_values(ascending=False)
rc = pd.Series(final["risk_contribution"])
sec_w = pd.Series(final["sector_weights"])
left, right = st.columns(2)
with left:
    st.subheader(f"Allocation ({final['date']})")
    fig = go.Figure(go.Pie(labels=w.index, values=w.values, hole=0.45,
                           textinfo="label+percent", textposition="outside"))
    st.plotly_chart(ui.base_layout(fig, height=430), use_container_width=True)
with right:
    st.subheader("Sector exposure & risk contribution")
    fig = go.Figure()
    fig.add_trace(go.Bar(y=sec_w.index, x=sec_w.values * 100, orientation="h",
                         name="capital %", marker_color="#2e86c1"))
    fig.update_xaxes(title="% of portfolio")
    st.plotly_chart(ui.base_layout(fig, height=220), use_container_width=True)
    st.caption(f"Forecast portfolio volatility: {ui.fmt_pct(final['portfolio_vol_forecast'])}")
    top_rc = rc.head(8)
    fig = go.Figure(go.Bar(y=top_rc.index, x=top_rc.values * 100, orientation="h",
                           marker_color="#ca6f1e"))
    fig.update_xaxes(title="% of portfolio risk")
    st.plotly_chart(ui.base_layout(fig, height=220), use_container_width=True)

with st.expander("Why these weights? (per-holding rationale)"):
    rows = []
    for sym, weight in w.items():
        rows.append({
            "Stock": sym,
            "Weight": f"{weight*100:.1f}%",
            "Risk contribution": f"{rc.get(sym, np.nan)*100:.1f}%",
            "Rationale": (
                "low covariance with the rest of the basket" if profile == "conservative"
                else "high shrunk expected return per unit of risk" if profile == "balanced"
                else "top momentum + model score, sized inversely to volatility"
            ),
        })
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
    st.caption(
        "Risk contribution is w_i(Σw)_i / wᵀΣw with the Ledoit-Wolf covariance — "
        "if a stock's risk share far exceeds its capital share, it is the "
        "portfolio's true risk driver."
    )

# ---- backtest ------------------------------------------------------------------
st.subheader("Walk-forward backtest (monthly rebalance, 10 bps/side costs)")
sel = st.multiselect(
    "Strategies", sorted(curves["profile"].unique()),
    default=[profile, "benchmark_ew", "hrp"],
)
st.plotly_chart(ui.wealth_chart(curves[curves["profile"].isin(sel)]),
                use_container_width=True)
st.caption(
    f"Growth of ₹1 since {summary['assumptions']['backtest_start']}, net of "
    f"{summary['assumptions']['cost_bps_per_side']} bps/side transaction costs. "
    "Weights at each rebalance use only information available the prior day; the "
    "model score inside the aggressive profile is strictly out-of-sample. HRP = "
    "hierarchical risk parity comparison."
)

rb = rebal[rebal["profile"] == profile]
with st.expander("Rebalance log"):
    show = rb.copy()
    show["turnover"] = (show["turnover"] * 100).round(1).astype(str) + "%"
    show["cost"] = (show["cost"] * 1e4).round(1).astype(str) + " bps"
    st.dataframe(show.rename(columns={"date": "Date", "n_names": "Holdings",
                                      "risky_scale": "Risky scale"}),
                 use_container_width=True, hide_index=True, height=240)

# ---- efficient frontier ----------------------------------------------------------
st.subheader("Efficient frontier (as of " + frontier["as_of"] + ")")
fig = go.Figure()
fr = pd.DataFrame(frontier["frontier"])
stocks = pd.DataFrame(frontier["stocks"])
fig.add_trace(go.Scatter(x=stocks["vol"] * 100, y=stocks["ret"] * 100, mode="markers+text",
                         text=stocks["symbol"], textposition="top center",
                         textfont=dict(size=8), marker=dict(size=5, color="#aab7b8"),
                         name="stocks"))
fig.add_trace(go.Scatter(x=fr["vol"] * 100, y=fr["ret"] * 100, mode="lines",
                         name="frontier", line=dict(color="#1a5276", width=2)))
for name, pt in frontier["portfolios"].items():
    fig.add_trace(go.Scatter(x=[pt["vol"] * 100], y=[pt["ret"] * 100], mode="markers+text",
                             text=[name], textposition="bottom right",
                             marker=dict(size=12, symbol="star",
                                         color=ui.PROFILE_COLORS.get(name, "#000")),
                             showlegend=False))
fig.update_xaxes(title="volatility (% ann.)")
fig.update_yaxes(title="expected return (% ann., shrunk trailing)")
st.plotly_chart(ui.base_layout(fig, height=480), use_container_width=True)
st.caption(frontier["note"])
