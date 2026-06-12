"""Risk metrics, drawdowns, rolling diagnostics, correlations, VaR backtest."""
from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from app import loaders, ui

st.set_page_config(page_title="Risk Dashboard", page_icon="🛡️", layout="wide")
st.title("Risk Dashboard")

summary = loaders.jsn("portfolio_summary.json")
rolling = loaders.parquet("rolling_risk.parquet")
dds = loaders.parquet("drawdowns.parquet")
corr = loaders.parquet("correlation.parquet")
var_bt = loaders.jsn("var_backtest.json")
vol = loaders.parquet("vol_forecasts.parquet")
garch = loaders.jsn("garch_params.json")
stock_risk = loaders.parquet("risk_stock.parquet")

entity = st.selectbox(
    "Entity", ["conservative", "balanced", "aggressive", "benchmark_ew", "hrp", "market_ew"],
    format_func=lambda s: {"benchmark_ew": "Equal-weight benchmark",
                           "market_ew": "Market proxy (no costs)"}.get(s, s.title()),
)

if entity != "market_ew":
    p = summary["profiles"][entity]
    c = st.columns(7)
    c[0].metric("CAGR", ui.fmt_pct(p["cagr"]))
    c[1].metric("Volatility", ui.fmt_pct(p["ann_vol"]))
    c[2].metric("Sharpe (rf 6%)", ui.fmt_num(p["sharpe"]))
    c[3].metric("Sortino", ui.fmt_num(p["sortino"]))
    c[4].metric("Calmar", ui.fmt_num(p["calmar"]))
    c[5].metric("Daily VaR 95%", ui.fmt_pct(p["var_95"], 2))
    c[6].metric("Daily CVaR 95%", ui.fmt_pct(p["cvar_95"], 2))
    st.caption(
        f"Sharpe sensitivity to the risk-free assumption: rf 4% → "
        f"{ui.fmt_num(p.get('sharpe_rf4'))}, rf 6% → {ui.fmt_num(p['sharpe'])}, "
        f"rf 7% → {ui.fmt_num(p.get('sharpe_rf7'))}. Monthly VaR ≈ daily × √21."
    )

# ---- underwater chart -----------------------------------------------------------
st.subheader("Drawdown (underwater) curve")
d = ui.downsample(dds[dds["entity"] == entity].sort_values("date"))
fig = go.Figure(go.Scatter(x=d["date"], y=d["drawdown"] * 100, fill="tozeroy",
                           line=dict(color="#c0392b", width=1), name="drawdown"))
fig.update_yaxes(title="% below peak")
st.plotly_chart(ui.base_layout(fig, height=300), use_container_width=True)

# ---- rolling metrics -------------------------------------------------------------
st.subheader("Rolling 252-day diagnostics")
r = ui.downsample(rolling[rolling["entity"] == entity].sort_values("date"))
c1, c2 = st.columns(2)
with c1:
    fig = ui.line(r, "date", "roll_vol", name="rolling vol", color="#2e86c1")
    fig.update_yaxes(title="ann. vol", tickformat=".0%")
    st.plotly_chart(ui.base_layout(fig, "Rolling volatility", 300), use_container_width=True)
with c2:
    col = "roll_beta" if "roll_beta" in r.columns and entity != "market_ew" else "roll_sharpe"
    fig = ui.line(r.dropna(subset=[col]), "date", col, name=col, color="#117a65")
    st.plotly_chart(ui.base_layout(fig, f"Rolling {'beta' if col=='roll_beta' else 'Sharpe'}", 300),
                    use_container_width=True)

# ---- VaR backtest ----------------------------------------------------------------
st.subheader("VaR model backtest (Kupiec proportion-of-failures)")
rows = []
for k, v in var_bt.items():
    rows.append({
        "Strategy": k, "Days": v["n"], "Exceptions": v["exceptions"],
        "Expected": round(v["expected_exceptions"]),
        "Exception rate": f"{v['exception_rate']*100:.2f}%",
        "Kupiec p-value": round(v["p_value"], 3),
        "Verdict": "✅ calibrated" if v["p_value"] > 0.05 else "❌ rejected",
    })
st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
st.caption(
    "A trailing-252-day historical VaR(95) is applied to the *next* day; Kupiec "
    "tests whether the realized exception rate matches the nominal 5%."
)

# ---- volatility forecasting --------------------------------------------------------
st.subheader("Volatility forecasting (the predictable risk dimension)")
c1, c2 = st.columns([3, 2])
with c1:
    sym = st.selectbox("Stock", sorted(vol["symbol"].unique()),
                       index=sorted(vol["symbol"].unique()).index("RELIANCE"))
    v = ui.downsample(vol[vol["symbol"] == sym].sort_values("date"))
    fig = ui.line(v, "date", "sigma_ann", name="EWMA σ (annualized)", color="#8e44ad")
    fig.update_yaxes(title="ann. vol", tickformat=".0%")
    st.plotly_chart(ui.base_layout(fig, height=320), use_container_width=True)
with c2:
    st.markdown("**EWMA (λ=0.94) vs GARCH(1,1)-t** — one-step forecasts on the locked test window:")
    rows = []
    for k, v_ in garch["showcase"].items():
        if "garch" in v_:
            rows.append({"Series": k, "EWMA MZ-R²": round(v_["ewma"]["mz_r2"], 3),
                         "GARCH MZ-R²": round(v_["garch"]["mz_r2"], 3),
                         "QLIKE gap": f"{(v_['garch']['qlike']-v_['ewma']['qlike'])/abs(v_['ewma']['qlike'])*100:+.1f}%"})
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
    st.caption(
        "GARCH is modestly sharper on liquid names; EWMA is kept as the production "
        "layer because it cannot fail to converge across all 49 names and is fully "
        "deterministic. Volatility shows real predictability (MZ-R² ≫ 0) — unlike "
        "short-horizon returns."
    )

# ---- correlation heatmap ------------------------------------------------------------
st.subheader("Cross-stock correlation (last 3 years, sector-ordered)")
cm = corr.pivot(index="symbol", columns="symbol2", values="corr")
order = stock_risk.sort_values("sector")["symbol"].tolist()
cm = cm.reindex(index=order, columns=order)
fig = go.Figure(go.Heatmap(z=cm.to_numpy(), x=cm.columns, y=cm.index,
                           colorscale="RdBu", zmid=0, zmin=-0.2, zmax=1))
st.plotly_chart(ui.base_layout(fig, height=620), use_container_width=True)
st.caption("Sector blocks are clearly visible — the diversification the sector caps exploit.")
