"""Per-stock charts, indicators and the corporate-action audit trail."""
from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

try:
    from app import loaders, ui
except ModuleNotFoundError:  # Streamlit Cloud doesn't put the repo root on sys.path
    import sys
    from pathlib import Path as _P
    sys.path.append(str(_P(__file__).resolve().parents[2]))
    from app import loaders, ui

st.set_page_config(page_title="Stock Explorer", page_icon="🔍", layout="wide")
st.title("Stock Explorer")

disp = loaders.parquet("display_indicators.parquet")
actions = loaders.csv("corporate_actions.csv")
stock_risk = loaders.parquet("risk_stock.parquet")
px = None  # loaded lazily for sector names

symbols = sorted(disp["symbol"].unique())
left, mid, right = st.columns([2, 2, 3])
symbol = left.selectbox("Stock", symbols, index=symbols.index("RELIANCE") if "RELIANCE" in symbols else 0)
years = right.slider("Window (years)", 1, 21, 10)
adjusted = mid.toggle("Split/bonus adjusted", value=True,
                      help="Raw NSE prices contain artificial cliffs at splits and "
                           "bonus issues; the pipeline detects and removes them.")

d = disp[disp["symbol"] == symbol].sort_values("date")
d = d[d["date"] >= d["date"].max() - pd.DateOffset(years=years)]
d = ui.downsample(d, 4000)
ev = actions[actions["symbol"] == symbol]

price_col = "close" if adjusted else "close_raw"
fig = make_subplots(rows=3, cols=1, shared_xaxes=True, row_heights=[0.62, 0.18, 0.20],
                    vertical_spacing=0.03)
fig.add_trace(go.Scatter(x=d["date"], y=d[price_col], name="close",
                         line=dict(color="#1a5276", width=1.6)), 1, 1)
if adjusted:
    fig.add_trace(go.Scatter(x=d["date"], y=d["sma10"], name="SMA-10",
                             line=dict(color="#e67e22", width=1, dash="dot")), 1, 1)
    fig.add_trace(go.Scatter(x=d["date"], y=d["sma50"], name="SMA-50",
                             line=dict(color="#8e44ad", width=1, dash="dot")), 1, 1)
    fig.add_trace(go.Scatter(x=d["date"], y=d["boll_up"], name="Bollinger",
                             line=dict(color="rgba(127,140,141,0.5)", width=0.8),
                             showlegend=True), 1, 1)
    fig.add_trace(go.Scatter(x=d["date"], y=d["boll_lo"], name="Bollinger lower",
                             line=dict(color="rgba(127,140,141,0.5)", width=0.8),
                             fill="tonexty", fillcolor="rgba(127,140,141,0.06)",
                             showlegend=False), 1, 1)
for _, e in ev.iterrows():
    if d["date"].min() <= e["date"] <= d["date"].max():
        fig.add_vline(x=e["date"], line_color="#c0392b", line_dash="dash",
                      line_width=1, row=1, col=1)
fig.add_trace(go.Bar(x=d["date"], y=d["volume"], name="volume",
                     marker_color="#85929e"), 2, 1)
fig.add_trace(go.Scatter(x=d["date"], y=d["rsi_14"], name="RSI-14",
                         line=dict(color="#117a65", width=1.2)), 3, 1)
fig.add_hline(y=70, line_dash="dot", line_color="#c0392b", row=3, col=1)
fig.add_hline(y=30, line_dash="dot", line_color="#27ae60", row=3, col=1)
fig.update_yaxes(title="price (₹)", type="log" if years > 8 else "linear", row=1, col=1)
fig.update_yaxes(title="volume", row=2, col=1)
fig.update_yaxes(title="RSI", range=[0, 100], row=3, col=1)
st.plotly_chart(ui.base_layout(fig, height=680), use_container_width=True)
if len(ev):
    st.caption(f"Red dashed lines: {len(ev)} detected corporate action(s) — see audit below.")

# ---- stats + corporate actions ----------------------------------------------
c1, c2 = st.columns([2, 3])
with c1:
    st.subheader("Risk profile (full history)")
    r = stock_risk[stock_risk["symbol"] == symbol].iloc[0]
    rows = {
        "Sector": r["sector"],
        "CAGR": ui.fmt_pct(r["cagr"]),
        "Volatility (ann.)": ui.fmt_pct(r["ann_vol"]),
        "Sharpe (rf 6%)": ui.fmt_num(r["sharpe"]),
        "Sortino": ui.fmt_num(r["sortino"]),
        "Max drawdown": f"−{ui.fmt_pct(r['max_drawdown'])}",
        "Daily VaR 95%": ui.fmt_pct(r["var_95"], 2),
        "Daily CVaR 95%": ui.fmt_pct(r["cvar_95"], 2),
        "Beta vs market": ui.fmt_num(r["beta"]),
    }
    st.table(pd.Series(rows, name=symbol))
with c2:
    st.subheader("Corporate-action audit")
    if len(ev):
        show = ev.copy()
        show["implied"] = (1 / show["factor"]).round(2).astype(str) + "×"
        show = show.rename(columns={
            "date": "Date", "observed_ratio": "Price ratio", "factor": "Factor",
            "kind": "Detection", "implied": "Share multiplier",
        })[["Date", "Price ratio", "Factor", "Share multiplier", "Detection"]]
        st.dataframe(show, use_container_width=True, hide_index=True)
        st.caption(
            "Detected from price ratios alone (no external data): a close-to-close "
            "drop snapping to a plausible split/bonus fraction, with the open "
            "confirming the adjusted level. Historical prices and volumes are "
            "back-adjusted so returns contain no artificial cliffs."
        )
    else:
        st.write("No corporate actions detected for this stock.")
