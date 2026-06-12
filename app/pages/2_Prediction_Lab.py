"""Out-of-sample forecasts, uncertainty bands, baselines and reason codes."""
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

st.set_page_config(page_title="Prediction Lab", page_icon="🎯", layout="wide")
st.title("Prediction Lab")
st.warning(
    "Everything on this page is **historical and out-of-sample**: each prediction "
    "comes from a model trained only on data *before* the window it is shown in. "
    "The dataset ends 2021-04-30 — these are not live trading signals.",
    icon="⚠️",
)

preds = loaders.parquet("predictions.parquet")
metrics = loaders.jsn("model_metrics.json")
shap_latest = loaders.parquet("shap_latest.parquet")

symbols = sorted(preds["symbol"].unique())
c1, c2, c3 = st.columns([2, 2, 3])
symbol = c1.selectbox("Stock", symbols, index=symbols.index("RELIANCE") if "RELIANCE" in symbols else 0)
horizon = c2.radio("Horizon (trading days)", [1, 5, 21], index=2, horizontal=True)
folds = [f for f in preds["fold"].unique()]
fold = c3.selectbox("Evaluation window", ["TEST"] + sorted(f for f in folds if f != "TEST"),
                    help="TEST = locked 2020-01 → 2021-04 window (COVID crash + recovery), "
                         "evaluated exactly once after all modeling choices were frozen.")

d = preds[(preds["symbol"] == symbol) & (preds["horizon"] == horizon)
          & (preds["fold"] == fold)].sort_values("date")

# ---- price-space fan chart ----------------------------------------------------
st.subheader(f"{symbol}: realized vs predicted {horizon}-day-ahead price")
dd = ui.downsample(d, 1200).copy()
dd["p_real"] = dd["close"] * np.exp(dd["y_true"])
dd["p_hat"] = dd["close"] * np.exp(dd["y_pred"])
dd["p_lo"] = dd["close"] * np.exp(dd["conf_lo"])
dd["p_hi"] = dd["close"] * np.exp(dd["conf_hi"])
fig = go.Figure()
fig.add_trace(go.Scatter(x=dd["date"], y=dd["p_hi"], line=dict(width=0), showlegend=False))
fig.add_trace(go.Scatter(x=dd["date"], y=dd["p_lo"], line=dict(width=0), fill="tonexty",
                         fillcolor="rgba(41,128,185,0.15)", name="90% conformal band"))
fig.add_trace(go.Scatter(x=dd["date"], y=dd["p_real"], name="realized",
                         line=dict(color="#1a5276", width=1.5)))
fig.add_trace(go.Scatter(x=dd["date"], y=dd["p_hat"], name="predicted",
                         line=dict(color="#ca6f1e", width=1.2, dash="dot")))
fig.update_yaxes(title=f"price {horizon}d ahead (₹)")
st.plotly_chart(ui.base_layout(fig, height=420), use_container_width=True)
cov = float(((d["y_true"] >= d["conf_lo"]) & (d["y_true"] <= d["conf_hi"])).mean())
st.caption(
    f"The shaded band is a **volatility-scaled conformal interval** calibrated on "
    f"earlier out-of-sample errors only. Empirical coverage in this window for "
    f"{symbol}: **{cov*100:.1f}%** (target 90%)."
)

# ---- metrics vs baselines -----------------------------------------------------
st.subheader("Model vs baselines (all stocks, this window)")
blk = metrics["test" if fold == "TEST" else "folds"]
blk = blk[str(horizon)] if fold == "TEST" else blk[str(horizon)][fold]
rs, ps, dr = blk["return_space"], blk["price_space"], blk["direction"]["model"]
t1 = pd.DataFrame({
    "MAE (return)": [rs["model"]["mae"], rs["naive"]["mae"], rs["drift"]["mae"], rs["ridge"]["mae"]],
    "RMSE (return)": [rs["model"]["rmse"], rs["naive"]["rmse"], rs["drift"]["rmse"], rs["ridge"]["rmse"]],
    "R² (return)": [rs["model"]["r2"], rs["naive"]["r2"], rs["drift"]["r2"], rs["ridge"]["r2"]],
}, index=["LightGBM (calibrated)", "Naive (zero return)", "Drift", "Ridge"]).round(5)
st.dataframe(t1, use_container_width=True)
m1, m2, m3, m4, m5 = st.columns(5)
m1.metric("Price MAE", f"₹{ps['model']['mae_rupees']:.1f}",
          f"{(1 - ps['model']['mae_rupees']/ps['naive']['mae_rupees'])*100:+.1f}% vs naive",
          delta_color="normal")
m2.metric("Price R²", ui.fmt_num(ps["model"]["r2"], 4))
m3.metric("Median APE", ui.fmt_pct(ps["model"]["median_ape_pct"] / 100, 2))
m4.metric("Direction accuracy", ui.fmt_pct(dr["accuracy"]),
          f"always-up: {dr['always_up_accuracy']*100:.1f}%", delta_color="off")
m5.metric("Brier / ECE", f"{dr['brier']:.4f} / {dr['ece']:.3f}",
          "coin flip: 0.2500", delta_color="off")
st.caption(
    "Price-level R² is dominated by persistence (today's price is an excellent "
    "predictor of next week's) — skill must be read from the **return-space rows "
    "against the baselines** and the calibration quality. Direction accuracy is "
    "shown next to the degenerate always-up rule because the market rises on "
    f"{dr['base_rate_up']*100:.0f}% of {horizon}-day windows."
)

# ---- reliability + reasons -----------------------------------------------------
c1, c2 = st.columns(2)
with c1:
    st.subheader("Probability calibration")
    rel = pd.DataFrame(dr["reliability"])
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="perfect",
                             line=dict(color="#7f8c8d", dash="dash")))
    fig.add_trace(go.Scatter(x=rel["confidence"], y=rel["observed"], mode="markers+lines",
                             name="model", marker=dict(size=np.sqrt(rel["n"]).clip(4, 18)),
                             line=dict(color="#117a65")))
    fig.update_xaxes(title="predicted P(up)", range=[0, 1])
    fig.update_yaxes(title="observed frequency", range=[0, 1])
    st.plotly_chart(ui.base_layout(fig, height=360), use_container_width=True)
with c2:
    st.subheader(f"Why the model leans where it does — {symbol}")
    sl = shap_latest[(shap_latest["symbol"] == symbol) & (shap_latest["horizon"] == horizon)]
    if len(sl):
        pred = sl["prediction"].iloc[0]
        st.metric(f"Model {horizon}-day return lean (last data date)", ui.fmt_pct(pred, 2))
        for _, row in sl.sort_values("rank").head(5).iterrows():
            arrow = "🟢" if row["shap"] > 0 else "🔴"
            st.write(f"{arrow} {row['reason']}")
        st.caption("Exact TreeSHAP attributions from the locked-test model (trained through 2019).")
    else:
        st.write("No attribution available for this stock.")

# ---- where the model fails ------------------------------------------------------
st.subheader("Where the model fails (honesty panel)")
worst = d.assign(abs_err=(d["y_true"] - d["y_pred"]).abs()).nlargest(5, "abs_err")
worst_t = pd.DataFrame({
    "Date": worst["date"].dt.date,
    "Realized": (np.exp(worst["y_true"]) - 1).map(lambda v: f"{v*100:+.1f}%"),
    "Predicted": (np.exp(worst["y_pred"]) - 1).map(lambda v: f"{v*100:+.1f}%"),
    "Band covered it": np.where((worst["y_true"] >= worst["conf_lo"])
                                & (worst["y_true"] <= worst["conf_hi"]), "yes", "no"),
})
st.dataframe(worst_t, use_container_width=True, hide_index=True)
st.caption(
    "Largest misses are crash/rally days — point forecasts cannot anticipate news. "
    "The honest mitigation is the uncertainty band, which widens with volatility."
)
