"""All report figures, rendered with matplotlib (Agg) from artifacts only."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

BLUE, ORANGE, GREEN, RED, GRAY, PURPLE = (
    "#1a5276", "#ca6f1e", "#117a65", "#c0392b", "#7f8c8d", "#8e44ad",
)
PROFILE_COLORS = {"conservative": "#2e86c1", "balanced": GREEN, "aggressive": ORANGE,
                  "benchmark_ew": GRAY, "hrp": PURPLE}

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 8.5, "axes.titlesize": 9.5,
    "axes.labelsize": 8.5, "axes.grid": True, "grid.alpha": 0.25,
    "axes.spines.top": False, "axes.spines.right": False,
    "figure.dpi": 200, "savefig.bbox": "tight", "savefig.dpi": 200,
})

FULL = (7.4, 3.0)
HALF = (3.6, 2.7)
TALL = (7.4, 3.8)


def _save(fig, out: Path, name: str) -> Path:
    path = out / name
    fig.savefig(path)
    plt.close(fig)
    return path


def _shade_regimes(ax, regimes: pd.DataFrame):
    reg = regimes[["date", "regime"]].dropna().reset_index(drop=True)
    start, cur = None, None
    for _, row in reg.iterrows():
        if row["regime"] != cur:
            if cur == 2 and start is not None:
                ax.axvspan(start, row["date"], color=RED, alpha=0.10, lw=0)
            if cur == 0 and start is not None:
                ax.axvspan(start, row["date"], color=GREEN, alpha=0.08, lw=0)
            start, cur = row["date"], row["regime"]
    if cur in (0, 2) and start is not None:
        ax.axvspan(start, reg["date"].iloc[-1],
                   color=RED if cur == 2 else GREEN, alpha=0.10 if cur == 2 else 0.08, lw=0)


def make_all(art: Path, out: Path, cfg) -> dict[str, Path]:
    out.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}

    px = pd.read_parquet(art / "prices_adjusted.parquet")
    px["date"] = pd.to_datetime(px["date"])
    proxy = pd.read_parquet(art / "market_proxy.parquet")
    proxy["date"] = pd.to_datetime(proxy["date"])
    regimes = pd.read_parquet(art / "regimes.parquet")
    regimes["date"] = pd.to_datetime(regimes["date"])
    preds = pd.read_parquet(art / "predictions.parquet")
    preds["date"] = pd.to_datetime(preds["date"])
    metrics = json.loads((art / "model_metrics.json").read_text())
    curves = pd.read_parquet(art / "backtest_curves.parquet")
    curves["date"] = pd.to_datetime(curves["date"])
    dds = pd.read_parquet(art / "drawdowns.parquet")
    dds["date"] = pd.to_datetime(dds["date"])
    frontier = json.loads((art / "frontier.json").read_text())
    summary = json.loads((art / "portfolio_summary.json").read_text())
    shap_global = pd.read_csv(art / "shap_global.csv")
    anoms = pd.read_parquet(art / "anomalies.parquet")
    anoms["date"] = pd.to_datetime(anoms["date"])
    vol = pd.read_parquet(art / "vol_forecasts.parquet")
    vol["date"] = pd.to_datetime(vol["date"])

    # 1 -- market with regimes ------------------------------------------------
    fig, ax = plt.subplots(figsize=FULL)
    d = proxy.dropna(subset=["mkt_level"])
    ax.plot(d["date"], d["mkt_level"], color=BLUE, lw=1.0)
    _shade_regimes(ax, regimes)
    ax.set_yscale("log")
    ax.set_title("Equal-weight NIFTY-50 proxy, 2000–2021 (log scale; red = Turbulent regime, green = Calm)")
    ax.set_ylabel("index level")
    paths["market"] = _save(fig, out, "fig01_market_regimes.png")

    # 2 -- adjustment evidence -------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 2.6))
    for ax, sym, evd in [(axes[0], "ITC", "2005-09-21"), (axes[1], "VEDL", "2008-08-08")]:
        s = px[px["symbol"] == sym]
        win = s[(s["date"] >= pd.Timestamp(evd) - pd.DateOffset(months=10))
                & (s["date"] <= pd.Timestamp(evd) + pd.DateOffset(months=10))]
        ax.plot(win["date"], win["close_raw"], color=RED, lw=1.0, label="raw NSE close")
        ax.plot(win["date"], win["close"], color=BLUE, lw=1.0, label="back-adjusted")
        ax.axvline(pd.Timestamp(evd), color=GRAY, ls="--", lw=0.8)
        ax.set_title(f"{sym}: corporate action {evd}")
        ax.legend(fontsize=7)
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=4))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    paths["adjustment"] = _save(fig, out, "fig02_adjustment_evidence.png")

    # 3 -- sector x year heatmap ----------------------------------------------
    sec_cols = [c for c in proxy.columns if c.startswith("sec_ret_")]
    sec = proxy.set_index("date")[sec_cols]
    yearly = ((1 + sec).groupby(sec.index.year).prod() - 1) * 100
    yearly.columns = [c.removeprefix("sec_ret_").replace(" & ", "&")[:14] for c in yearly.columns]
    fig, ax = plt.subplots(figsize=(7.4, 3.2))
    im = ax.imshow(yearly.T, cmap="RdYlGn", vmin=-60, vmax=60, aspect="auto")
    ax.set_xticks(range(len(yearly.index)), yearly.index, rotation=90, fontsize=6.5)
    ax.set_yticks(range(len(yearly.columns)), yearly.columns, fontsize=6.5)
    ax.grid(False)
    fig.colorbar(im, ax=ax, label="annual return (%)", fraction=0.025)
    ax.set_title("Sector returns by calendar year (%)")
    paths["sector_heatmap"] = _save(fig, out, "fig03_sector_heatmap.png")

    # 4 -- walk-forward diagram -------------------------------------------------
    folds_cfg = cfg["cv"]["folds"] + [cfg["cv"]["test"]]
    fig, ax = plt.subplots(figsize=(7.4, 2.2))
    for i, f in enumerate(folds_cfg):
        t0 = pd.Timestamp(cfg["cv"]["feature_start"])
        te = pd.Timestamp(f["train_end"])
        vs, ve = pd.Timestamp(f["val_start"]), pd.Timestamp(f["val_end"])
        ax.barh(i, (te - t0).days, left=t0, height=0.62, color=BLUE, alpha=0.75)
        ax.barh(i, (ve - vs).days, left=vs, height=0.62,
                color=ORANGE if f["name"] != "TEST" else RED, alpha=0.9)
    ax.set_yticks(range(len(folds_cfg)), [f["name"] for f in folds_cfg], fontsize=7)
    ax.invert_yaxis()
    ax.set_title("Expanding walk-forward design — blue: train, orange: validation, red: locked test "
                 "(purge = horizon + 5d embargo at each boundary)")
    ax.grid(axis="y", visible=False)
    paths["walkforward"] = _save(fig, out, "fig04_walkforward.png")

    # 5 -- prediction vs realized fan (TEST, h=21, RELIANCE) ---------------------
    d = preds[(preds["symbol"] == "RELIANCE") & (preds["horizon"] == 21)
              & (preds["fold"] == "TEST")].sort_values("date")
    fig, ax = plt.subplots(figsize=FULL)
    p_real = d["close"] * np.exp(d["y_true"])
    p_hat = d["close"] * np.exp(d["y_pred"])
    lo = d["close"] * np.exp(d["conf_lo"])
    hi = d["close"] * np.exp(d["conf_hi"])
    ax.fill_between(d["date"], lo, hi, color=BLUE, alpha=0.15, label="90% conformal band")
    ax.plot(d["date"], p_real, color=BLUE, lw=1.0, label="realized")
    ax.plot(d["date"], p_hat, color=ORANGE, lw=0.9, ls="--", label="predicted")
    ax.set_title("RELIANCE — 21-day-ahead price, locked test window (COVID crash + recovery)")
    ax.set_ylabel("price (₹)")
    ax.legend(fontsize=7)
    cov = ((d["y_true"] >= d["conf_lo"]) & (d["y_true"] <= d["conf_hi"])).mean()
    ax.text(0.02, 0.05, f"band coverage: {cov*100:.1f}% (target 90%)",
            transform=ax.transAxes, fontsize=7.5,
            bbox=dict(facecolor="white", alpha=0.8, edgecolor=GRAY))
    paths["prediction_fan"] = _save(fig, out, "fig05_prediction_fan.png")

    # 6 -- bias-variance: learning + complexity curves (skipped in smoke runs) ----
    if metrics["learning_curve"] and metrics["complexity_curve"]:
        fig, axes = plt.subplots(1, 2, figsize=(7.4, 2.5))
        lc = pd.DataFrame(metrics["learning_curve"])
        axes[0].plot(lc["n_train"] / 1000, lc["train_mae"], "o-", color=BLUE, label="train MAE", ms=3)
        axes[0].plot(lc["n_train"] / 1000, lc["val_mae"], "o-", color=ORANGE, label="validation MAE", ms=3)
        axes[0].set_xlabel("training rows (thousands)")
        axes[0].set_title("Learning curve (F3, h=5)")
        axes[0].legend(fontsize=7)
        cc = pd.DataFrame(metrics["complexity_curve"])
        axes[1].plot(cc["num_leaves"], cc["train_mae"], "o-", color=BLUE, label="train MAE", ms=3)
        axes[1].plot(cc["num_leaves"], cc["val_mae"], "o-", color=ORANGE, label="validation MAE", ms=3)
        axes[1].set_xscale("log", base=2)
        axes[1].set_xlabel("num_leaves (model capacity)")
        axes[1].set_title("Complexity curve (F3, h=5)")
        axes[1].legend(fontsize=7)
        paths["bias_variance"] = _save(fig, out, "fig06_bias_variance.png")

    # 7 -- reliability + coverage ---------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 2.7))
    for h, color in [(1, BLUE), (5, GREEN), (21, ORANGE)]:
        rel = pd.DataFrame(metrics["test"][str(h)]["direction"]["model"]["reliability"])
        rel = rel[rel["n"] >= 0.005 * rel["n"].sum()]  # drop near-empty bins
        sizes = 2 + 10 * np.sqrt(rel["n"] / rel["n"].max())
        axes[0].plot(rel["confidence"], rel["observed"], "-", lw=1, color=color,
                     label=f"h={h}")
        axes[0].scatter(rel["confidence"], rel["observed"], s=sizes**1.5, color=color)
    axes[0].plot([0.3, 0.8], [0.3, 0.8], ls="--", color=GRAY, lw=0.8)
    axes[0].set_xlabel("predicted P(up)")
    axes[0].set_ylabel("observed frequency")
    axes[0].set_title("Direction probability calibration (TEST)")
    axes[0].legend(fontsize=7)
    rows = []
    for h in (1, 5, 21):
        for fold, blk in metrics["folds"][str(h)].items():
            rows.append({"h": h, "fold": fold, "cov": blk["conformal_coverage"]})
        rows.append({"h": h, "fold": "TEST", "cov": metrics["test"][str(h)]["conformal_coverage"]})
    cov_df = pd.DataFrame(rows)
    x = np.arange(cov_df["fold"].nunique())
    width = 0.27
    for i, h in enumerate((1, 5, 21)):
        sub = cov_df[cov_df["h"] == h]
        axes[1].bar(x + (i - 1) * width, sub["cov"] * 100, width,
                    color=[BLUE, GREEN, ORANGE][i], label=f"h={h}")
    axes[1].axhline(90, ls="--", color=GRAY, lw=0.8)
    axes[1].set_xticks(x, cov_df["fold"].unique(), fontsize=7)
    axes[1].set_ylim(75, 100)
    axes[1].set_title("Conformal 90%-band empirical coverage")
    axes[1].legend(fontsize=7)
    paths["calibration"] = _save(fig, out, "fig07_calibration_coverage.png")

    # 8 -- volatility forecasting -----------------------------------------------------
    fig, ax = plt.subplots(figsize=FULL)
    v = vol[vol["symbol"] == "RELIANCE"].sort_values("date")
    realized = px[px["symbol"] == "RELIANCE"].set_index("date")["ret"].rolling(21).std() * np.sqrt(252)
    ax.plot(realized.index, realized * 100, color=GRAY, lw=0.8, label="realized vol (21d, fwd-looking proxy)")
    ax.plot(v["date"], v["sigma_ann"] * 100, color=PURPLE, lw=0.9, label="EWMA forecast (λ=0.94)")
    ax.set_title("RELIANCE — volatility is the predictable dimension")
    ax.set_ylabel("annualized vol (%)")
    ax.legend(fontsize=7)
    paths["volatility"] = _save(fig, out, "fig08_volatility.png")

    # 9 -- frontier + weights -----------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.0))
    fr = pd.DataFrame(frontier["frontier"])
    stocks = pd.DataFrame(frontier["stocks"])
    if len(stocks):
        axes[0].scatter(stocks["vol"] * 100, stocks["ret"] * 100, s=6, color=GRAY, alpha=0.6)
    if len(fr):
        axes[0].plot(fr["vol"] * 100, fr["ret"] * 100, color=BLUE, lw=1.4, label="efficient frontier")
    for name, pt in frontier["portfolios"].items():
        axes[0].scatter(pt["vol"] * 100, pt["ret"] * 100, marker="*", s=90,
                        color=PROFILE_COLORS.get(name, "k"), zorder=5)
        axes[0].annotate(name.replace("benchmark_ew", "equal wt"), (pt["vol"] * 100, pt["ret"] * 100),
                         fontsize=6.5, xytext=(4, -8), textcoords="offset points")
    axes[0].set_xlabel("volatility (% ann.)")
    axes[0].set_ylabel("expected return (% ann.)")
    axes[0].set_title(f"Efficient frontier as of {frontier['as_of']}")
    w = pd.Series(summary["final_weights"]["balanced"]["weights"]).sort_values()
    axes[1].barh(w.index, w.values * 100, color=GREEN, alpha=0.85)
    axes[1].set_title("Balanced profile — final allocation (%)")
    axes[1].tick_params(axis="y", labelsize=6.5)
    paths["frontier"] = _save(fig, out, "fig09_frontier_weights.png")

    # 10 -- backtest + drawdown -----------------------------------------------------------
    fig, axes = plt.subplots(2, 1, figsize=(7.4, 4.6), sharex=True,
                             gridspec_kw={"height_ratios": [2.1, 1.0]})
    for prof, d in curves.groupby("profile"):
        d = d.sort_values("date")
        axes[0].plot(d["date"], d["wealth"], lw=1.0, color=PROFILE_COLORS.get(prof),
                     label=prof.replace("benchmark_ew", "equal-weight benchmark"))
    axes[0].set_yscale("log")
    axes[0].set_ylabel("growth of ₹1 (log)")
    axes[0].legend(fontsize=7, ncol=3)
    axes[0].set_title("Walk-forward backtest, monthly rebalance, 10 bps/side costs (2007–2021)")
    for prof in ("conservative", "benchmark_ew"):
        d = dds[dds["entity"] == prof].sort_values("date")
        axes[1].plot(d["date"], d["drawdown"] * 100, lw=0.9, color=PROFILE_COLORS.get(prof),
                     label=prof.replace("benchmark_ew", "equal-weight benchmark"))
    axes[1].fill_between(dds[dds["entity"] == "benchmark_ew"].sort_values("date")["date"],
                         dds[dds["entity"] == "benchmark_ew"].sort_values("date")["drawdown"] * 100,
                         0, color=GRAY, alpha=0.15)
    axes[1].set_ylabel("drawdown (%)")
    axes[1].legend(fontsize=7)
    paths["backtest"] = _save(fig, out, "fig10_backtest.png")

    # 11 -- SHAP global -----------------------------------------------------------------
    g = shap_global[shap_global["horizon"] == 21].nlargest(14, "mean_abs_shap")
    fig, ax = plt.subplots(figsize=(7.4, 2.8))
    ax.barh(g["label"][::-1], g["mean_abs_shap"][::-1] * 100, color=GREEN, alpha=0.9)
    ax.set_xlabel("mean |SHAP| (percentage points of 21-day return)")
    ax.set_title("What drives the 21-day forecasts (exact TreeSHAP, 20k-row stratified sample)")
    ax.tick_params(axis="y", labelsize=7)
    paths["shap"] = _save(fig, out, "fig11_shap_global.png")

    # 12 -- anomaly timeline ---------------------------------------------------------------
    fig, ax = plt.subplots(figsize=FULL)
    mw = anoms[anoms["type"] == "market_wide"]
    rs = anoms[anoms["type"] == "return_shock"].nlargest(600, "severity")
    ax.scatter(rs["date"], rs["severity"], s=5, color=ORANGE, alpha=0.45, label="stock return shocks")
    ax.scatter(mw["date"], mw["severity"], s=16, color=RED, alpha=0.85, label="market-wide shocks")
    for d_, label in [("2004-05-17", "2004 election"), ("2008-10-24", "GFC"),
                      ("2013-08-16", "taper tantrum"), ("2016-11-09", "demonetization"),
                      ("2020-03-23", "COVID-19")]:
        ax.annotate(label, (pd.Timestamp(d_), mw[mw["date"] == d_]["severity"].max()
                            if len(mw[mw["date"] == d_]) else 8),
                    fontsize=6.5, rotation=0, xytext=(2, 6), textcoords="offset points")
    _shade_regimes(ax, regimes)
    ax.set_ylabel("robust z severity")
    ax.set_title("Anomaly register — known crises re-discovered from the data alone")
    ax.legend(fontsize=7, loc="upper left")
    paths["anomalies"] = _save(fig, out, "fig12_anomalies.png")

    return paths
