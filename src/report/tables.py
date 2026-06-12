"""Formatted table rows for the PDF, extracted from artifacts (no hand-typed numbers)."""
from __future__ import annotations


def fmt(x, kind="num", digits=2):
    if x is None or x != x:
        return "—"
    if kind == "pct":
        return f"{x*100:.{digits}f}%"
    if kind == "num":
        return f"{x:.{digits}f}"
    return str(x)


def headline_model_table(metrics: dict) -> list[list[str]]:
    rows = [["Horizon", "Window", "relMAE vs naive", "Dir. acc", "Always-up",
             "Brier", "ECE", "90% coverage"]]
    for h in ("1", "5", "21"):
        for name, blk in (("CV pooled", metrics["pooled_cv"][h]), ("TEST", metrics["test"][h])):
            rs, dr = blk["return_space"], blk["direction"]["model"]
            rows.append([
                f"{h}d", name, fmt(rs["rel_mae_vs_naive"], digits=4),
                fmt(dr["accuracy"], "pct", 1), fmt(dr["always_up_accuracy"], "pct", 1),
                fmt(dr["brier"], digits=4), fmt(dr["ece"], digits=3),
                fmt(blk["conformal_coverage"], "pct", 1),
            ])
    return rows


def baseline_table(metrics: dict, h: str, window: str = "test") -> list[list[str]]:
    blk = metrics["test"][h] if window == "test" else metrics["pooled_cv"][h]
    rs, ps = blk["return_space"], blk["price_space"]
    price_key = {"model": "model", "naive": "naive"}
    rows = [["Forecaster", "MAE (ret)", "RMSE (ret)", "R² (ret)", "Price MAE ₹", "Price R²"]]
    for label, key in [("LightGBM (calibrated)", "model"), ("LightGBM (raw)", "model_raw"),
                       ("Naive (zero return)", "naive"), ("Drift", "drift"), ("Ridge", "ridge")]:
        r = rs[key]
        p = ps.get(price_key.get(key, ""))
        rows.append([
            label, fmt(r["mae"], digits=5), fmt(r["rmse"], digits=5), fmt(r["r2"], digits=4),
            fmt(p["mae_rupees"], digits=1) if p else "—",
            fmt(p["r2"], digits=4) if p else "—",
        ])
    return rows


def portfolio_table(summary: dict) -> list[list[str]]:
    rows = [["Strategy", "CAGR", "Vol", "Sharpe", "Sortino", "MaxDD", "Calmar",
             "VaR95(d)", "Beta", "Turnover/yr"]]
    order = ["conservative", "balanced", "aggressive", "hrp", "benchmark_ew"]
    for k in order:
        v = summary["profiles"][k]
        rows.append([
            k.replace("benchmark_ew", "equal-weight bench"),
            fmt(v["cagr"], "pct", 1), fmt(v["ann_vol"], "pct", 1), fmt(v["sharpe"]),
            fmt(v["sortino"]), fmt(v["max_drawdown"], "pct", 1), fmt(v["calmar"]),
            fmt(v["var_95"], "pct", 2), fmt(v.get("beta")), f"{v['annual_turnover']:.1f}x",
        ])
    return rows


def kupiec_table(var_bt: dict) -> list[list[str]]:
    rows = [["Strategy", "Days", "Exceptions", "Expected", "Rate", "Kupiec p", "Verdict"]]
    for k, v in var_bt.items():
        rows.append([
            k.replace("benchmark_ew", "equal-weight"), str(v["n"]), str(v["exceptions"]),
            f"{v['expected_exceptions']:.0f}", fmt(v["exception_rate"], "pct", 2),
            fmt(v["p_value"], digits=3),
            "calibrated" if v["p_value"] > 0.05 else "rejected",
        ])
    return rows


def garch_table(garch: dict) -> list[list[str]]:
    rows = [["Series", "EWMA QLIKE", "GARCH QLIKE", "EWMA MZ-R²", "GARCH MZ-R²"]]
    for k, v in garch["showcase"].items():
        if "garch" not in v:
            continue
        rows.append([
            k.replace("_MARKET_", "market proxy"),
            fmt(v["ewma"]["qlike"], digits=3), fmt(v["garch"]["qlike"], digits=3),
            fmt(v["ewma"]["mz_r2"], digits=3), fmt(v["garch"]["mz_r2"], digits=3),
        ])
    return rows


def quality_table(quality: dict) -> list[list[str]]:
    return [
        ["Check", "Value"],
        ["Rows / companies", f"{quality['n_rows']:,} / {quality['n_symbols']}"],
        ["Span", f"{quality['date_min']} → {quality['date_max']}"],
        ["Corporate actions detected", str(quality["events_total"])],
        ["… large split/bonus", str(quality["events_by_kind"].get("split_bonus_large", 0))],
        ["… moderate bonus (4-condition rule)", str(quality["events_by_kind"].get("split_bonus_moderate", 0))],
        ["… exchange-published (rights)", str(quality["events_by_kind"].get("exchange_prevclose", 0))],
        ["Residual fake cliffs (gate)", f"{quality['residual_down_gt30pct']} (must be 0)"],
        ["Genuine >+30% rallies kept", str(len(quality["up_gt30pct_days"]))],
    ]
