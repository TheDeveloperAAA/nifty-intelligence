"""Stage 1 — data: stitch renames, back-adjust corporate actions, build proxies.

Outputs
-------
artifacts/prices_adjusted.parquet   canonical long panel (adjusted OHLCV + returns)
artifacts/market_proxy.parquet      EW market + sector index levels/returns
artifacts/corporate_actions.csv     detected events with evidence columns
artifacts/event_register.csv        every remaining |ret|>20% day, classified
artifacts/data_quality.json         gate results and dataset stats
"""
from __future__ import annotations

import time

import numpy as np
import pandas as pd

from src.core.adjustments import adjust_universe
from src.core.stitching import load_metadata, load_universe
from src.io_utils import StageCache, write_json, write_parquet

OUT_COLUMNS = {
    "Date": "date",
    "symbol": "symbol",
    "Prev Close": "prev_close",
    "Open": "open",
    "High": "high",
    "Low": "low",
    "Last": "last",
    "Close": "close",
    "VWAP": "vwap",
    "Volume": "volume",
    "Turnover": "turnover",
    "Trades": "trades",
    "Deliverable Volume": "deliv_volume",
    "%Deliverble": "pct_deliv",
    "adj_factor": "adj_factor",
    "close_raw": "close_raw",
    "symbol_raw": "symbol_raw",
}


def run(cfg, force: bool = False) -> bool:
    art = cfg.path_for("artifacts")
    raw = cfg.path_for("raw")
    outputs = [
        art / "prices_adjusted.parquet",
        art / "market_proxy.parquet",
        art / "corporate_actions.csv",
        art / "event_register.csv",
        art / "data_quality.json",
    ]
    inputs = sorted(raw.glob("*.csv"))
    cache = StageCache(art, "data")
    chash = cfg.section_hash("data", "run")
    if not force and cache.is_fresh(chash, inputs, outputs):
        return True
    t0 = time.time()

    meta = load_metadata(raw)
    panel = load_universe(
        raw,
        rename_map=dict(cfg.get("data", "rename_map") or {}),
        drop_symbols=list(cfg.get("data", "drop_symbols") or []),
        symbols=cfg.get("data", "symbols"),
    )
    start, end = pd.Timestamp(cfg["data"]["start"]), pd.Timestamp(cfg["data"]["end"])
    panel = panel[(panel["Date"] >= start) & (panel["Date"] <= end)]

    adjusted, events = adjust_universe(panel, dict(cfg.get("data", "adjust")))

    out = adjusted.rename(columns=OUT_COLUMNS)[list(OUT_COLUMNS.values())]
    out = out.merge(meta[["symbol", "sector", "company_name"]], on="symbol", how="left")
    out = out.sort_values(["symbol", "date"]).reset_index(drop=True)
    g = out.groupby("symbol", sort=False)
    out["ret"] = np.log(out["close"]) - np.log(g["close"].shift(1))
    out["simple_ret"] = g["close"].pct_change()

    # --- market & sector proxies (equal-weight of adjusted simple returns) ---
    mkt = out.groupby("date")["simple_ret"].mean().rename("mkt_ret").to_frame()
    mkt["breadth"] = out.groupby("date")["simple_ret"].count()
    mkt["mkt_level"] = (1 + mkt["mkt_ret"].fillna(0)).cumprod()
    sec = (
        out.dropna(subset=["sector"])
        .groupby(["date", "sector"])["simple_ret"]
        .mean()
        .unstack()
        .add_prefix("sec_ret_")
    )
    proxy = mkt.join(sec).reset_index()

    # --- event register: classify every remaining |ret| > 20% day -----------
    big = out[out["ret"].abs() > np.log(1.20)][
        ["date", "symbol", "ret", "close_raw", "close"]
    ].copy()
    mkt_ret_map = mkt["mkt_ret"]
    big["market_ret"] = big["date"].map(mkt_ret_map)
    big["classification"] = np.where(
        big["market_ret"].abs() > 0.04, "market_stress", "idiosyncratic"
    )
    big["direction"] = np.where(big["ret"] > 0, "up", "down")
    big = big.sort_values("ret")

    # --- gate ----------------------------------------------------------------
    resid_down = big[(big["ret"] < np.log(0.70))]
    up_30 = big[big["ret"] > np.log(1.30)]
    quality = {
        "n_rows": int(len(out)),
        "n_symbols": int(out["symbol"].nunique()),
        "date_min": str(out["date"].min().date()),
        "date_max": str(out["date"].max().date()),
        "events_total": int(len(events)),
        "events_by_kind": {} if events.empty else events["kind"].value_counts().to_dict(),
        "residual_down_gt30pct": int(len(resid_down)),
        "residual_down_days": resid_down[["date", "symbol", "ret"]].astype(str).to_dict("records"),
        "up_gt30pct_days": up_30[["date", "symbol", "ret"]].astype(str).to_dict("records"),
        "max_abs_ret_post_adjustment": float(out["ret"].abs().max()),
        "n_days_abs_ret_gt20pct": int(len(big)),
        "missing_trades_rows": int(out["trades"].isna().sum()),
        "missing_deliv_rows": int(out["pct_deliv"].isna().sum()),
        "gate_pass": bool(len(resid_down) == 0),
    }

    write_parquet(out, outputs[0])
    write_parquet(proxy, outputs[1])
    outputs[2].parent.mkdir(parents=True, exist_ok=True)
    events.to_csv(outputs[2], index=False)
    big.to_csv(outputs[3], index=False)
    write_json(quality, outputs[4])

    if not quality["gate_pass"]:
        raise RuntimeError(
            f"data gate FAILED: {quality['residual_down_gt30pct']} residual "
            f"downward >30% days — see {outputs[3]}"
        )
    cache.record(chash, inputs, outputs, time.time() - t0,
                 extra={"n_rows": quality["n_rows"], "events": quality["events_total"]})
    return False
