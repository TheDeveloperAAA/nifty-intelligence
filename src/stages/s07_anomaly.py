"""Stage 7 — anomaly detection with built-in validation against known events."""
from __future__ import annotations

import time

import pandas as pd

from src.core.anomalies import detect_anomalies
from src.io_utils import StageCache, read_parquet, write_json, write_parquet

KNOWN_EVENTS = {
    "2004-05-17": "Election-result crash (UPA upset); NIFTY hit lower circuits",
    "2008-01-21": "Global selloff; biggest single-day fall of the 2008 cycle start",
    "2008-10-24": "Global financial crisis capitulation",
    "2009-05-18": "Election-result rally; upper circuit, trading halted",
    "2013-08-16": "Taper-tantrum rupee crisis selloff",
    "2016-11-09": "Demonetization announcement + US election",
    "2020-03-12": "COVID-19 pandemic declaration selloff",
    "2020-03-23": "COVID-19 national-lockdown crash (worst day of the sample)",
}


def run(cfg, force: bool = False) -> bool:
    art = cfg.path_for("artifacts")
    inputs = [art / "prices_adjusted.parquet", art / "market_proxy.parquet"]
    outputs = [art / "anomalies.parquet", art / "anomaly_validation.json"]
    cache = StageCache(art, "anomaly")
    chash = cfg.section_hash("anomaly", "run")
    if not force and cache.is_fresh(chash, inputs, outputs):
        return True
    t0 = time.time()

    px = read_parquet(inputs[0])
    px["date"] = pd.to_datetime(px["date"])
    proxy = read_parquet(inputs[1])
    proxy["date"] = pd.to_datetime(proxy["date"])

    events = detect_anomalies(px, proxy, dict(cfg["anomaly"]))
    # annotate known historical events (derived from the data, labeled post hoc)
    events["known_event"] = events["date"].dt.strftime("%Y-%m-%d").map(KNOWN_EVENTS)
    write_parquet(events, outputs[0])

    mkt_days = (
        events[events["type"] == "market_wide"]
        .nlargest(20, "severity")["date"].dt.strftime("%Y-%m-%d").tolist()
    )
    recovered = [d for d in KNOWN_EVENTS if any(abs(
        (pd.Timestamp(d) - pd.Timestamp(m)).days) <= 5 for m in mkt_days)]
    validation = {
        "n_events_total": int(len(events)),
        "by_type": events["type"].value_counts().to_dict(),
        "top20_market_days": mkt_days,
        "known_events_recovered": recovered,
        "known_events_expected": list(KNOWN_EVENTS),
        "gate_pass": bool(
            any(d.startswith("2008") for d in recovered)
            and any(d.startswith("2020-03") for d in recovered)
        ),
    }
    write_json(validation, outputs[1])
    if not validation["gate_pass"]:
        raise RuntimeError("anomaly gate FAILED: 2008/2020 crises not recovered")
    cache.record(chash, inputs, outputs, time.time() - t0,
                 extra={"n_events": int(len(events))})
    return False
