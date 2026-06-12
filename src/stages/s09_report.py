"""Stage 9 — figures + the 12-page PDF technical report."""
from __future__ import annotations

import time
from pathlib import Path

from src.io_utils import StageCache
from src.report.builder import build_report
from src.report.figures import make_all


def run(cfg, force: bool = False) -> bool:
    art = cfg.path_for("artifacts")
    reports = cfg.path_for("reports")
    figs_dir = reports / "figures"
    out_pdf = reports / "technical_report.pdf"
    fonts_dir = Path(cfg.root) / "assets" / "fonts"

    inputs = [art / "model_metrics.json", art / "portfolio_summary.json",
              art / "data_quality.json", art / "predictions.parquet"]
    outputs = [out_pdf]
    cache = StageCache(art, "report")
    chash = cfg.section_hash("run", "cv", "labels", "portfolio", "risk")
    if not force and cache.is_fresh(chash, inputs, outputs):
        return True
    t0 = time.time()

    figs = make_all(art, figs_dir, cfg)
    n_pages = build_report(art, figs, fonts_dir, out_pdf, cfg)
    print(f"  report: {n_pages} pages -> {out_pdf}")
    cache.record(chash, inputs, outputs, time.time() - t0, extra={"pages": n_pages})
    return False
