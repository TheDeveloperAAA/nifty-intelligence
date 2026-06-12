"""Pipeline entrypoint.

Usage:
    python -m src.cli run all [--config config/config.yaml] [--force]
    python -m src.cli run data|features|models|volatility|portfolio|risk|anomaly|explain|report
"""
from __future__ import annotations

import argparse
import importlib
import sys
import time

from src.config import load_config

STAGES = [
    "data",
    "features",
    "models",
    "volatility",
    "portfolio",
    "risk",
    "anomaly",
    "explain",
    "report",
]

STAGE_MODULES = {
    "data": "src.stages.s01_data",
    "features": "src.stages.s02_features",
    "models": "src.stages.s03_models",
    "volatility": "src.stages.s04_volatility",
    "portfolio": "src.stages.s05_portfolio",
    "risk": "src.stages.s06_risk",
    "anomaly": "src.stages.s07_anomaly",
    "explain": "src.stages.s08_explain",
    "report": "src.stages.s09_report",
}


def run_stage(name: str, cfg, force: bool) -> None:
    module = importlib.import_module(STAGE_MODULES[name])
    t0 = time.time()
    print(f"=== stage:{name} start ===", flush=True)
    skipped = module.run(cfg, force=force)
    status = "cached" if skipped else "done"
    print(f"=== stage:{name} {status} in {time.time() - t0:.1f}s ===", flush=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="src.cli")
    sub = parser.add_subparsers(dest="cmd", required=True)
    runp = sub.add_parser("run")
    runp.add_argument("stage", choices=STAGES + ["all"])
    runp.add_argument("--config", default=None)
    runp.add_argument("--force", action="store_true")
    args = parser.parse_args(argv)

    cfg = load_config(args.config)
    stages = STAGES if args.stage == "all" else [args.stage]
    for name in stages:
        run_stage(name, cfg, force=args.force)
    return 0


if __name__ == "__main__":
    sys.exit(main())
