"""Reproducibility drill: hash the result artifacts and compare against the
committed manifest. Run after `make all` on a fresh clone.

Usage:
    python scripts/verify_repro.py --write   # record current hashes (maintainer)
    python scripts/verify_repro.py           # verify against recorded hashes
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "artifacts" / "repro_manifest.json"

# deterministic result files (parquet binary layout can vary across pyarrow
# versions, so we hash the JSON results, which capture every headline number)
TRACKED = [
    "artifacts/model_metrics.json",
    "artifacts/conformal.json",
    "artifacts/portfolio_summary.json",
    "artifacts/risk_metrics.json",
    "artifacts/var_backtest.json",
    "artifacts/data_quality.json",
    "artifacts/anomaly_validation.json",
    "artifacts/garch_params.json",
]


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--write", action="store_true")
    args = parser.parse_args()

    current = {rel: sha(ROOT / rel) for rel in TRACKED if (ROOT / rel).exists()}
    if args.write:
        MANIFEST.write_text(json.dumps(current, indent=2, sort_keys=True))
        print(f"wrote {MANIFEST} ({len(current)} files)")
        return 0

    if not MANIFEST.exists():
        print("no manifest recorded — run with --write first")
        return 1
    recorded = json.loads(MANIFEST.read_text())
    ok = True
    for rel, digest in recorded.items():
        cur = current.get(rel)
        status = "OK " if cur == digest else "DIFF" if cur else "MISSING"
        if status != "OK ":
            ok = False
        print(f"  [{status}] {rel}")
    print("REPRODUCIBLE ✔" if ok else "MISMATCH ✘ — results differ from recorded run")
    return 0 if ok else 2


if __name__ == "__main__":
    sys.exit(main())
