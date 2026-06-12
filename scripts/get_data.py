"""Optional provenance tool: re-download the dataset from Kaggle and verify it.

The repo already commits the raw CSVs (CC0 license), so judges never need this.
Requires the kaggle CLI and ~/.kaggle/kaggle.json.

Usage: python scripts/get_data.py [--dest data/raw]
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATASET = "rohanrao/nifty50-stock-market-data"
EXCLUDE = {"NIFTY50_all.csv", "INFRATEL.csv"}  # redundant concat / empty file


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dest", default=str(ROOT / "data" / "raw"))
    args = parser.parse_args()
    dest = Path(args.dest)
    dest.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory() as tmp:
        try:
            subprocess.run(
                ["kaggle", "datasets", "download", "-d", DATASET, "-p", tmp, "--unzip"],
                check=True,
            )
        except FileNotFoundError:
            print("kaggle CLI not found — `pip install kaggle` and add ~/.kaggle/kaggle.json")
            return 1
        for csv in sorted(Path(tmp).glob("*.csv")):
            if csv.name in EXCLUDE:
                continue
            shutil.copy2(csv, dest / csv.name)
            print(f"  {csv.name}")

    checksums = ROOT / "data" / "checksums.sha256"
    if checksums.exists():
        print("verifying checksums…")
        res = subprocess.run(["shasum", "-a", "256", "-c", str(checksums)],
                             cwd=dest, capture_output=True, text=True)
        print(res.stdout[-500:])
        if res.returncode != 0:
            print("CHECKSUM MISMATCH — upstream dataset changed?")
            return 2
    print("done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
