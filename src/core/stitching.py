"""Load per-stock CSVs and stitch historical ticker renames to current symbols.

Each per-stock CSV already contains the company's full history including rows
recorded under former tickers (verified: TATAMOTORS.csv contains TELCO rows,
VEDL.csv contains SESAGOA and SSLT). Stitching therefore only normalizes the
Symbol column and asserts continuity.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

PRICE_COLUMNS = [
    "Prev Close",
    "Open",
    "High",
    "Low",
    "Last",
    "Close",
    "VWAP",
]

NUMERIC_COLUMNS = PRICE_COLUMNS + [
    "Volume",
    "Turnover",
    "Trades",
    "Deliverable Volume",
    "%Deliverble",
]


def load_metadata(raw_dir: Path) -> pd.DataFrame:
    """Sector map keyed by current symbol; fixes M&M -> MM file naming."""
    meta = pd.read_csv(raw_dir / "stock_metadata.csv")
    meta = meta.rename(
        columns={"Company Name": "company_name", "Industry": "sector", "Symbol": "symbol"}
    )[["company_name", "sector", "symbol"]]
    meta["symbol"] = meta["symbol"].replace({"M&M": "MM"})
    meta = meta[meta["symbol"] != "INFRATEL"].reset_index(drop=True)
    return meta


def load_stock_csv(path: Path, rename_map: dict[str, str]) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["Date"])
    if df.empty:
        return df
    symbol = path.stem  # file name is the current ticker (M&M lives in MM.csv)
    df["symbol_raw"] = df["Symbol"]
    df["symbol"] = symbol
    # Sanity: every raw symbol in the file must be the ticker itself or a known
    # former name mapping to it.
    mapped = df["Symbol"].map(lambda s: rename_map.get(s, s)).unique()
    unexpected = [m for m in mapped if m not in (symbol, symbol.replace("MM", "M&M"))]
    # M&M quirk: file MM.csv carries Symbol "M&M"
    if symbol == "MM":
        unexpected = [m for m in mapped if m != "M&M"]
    if unexpected:
        raise ValueError(f"{path.name}: unexpected symbols {unexpected}")
    return df


def load_universe(
    raw_dir: Path,
    rename_map: dict[str, str],
    drop_symbols: list[str],
    symbols: list[str] | None = None,
) -> pd.DataFrame:
    """Long panel of all companies, Symbol normalized to current ticker."""
    frames = []
    for path in sorted(raw_dir.glob("*.csv")):
        if path.stem in ("stock_metadata", "NIFTY50_all") or path.stem in drop_symbols:
            continue
        if symbols is not None and path.stem not in symbols:
            continue
        df = load_stock_csv(path, rename_map)
        if df.empty:
            continue
        frames.append(df)
    panel = pd.concat(frames, ignore_index=True)
    panel = panel.sort_values(["symbol", "Date"]).reset_index(drop=True)

    dup = panel.duplicated(subset=["symbol", "Date"])
    if dup.any():
        raise ValueError(f"{dup.sum()} duplicate (symbol, Date) rows after stitching")

    for col in NUMERIC_COLUMNS:
        if col in panel.columns:
            panel[col] = pd.to_numeric(panel[col], errors="coerce")
    return panel
