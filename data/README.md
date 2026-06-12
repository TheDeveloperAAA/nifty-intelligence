# Data provenance

**Source:** [NIFTY-50 Stock Market Data (Kaggle, rohanrao/nifty50-stock-market-data)](https://www.kaggle.com/datasets/rohanrao/nifty50-stock-market-data) — the dataset designated by the Cult Open Projects 2026 problem statement.

**License:** CC0 (public domain) — redistribution permitted, which is why the raw
CSVs are committed here so that `git clone` is fully self-contained for judges.

**Span:** 2000-01-03 → 2021-04-30, daily OHLCV + VWAP/Turnover/Trades/Deliverable
volume for companies in the NIFTY-50 index as of April 2021.

**Contents of `raw/`:** the 49 per-stock CSVs plus `stock_metadata.csv`
(sector/industry mapping). Each per-stock file contains the company's complete
history *including rows under its former ticker symbols* (e.g. `TATAMOTORS.csv`
contains TELCO rows; `VEDL.csv` contains SESAGOA and SSLT rows).

**Intentionally omitted:**
- `NIFTY50_all.csv` (27 MB) — a pure concatenation of the per-stock files; zero
  additional information. The pipeline ignores it if present.
- `INFRATEL.csv` — header-only (empty) in the published dataset; Bharti Infratel
  is excluded from the universe (49 usable companies).

**Integrity:** `checksums.sha256` holds SHA-256 digests of every committed file.
Verify with:

```bash
cd data/raw && shasum -a 256 -c ../checksums.sha256
```

**Optional re-download:** `python scripts/get_data.py` fetches the dataset via the
Kaggle CLI (requires `~/.kaggle/kaggle.json`) and verifies it against the same
checksums. This is provenance tooling, not a required step.

**Known quirks handled by the pipeline (stage `data`):**
- Prices are **not adjusted** for splits/bonus issues — corporate actions are
  detected from price ratios and back-adjusted (see `src/core/adjustments.py`).
- 16 historical ticker renames are stitched to current symbols.
- `Trades` is missing before ~2011; `Deliverable Volume`/`%Deliverble` missing in
  early years — left as NaN (LightGBM handles natively).
- `stock_metadata.csv` uses symbol `M&M`, whose price file is `MM.csv`.
