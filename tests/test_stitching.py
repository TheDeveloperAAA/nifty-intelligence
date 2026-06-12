"""Stitching: real raw files collapse to 49 companies with clean continuity."""
from __future__ import annotations

from pathlib import Path

import pytest

from src.config import load_config
from src.core.stitching import load_metadata, load_universe

RAW = Path(__file__).resolve().parents[1] / "data" / "raw"

pytestmark = pytest.mark.skipif(not RAW.exists(), reason="raw data not present")


@pytest.fixture(scope="module")
def cfg():
    return load_config()


@pytest.fixture(scope="module")
def panel(cfg):
    return load_universe(
        RAW,
        rename_map=dict(cfg.get("data", "rename_map")),
        drop_symbols=list(cfg.get("data", "drop_symbols")),
    )


def test_universe_has_49_companies(panel):
    assert panel["symbol"].nunique() == 49


def test_no_duplicate_symbol_dates(panel):
    assert not panel.duplicated(subset=["symbol", "Date"]).any()


def test_vedl_merge_is_monotone(panel):
    vedl = panel[panel["symbol"] == "VEDL"]
    assert vedl["Date"].is_monotonic_increasing
    assert set(vedl["symbol_raw"].unique()) == {"SESAGOA", "SSLT", "VEDL"}


def test_tatamotors_contains_telco_history(panel):
    tm = panel[panel["symbol"] == "TATAMOTORS"]
    assert "TELCO" in set(tm["symbol_raw"].unique())
    assert str(tm["Date"].min().date()) == "2000-01-03"


def test_metadata_covers_universe(panel):
    meta = load_metadata(RAW)
    missing = set(panel["symbol"].unique()) - set(meta["symbol"])
    assert missing == set(), f"symbols without sector: {missing}"
