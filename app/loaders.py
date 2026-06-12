"""Single artifact gateway for the app. Only pandas/numpy/yaml/streamlit here —
the app process never imports lightgbm/shap/sklearn/arch (test-enforced)."""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
ART = ROOT / "artifacts"


def _mtime(path: Path) -> float:
    return path.stat().st_mtime if path.exists() else -1.0


@st.cache_data(show_spinner=False)
def load_parquet(name: str, _mt: float = 0.0) -> pd.DataFrame:
    df = pd.read_parquet(ART / name)
    for c in ("date",):
        if c in df.columns:
            df[c] = pd.to_datetime(df[c])
    return df


@st.cache_data(show_spinner=False)
def load_json(name: str, _mt: float = 0.0) -> dict:
    with open(ART / name) as fh:
        return json.load(fh)


@st.cache_data(show_spinner=False)
def load_csv(name: str, _mt: float = 0.0) -> pd.DataFrame:
    df = pd.read_csv(ART / name)
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"])
    return df


def parquet(name: str) -> pd.DataFrame:
    return load_parquet(name, _mtime(ART / name))


def jsn(name: str) -> dict:
    return load_json(name, _mtime(ART / name))


def csv(name: str) -> pd.DataFrame:
    return load_csv(name, _mtime(ART / name))


def artifacts_ready() -> bool:
    return (ART / "prices_adjusted.parquet").exists()
