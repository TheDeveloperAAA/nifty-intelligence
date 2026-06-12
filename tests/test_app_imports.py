"""The app process must never import ML libraries (OpenMP quarantine).

Training happens only in short-lived CLI processes; the Streamlit app reads
precomputed artifacts. This test scans every file under app/ for forbidden
imports so the rule cannot regress silently.
"""
from __future__ import annotations

import re
from pathlib import Path

APP_DIR = Path(__file__).resolve().parents[1] / "app"
FORBIDDEN = ("lightgbm", "shap", "sklearn", "arch", "statsmodels", "torch", "scipy")

IMPORT_RE = re.compile(r"^\s*(?:import|from)\s+([a-zA-Z_][\w.]*)", re.MULTILINE)


def test_app_never_imports_ml_libraries():
    offenders = []
    for py in APP_DIR.rglob("*.py"):
        for match in IMPORT_RE.finditer(py.read_text()):
            root = match.group(1).split(".")[0]
            if root in FORBIDDEN:
                offenders.append(f"{py.name}: {match.group(0).strip()}")
    assert offenders == [], f"forbidden imports in app/: {offenders}"


def test_app_pages_exist():
    assert (APP_DIR / "Home.py").exists()
    pages = list((APP_DIR / "pages").glob("*.py"))
    assert len(pages) == 6
