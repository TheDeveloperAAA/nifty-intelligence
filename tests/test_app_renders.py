"""Every dashboard page must render without exceptions against real artifacts."""
from __future__ import annotations

from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
ART = ROOT / "artifacts"

pytestmark = pytest.mark.skipif(
    not (ART / "predictions.parquet").exists(),
    reason="artifacts not built (run `make all`)",
)

PAGES = [
    "app/Home.py",
    "app/pages/1_Stock_Explorer.py",
    "app/pages/2_Prediction_Lab.py",
    "app/pages/3_Portfolio_Builder.py",
    "app/pages/4_Risk_Dashboard.py",
    "app/pages/5_Anomalies_and_Regimes.py",
    "app/pages/6_Methodology_and_Model_Card.py",
]


@pytest.mark.parametrize("page", PAGES)
def test_page_renders_without_exception(page):
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_file(str(ROOT / page), default_timeout=60)
    at.run()
    assert not at.exception, f"{page} raised: {[e.value for e in at.exception]}"
