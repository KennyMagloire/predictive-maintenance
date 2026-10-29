"""
System tests: the whole dashboard, run headlessly with Streamlit's AppTest.

These exercise every layer together: CSV -> pipeline -> models -> engine -> screen.
(File-upload is covered by the manual tests T2/T3 with screenshots, because
AppTest cannot simulate the upload widget.)
"""

from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

DASHBOARD = str(Path(__file__).resolve().parent.parent / "src" / "dashboard.py")


@pytest.fixture(scope="module")
def app():
    at = AppTest.from_file(DASHBOARD, default_timeout=120)
    at.run()
    return at


def metric_value(app, word):
    metrics = {m.label: m.value for m in app.metric}
    return next(value for label, value in metrics.items() if word in label)


def test_s01_dashboard_starts_without_errors(app):
    assert not app.exception


def test_s02_all_four_tabs_are_present(app):
    labels = [tab.label for tab in app.tabs]
    assert labels == ["Overview", "Machine detail", "Alerts", "Remaining useful life"]


def test_s03_overview_shows_the_expected_fleet_counts(app):
    assert metric_value(app, "Critical") == "329"
    assert metric_value(app, "Warning") == "20"
    assert metric_value(app, "Normal") == "9651"
    assert metric_value(app, "Rejected") == "0"


def test_s04_data_quality_summary_line_is_shown(app):
    captions = [c.value for c in app.caption]
    assert any("Records loaded: 10,000" in text for text in captions)


def test_s05_machine_detail_shows_a_recommended_action(app):
    assert len(app.selectbox) >= 1
    assert any("Recommended action" in info.value for info in app.info)


def test_s06_rul_tab_has_a_cycle_slider(app):
    assert len(app.slider) == 1