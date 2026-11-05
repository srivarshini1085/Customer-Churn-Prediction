"""Smoke tests: every dashboard page renders without an exception.

Uses Streamlit's headless ``AppTest`` against the isolated test registry, whose
model has no survival model or fairness report, so the "retrain to add one"
empty states are exercised too.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

APP = str(Path(__file__).resolve().parent.parent / "app.py")

PAGES = [
    "Prediction",
    "Churn Analysis",
    "Model Performance",
    "Batch Prediction",
    "Retention Planner",
    "Time to Churn",
    "Monitoring",
]


def _open(page: str) -> AppTest:
    app = AppTest.from_file(APP, default_timeout=180).run()
    app.sidebar.radio[0].set_value(page).run()
    return app


@pytest.mark.parametrize("page", PAGES)
def test_page_renders_without_exceptions(page, registered_model):
    app = _open(page)
    assert not app.exception, [e.value for e in app.exception]


def test_sidebar_shows_the_loaded_model_version(registered_model):
    version, _ = registered_model
    app = _open("Prediction")
    assert any(version in caption.value for caption in app.sidebar.caption)


def test_prediction_form_shows_result_time_to_churn_and_what_if(registered_model):
    app = _open("Prediction")
    app.button[0].click().run()  # the form's Predict button
    assert not app.exception
    labels = {metric.label for metric in app.metric}
    assert {"Churn Probability", "Prediction", "Risk Tier"} <= labels
    headers = {h.value for h in app.subheader}
    assert {"Prediction Result", "Top Churn Factors", "What would reduce this risk?"} <= headers


def test_missing_survival_model_shows_an_empty_state(registered_model):
    app = _open("Time to Churn")
    assert any("python -m churn train" in info.value for info in app.info)


def test_monitoring_sample_needs_no_retraining(registered_model):
    app = _open("Monitoring")
    app.radio[0].set_value("Sample: last 1,000 rows of the Telco data").run()
    assert not app.exception
    assert any("No retraining needed" in s.value for s in app.success)
