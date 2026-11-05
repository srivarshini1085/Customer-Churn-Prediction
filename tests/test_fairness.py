"""Fairness report: per-group metrics, gaps, flags and generated interpretation."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from churn import config
from churn.data import split
from churn.fairness import fairness_report, group_metrics, interpret, model_card_section


def test_group_metrics_on_a_hand_checked_example():
    y_true = np.array([1, 1, 1, 0, 0, 0, 0, 0])
    y_pred = np.array([1, 1, 0, 1, 0, 0, 0, 0])  # tp=2 fn=1 fp=1 tn=4
    m = group_metrics(y_true, y_pred)
    assert m["n"] == 8 and m["churners"] == 3
    assert m["churn_rate"] == pytest.approx(3 / 8)
    assert m["flagged_rate"] == pytest.approx(3 / 8)
    assert m["recall"] == pytest.approx(2 / 3)
    assert m["precision"] == pytest.approx(2 / 3)
    assert m["false_positive_rate"] == pytest.approx(1 / 5)


def test_undefined_metrics_are_none_not_zero():
    m = group_metrics(np.array([0, 0]), np.array([0, 0]))  # no churners, nobody flagged
    assert m["recall"] is None and m["precision"] is None
    assert m["false_positive_rate"] == 0.0


def _toy_frame() -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    x = pd.DataFrame(
        {
            "gender": ["Female"] * 4 + ["Male"] * 4,
            "SeniorCitizen": [0, 1] * 4,
            "Partner": ["Yes"] * 8,
        }
    )
    y = np.array([1, 1, 0, 0, 1, 1, 0, 0])
    # Every female churner is caught, no male churner is.
    proba = np.array([0.9, 0.9, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1])
    return x, y, proba


def test_report_structure_gaps_and_flags():
    x, y, proba = _toy_frame()
    report = fairness_report(x, y, proba, threshold=0.5)
    assert report["threshold"] == 0.5
    assert set(report["attributes"]) == set(config.FAIRNESS_ATTRIBUTES)

    gender = report["attributes"]["gender"]
    assert set(gender["groups"]) == {"Female", "Male"}
    assert gender["gaps"]["recall"] == pytest.approx(1.0)
    assert "recall" in gender["flagged"]

    partner = report["attributes"]["Partner"]  # a single group cannot have a gap
    assert all(gap is None for gap in partner["gaps"].values())
    assert partner["flagged"] == []


def test_interpretation_is_generated_from_the_numbers():
    x, y, proba = _toy_frame()
    findings = interpret(fairness_report(x, y, proba, threshold=0.5))
    gender = next(f for f in findings if f.startswith("gender"))
    assert "0.00 (gender=Male) to 1.00 (gender=Female)" in gender
    assert "worth reviewing" in gender
    assert any(f.startswith("Partner") and "not enough groups" in f for f in findings)


def test_report_on_the_real_model(fitted_model, xy):
    x, y = xy
    _, x_test, _, y_test = split(x, y)
    proba = fitted_model.pipeline.predict_proba(x_test)[:, 1]
    report = fairness_report(x_test, y_test, proba, fitted_model.threshold)
    for result in report["attributes"].values():
        assert sum(g["n"] for g in result["groups"].values()) == len(x_test)
        for metrics in result["groups"].values():
            for name in ("recall", "precision", "false_positive_rate"):
                assert metrics[name] is None or 0.0 <= metrics[name] <= 1.0
        for metric in result["flagged"]:
            assert result["gaps"][metric] > config.FAIRNESS_GAP_FLAG

    section = model_card_section(report)
    assert section.startswith("## Fairness")
    assert "**Interpretation**" in section
    for attribute in config.FAIRNESS_ATTRIBUTES:
        assert f"| {attribute} |" in section
