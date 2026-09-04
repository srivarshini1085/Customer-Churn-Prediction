"""Tests for churn.validation — input checks and PSI drift."""

from __future__ import annotations

from churn.validation import (
    PSI_MAJOR,
    TrainingStats,
    drift_report,
    population_stability_index,
    validate_features,
)
from tests.conftest import HIGH_RISK


def _stats(xy) -> TrainingStats:
    x, _ = xy
    return TrainingStats.from_frame(x)


def test_training_stats_cover_every_feature(xy):
    stats = _stats(xy)
    assert set(stats.numeric) == {"tenure", "MonthlyCharges", "TotalCharges"}
    assert "Contract" in stats.categorical
    assert "Month-to-month" in stats.categorical["Contract"].categories


def test_clean_payload_has_no_warnings(xy):
    assert validate_features(HIGH_RISK, _stats(xy)) == []


def test_numeric_out_of_range_is_flagged(xy):
    warnings = validate_features({**HIGH_RISK, "tenure": 999}, _stats(xy))
    assert any("tenure" in w and "range" in w for w in warnings)


def test_unseen_category_is_flagged(xy):
    warnings = validate_features({**HIGH_RISK, "PaymentMethod": "Crypto"}, _stats(xy))
    assert any("PaymentMethod" in w for w in warnings)


def test_psi_is_zero_for_identical_distributions():
    freq = [0.2, 0.3, 0.5]
    assert population_stability_index(freq, freq) == 0.0


def test_psi_grows_with_shift():
    small = population_stability_index([0.5, 0.5], [0.45, 0.55])
    large = population_stability_index([0.5, 0.5], [0.05, 0.95])
    assert 0 < small < large


def test_drift_report_no_drift_on_same_data(xy):
    x, _ = xy
    report = drift_report(_stats(xy), x)
    assert report  # not empty
    assert all(row["psi"] < PSI_MAJOR for row in report.values())
    assert not any(row["drifted"] for row in report.values())


def test_drift_report_detects_shift(xy):
    x, _ = xy
    shifted = x.copy()
    shifted["Contract"] = "Two year"  # collapse a whole categorical
    report = drift_report(_stats(xy), shifted)
    assert report["Contract"]["drifted"] is True
