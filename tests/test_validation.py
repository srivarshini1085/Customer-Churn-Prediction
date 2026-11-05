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


# --------------------------------------------------------------------------- #
# Traffic-light status and retrain recommendation
# --------------------------------------------------------------------------- #


def test_drift_status_bands():
    from churn.validation import PSI_MINOR, drift_status

    assert drift_status(0.0) == "green"
    assert drift_status(PSI_MINOR - 1e-9) == "green"
    assert drift_status(PSI_MINOR) == "amber"
    assert drift_status(PSI_MAJOR - 1e-9) == "amber"
    assert drift_status(PSI_MAJOR) == "red"


def _report(**psi: float) -> dict:
    return {f: {"psi": v, "drifted": v >= PSI_MAJOR} for f, v in psi.items()}


def test_summary_recommends_retrain_on_any_red():
    from churn.validation import drift_summary

    summary = drift_summary(_report(tenure=0.01, Contract=0.35))
    assert summary["retrain_recommended"] is True
    assert summary["counts"] == {"green": 1, "amber": 0, "red": 1}
    assert "Contract" in summary["reason"]


def test_summary_recommends_retrain_on_many_ambers(monkeypatch):
    from churn import config
    from churn.validation import drift_summary

    monkeypatch.setattr(config, "DRIFT_AMBER_RETRAIN_COUNT", 2)
    assert drift_summary(_report(a=0.15))["retrain_recommended"] is False
    summary = drift_summary(_report(a=0.15, b=0.12))
    assert summary["retrain_recommended"] is True
    assert "2 features show moderate drift" in summary["reason"]


def test_summary_is_green_on_training_data(xy):
    from churn.validation import drift_summary

    x, _ = xy
    summary = drift_summary(drift_report(_stats(xy), x))
    assert summary["retrain_recommended"] is False
    assert summary["counts"]["red"] == 0
    assert set(summary["statuses"]) == set(drift_report(_stats(xy), x))
