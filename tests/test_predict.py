"""Tests for churn.predict: ChurnModel scoring API and risk tiers."""

from __future__ import annotations

import numpy as np
import pytest

from churn.predict import risk_tier
from tests.conftest import HIGH_RISK, LOW_RISK


@pytest.mark.parametrize(
    ("probability", "expected"),
    [(0.0, "Low"), (0.39, "Low"), (0.40, "Moderate"), (0.69, "Moderate"),
     (0.70, "High"), (0.999, "High")],
)
def test_risk_tier_bands(probability, expected):
    assert risk_tier(probability) == expected


def test_high_risk_scores_above_low_risk(fitted_model):
    assert fitted_model.predict_proba(HIGH_RISK) > fitted_model.predict_proba(LOW_RISK)


def test_predict_returns_full_result(fitted_model):
    result = fitted_model.predict(HIGH_RISK)
    assert set(result) == {
        "churn", "probability", "risk_tier", "threshold", "model_version", "warnings"
    }
    assert isinstance(result["churn"], bool)
    assert 0.0 <= result["probability"] <= 1.0
    assert result["warnings"] == []


def test_predict_flags_out_of_range_input(fitted_model):
    result = fitted_model.predict({**HIGH_RISK, "tenure": 5000})
    assert any("tenure" in w for w in result["warnings"])


def test_unknown_feature_rejected(fitted_model):
    with pytest.raises(ValueError, match="Unknown feature"):
        fitted_model.predict_proba({**HIGH_RISK, "bogus": 1})


def test_unseen_category_value_does_not_crash(fitted_model):
    weird = {**HIGH_RISK, "PaymentMethod": "Crypto wallet"}
    assert 0.0 <= fitted_model.predict_proba(weird) <= 1.0


def test_missing_fields_are_filled(fitted_model):
    proba = fitted_model.predict_proba({"tenure": 3, "Contract": "Month-to-month"})
    assert np.isfinite(proba)


def test_top_factors_are_ranked_by_magnitude(fitted_model):
    factors = fitted_model.top_factors(HIGH_RISK, n=5)
    assert 0 < len(factors) <= 5
    magnitudes = [abs(value) for _, value in factors]
    assert magnitudes == sorted(magnitudes, reverse=True)


def test_top_factors_are_signed_for_linear_model(fitted_model):
    # a fiber-optic month-to-month new customer: at least one factor pushes up,
    # at least one pushes down.
    values = [v for _, v in fitted_model.top_factors(HIGH_RISK, n=6)]
    assert any(v > 0 for v in values)
    assert any(v < 0 for v in values)
