"""What-if search: realistic, immutable-safe, and consistent with the model."""

from __future__ import annotations

import hypothesis.strategies as st
import pytest
from hypothesis import HealthCheck, given, settings

from churn import config
from churn.counterfactual import candidate_changes, complete_features, what_if
from churn.schema import INPUT_FIELDS
from tests.conftest import HIGH_RISK, LOW_RISK


def test_config_never_marks_an_immutable_feature_as_actionable():
    assert not set(config.ACTIONABLE_CHANGES) & set(config.IMMUTABLE_FEATURES)


def test_high_risk_customer_gets_risk_reducing_scenarios(fitted_model):
    result = what_if(fitted_model, HIGH_RISK)
    assert result.baseline_probability == pytest.approx(fitted_model.predict_proba(HIGH_RISK))
    assert result.scenarios
    probabilities = [s.probability for s in result.scenarios]
    assert probabilities == sorted(probabilities)
    assert all(s.delta < 0 for s in result.scenarios)


def test_reported_probability_matches_rescoring_the_changed_customer(fitted_model):
    for scenario in what_if(fitted_model, HIGH_RISK).scenarios:
        changed = {**HIGH_RISK, **{c.feature: c.to_value for c in scenario.changes}}
        assert scenario.probability == pytest.approx(fitted_model.predict_proba(changed))
        assert scenario.delta == pytest.approx(
            scenario.probability - fitted_model.predict_proba(HIGH_RISK)
        )
        for change in scenario.changes:
            assert change.from_value == str(HIGH_RISK[change.feature])


def test_every_change_in_a_combination_pays_its_way(fitted_model):
    min_gain = 0.005
    result = what_if(fitted_model, HIGH_RISK, max_changes=3, top_k=20, min_gain=min_gain)
    combos = [s for s in result.scenarios if len(s.changes) > 1]
    assert combos
    for scenario in combos:
        changed = {**HIGH_RISK, **{c.feature: c.to_value for c in scenario.changes}}
        for dropped in scenario.changes:
            without = {**changed, dropped.feature: dropped.from_value}
            assert fitted_model.predict_proba(without) - scenario.probability >= min_gain - 1e-9


def test_limits_are_respected(fitted_model):
    result = what_if(fitted_model, HIGH_RISK, max_changes=1, top_k=2)
    assert len(result.scenarios) <= 2
    assert all(len(s.changes) == 1 for s in result.scenarios)


def test_invalid_max_changes_is_rejected(fitted_model):
    with pytest.raises(ValueError, match="max_changes"):
        what_if(fitted_model, HIGH_RISK, max_changes=0)


def test_contracts_only_get_longer():
    month = candidate_changes(complete_features({"Contract": "Month-to-month"}))
    one_year = candidate_changes(complete_features({"Contract": "One year"}))
    two_year = candidate_changes(complete_features({"Contract": "Two year"}))
    assert month["Contract"] == ["One year", "Two year"]
    assert one_year["Contract"] == ["Two year"]
    assert "Contract" not in two_year


def test_addons_need_internet_and_are_only_added():
    no_internet = complete_features(
        {
            "InternetService": "No",
            "OnlineSecurity": "No internet service",
            "TechSupport": "No internet service",
        }
    )
    options = candidate_changes(no_internet)
    assert "OnlineSecurity" not in options and "TechSupport" not in options

    has_both = candidate_changes(complete_features({"OnlineSecurity": "Yes", "TechSupport": "Yes"}))
    assert "OnlineSecurity" not in has_both and "TechSupport" not in has_both


def test_payment_moves_only_to_automatic_methods():
    options = candidate_changes(complete_features({"PaymentMethod": "Mailed check"}))
    assert set(options["PaymentMethod"]) == {"Bank transfer (automatic)", "Credit card (automatic)"}


def test_immutable_feature_is_skipped_even_if_misconfigured(monkeypatch):
    monkeypatch.setitem(config.ACTIONABLE_CHANGES, "gender", ("Male",))
    assert "gender" not in candidate_changes(complete_features({"gender": "Female"}))


def test_no_actionable_options_returns_no_scenarios(fitted_model, monkeypatch):
    monkeypatch.setattr(config, "ACTIONABLE_CHANGES", {})
    result = what_if(fitted_model, HIGH_RISK)
    assert result.scenarios == []
    assert result.baseline_risk_tier in {"High", "Moderate", "Low"}


def test_low_risk_customer_may_have_nothing_to_improve(fitted_model):
    result = what_if(fitted_model, LOW_RISK)
    assert all(s.delta <= -0.005 for s in result.scenarios)


# --------------------------------------------------------------------------- #
# Property: whatever the customer, fixed attributes are never changed
# --------------------------------------------------------------------------- #


@st.composite
def customers(draw: st.DrawFn) -> dict[str, object]:
    customer: dict[str, object] = {}
    for field in INPUT_FIELDS:
        if field.choices:
            customer[field.name] = field.coerce(draw(st.sampled_from(field.choices)))
        else:
            customer[field.name] = draw(st.integers(min_value=0, max_value=100))
    return customer


@settings(
    max_examples=25, deadline=None, suppress_health_check=[HealthCheck.function_scoped_fixture]
)
@given(customer=customers())
def test_what_if_never_changes_fixed_attributes(fitted_model, customer):
    result = what_if(fitted_model, customer, max_changes=3, top_k=20)
    for scenario in result.scenarios:
        changed = {c.feature for c in scenario.changes}
        assert not changed & set(config.IMMUTABLE_FEATURES)
        assert changed <= set(config.ACTIONABLE_CHANGES)
        assert len(changed) == len(scenario.changes) <= 3
        assert scenario.probability < result.baseline_probability
