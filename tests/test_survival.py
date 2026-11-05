"""Survival analysis: design matrix, Kaplan-Meier, Cox fit, per-customer estimates."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from churn import config
from churn.data import split
from churn.registry import ModelRegistry
from churn.survival import (
    SurvivalModel,
    covariate_names,
    design_matrix,
    fit_survival,
    kaplan_meier_by_contract,
    model_card_section,
)
from tests.conftest import HIGH_RISK, LOW_RISK


@pytest.fixture(scope="module")
def survival(xy) -> SurvivalModel:
    x, y = xy
    x_train, x_test, y_train, y_test = split(x, y)
    return fit_survival(x_train, y_train, x_test, y_test)


# --------------------------------------------------------------------------- #
# Design matrix
# --------------------------------------------------------------------------- #


def test_design_matrix_never_uses_the_duration():
    names = covariate_names()
    assert len(names) == len(set(names))
    for leaky in config.SURVIVAL_EXCLUDED:
        assert not any(name.startswith(leaky) for name in names)


def test_design_matrix_encoding():
    frame = pd.DataFrame(
        [
            {**HIGH_RISK, "Contract": "Two year", "PaymentMethod": "Mailed check"},
            {**LOW_RISK, "InternetService": "No", "OnlineSecurity": "No internet service"},
        ]
    )
    matrix = design_matrix(frame)
    assert list(matrix.columns) == covariate_names()
    first, second = matrix.iloc[0], matrix.iloc[1]
    assert first["Contract=Two year"] == 1 and first["Contract=One year"] == 0
    assert first["PaymentMethod=Mailed check"] == 1
    assert first["MonthlyCharges_per_10"] == pytest.approx(9.5)
    assert second["InternetService=No"] == 1
    assert second["OnlineSecurity=Yes"] == 0  # "No internet service" is not "Yes"


def test_design_matrix_fills_missing_columns_with_defaults():
    matrix = design_matrix(pd.DataFrame([{"tenure": 5}]))
    assert list(matrix.columns) == covariate_names()
    assert not matrix.isna().any().any()


# --------------------------------------------------------------------------- #
# Kaplan-Meier
# --------------------------------------------------------------------------- #


def test_kaplan_meier_curves_are_valid_survival_functions(xy):
    x, y = xy
    curves = kaplan_meier_by_contract(x, y)
    assert set(curves) == set(config.SURVIVAL_CATEGORICAL["Contract"])
    for curve in curves.values():
        assert list(curve.columns) == ["month", "survival", "lower_95", "upper_95"]
        assert curve["survival"].iloc[0] == pytest.approx(1.0)
        assert (np.diff(curve["survival"]) <= 1e-12).all()  # never increases
        assert (curve["lower_95"] <= curve["survival"] + 1e-12).all()
        assert (curve["survival"] <= curve["upper_95"] + 1e-12).all()


def test_longer_contracts_survive_longer(survival):
    def at(contract: str, month: int) -> float:
        curve = survival.km_curves[contract].set_index("month")["survival"]
        return float(curve[curve.index <= month].iloc[-1])

    assert at("Month-to-month", 24) < at("One year", 24) < at("Two year", 24)


# --------------------------------------------------------------------------- #
# Cox model
# --------------------------------------------------------------------------- #


def test_cox_metrics_are_reported(survival):
    m = survival.metrics
    assert 0.5 < m["concordance_test"] <= 1.0
    assert 0.5 < m["concordance_train"] <= 1.0
    assert set(m["ph_test_p"]) == set(covariate_names())
    assert set(m["ph_violations"]) <= set(covariate_names())
    assert all(m["ph_test_p"][v] < config.SURVIVAL_PH_ALPHA for v in m["ph_violations"])
    assert m["excluded_covariates"] == list(config.SURVIVAL_EXCLUDED)


def test_hazard_ratios_match_coefficients(survival):
    hr = survival.hazard_ratios().set_index("covariate")
    np.testing.assert_allclose(hr["hazard_ratio"], np.exp(survival.cox.params_[hr.index]))
    assert (hr["lower_95"] <= hr["hazard_ratio"]).all()
    assert (hr["hazard_ratio"] <= hr["upper_95"]).all()
    assert hr.loc["Contract=Two year", "hazard_ratio"] < 1  # long contracts churn later


# --------------------------------------------------------------------------- #
# Per-customer estimates
# --------------------------------------------------------------------------- #


def test_riskier_customer_is_expected_to_churn_sooner(survival):
    same_tenure = {"tenure": 6}
    high, low = survival.predict([{**HIGH_RISK, **same_tenure}, {**LOW_RISK, **same_tenure}])
    assert high["churn_within_horizon"] > low["churn_within_horizon"]
    assert high["months_until_churn"] is not None
    if low["months_until_churn"] is not None:
        assert high["months_until_churn"] < low["months_until_churn"]


def test_estimates_are_conditional_on_current_tenure(survival):
    new, settled = survival.predict([{**HIGH_RISK, "tenure": 0}, {**HIGH_RISK, "tenure": 12}])
    for estimate in (new, settled):
        assert 0.0 <= estimate["churn_within_horizon"] <= 1.0
        assert estimate["horizon_months"] == config.SURVIVAL_HORIZON_MONTHS
    # The first year is the riskiest, so having survived it lowers next-year risk.
    assert settled["churn_within_horizon"] < new["churn_within_horizon"]


def test_risk_rises_again_at_long_tenure_as_in_the_data(survival):
    """The data's hazard is U-shaped: churn picks up again for 60+ month customers.

    This is a property of the dataset (Kaplan-Meier shows it too), documented as a
    limitation; the test pins it so a change in behaviour is noticed.
    """
    middle, late = survival.predict([{**HIGH_RISK, "tenure": 24}, {**HIGH_RISK, "tenure": 60}])
    assert late["churn_within_horizon"] > middle["churn_within_horizon"]


def test_no_extrapolation_beyond_the_observed_tenure(survival):
    at_max = int(survival.max_tenure)
    estimate = survival.predict([{**LOW_RISK, "tenure": at_max}])[0]
    assert estimate["months_until_churn"] is None
    assert estimate["beyond_months"] == 0
    assert estimate["horizon_months"] == 0
    assert estimate["churn_within_horizon"] == pytest.approx(0.0)

    near = survival.predict([{**LOW_RISK, "tenure": at_max - 5}])[0]
    assert near["horizon_months"] == 5  # the horizon is cut where the data ends


def test_predict_with_defaults_only(survival):
    (estimate,) = survival.predict([{}])
    assert set(estimate) == {
        "months_until_churn",
        "beyond_months",
        "churn_within_horizon",
        "horizon_months",
    }


# --------------------------------------------------------------------------- #
# Persistence, registry and reporting
# --------------------------------------------------------------------------- #


def test_save_load_roundtrip(survival, tmp_path):
    path = survival.save(tmp_path / "s" / "survival.joblib")
    restored = SurvivalModel.load(path)
    assert restored.predict([HIGH_RISK]) == survival.predict([HIGH_RISK])


def test_load_rejects_other_objects(tmp_path):
    import joblib

    joblib.dump({"not": "a model"}, tmp_path / "bad.joblib")
    with pytest.raises(TypeError, match="SurvivalModel"):
        SurvivalModel.load(tmp_path / "bad.joblib")


def test_registry_stores_survival_per_version(fitted_model, survival, tmp_path):
    registry = ModelRegistry(tmp_path / "registry")
    with_survival = registry.save(fitted_model, {}, "card", survival=survival)
    assert registry.load_survival(with_survival).metrics == survival.metrics

    without = registry.save(fitted_model, {}, "card")
    with pytest.raises(FileNotFoundError, match="no survival model"):
        registry.load_survival(without)


def test_model_card_section_reports_results_and_limitations(survival):
    section = model_card_section(survival)
    assert f"{survival.metrics['concordance_test']:.3f}" in section
    assert "Limitations" in section
    for covariate in covariate_names():
        assert covariate in section
