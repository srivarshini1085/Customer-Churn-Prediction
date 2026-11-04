"""Per-customer explanations: coefficients for linear models, SHAP for trees."""

from __future__ import annotations

import pickle

import numpy as np
import pytest
from sklearn.ensemble import HistGradientBoostingClassifier

from churn.features import build_feature_pipeline
from churn.predict import ChurnModel, churn_class_shap
from tests.conftest import HIGH_RISK, LOW_RISK


@pytest.fixture(scope="module")
def tree_model(sample_xy) -> ChurnModel:
    """A small gradient-boosting ChurnModel (uncalibrated, fast to fit)."""
    x, y = sample_xy
    pipe = build_feature_pipeline()
    pipe.steps.append(("clf", HistGradientBoostingClassifier(max_iter=40, random_state=0)))
    pipe.fit(x, y)
    return ChurnModel(
        pipeline=pipe,
        base_pipeline=pipe,
        threshold=0.5,
        feature_names=list(pipe.named_steps["prep"].get_feature_names_out()),
        metadata={"version": "tree-test"},
    )


def test_explanation_method_matches_model_family(fitted_model, tree_model):
    assert fitted_model.explanation_method == "coefficients"
    assert tree_model.explanation_method == "shap"


def test_tree_factors_are_signed_and_ranked(tree_model):
    factors = tree_model.top_factors(HIGH_RISK, n=8)
    assert factors
    magnitudes = [abs(v) for _, v in factors]
    assert magnitudes == sorted(magnitudes, reverse=True)
    # SHAP gives direction: a month-to-month contract pushes risk up.
    contributions = dict(factors)
    assert contributions["Contract = Month-to-month"] > 0


def test_shap_values_add_up_to_the_model_output(tree_model):
    """SHAP's defining property: expected value + contributions = raw model output."""
    transformed = np.asarray(
        tree_model.base_pipeline[:-1].transform(tree_model._row(LOW_RISK)), dtype=float
    )
    shap_values = tree_model._shap_contributions(transformed[0])
    expected = float(np.ravel(tree_model._explainer.expected_value)[-1])
    raw = float(tree_model.base_pipeline[-1].decision_function(transformed)[0])
    assert expected + shap_values.sum() == pytest.approx(raw, abs=1e-4)


def test_explainer_is_not_pickled(tree_model):
    tree_model.top_factors(HIGH_RISK)
    assert tree_model._explainer is not None
    restored = pickle.loads(pickle.dumps(tree_model))
    assert restored._explainer is None
    assert restored.top_factors(HIGH_RISK) == tree_model.top_factors(HIGH_RISK)


@pytest.mark.parametrize(
    "raw",
    [
        [np.zeros((1, 3)), np.array([[1.0, -2.0, 3.0]])],  # list: one array per class
        np.array([[[0.0, 1.0], [0.0, -2.0], [0.0, 3.0]]]),  # (rows, features, classes)
        np.array([[1.0, -2.0, 3.0]]),  # (rows, features) for the positive class
    ],
)
def test_churn_class_shap_normalises_every_output_shape(raw):
    np.testing.assert_allclose(churn_class_shap(raw), [1.0, -2.0, 3.0])


def test_predict_proba_many_matches_single_scoring(fitted_model):
    batch = fitted_model.predict_proba_many([HIGH_RISK, LOW_RISK, {}])
    singles = [fitted_model.predict_proba(c) for c in (HIGH_RISK, LOW_RISK, {})]
    np.testing.assert_allclose(batch, singles)


def test_predict_proba_many_rejects_unknown_features(fitted_model):
    with pytest.raises(ValueError, match="Unknown feature"):
        fitted_model.predict_proba_many([HIGH_RISK, {"bogus": 1}])
