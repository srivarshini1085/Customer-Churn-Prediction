"""End-to-end pipeline behaviour (fast — the shared fitted_model fixture, no search)."""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.metrics import roc_auc_score

from churn.data import split
from churn.predict import ChurnModel


def test_predict_proba_in_unit_interval(fitted_model, xy):
    _, x_test, _, _ = split(*xy)
    proba = fitted_model.pipeline.predict_proba(x_test)[:, 1]
    assert proba.min() >= 0.0
    assert proba.max() <= 1.0


def test_holdout_roc_auc_is_reasonable(fitted_model, xy):
    _, x_test, _, y_test = split(*xy)
    auc = roc_auc_score(y_test, fitted_model.pipeline.predict_proba(x_test)[:, 1])
    assert auc > 0.80


def test_is_linear_true_for_logistic_regression(fitted_model):
    assert fitted_model.is_linear is True


def test_churnmodel_roundtrips_through_joblib(fitted_model, xy, tmp_path):
    _, x_test, _, _ = split(*xy)
    path = fitted_model.save(tmp_path / "m.joblib")
    reloaded = ChurnModel.load(path)

    assert reloaded.threshold == fitted_model.threshold
    assert reloaded.feature_names == fitted_model.feature_names
    row = x_test.iloc[0].to_dict()
    assert np.isclose(reloaded.predict_proba(row), fitted_model.predict_proba(row))


def test_load_missing_model_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        ChurnModel.load(tmp_path / "does-not-exist.joblib")


def test_load_wrong_content_raises(tmp_path):
    import joblib

    bad = tmp_path / "bad.joblib"
    joblib.dump({"not": "a model"}, bad)
    with pytest.raises(TypeError):
        ChurnModel.load(bad)
