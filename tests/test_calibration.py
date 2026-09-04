"""Tests for probability calibration (churn.model)."""

from __future__ import annotations

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss

from churn.data import split
from churn.features import build_feature_pipeline
from churn.model import calibrate, choose_calibration_method


def _base_pipeline():
    pipe = build_feature_pipeline()
    pipe.steps.append(("clf", LogisticRegression(max_iter=1000, class_weight="balanced")))
    return pipe


def test_choose_calibration_method_returns_valid_choice(sample_xy):
    x, y = sample_xy
    method = choose_calibration_method(_base_pipeline(), x, y)
    assert method in ("sigmoid", "isotonic")


def test_calibration_improves_brier_on_holdout(xy):
    x, y = xy
    x_train, x_test, y_train, y_test = split(x, y)

    base = _base_pipeline().fit(x_train, y_train)
    calibrated = calibrate(base, x_train, y_train, method="sigmoid")

    base_brier = brier_score_loss(y_test, base.predict_proba(x_test)[:, 1])
    cal_brier = brier_score_loss(y_test, calibrated.predict_proba(x_test)[:, 1])
    # class_weight="balanced" leaves the base probabilities badly miscalibrated,
    # so calibration should not make Brier worse (and usually improves it).
    assert cal_brier <= base_brier + 1e-3


def test_calibrated_probabilities_are_valid(sample_xy):
    x, y = sample_xy
    calibrated = calibrate(_base_pipeline().fit(x, y), x, y, method="isotonic")
    proba = calibrated.predict_proba(x)[:, 1]
    assert np.all((proba >= 0) & (proba <= 1))
