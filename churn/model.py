"""Candidate models and their (deliberately small) hyper-parameter grids.

Every candidate is a full pipeline: ``engineer -> preprocess -> classifier``.
All classifiers use ``class_weight="balanced"`` because the target is ~27% churn
and the cost of missing a churner outweighs a false alarm.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import ClassifierMixin, clone
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline

from churn import config
from churn.features import build_feature_pipeline

logger = logging.getLogger(__name__)


@dataclass
class Candidate:
    """A named classifier plus the grid to search over it."""

    name: str
    estimator: ClassifierMixin
    param_grid: dict[str, list[Any]] = field(default_factory=dict)

    def build_pipeline(self) -> Pipeline:
        """Full pipeline: engineer -> preprocess -> this classifier."""
        steps = [*build_feature_pipeline().steps, ("clf", self.estimator)]
        return Pipeline(steps)

    def prefixed_grid(self) -> dict[str, list[Any]]:
        """The grid with keys namespaced to the ``clf`` pipeline step."""
        return {f"clf__{key}": value for key, value in self.param_grid.items()}


def candidates() -> list[Candidate]:
    """The models compared on every training run."""
    rs = config.RANDOM_STATE
    return [
        Candidate(
            name="logistic_regression",
            estimator=LogisticRegression(max_iter=2000, class_weight="balanced"),
            param_grid={"C": [0.03, 0.1, 0.3, 1.0]},
        ),
        Candidate(
            name="random_forest",
            # balanced_subsample re-weights per bootstrap sample - the RF-native
            # analogue of class_weight="balanced".
            estimator=RandomForestClassifier(
                n_estimators=400,
                class_weight="balanced_subsample",
                random_state=rs,
                n_jobs=-1,
            ),
            param_grid={
                "max_depth": [10, None],
                "min_samples_leaf": [1, 5],
            },
        ),
        Candidate(
            name="hist_gradient_boosting",
            estimator=HistGradientBoostingClassifier(
                class_weight="balanced", random_state=rs
            ),
            param_grid={
                "learning_rate": [0.05, 0.1],
                "max_depth": [None, 6],
                "l2_regularization": [0.0, 1.0],
            },
        ),
    ]


# --------------------------------------------------------------------------- #
# Probability calibration
# --------------------------------------------------------------------------- #

_CALIBRATION_METHODS = ("sigmoid", "isotonic")


def _cv() -> StratifiedKFold:
    return StratifiedKFold(
        n_splits=config.CV_FOLDS, shuffle=True, random_state=config.RANDOM_STATE
    )


def choose_calibration_method(
    pipeline: Pipeline, x: pd.DataFrame, y: pd.Series
) -> str:
    """Return ``"sigmoid"`` or ``"isotonic"`` — whichever gives the lower CV Brier."""
    y_arr = np.asarray(y)
    scores: dict[str, float] = {}
    for method in _CALIBRATION_METHODS:
        proba = cross_val_predict(
            CalibratedClassifierCV(clone(pipeline), method=method, cv=3),
            x,
            y,
            cv=_cv(),
            method="predict_proba",
            n_jobs=-1,
        )[:, 1]
        scores[method] = float(brier_score_loss(y_arr, proba))
    best = min(_CALIBRATION_METHODS, key=lambda m: scores[m])
    logger.info("Calibration Brier: %s -> chose %s", scores, best)
    return best


def calibrate(
    pipeline: Pipeline, x: pd.DataFrame, y: pd.Series, method: str
) -> CalibratedClassifierCV:
    """Fit a calibrated wrapper around a *fresh clone* of ``pipeline``.

    ``ensemble=False`` keeps a single base estimator (fit on all data) plus one
    calibrator, so serving ``predict_proba`` costs one pipeline pass, not five.
    """
    calibrated = CalibratedClassifierCV(
        clone(pipeline), method=method, cv=_cv(), ensemble=False
    )
    calibrated.fit(x, y)
    return calibrated
