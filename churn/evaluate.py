"""Cross-validation, model comparison, decision-threshold tuning and reporting."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import ClassifierMixin
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_predict

from churn import config
from churn.model import Candidate

logger = logging.getLogger(__name__)

#: CV metric name -> scikit-learn scorer string.
SCORING: dict[str, str] = {
    "roc_auc": "roc_auc",
    "pr_auc": "average_precision",
    "f1": "f1",
    "recall": "recall",
    "precision": "precision",
}

THRESHOLD_OBJECTIVES = ("f1", "recall_at_precision")


@dataclass
class CandidateResult:
    """Outcome of grid-searching and scoring one candidate."""

    name: str
    best_params: dict[str, Any]
    cv_scores: dict[str, float]
    cv_score_std: dict[str, float]
    estimator: Any = field(repr=False, default=None)

    @property
    def selection_score(self) -> float:
        return self.cv_scores[config.SELECTION_METRIC]


def _cv() -> StratifiedKFold:
    return StratifiedKFold(
        n_splits=config.CV_FOLDS, shuffle=True, random_state=config.RANDOM_STATE
    )


def tune_and_score(
    candidate: Candidate, x: pd.DataFrame, y: pd.Series
) -> CandidateResult:
    """Grid-search one candidate in a single multi-metric CV pass.

    ``refit`` targets :data:`config.SELECTION_METRIC`, so ``best_estimator_`` is
    already fitted on the whole of ``x``/``y`` when this returns.
    """
    search = GridSearchCV(
        estimator=candidate.build_pipeline(),
        param_grid=candidate.prefixed_grid(),
        scoring=SCORING,
        refit=config.SELECTION_METRIC,
        cv=_cv(),
        n_jobs=-1,
    )
    search.fit(x, y)

    idx = search.best_index_
    cv_scores = {m: float(search.cv_results_[f"mean_test_{m}"][idx]) for m in SCORING}
    cv_std = {m: float(search.cv_results_[f"std_test_{m}"][idx]) for m in SCORING}
    best_params = {k.removeprefix("clf__"): v for k, v in search.best_params_.items()}

    logger.info(
        "%-24s CV %s=%.4f  params=%s",
        candidate.name,
        config.SELECTION_METRIC,
        cv_scores[config.SELECTION_METRIC],
        best_params,
    )
    return CandidateResult(
        name=candidate.name,
        best_params=best_params,
        cv_scores=cv_scores,
        cv_score_std=cv_std,
        estimator=search.best_estimator_,
    )


def selection_key(result: CandidateResult) -> tuple[float, float, float]:
    """Ranking key for candidates (higher is better).

    The selection metric is compared only to 3 decimals: sub-0.001 differences in
    CV ROC-AUC are within fold noise, so when models are effectively tied the
    business priority — catching churners (recall) — breaks the tie, then PR-AUC.
    """
    return (
        round(result.selection_score, 3),
        round(result.cv_scores["recall"], 3),
        round(result.cv_scores["pr_auc"], 3),
    )


def select_best(results: list[CandidateResult]) -> CandidateResult:
    """Pick the winning candidate by :func:`selection_key`."""
    return max(results, key=selection_key)


def optimal_threshold(
    estimator: ClassifierMixin,
    x: pd.DataFrame,
    y: pd.Series,
    objective: str = "f1",
    min_precision: float = 0.5,
) -> tuple[float, dict[str, float]]:
    """Choose a decision threshold from out-of-fold probabilities.

    * ``"f1"`` – maximise F1 for the churn class.
    * ``"recall_at_precision"`` – maximise recall subject to
      precision >= ``min_precision``.
    """
    if objective not in THRESHOLD_OBJECTIVES:
        raise ValueError(f"objective must be one of {THRESHOLD_OBJECTIVES}")

    y_arr = np.asarray(y)
    proba = cross_val_predict(
        estimator, x, y, cv=_cv(), method="predict_proba", n_jobs=-1
    )[:, 1]

    best_t, best_key = 0.5, -1.0
    for t in np.round(np.arange(0.05, 0.96, 0.01), 2):
        pred = (proba >= t).astype(int)
        precision = precision_score(y_arr, pred, zero_division=0)
        recall = recall_score(y_arr, pred, zero_division=0)
        if objective == "recall_at_precision":
            key = recall if precision >= min_precision else -1.0
        else:
            key = f1_score(y_arr, pred, zero_division=0)
        if key > best_key:
            best_key, best_t = key, float(t)

    pred = (proba >= best_t).astype(int)
    stats = {
        "threshold": best_t,
        "cv_precision": float(precision_score(y_arr, pred, zero_division=0)),
        "cv_recall": float(recall_score(y_arr, pred, zero_division=0)),
        "cv_f1": float(f1_score(y_arr, pred, zero_division=0)),
    }
    logger.info(
        "Tuned threshold=%.2f (%s): precision=%.3f recall=%.3f f1=%.3f",
        best_t,
        objective,
        stats["cv_precision"],
        stats["cv_recall"],
        stats["cv_f1"],
    )
    return best_t, stats


def test_report(
    estimator: ClassifierMixin,
    x_test: pd.DataFrame,
    y_test: pd.Series,
    threshold: float,
) -> dict[str, Any]:
    """Held-out test metrics at both the tuned threshold and the default 0.5."""
    y_arr = np.asarray(y_test)
    proba = estimator.predict_proba(x_test)[:, 1]

    def _at(t: float) -> dict[str, Any]:
        pred = (proba >= t).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_arr, pred, labels=[0, 1]).ravel()
        return {
            "threshold": float(t),
            "precision": float(precision_score(y_arr, pred, zero_division=0)),
            "recall": float(recall_score(y_arr, pred, zero_division=0)),
            "f1": float(f1_score(y_arr, pred, zero_division=0)),
            "accuracy": float(np.mean(pred == y_arr)),
            "confusion_matrix": {
                "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)
            },
        }

    prob_true, prob_pred = calibration_curve(y_arr, proba, n_bins=10, strategy="quantile")
    return {
        "roc_auc": float(roc_auc_score(y_arr, proba)),
        "pr_auc": float(average_precision_score(y_arr, proba)),
        "brier": float(brier_score_loss(y_arr, proba)),
        "reliability": {
            "prob_pred": [float(v) for v in prob_pred],
            "prob_true": [float(v) for v in prob_true],
        },
        "tuned": _at(threshold),
        "default_0.5": _at(0.5),
    }
