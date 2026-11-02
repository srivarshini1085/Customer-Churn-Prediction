"""Unit tests for evaluation, selection metrics, threshold tuning, and reporting."""

from __future__ import annotations

import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression

from churn.evaluate import (
    CandidateResult,
    optimal_threshold,
    select_best,
    selection_key,
    tune_and_score,
)
from churn.evaluate import (
    test_report as generate_test_report,
)
from churn.model import Candidate


def test_candidate_result_and_selection_key() -> None:
    res1 = CandidateResult(
        name="model1",
        best_params={"C": 0.1},
        cv_scores={"roc_auc": 0.850, "recall": 0.700, "pr_auc": 0.650},
        cv_score_std={},
    )
    res2 = CandidateResult(
        name="model2",
        best_params={"C": 1.0},
        cv_scores={"roc_auc": 0.850, "recall": 0.750, "pr_auc": 0.640},
        cv_score_std={},
    )
    # When ROC-AUC is tied, recall breaks the tie
    assert selection_key(res2) > selection_key(res1)
    assert select_best([res1, res2]).name == "model2"


def test_tune_and_score_small(sample_xy: tuple[pd.DataFrame, pd.Series]) -> None:
    x, y = sample_xy
    cand = Candidate(
        name="lr_test",
        estimator=LogisticRegression(max_iter=100),
        param_grid={"C": [1.0]},
    )
    res = tune_and_score(cand, x.iloc[:200], y.iloc[:200])
    assert res.name == "lr_test"
    assert "roc_auc" in res.cv_scores
    assert res.estimator is not None


def test_optimal_threshold_invalid_objective(
    sample_xy: tuple[pd.DataFrame, pd.Series], fitted_model
) -> None:
    x, y = sample_xy
    with pytest.raises(ValueError, match="objective must be one of"):
        optimal_threshold(
            fitted_model.pipeline, x.iloc[:100], y.iloc[:100], objective="invalid_obj"
        )


def test_optimal_threshold_f1_and_recall_at_precision(
    sample_xy: tuple[pd.DataFrame, pd.Series], fitted_model
) -> None:
    x, y = sample_xy
    t_f1, stats_f1 = optimal_threshold(
        fitted_model.pipeline, x.iloc[:200], y.iloc[:200], objective="f1"
    )
    assert 0.05 <= t_f1 <= 0.95
    assert "cv_f1" in stats_f1

    t_rec, stats_rec = optimal_threshold(
        fitted_model.pipeline,
        x.iloc[:200],
        y.iloc[:200],
        objective="recall_at_precision",
        min_precision=0.4,
    )
    assert 0.05 <= t_rec <= 0.95
    assert "cv_recall" in stats_rec


def test_test_report_metrics(sample_xy: tuple[pd.DataFrame, pd.Series], fitted_model) -> None:
    x, y = sample_xy
    report = generate_test_report(fitted_model.pipeline, x.iloc[:200], y.iloc[:200], threshold=0.45)
    assert "roc_auc" in report
    assert "pr_auc" in report
    assert "brier" in report
    assert "reliability" in report
    assert "tuned" in report
    assert report["tuned"]["threshold"] == 0.45
    assert "confusion_matrix" in report["tuned"]
