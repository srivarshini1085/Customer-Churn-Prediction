"""Tests for the train pipeline execution and model card generation."""

from __future__ import annotations

import pandas as pd
from sklearn.linear_model import LogisticRegression

from churn.model import Candidate
from churn.train import _model_card, train


def test_model_card_generation() -> None:
    manifest = {
        "version": "test-v1",
        "created_at": "2026-01-01T00:00:00Z",
        "model_name": "logistic_regression",
        "best_params": {"C": 1.0},
        "calibration": "sigmoid",
        "git_sha": "abc1234",
        "python": "3.10",
        "sklearn": "1.3.0",
        "n_train": 100,
        "n_test": 20,
        "data_sha256": "1234567890abcdef",
        "threshold": 0.42,
        "threshold_objective": "f1",
        "threshold_stats": {"cv_precision": 0.6, "cv_recall": 0.7, "cv_f1": 0.65},
        "test_metrics": {
            "roc_auc": 0.85,
            "pr_auc": 0.65,
            "brier": 0.12,
            "tuned": {
                "threshold": 0.42,
                "precision": 0.62,
                "recall": 0.72,
                "f1": 0.67,
                "accuracy": 0.80,
            },
            "default_0.5": {
                "threshold": 0.5,
                "precision": 0.65,
                "recall": 0.65,
                "f1": 0.65,
                "accuracy": 0.81,
            },
        },
        "global_importances": {"tenure": -0.5, "MonthlyCharges": 0.3},
    }
    card = _model_card(manifest)
    assert "# Churn model card" in card
    assert "test-v1" in card
    assert "logistic_regression" in card


def test_train_end_to_end_fast(monkeypatch, sample_xy: tuple[pd.DataFrame, pd.Series]) -> None:
    x, y = sample_xy
    x_sample = x.iloc[:200]
    y_sample = y.iloc[:200]

    monkeypatch.setattr(
        "churn.train.load_clean_split",
        lambda: (
            x_sample.iloc[:150],
            x_sample.iloc[150:],
            y_sample.iloc[:150],
            y_sample.iloc[150:],
        ),
    )
    monkeypatch.setattr(
        "churn.train.candidates",
        lambda: [
            Candidate(
                name="logistic_regression",
                estimator=LogisticRegression(max_iter=100),
                param_grid={"C": [1.0]},
            )
        ],
    )
    monkeypatch.setattr("churn.train._resolve_calibration_method", lambda *args, **kwargs: None)

    trained_model = train(threshold_objective="f1")
    assert trained_model is not None
    assert trained_model.threshold > 0.0
    assert trained_model.version is not None

    # Every registered version also carries its time-to-churn model and results.
    from churn.registry import ModelRegistry

    survival = ModelRegistry().load_survival(trained_model.version)
    assert 0.5 <= survival.metrics["concordance_test"] <= 1.0
    assert "concordance_test" in trained_model.metadata["survival"]
    card = (ModelRegistry().root_dir / trained_model.version / "model_card.md").read_text(
        encoding="utf-8"
    )
    assert "## Time to churn (survival analysis)" in card
    assert "## Fairness" in card
    fairness = trained_model.metadata["fairness"]
    assert set(fairness["attributes"]) == {"gender", "SeniorCitizen", "Partner"}
