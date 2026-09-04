"""Shared fixtures, reference profiles and an isolated model registry."""

from __future__ import annotations

import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression

from churn import config
from churn.data import clean, load_raw, split
from churn.features import build_feature_pipeline
from churn.predict import ChurnModel, clear_cache
from churn.validation import TrainingStats


@pytest.fixture(scope="session", autouse=True)
def isolated_registry(tmp_path_factory):
    """Point the whole suite at a throwaway registry under tmp."""
    original = config.settings.models_dir
    config.settings.models_dir = tmp_path_factory.mktemp("models")
    clear_cache()
    yield
    config.settings.models_dir = original
    clear_cache()


@pytest.fixture(scope="session")
def raw_df() -> pd.DataFrame:
    return load_raw()


@pytest.fixture(scope="session")
def xy(raw_df: pd.DataFrame):
    return clean(raw_df)


@pytest.fixture(scope="session")
def sample_xy(xy):
    x, y = xy
    x_s = x.sample(n=1500, random_state=0)
    return x_s, y.loc[x_s.index]


@pytest.fixture(scope="session")
def fitted_model(xy) -> ChurnModel:
    """A real (fast, un-tuned, un-calibrated) linear ChurnModel."""
    x, y = xy
    x_train, _, y_train, _ = split(x, y)
    pipe = build_feature_pipeline()
    pipe.steps.append(("clf", LogisticRegression(max_iter=1000, class_weight="balanced")))
    pipe.fit(x_train, y_train)
    names = list(pipe.named_steps["prep"].get_feature_names_out())
    return ChurnModel(
        pipeline=pipe,
        base_pipeline=pipe,
        threshold=0.5,
        feature_names=names,
        training_stats=TrainingStats.from_frame(x_train),
        metadata={"version": "test-0001", "model_name": "logistic_regression"},
    )


@pytest.fixture(scope="session")
def registered_model(fitted_model):
    """The fitted model saved into the (isolated) registry and promoted to latest."""
    from churn.registry import ModelRegistry

    registry = ModelRegistry()
    version = registry.save(
        fitted_model,
        {**fitted_model.metadata, "test_metrics": {"roc_auc": 0.84, "brier": 0.13,
                                                   "tuned": {"recall": 0.7, "precision": 0.5}}},
        "# test model card\n",
    )
    fitted_model.metadata["version"] = version
    clear_cache()
    return version, fitted_model


HIGH_RISK = {
    "gender": "Female", "SeniorCitizen": 1, "Partner": "No", "Dependents": "No",
    "tenure": 1, "PhoneService": "Yes", "MultipleLines": "No",
    "InternetService": "Fiber optic", "OnlineSecurity": "No", "OnlineBackup": "No",
    "DeviceProtection": "No", "TechSupport": "No", "StreamingTV": "No",
    "StreamingMovies": "No", "Contract": "Month-to-month", "PaperlessBilling": "Yes",
    "PaymentMethod": "Electronic check", "MonthlyCharges": 95.0, "TotalCharges": 95.0,
}

LOW_RISK = {
    "gender": "Male", "SeniorCitizen": 0, "Partner": "Yes", "Dependents": "Yes",
    "tenure": 68, "PhoneService": "Yes", "MultipleLines": "Yes",
    "InternetService": "DSL", "OnlineSecurity": "Yes", "OnlineBackup": "Yes",
    "DeviceProtection": "Yes", "TechSupport": "Yes", "StreamingTV": "Yes",
    "StreamingMovies": "Yes", "Contract": "Two year", "PaperlessBilling": "No",
    "PaymentMethod": "Bank transfer (automatic)", "MonthlyCharges": 65.0,
    "TotalCharges": 4400.0,
}
