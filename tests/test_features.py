"""Tests for churn.features."""

from __future__ import annotations

import numpy as np
import pandas as pd

from churn.features import add_engineered, build_feature_pipeline, pretty_feature_name
from churn.schema import defaults


def test_add_engineered_creates_expected_columns(xy):
    x, _ = xy
    out = add_engineered(x.head(50))
    for col in ("tenure_group", "has_internet", "num_addons"):
        assert col in out.columns
    assert out["num_addons"].between(0, 6).all()
    assert set(out["has_internet"].unique()) <= {0, 1}


def test_add_engineered_is_robust_to_missing_values():
    row = pd.DataFrame([defaults()])
    row.loc[0, "tenure"] = np.nan
    row.loc[0, "TotalCharges"] = np.nan
    out = add_engineered(row)
    assert out["num_addons"].iloc[0] == 0
    assert out["tenure_group"].notna().all()


def test_pretty_feature_name():
    assert pretty_feature_name("cat__Contract_Month-to-month") == "Contract = Month-to-month"
    assert pretty_feature_name("cat__tenure_group_48m+") == "tenure_group = 48m+"
    assert pretty_feature_name("num__num_addons") == "num_addons"
    assert pretty_feature_name("pass__SeniorCitizen") == "SeniorCitizen"


def test_pipeline_transforms_and_imputes(sample_xy):
    x, y = sample_xy
    pipe = build_feature_pipeline()
    matrix = pipe.fit_transform(x, y)
    assert matrix.shape[0] == len(x)
    assert np.isfinite(matrix).all()  # SimpleImputer removed every NaN


def test_unseen_category_does_not_raise(sample_xy):
    x, y = sample_xy
    pipe = build_feature_pipeline()
    pipe.fit(x, y)

    weird = x.head(1).copy()
    weird.loc[weird.index[0], "Contract"] = "Lifetime membership"  # never seen
    matrix = pipe.transform(weird)
    assert np.isfinite(matrix).all()
