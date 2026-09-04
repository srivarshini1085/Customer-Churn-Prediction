"""Tests for churn.data.clean."""

from __future__ import annotations

import numpy as np
import pandas as pd

from churn import config
from churn.data import clean


def test_clean_drops_id_and_target_from_features(xy):
    x, y = xy
    assert config.ID_COLUMN not in x.columns
    assert config.TARGET_COLUMN not in x.columns
    assert list(x.columns) == config.FEATURE_COLUMNS
    assert x.shape[1] == 19


def test_target_is_binary_int(xy):
    _, y = xy
    assert set(y.unique()) == {0, 1}
    assert y.dtype == np.int64
    assert not y.isna().any()


def test_totalcharges_coerced_to_numeric(raw_df):
    x, _ = clean(raw_df)
    assert pd.api.types.is_numeric_dtype(x["TotalCharges"])
    # the 11 blank strings in the source become NaN (imputed later in the pipeline)
    assert x["TotalCharges"].isna().sum() == 11


def test_clean_does_not_mutate_input(raw_df):
    before = raw_df.copy()
    clean(raw_df)
    pd.testing.assert_frame_equal(raw_df, before)


def test_churn_rate_is_imbalanced(xy):
    _, y = xy
    assert 0.2 < y.mean() < 0.35
