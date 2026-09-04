"""Tests for churn.batch."""

from __future__ import annotations

import pandas as pd

from churn.batch import score_csv, score_frame, score_records
from tests.conftest import HIGH_RISK, LOW_RISK


def test_score_frame_adds_columns_and_keeps_order(raw_df, fitted_model):
    sample = raw_df.head(20).copy()
    scored = score_frame(sample, fitted_model)

    assert len(scored) == 20
    assert list(scored["customerID"]) == list(sample["customerID"])
    for col in ("churn_probability", "risk_tier", "churn_prediction", "warnings"):
        assert col in scored.columns
    assert scored["churn_probability"].between(0, 1).all()
    assert set(scored["churn_prediction"].unique()) <= {0, 1}


def test_score_frame_handles_blank_totalcharges(raw_df, fitted_model):
    # rows 488/... have a blank TotalCharges in the source data
    blanks = raw_df[raw_df["TotalCharges"] == " "]
    scored = score_frame(blanks, fitted_model)
    assert scored["churn_probability"].notna().all()


def test_score_records(fitted_model):
    results = score_records([HIGH_RISK, LOW_RISK], fitted_model)
    assert len(results) == 2
    assert results[0]["probability"] > results[1]["probability"]


def test_score_csv_roundtrip(raw_df, registered_model, tmp_path):
    src = tmp_path / "in.csv"
    raw_df.head(30).to_csv(src, index=False)
    out = score_csv(src, tmp_path / "out.csv")

    result = pd.read_csv(out)
    assert len(result) == 30
    assert "risk_tier" in result.columns
