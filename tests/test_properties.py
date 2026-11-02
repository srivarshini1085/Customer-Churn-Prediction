"""Property-based tests with Hypothesis for feature engineering, risk tiers, and input validation."""

from __future__ import annotations

import hypothesis.strategies as st
import numpy as np
import pandas as pd
import pytest
from hypothesis import given
from pydantic import ValidationError

from churn import config
from churn.features import add_engineered
from churn.predict import risk_tier
from churn.schema import INPUT_FIELDS
from churn.service.models import CustomerFeatures
from churn.validation import TrainingStats, validate_features

# --------------------------------------------------------------------------- #
# Strategies
# --------------------------------------------------------------------------- #

_VALID_INTERNET = st.sampled_from(["DSL", "Fiber optic", "No"])
_YES_NO = st.sampled_from(["Yes", "No", "No internet service"])


@st.composite
def customer_dataframe_strategy(draw: st.DrawFn) -> pd.DataFrame:
    """Generate a DataFrame with valid customer columns for add_engineered."""
    n_rows = draw(st.integers(min_value=1, max_value=20))
    tenure_values = draw(
        st.lists(
            st.floats(min_value=0.0, max_value=120.0, allow_nan=False, allow_infinity=False),
            min_size=n_rows,
            max_size=n_rows,
        )
    )
    internet_values = draw(st.lists(_VALID_INTERNET, min_size=n_rows, max_size=n_rows))

    data: dict[str, list[object]] = {
        "tenure": list(tenure_values),
        "InternetService": list(internet_values),
    }

    for col in config.ADDON_SERVICE_COLUMNS:
        data[col] = draw(st.lists(_YES_NO, min_size=n_rows, max_size=n_rows))

    return pd.DataFrame(data)


# --------------------------------------------------------------------------- #
# add_engineered properties
# --------------------------------------------------------------------------- #


@given(df=customer_dataframe_strategy())
def test_add_engineered_invariants(df: pd.DataFrame) -> None:
    """Engineered features must satisfy domain invariants across all inputs."""
    result = add_engineered(df)

    # 1. Output preserves row count and index
    assert len(result) == len(df)
    assert list(result.index) == list(df.index)

    # 2. has_internet is binary 0 or 1, and matches InternetService != 'No'
    assert set(result["has_internet"].unique()).issubset({0, 1})
    expected_internet = (df["InternetService"] != "No").astype("int64")
    pd.testing.assert_series_equal(result["has_internet"], expected_internet, check_names=False)

    # 3. num_addons is between 0 and 6, and exactly counts 'Yes' add-ons
    assert (result["num_addons"] >= 0).all()
    assert (result["num_addons"] <= len(config.ADDON_SERVICE_COLUMNS)).all()
    expected_addons = df[config.ADDON_SERVICE_COLUMNS].eq("Yes").sum(axis=1).astype("int64")
    pd.testing.assert_series_equal(result["num_addons"], expected_addons, check_names=False)

    # 4. tenure_group is assigned from valid labels
    valid_labels = set(config.TENURE_LABELS)
    assert set(result["tenure_group"].dropna().unique()).issubset(valid_labels)


# --------------------------------------------------------------------------- #
# risk_tier properties
# --------------------------------------------------------------------------- #


@given(prob=st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False))
def test_risk_tier_output_domain(prob: float) -> None:
    """risk_tier must always map probabilities to one of the 3 configured tiers."""
    tier = risk_tier(prob)
    assert tier in {"High", "Moderate", "Low"}


@given(
    p1=st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False),
    p2=st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False),
)
def test_risk_tier_monotonicity(p1: float, p2: float) -> None:
    """Higher churn probability must yield an equal or higher risk tier."""
    rank = {"Low": 0, "Moderate": 1, "High": 2}
    tier1 = risk_tier(p1)
    tier2 = risk_tier(p2)

    if p1 >= p2:
        assert rank[tier1] >= rank[tier2]
    else:
        assert rank[tier1] <= rank[tier2]


@pytest.mark.parametrize(
    ("prob", "expected"),
    [
        (0.0, "Low"),
        (0.399999, "Low"),
        (0.40, "Moderate"),
        (0.699999, "Moderate"),
        (0.70, "High"),
        (1.0, "High"),
    ],
)
def test_risk_tier_boundaries(prob: float, expected: str) -> None:
    assert risk_tier(prob) == expected


# --------------------------------------------------------------------------- #
# Input validation properties
# --------------------------------------------------------------------------- #


@given(
    tenure=st.integers(min_value=0, max_value=100),
    monthly=st.floats(min_value=0.0, max_value=500.0, allow_nan=False, allow_infinity=False),
    senior=st.sampled_from([0, 1]),
    contract=st.sampled_from(["Month-to-month", "One year", "Two year"]),
)
def test_valid_customer_features_payload(
    tenure: int, monthly: float, senior: int, contract: str
) -> None:
    """Pydantic model must accept valid generated fields without error."""
    payload = {
        "tenure": tenure,
        "MonthlyCharges": monthly,
        "SeniorCitizen": senior,
        "Contract": contract,
    }
    model = CustomerFeatures(**payload)
    dumped = model.model_dump(exclude_none=True)
    assert dumped["tenure"] == tenure
    assert np.isclose(dumped["MonthlyCharges"], monthly)
    assert dumped["SeniorCitizen"] == senior
    assert dumped["Contract"] == contract


@given(negative_tenure=st.integers(max_value=-1))
def test_customer_features_rejects_negative_tenure(negative_tenure: int) -> None:
    """Pydantic model must reject negative tenure with ValidationError."""
    with pytest.raises(ValidationError):
        CustomerFeatures(tenure=negative_tenure)


@given(
    extra_key=st.text(min_size=1, max_size=15).filter(
        lambda s: s not in {f.name for f in INPUT_FIELDS}
    )
)
def test_customer_features_forbids_extra_keys(extra_key: str) -> None:
    """Pydantic model config forbids extra unexpected fields."""
    with pytest.raises(ValidationError):
        CustomerFeatures(**{extra_key: "unexpected_val"})


@pytest.fixture(scope="module")
def sample_stats(raw_df: pd.DataFrame) -> TrainingStats:
    return TrainingStats.from_frame(raw_df)


@given(
    tenure_raw=st.one_of(
        st.integers(), st.floats(allow_nan=False, allow_infinity=False), st.text()
    ),
    charges_raw=st.one_of(st.floats(allow_nan=False, allow_infinity=False), st.text()),
)
def test_validate_features_never_crashes(
    tenure_raw: object, charges_raw: object, sample_stats: TrainingStats
) -> None:
    """validate_features must robustly return warnings list for arbitrary input types."""
    features = {
        "tenure": tenure_raw,
        "MonthlyCharges": charges_raw,
        "Contract": "NonexistentContractType",
    }
    warnings = validate_features(features, sample_stats)
    assert isinstance(warnings, list)
    assert all(isinstance(w, str) for w in warnings)
