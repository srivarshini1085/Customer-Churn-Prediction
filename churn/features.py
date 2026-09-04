"""Feature engineering, the preprocessing ``ColumnTransformer`` and feature-name
formatting.

``add_engineered`` is a module-level function (so it pickles cleanly inside a
``FunctionTransformer``) that appends a few domain features. The
``ColumnTransformer`` then imputes, scales and one-hot encodes. The encoder uses
``handle_unknown="ignore"`` so an unseen category at inference time becomes an
all-zero group instead of raising or silently corrupting the row.
"""

from __future__ import annotations

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

from churn import config


def add_engineered(x: pd.DataFrame) -> pd.DataFrame:
    """Append engineered columns to a copy of ``x``.

    * ``tenure_group`` – categorical tenure bucket
    * ``has_internet`` – 1 if the customer has any internet service
    * ``num_addons``   – count of the six add-on services set to "Yes"

    Everything here is derived from a single row and is robust to partially
    specified inputs (a ratio like total/tenure is deliberately avoided because a
    caller can easily supply an inconsistent ``TotalCharges``/``tenure`` pair).
    """
    x = x.copy()

    tenure = pd.to_numeric(x["tenure"], errors="coerce").fillna(0)

    x["tenure_group"] = pd.cut(
        tenure, bins=config.TENURE_BINS, labels=config.TENURE_LABELS
    ).astype(object)
    x["has_internet"] = x["InternetService"].astype(str).ne("No").astype("int64")
    x["num_addons"] = (
        x[config.ADDON_SERVICE_COLUMNS].astype(str).eq("Yes").sum(axis=1).astype("int64")
    )

    return x


def build_preprocessor() -> ColumnTransformer:
    """Column transformer: scale numerics, one-hot categoricals, pass binaries."""
    numeric = config.NUMERIC_COLUMNS + config.ENGINEERED_NUMERIC_COLUMNS
    categorical = config.CATEGORICAL_COLUMNS + config.ENGINEERED_CATEGORICAL_COLUMNS
    passthrough = config.PASSTHROUGH_COLUMNS + config.ENGINEERED_PASSTHROUGH_COLUMNS

    numeric_pipe = Pipeline(
        [
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
        ]
    )
    categorical_pipe = Pipeline(
        [
            ("impute", SimpleImputer(strategy="most_frequent")),
            ("encode", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("num", numeric_pipe, numeric),
            ("cat", categorical_pipe, categorical),
            ("pass", "passthrough", passthrough),
        ],
        remainder="drop",
        verbose_feature_names_out=True,
    )


def build_feature_pipeline() -> Pipeline:
    """The shared prefix of every candidate pipeline: engineer -> preprocess."""
    return Pipeline(
        [
            ("engineer", FunctionTransformer(add_engineered)),
            ("prep", build_preprocessor()),
        ]
    )


_NAME_PREFIXES = ("num__", "cat__", "pass__", "remainder__")
_CATEGORICAL_NAMES = tuple(
    sorted(
        config.CATEGORICAL_COLUMNS + config.ENGINEERED_CATEGORICAL_COLUMNS,
        key=len,
        reverse=True,
    )
)


def pretty_feature_name(name: str) -> str:
    """Format a transformed feature name for display.

    ``cat__Contract_Month-to-month`` -> ``Contract = Month-to-month``;
    ``cat__tenure_group_48m+``       -> ``tenure_group = 48m+``;
    ``num__num_addons``              -> ``num_addons``.
    """
    is_categorical = name.startswith("cat__")
    for prefix in _NAME_PREFIXES:
        name = name.removeprefix(prefix)
    if is_categorical:
        for column in _CATEGORICAL_NAMES:
            if name.startswith(f"{column}_"):
                return f"{column} = {name[len(column) + 1:]}"
    return name
