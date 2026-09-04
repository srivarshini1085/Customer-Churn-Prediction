"""Dataset loading, cleaning and splitting.

The cleaning here is deliberately minimal — only the transforms that must happen
*before* the data reaches the modelling pipeline (dropping the ID column,
coercing the mistyped ``TotalCharges`` column, decoding the target). Imputation,
scaling and encoding all live inside the pipeline so they are refit per CV fold
and travel with the saved model.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from churn import config

logger = logging.getLogger(__name__)


def load_raw(path: str | Path | None = None) -> pd.DataFrame:
    """Load the raw Telco CSV exactly as shipped."""
    csv_path = Path(path) if path is not None else config.DATA_PATH
    if not csv_path.exists():
        raise FileNotFoundError(f"Dataset not found at {csv_path}")
    df = pd.read_csv(csv_path)
    logger.info("Loaded raw dataset %s rows x %s cols from %s", *df.shape, csv_path)
    return df


def _decode_target(target: pd.Series) -> pd.Series:
    """Decode the churn target to an ``int64`` 0/1 series.

    Handles both the raw ``Yes``/``No`` strings and an already-numeric column
    (pandas 3 reads the CSV column as ``str``, not ``object``, so we cannot
    branch on dtype alone).
    """
    if pd.api.types.is_numeric_dtype(target):
        return target.astype("int64")
    y = target.astype(str).map({"Yes": 1, "No": 0})
    if y.isna().any():
        bad = sorted(set(target.astype(str)) - {"Yes", "No"})
        raise ValueError(f"Unexpected values in target column: {bad}")
    return y.astype("int64")


def coerce_features(df: pd.DataFrame) -> pd.DataFrame:
    """Feature-side cleaning shared by training and batch scoring.

    Drops ``customerID`` and coerces the mistyped ``TotalCharges`` column to
    numeric (blank strings become ``NaN``, imputed later inside the pipeline).
    """
    df = df.copy()
    if config.ID_COLUMN in df.columns:
        df = df.drop(columns=config.ID_COLUMN)
    if "TotalCharges" in df.columns:
        df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    return df


def clean(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Return ``(X, y)`` ready for the modelling pipeline.

    * ``customerID`` is dropped, ``TotalCharges`` coerced (see :func:`coerce_features`).
    * ``Churn`` is decoded from ``Yes``/``No`` to ``1``/``0``.
    """
    df = coerce_features(df)
    y = _decode_target(df[config.TARGET_COLUMN])
    x = df[config.FEATURE_COLUMNS].copy()
    logger.info(
        "Cleaned dataset: %s features, churn rate %.1f%%",
        x.shape[1],
        100 * y.mean(),
    )
    return x, y


def split(
    x: pd.DataFrame, y: pd.Series
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Stratified train/test split using the configured seed and size."""
    return train_test_split(
        x,
        y,
        test_size=config.TEST_SIZE,
        random_state=config.RANDOM_STATE,
        stratify=y,
    )


def load_clean_split() -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Convenience: raw CSV -> cleaned -> stratified split."""
    x, y = clean(load_raw())
    return split(x, y)
