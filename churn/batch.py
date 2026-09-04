"""Batch scoring: score a table of customers in one pass.

Used by ``python -m churn score --input customers.csv --output scored.csv`` and by
the ``POST /predict/batch`` endpoint (which calls :func:`score_records`).
"""

from __future__ import annotations

import logging
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

import pandas as pd

from churn import config
from churn.data import coerce_features
from churn.predict import ChurnModel, load_model, log_prediction, risk_tier
from churn.schema import defaults
from churn.validation import validate_features

logger = logging.getLogger(__name__)


def score_records(
    records: Iterable[Mapping[str, Any]],
    model: ChurnModel | None = None,
    *,
    audit: bool = True,
) -> list[dict[str, Any]]:
    """Score an iterable of ``{column: value}`` mappings."""
    model = model or load_model()
    results: list[dict[str, Any]] = []
    for record in records:
        features = {k: v for k, v in record.items() if k in config.FEATURE_COLUMNS}
        result = model.predict(features)
        if audit:
            log_prediction(features, result)
        results.append(result)
    return results


def score_frame(df: pd.DataFrame, model: ChurnModel | None = None) -> pd.DataFrame:
    """Return ``df`` with churn-probability / tier / prediction / warnings columns.

    Row order is preserved. Columns not in the feature set are passed through
    untouched; missing feature columns are filled from the schema defaults.
    Vectorised — one ``predict_proba`` call for the whole frame.
    """
    model = model or load_model()
    source = coerce_features(df)
    known = [c for c in config.FEATURE_COLUMNS if c in source.columns]
    fallback = defaults()

    features = pd.DataFrame(
        {
            col: (source[col].to_numpy() if col in known else fallback[col])
            for col in config.FEATURE_COLUMNS
        }
    )

    proba = model.pipeline.predict_proba(features)[:, 1]
    out = df.copy()
    out["churn_probability"] = proba
    out["risk_tier"] = [risk_tier(p) for p in proba]
    out["churn_prediction"] = (proba >= model.threshold).astype(int)

    if model.training_stats is not None:
        out["warnings"] = [
            "; ".join(validate_features(row.to_dict(), model.training_stats))
            for _, row in features.iterrows()
        ]
    else:
        out["warnings"] = ""
    return out


def score_csv(
    input_path: str | Path,
    output_path: str | Path,
    version: str | None = None,
) -> Path:
    """Read a CSV, score every row, write the augmented CSV."""
    input_path, output_path = Path(input_path), Path(output_path)
    df = pd.read_csv(input_path)
    model = load_model(version)
    scored = score_frame(df, model)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    scored.to_csv(output_path, index=False)
    logger.info(
        "Scored %s rows from %s -> %s (model %s)",
        len(scored),
        input_path,
        output_path,
        model.version,
    )
    return output_path
