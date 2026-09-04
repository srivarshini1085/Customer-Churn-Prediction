"""Inference: the persisted :class:`ChurnModel` and the module-level scoring API.

A model bundles two fitted pipelines:

* ``pipeline``       – **calibrated** (``CalibratedClassifierCV``); its
  ``predict_proba`` output is what drives the label and the risk tier.
* ``base_pipeline``  – the **uncalibrated** winning estimator, refit on all of
  the training data, kept only so per-prediction explanations can read its
  coefficients / feature importances.

Plus the tuned ``threshold``, the transformed ``feature_names``, a
:class:`~churn.validation.TrainingStats` snapshot (for input validation), and a
free-form ``metadata`` dict.
"""

from __future__ import annotations

import datetime as dt
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

from churn import config
from churn.features import pretty_feature_name
from churn.schema import defaults as _schema_defaults
from churn.validation import TrainingStats, validate_features

logger = logging.getLogger(__name__)


@dataclass
class ChurnModel:
    """Everything needed to score a customer, in one picklable object."""

    pipeline: Any  # CalibratedClassifierCV wrapping the feature pipeline
    base_pipeline: Pipeline  # uncalibrated winner, for explanations only
    threshold: float
    feature_names: list[str]
    training_stats: TrainingStats | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    # -- persistence --------------------------------------------------- #

    def save(self, path: str | Path) -> Path:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, target)
        logger.info("Saved model to %s", target)
        return target

    @classmethod
    def load(cls, path: str | Path) -> ChurnModel:
        source = Path(path)
        if not source.exists():
            raise FileNotFoundError(f"No model artifact at {source}")
        model = joblib.load(source)
        if not isinstance(model, cls):
            raise TypeError(f"{source} does not contain a ChurnModel")
        return model

    # -- metadata ---------------------------------------------------- #

    @property
    def version(self) -> str:
        return self.metadata.get("version", "unversioned")

    @property
    def is_linear(self) -> bool:
        """True when the base estimator exposes signed coefficients."""
        return hasattr(self.base_pipeline[-1], "coef_")

    # -- scoring --------------------------------------------------------- #

    def _row(self, features: dict[str, Any]) -> pd.DataFrame:
        """1-row frame; omitted fields fall back to their schema default."""
        unknown = set(features) - set(config.FEATURE_COLUMNS)
        if unknown:
            raise ValueError(f"Unknown feature(s): {sorted(unknown)}")
        row = _schema_defaults()
        row.update({k: v for k, v in features.items() if v is not None})
        return pd.DataFrame([{col: row[col] for col in config.FEATURE_COLUMNS}])

    def predict_proba(self, features: dict[str, Any]) -> float:
        """Calibrated probability that the customer churns."""
        return float(self.pipeline.predict_proba(self._row(features))[0, 1])

    def warnings_for(self, features: dict[str, Any]) -> list[str]:
        if self.training_stats is None:
            return []
        return validate_features(features, self.training_stats)

    def predict(self, features: dict[str, Any]) -> dict[str, Any]:
        """Full result for one customer."""
        proba = self.predict_proba(features)
        return {
            "churn": bool(proba >= self.threshold),
            "probability": proba,
            "risk_tier": risk_tier(proba),
            "threshold": self.threshold,
            "model_version": self.version,
            "warnings": self.warnings_for(features),
        }

    def top_factors(
        self, features: dict[str, Any], n: int = 6
    ) -> list[tuple[str, float]]:
        """The ``n`` features that weigh most on this customer's score.

        Linear model → signed contribution (``value * coefficient``, + pushes
        churn up). Tree model → global importance restricted to the features
        active for this row (unsigned — see :attr:`is_linear`).
        """
        transformed = np.asarray(
            self.base_pipeline[:-1].transform(self._row(features))
        )[0]

        if self.is_linear:
            contrib = transformed * np.ravel(self.base_pipeline[-1].coef_)
        else:
            importances = self.metadata.get("global_importances") or {}
            imp = np.array([importances.get(f, 0.0) for f in self.feature_names])
            contrib = np.where(np.abs(transformed) > 1e-9, imp, 0.0)

        order = np.argsort(np.abs(contrib))[::-1][:n]
        return [
            (pretty_feature_name(self.feature_names[i]), float(contrib[i]))
            for i in order
            if abs(contrib[i]) > 1e-9
        ]


def risk_tier(probability: float) -> str:
    """Map a churn probability to a fixed business risk band (see config)."""
    for cutoff, label in config.RISK_BANDS:
        if probability >= cutoff:
            return label
    return config.RISK_BANDS[-1][1]


# --------------------------------------------------------------------------- #
# Module-level API (registry-backed, process-cached)
# --------------------------------------------------------------------------- #

_CACHE: dict[str, ChurnModel] = {}


def load_model(version: str | None = None) -> ChurnModel:
    """Load a model from the registry (``version`` defaults to ``settings.model_version``)."""
    from churn.registry import ModelRegistry

    registry = ModelRegistry()
    resolved = registry.resolve(version or config.settings.model_version)
    if resolved not in _CACHE:
        _CACHE[resolved] = registry.load(resolved)
    return _CACHE[resolved]


def clear_cache() -> None:
    _CACHE.clear()


def predict_one(
    features: dict[str, Any], version: str | None = None
) -> dict[str, Any]:
    """Score one customer with the current (or a pinned) model."""
    result = load_model(version).predict(features)
    log_prediction(features, result)
    return result


def log_prediction(features: dict[str, Any], result: dict[str, Any]) -> None:
    """Append one prediction to the JSONL audit log when configured."""
    path = config.settings.prediction_log_path
    if path is None:
        return
    record = {
        "ts": dt.datetime.now(dt.timezone.utc).isoformat(timespec="milliseconds"),
        "model_version": result.get("model_version"),
        "features": features,
        "probability": result.get("probability"),
        "risk_tier": result.get("risk_tier"),
        "churn": result.get("churn"),
        "warnings": result.get("warnings"),
    }
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(record, default=str) + "\n")
    except OSError:  # pragma: no cover - logging must never break scoring
        logger.warning("Could not write prediction log to %s", path, exc_info=True)
