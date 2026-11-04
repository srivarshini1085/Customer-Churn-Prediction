"""Inference: the persisted :class:`ChurnModel` and the module-level scoring API.

A model bundles two fitted pipelines:

* ``pipeline``       – **calibrated** (``CalibratedClassifierCV``); its
  ``predict_proba`` output is what drives the label and the risk tier.
* ``base_pipeline``  – the **uncalibrated** winning estimator, refit on all of
  the training data, kept only for per-prediction explanations: signed
  coefficient contributions for a linear model, SHAP values for a tree model.

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
    # Lazily built SHAP explainer (tree models only). Never pickled: it is cheap
    # to rebuild and would tie saved artifacts to the installed shap version.
    _explainer: Any = field(default=None, init=False, repr=False, compare=False)

    def __getstate__(self) -> dict[str, Any]:
        state = self.__dict__.copy()
        state["_explainer"] = None
        return state

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
        return str(self.metadata.get("version", "unversioned"))

    @property
    def is_linear(self) -> bool:
        """True when the base estimator exposes signed coefficients."""
        return hasattr(self.base_pipeline[-1], "coef_")

    # -- scoring --------------------------------------------------------- #

    def _frame(self, rows: list[dict[str, Any]]) -> pd.DataFrame:
        """One row per customer; omitted fields fall back to their schema default."""
        records = []
        for features in rows:
            unknown = set(features) - set(config.FEATURE_COLUMNS)
            if unknown:
                raise ValueError(f"Unknown feature(s): {sorted(unknown)}")
            row = _schema_defaults()
            row.update({k: v for k, v in features.items() if v is not None})
            records.append({col: row[col] for col in config.FEATURE_COLUMNS})
        return pd.DataFrame(records, columns=config.FEATURE_COLUMNS)

    def _row(self, features: dict[str, Any]) -> pd.DataFrame:
        return self._frame([features])

    def predict_proba(self, features: dict[str, Any]) -> float:
        """Calibrated probability that the customer churns."""
        return float(self.pipeline.predict_proba(self._row(features))[0, 1])

    def predict_proba_many(self, rows: list[dict[str, Any]]) -> np.ndarray:
        """Calibrated churn probabilities for many customers in one pipeline pass."""
        return np.asarray(self.pipeline.predict_proba(self._frame(rows))[:, 1], dtype=float)

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

    def top_factors(self, features: dict[str, Any], n: int = 6) -> list[tuple[str, float]]:
        """The ``n`` features that weigh most on this customer's score.

        Both model families return *signed* contributions (positive pushes churn
        up), sorted by magnitude, in the base model's own output units (log-odds
        for logistic regression and gradient boosting, probability for a random
        forest):

        * Linear model → ``scaled value * coefficient`` (exact for a logistic
          regression, and free to compute).
        * Tree model → SHAP values from :class:`shap.TreeExplainer`, which
          split the prediction fairly across features, including interactions.

        Explanations describe the uncalibrated ``base_pipeline``. Calibration is a
        monotonic re-mapping, so the ranking and direction of factors still hold.
        """
        transformed = np.asarray(
            self.base_pipeline[:-1].transform(self._row(features)), dtype=float
        )[0]

        if self.is_linear:
            contrib = transformed * np.ravel(self.base_pipeline[-1].coef_)
        else:
            contrib = self._shap_contributions(transformed)

        order = np.argsort(np.abs(contrib))[::-1][:n]
        return [
            (pretty_feature_name(self.feature_names[i]), float(contrib[i]))
            for i in order
            if abs(contrib[i]) > 1e-9
        ]

    @property
    def explanation_method(self) -> str:
        """How :meth:`top_factors` is computed: ``"coefficients"`` or ``"shap"``."""
        return "coefficients" if self.is_linear else "shap"

    def _shap_contributions(self, transformed: np.ndarray) -> np.ndarray:
        """SHAP values for the churn class for one already-transformed row."""
        if getattr(self, "_explainer", None) is None:
            import shap  # imported lazily: slow to import and only needed for trees

            self._explainer = shap.TreeExplainer(self.base_pipeline[-1])
        values = self._explainer.shap_values(transformed.reshape(1, -1))
        return churn_class_shap(values)


def churn_class_shap(values: Any) -> np.ndarray:
    """Normalise ``TreeExplainer.shap_values`` output to one 1-D row for class 1.

    Depending on the estimator and the shap version, the result is a list with
    one array per class, an array of shape ``(rows, features, classes)``, or a
    single ``(rows, features)`` array that already refers to the positive class.
    """
    if isinstance(values, list):
        values = values[-1]
    array = np.asarray(values, dtype=float)
    if array.ndim == 3:
        array = array[..., -1]
    return np.asarray(array[0], dtype=float)


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


def predict_one(features: dict[str, Any], version: str | None = None) -> dict[str, Any]:
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
