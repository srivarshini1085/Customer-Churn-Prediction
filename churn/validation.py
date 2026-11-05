"""Input validation and data-drift monitoring.

At training time :meth:`TrainingStats.from_frame` records the range and
distribution of every feature. At serving time :func:`validate_features` flags a
payload whose values fall outside what the model has seen, and
:func:`drift_report` compares a fresh batch against the training distribution
using the Population Stability Index (PSI).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from churn import config

_EPS = 1e-6
_N_BINS = 10
#: PSI thresholds (industry rule of thumb).
PSI_MINOR, PSI_MAJOR = 0.1, 0.2


@dataclass
class NumericStat:
    min: float
    max: float
    mean: float
    std: float
    p01: float
    p99: float
    bin_edges: list[float]
    bin_freq: list[float]


@dataclass
class CategoricalStat:
    categories: list[str]
    freq: dict[str, float]


@dataclass
class TrainingStats:
    """Per-feature range and distribution snapshot of the training data."""

    n_rows: int
    numeric: dict[str, NumericStat] = field(default_factory=dict)
    categorical: dict[str, CategoricalStat] = field(default_factory=dict)

    @classmethod
    def from_frame(cls, x: pd.DataFrame) -> TrainingStats:
        stats = cls(n_rows=len(x))
        for col in config.NUMERIC_COLUMNS:
            s = pd.to_numeric(x[col], errors="coerce").dropna()
            edges = np.histogram_bin_edges(s, bins=_N_BINS)
            freq, _ = np.histogram(s, bins=edges)
            stats.numeric[col] = NumericStat(
                min=float(s.min()),
                max=float(s.max()),
                mean=float(s.mean()),
                std=float(s.std()),
                p01=float(s.quantile(0.01)),
                p99=float(s.quantile(0.99)),
                bin_edges=[float(e) for e in edges],
                bin_freq=_normalise(freq),
            )
        for col in config.CATEGORICAL_COLUMNS + config.PASSTHROUGH_COLUMNS:
            counts = x[col].astype(str).value_counts()
            total = float(counts.sum())
            stats.categorical[col] = CategoricalStat(
                categories=list(counts.index),
                freq={k: v / total for k, v in counts.items()},
            )
        return stats

    def summary(self) -> dict[str, Any]:
        """Compact form for the model manifest."""
        return {
            "n_rows": self.n_rows,
            "numeric": {
                c: {"min": s.min, "max": s.max, "mean": round(s.mean, 4)}
                for c, s in self.numeric.items()
            },
            "categorical": {c: s.categories for c, s in self.categorical.items()},
        }


def _normalise(counts: Sequence[float]) -> list[float]:
    total = float(sum(counts)) or 1.0
    return [c / total for c in counts]


def validate_features(
    features: dict[str, Any], stats: TrainingStats, tolerance: float = 0.25
) -> list[str]:
    """Return human-readable warnings for values outside the training envelope."""
    warnings: list[str] = []

    for col, s in stats.numeric.items():
        raw = features.get(col)
        if raw is None or raw == "":
            continue
        try:
            value = float(raw)
        except (TypeError, ValueError):
            warnings.append(f"{col}={raw!r} is not a number")
            continue
        span = max(s.max - s.min, _EPS)
        low, high = s.min - tolerance * span, s.max + tolerance * span
        if not low <= value <= high:
            warnings.append(f"{col}={value:g} is outside the training range [{s.min:g}, {s.max:g}]")

    for col, cat_s in stats.categorical.items():
        raw = features.get(col)
        if raw is None or raw == "":
            continue
        if str(raw) not in cat_s.categories:
            warnings.append(
                f"{col}={raw!r} was never seen in training "
                f"(known: {', '.join(map(str, cat_s.categories))})"
            )

    return warnings


def population_stability_index(
    expected_freq: Sequence[float], actual_freq: Sequence[float]
) -> float:
    """PSI between two aligned, normalised frequency vectors."""
    expected = np.clip(np.asarray(expected_freq, dtype=float), _EPS, None)
    actual = np.clip(np.asarray(actual_freq, dtype=float), _EPS, None)
    return float(np.sum((actual - expected) * np.log(actual / expected)))


def _numeric_psi(stat: NumericStat, values: pd.Series) -> float:
    series = pd.to_numeric(values, errors="coerce").dropna()
    counts, _ = np.histogram(series, bins=stat.bin_edges)
    return population_stability_index(stat.bin_freq, _normalise(counts))


def _categorical_psi(stat: CategoricalStat, values: pd.Series) -> float:
    actual = values.astype(str).value_counts()
    total = float(actual.sum()) or 1.0
    categories = list(dict.fromkeys([*stat.categories, *actual.index]))
    expected_freq = [stat.freq.get(c, 0.0) for c in categories]
    actual_freq = [actual.get(c, 0) / total for c in categories]
    return population_stability_index(expected_freq, actual_freq)


def drift_report(
    stats: TrainingStats, df: pd.DataFrame, threshold: float = PSI_MAJOR
) -> dict[str, dict[str, Any]]:
    """PSI for every feature present in ``df`` vs. the training distribution."""
    report: dict[str, dict[str, Any]] = {}
    for col, stat in stats.numeric.items():
        if col in df.columns:
            psi = _numeric_psi(stat, df[col])
            report[col] = {"psi": round(psi, 4), "drifted": psi >= threshold}
    for col, cat_stat in stats.categorical.items():
        if col in df.columns:
            psi = _categorical_psi(cat_stat, df[col])
            report[col] = {"psi": round(psi, 4), "drifted": psi >= threshold}
    return report


def drift_status(psi: float) -> str:
    """Traffic light for one feature: green < PSI_MINOR <= amber < PSI_MAJOR <= red."""
    if psi >= PSI_MAJOR:
        return "red"
    if psi >= PSI_MINOR:
        return "amber"
    return "green"


def drift_summary(report: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """Status per feature, counts per status, and whether retraining is recommended.

    Retraining is recommended when any feature is red (a major shift) or when at
    least ``config.DRIFT_AMBER_RETRAIN_COUNT`` features are amber (several moderate
    shifts that together may change what the model sees).
    """
    statuses = {feature: drift_status(row["psi"]) for feature, row in report.items()}
    counts = {s: sum(1 for v in statuses.values() if v == s) for s in ("green", "amber", "red")}
    red = sorted(f for f, s in statuses.items() if s == "red")
    amber = sorted(f for f, s in statuses.items() if s == "amber")
    if red:
        reason = f"major drift (PSI >= {PSI_MAJOR}) in: {', '.join(red)}"
    elif len(amber) >= config.DRIFT_AMBER_RETRAIN_COUNT:
        reason = f"{len(amber)} features show moderate drift: {', '.join(amber)}"
    else:
        reason = ""
    return {
        "statuses": statuses,
        "counts": counts,
        "retrain_recommended": bool(reason),
        "reason": reason or "no feature shows major drift",
    }
