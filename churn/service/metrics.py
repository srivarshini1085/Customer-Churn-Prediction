"""Prometheus metrics for the serving layer."""

from __future__ import annotations

from prometheus_client import CONTENT_TYPE_LATEST, Counter, Histogram, generate_latest

PREDICTIONS = Counter(
    "churn_predictions_total",
    "Predictions served, by risk tier and endpoint.",
    ["tier", "endpoint"],
)
PREDICTION_ERRORS = Counter(
    "churn_prediction_errors_total",
    "Requests that failed before a prediction was produced.",
    ["endpoint"],
)
PREDICTION_WARNINGS = Counter(
    "churn_prediction_warnings_total",
    "Predictions returned with at least one input-validation warning.",
    ["endpoint"],
)
LATENCY = Histogram(
    "churn_prediction_latency_seconds",
    "End-to-end prediction latency.",
    ["endpoint"],
    buckets=(0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0),
)


def render() -> tuple[bytes, str]:
    """Prometheus exposition payload + content-type."""
    return generate_latest(), CONTENT_TYPE_LATEST
