"""Tests for the FastAPI serving layer (churn.service)."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from churn.service.app import create_app


@pytest.fixture(scope="module")
def client(registered_model):
    """A TestClient with the lifespan run (so the model is loaded)."""
    with TestClient(create_app()) as c:
        yield c


def test_health_ok(client, registered_model):
    version, _ = registered_model
    body = client.get("/health").json()
    assert body["status"] == "ok"
    assert body["model_version"] == version


def test_model_info(client):
    body = client.get("/model").json()
    assert body["model_name"] == "logistic_regression"
    assert "roc_auc" in body["metrics"]
    assert len(body["features"]) > 19


def test_predict_happy_path(client):
    r = client.post(
        "/predict",
        json={"tenure": 3, "Contract": "Month-to-month", "InternetService": "Fiber optic"},
    )
    assert r.status_code == 200
    body = r.json()
    assert 0.0 <= body["probability"] <= 1.0
    assert body["risk_tier"] in ("High", "Moderate", "Low")
    assert body["request_id"]
    assert isinstance(body["top_factors"], list)
    assert r.headers["x-request-id"]


def test_predict_rejects_unknown_field(client):
    r = client.post("/predict", json={"bogus": 1})
    assert r.status_code == 422
    assert "request_id" in r.json()


def test_predict_rejects_bad_enum(client):
    r = client.post("/predict", json={"Contract": "Lifetime"})
    assert r.status_code == 422


def test_predict_out_of_range_returns_warning(client):
    body = client.post("/predict", json={"tenure": 9999}).json()
    assert any("tenure" in w for w in body["warnings"])


def test_batch(client):
    r = client.post(
        "/predict/batch",
        json={"customers": [{"tenure": 1}, {"tenure": 70, "Contract": "Two year"}]},
    )
    assert r.status_code == 200
    body = r.json()
    assert body["count"] == 2
    assert body["predictions"][0]["probability"] > body["predictions"][1]["probability"]


def test_batch_over_limit(client, monkeypatch):
    from churn import config

    monkeypatch.setattr(config.settings, "api_batch_limit", 1)
    r = client.post("/predict/batch", json={"customers": [{"tenure": 1}, {"tenure": 2}]})
    assert r.status_code == 413


def test_metrics_endpoint(client):
    client.post("/predict", json={"tenure": 5})
    text = client.get("/metrics").text
    assert "churn_predictions_total" in text
    assert "churn_prediction_latency_seconds" in text
