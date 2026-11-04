"""Comprehensive API contract tests for all endpoints and error response schemas."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from churn import config
from churn.service.app import create_app
from churn.service.deps import ModelState


@pytest.fixture(scope="module")
def client(registered_model: tuple[str, object]) -> TestClient:
    """A TestClient with the lifespan run (so the model is loaded)."""
    with TestClient(create_app()) as c:
        yield c


# --------------------------------------------------------------------------- #
# GET /health Contracts
# --------------------------------------------------------------------------- #


def test_contract_health_ok(client: TestClient, registered_model: tuple[str, object]) -> None:
    version, _ = registered_model
    response = client.get("/health")
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("application/json")

    data = response.json()
    assert isinstance(data, dict)
    assert set(data.keys()) == {"status", "model_version", "uptime_seconds"}
    assert data["status"] == "ok"
    assert data["model_version"] == version
    assert isinstance(data["uptime_seconds"], (int, float))
    assert data["uptime_seconds"] >= 0.0


def test_contract_health_degraded() -> None:
    """When model_state is unhealthy, /health must return 503 degraded contract."""
    app = create_app()
    with TestClient(app) as c:
        app.state.model_state = ModelState(model=None, started_at=0.0)
        response = c.get("/health")
        assert response.status_code == 503
        data = response.json()
        assert data["status"] == "degraded"
        assert data["model_version"] is None
        assert isinstance(data["uptime_seconds"], (int, float))


# --------------------------------------------------------------------------- #
# GET /model Contracts
# --------------------------------------------------------------------------- #


def test_contract_model_info_ok(client: TestClient, registered_model: tuple[str, object]) -> None:
    version, _ = registered_model
    response = client.get("/model")
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("application/json")

    data = response.json()
    expected_keys = {
        "version",
        "model_name",
        "created_at",
        "threshold",
        "threshold_objective",
        "calibration",
        "metrics",
        "features",
    }
    assert set(data.keys()) == expected_keys
    assert data["version"] == version
    assert isinstance(data["model_name"], str)
    assert isinstance(data["threshold"], float)
    assert 0.0 <= data["threshold"] <= 1.0
    assert isinstance(data["metrics"], dict)
    for metric_name in ("roc_auc", "pr_auc", "brier", "recall", "precision"):
        assert metric_name in data["metrics"]
    assert isinstance(data["features"], list)
    assert len(data["features"]) > 0


# --------------------------------------------------------------------------- #
# POST /predict Contracts
# --------------------------------------------------------------------------- #


def test_contract_predict_success(client: TestClient) -> None:
    payload = {
        "tenure": 12,
        "Contract": "Month-to-month",
        "MonthlyCharges": 70.0,
        "InternetService": "Fiber optic",
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    assert "x-request-id" in response.headers

    data = response.json()
    expected_keys = {
        "churn",
        "probability",
        "risk_tier",
        "threshold",
        "model_version",
        "warnings",
        "top_factors",
        "request_id",
        "latency_ms",
    }
    assert set(data.keys()) == expected_keys
    assert isinstance(data["churn"], bool)
    assert isinstance(data["probability"], float)
    assert 0.0 <= data["probability"] <= 1.0
    assert data["risk_tier"] in {"High", "Moderate", "Low"}
    assert isinstance(data["threshold"], float)
    assert isinstance(data["model_version"], str)
    assert isinstance(data["warnings"], list)
    assert isinstance(data["top_factors"], list)
    for factor in data["top_factors"]:
        assert "feature" in factor and isinstance(factor["feature"], str)
        assert "contribution" in factor and isinstance(factor["contribution"], float)
    assert isinstance(data["request_id"], str) and data["request_id"]
    assert data["request_id"] == response.headers["x-request-id"]
    assert isinstance(data["latency_ms"], float) and data["latency_ms"] >= 0.0


def test_contract_predict_error_unknown_field(client: TestClient) -> None:
    response = client.post("/predict", json={"unknown_feature": "bad_value"})
    assert response.status_code == 422
    data = response.json()
    assert "detail" in data
    assert "request_id" in data
    assert "unknown_feature" in data["detail"]


def test_contract_predict_error_invalid_type(client: TestClient) -> None:
    response = client.post("/predict", json={"tenure": "not-an-integer"})
    assert response.status_code == 422
    data = response.json()
    assert "detail" in data
    assert "request_id" in data
    assert "tenure" in data["detail"]


def test_contract_predict_error_negative_value(client: TestClient) -> None:
    response = client.post("/predict", json={"tenure": -5})
    assert response.status_code == 422
    data = response.json()
    assert "detail" in data
    assert "request_id" in data


# --------------------------------------------------------------------------- #
# POST /predict/batch Contracts
# --------------------------------------------------------------------------- #


def test_contract_batch_success(client: TestClient) -> None:
    payload = {
        "customers": [
            {"tenure": 5, "Contract": "Month-to-month"},
            {"tenure": 40, "Contract": "Two year"},
        ]
    }
    response = client.post("/predict/batch", json=payload)
    assert response.status_code == 200

    data = response.json()
    assert set(data.keys()) == {"model_version", "count", "predictions"}
    assert data["count"] == 2
    assert len(data["predictions"]) == 2
    for pred in data["predictions"]:
        assert "churn" in pred
        assert "probability" in pred
        assert "risk_tier" in pred
        assert "request_id" in pred


def test_contract_batch_error_empty_list(client: TestClient) -> None:
    response = client.post("/predict/batch", json={"customers": []})
    assert response.status_code == 422
    data = response.json()
    assert "detail" in data
    assert "request_id" in data


def test_contract_batch_error_limit_exceeded(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(config.settings, "api_batch_limit", 2)
    payload = {
        "customers": [
            {"tenure": 1},
            {"tenure": 2},
            {"tenure": 3},
        ]
    }
    response = client.post("/predict/batch", json=payload)
    assert response.status_code == 413
    data = response.json()
    assert "detail" in data
    assert "exceeds the limit" in data["detail"]


def test_contract_batch_error_unknown_field(client: TestClient) -> None:
    response = client.post("/predict/batch", json={"customers": [{"bogus": 1}]})
    assert response.status_code == 422
    data = response.json()
    assert set(data.keys()) == {"detail", "request_id"}
    assert "bogus" in data["detail"]


# --------------------------------------------------------------------------- #
# POST /explain/whatif Contracts
# --------------------------------------------------------------------------- #


def test_contract_whatif_success(client: TestClient) -> None:
    payload = {
        "customer": {"tenure": 2, "Contract": "Month-to-month", "InternetService": "Fiber optic"},
        "max_changes": 2,
        "top_k": 3,
    }
    response = client.post("/explain/whatif", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert set(data) == {
        "model_version",
        "baseline_probability",
        "baseline_risk_tier",
        "scenarios",
        "request_id",
    }
    assert 0.0 <= data["baseline_probability"] <= 1.0
    assert data["baseline_risk_tier"] in {"High", "Moderate", "Low"}
    assert data["request_id"] == response.headers["x-request-id"]
    assert 1 <= len(data["scenarios"]) <= 3
    for scenario in data["scenarios"]:
        assert set(scenario) == {"changes", "probability", "delta", "risk_tier"}
        assert scenario["probability"] < data["baseline_probability"]
        assert scenario["delta"] < 0
        assert 1 <= len(scenario["changes"]) <= 2
        for change in scenario["changes"]:
            assert set(change) == {"feature", "from_value", "to_value"}
            assert change["feature"] not in config.IMMUTABLE_FEATURES


def test_contract_whatif_defaults(client: TestClient) -> None:
    response = client.post("/explain/whatif", json={"customer": {}})
    assert response.status_code == 200
    assert len(response.json()["scenarios"]) <= 5


@pytest.mark.parametrize(
    ("payload", "field"),
    [
        ({}, "customer"),
        ({"customer": {}, "max_changes": 0}, "max_changes"),
        ({"customer": {}, "max_changes": 4}, "max_changes"),
        ({"customer": {}, "top_k": 21}, "top_k"),
        ({"customer": {"bogus": 1}}, "bogus"),
        ({"customer": {"tenure": -1}}, "tenure"),
        ({"customer": {}, "extra": True}, "extra"),
    ],
)
def test_contract_whatif_validation_errors(client: TestClient, payload: dict, field: str) -> None:
    response = client.post("/explain/whatif", json=payload)
    assert response.status_code == 422
    data = response.json()
    assert set(data) == {"detail", "request_id"}
    assert field in data["detail"]


# --------------------------------------------------------------------------- #
# GET /metrics Contracts
# --------------------------------------------------------------------------- #


def test_contract_prometheus_metrics(client: TestClient) -> None:
    response = client.get("/metrics")
    assert response.status_code == 200
    assert "text/plain" in response.headers["content-type"]
    body = response.text
    assert "churn_predictions_total" in body
    assert "churn_prediction_latency_seconds" in body
    assert "# TYPE churn_predictions_total counter" in body


# --------------------------------------------------------------------------- #
# Error contracts shared by scoring endpoints
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    ("method", "path", "body"),
    [
        ("get", "/model", None),
        ("post", "/predict", {"tenure": 5}),
        ("post", "/predict/batch", {"customers": [{"tenure": 5}]}),
        ("post", "/explain/whatif", {"customer": {"tenure": 5}}),
    ],
)
def test_contract_503_when_model_missing(method: str, path: str, body: object) -> None:
    """Model-dependent endpoints answer 503 with a detail when no model is loaded."""
    app = create_app()
    with TestClient(app) as c:
        app.state.model_state = ModelState(model=None, started_at=0.0, error="boom")
        response = c.request(method, path, json=body)
    assert response.status_code == 503
    assert "Model not available: boom" in response.json()["detail"]


def test_contract_500_unhandled_error(
    registered_model: tuple[str, object], monkeypatch: pytest.MonkeyPatch
) -> None:
    """Unexpected exceptions become a generic 500 that leaks no internals."""
    from churn.predict import ChurnModel

    def _explode(self: ChurnModel, features: dict[str, object]) -> dict[str, object]:
        raise RuntimeError("secret internals")

    monkeypatch.setattr(ChurnModel, "predict", _explode)
    with TestClient(create_app(), raise_server_exceptions=False) as c:
        response = c.post("/predict", json={"tenure": 5})
    assert response.status_code == 500
    data = response.json()
    assert set(data.keys()) == {"detail", "request_id"}
    assert data["detail"] == "internal error"
    assert "secret" not in response.text
