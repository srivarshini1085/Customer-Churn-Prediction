"""API endpoints."""

from __future__ import annotations

import time

from fastapi import APIRouter, Depends, HTTPException, Request, Response
from fastapi import status as http_status

from churn import config
from churn.logging_ import request_id_var
from churn.predict import ChurnModel, log_prediction
from churn.service import metrics
from churn.service.deps import ModelState, get_model
from churn.service.models import (
    BatchRequest,
    BatchResponse,
    CustomerFeatures,
    Factor,
    HealthResponse,
    ModelInfoResponse,
    PredictionResponse,
)

router = APIRouter()


def _predict_one(model: ChurnModel, payload: CustomerFeatures, endpoint: str) -> PredictionResponse:
    features = payload.model_dump(exclude_none=True)
    start = time.perf_counter()
    result = model.predict(features)
    latency_ms = (time.perf_counter() - start) * 1000

    log_prediction(features, result)
    metrics.PREDICTIONS.labels(tier=result["risk_tier"], endpoint=endpoint).inc()
    metrics.LATENCY.labels(endpoint=endpoint).observe(latency_ms / 1000)
    if result["warnings"]:
        metrics.PREDICTION_WARNINGS.labels(endpoint=endpoint).inc()

    return PredictionResponse(
        churn=result["churn"],
        probability=result["probability"],
        risk_tier=result["risk_tier"],
        threshold=result["threshold"],
        model_version=result["model_version"],
        warnings=result["warnings"],
        top_factors=[
            Factor(feature=name, contribution=value)
            for name, value in model.top_factors(features)
        ],
        request_id=request_id_var.get() or "",
        latency_ms=round(latency_ms, 3),
    )


@router.get("/health", response_model=HealthResponse, tags=["ops"])
def health(request: Request, response: Response) -> HealthResponse:
    state: ModelState = request.app.state.model_state
    if not state.healthy:
        response.status_code = http_status.HTTP_503_SERVICE_UNAVAILABLE
    return HealthResponse(
        status="ok" if state.healthy else "degraded",
        model_version=state.model.version if state.model else None,
        uptime_seconds=round(state.uptime_seconds, 1),
    )


@router.get("/model", response_model=ModelInfoResponse, tags=["ops"])
def model_info(model: ChurnModel = Depends(get_model)) -> ModelInfoResponse:
    meta = model.metadata
    test = meta.get("test_metrics", {})
    return ModelInfoResponse(
        version=model.version,
        model_name=meta.get("model_name", "unknown"),
        created_at=meta.get("created_at"),
        threshold=model.threshold,
        threshold_objective=meta.get("threshold_objective", "f1"),
        calibration=meta.get("calibration"),
        metrics={
            "roc_auc": test.get("roc_auc"),
            "pr_auc": test.get("pr_auc"),
            "brier": test.get("brier"),
            "recall": test.get("tuned", {}).get("recall"),
            "precision": test.get("tuned", {}).get("precision"),
        },
        features=model.feature_names,
    )


@router.post("/predict", response_model=PredictionResponse, tags=["scoring"])
def predict(
    payload: CustomerFeatures, model: ChurnModel = Depends(get_model)
) -> PredictionResponse:
    return _predict_one(model, payload, endpoint="predict")


@router.post("/predict/batch", response_model=BatchResponse, tags=["scoring"])
def predict_batch(
    payload: BatchRequest, model: ChurnModel = Depends(get_model)
) -> BatchResponse:
    limit = config.settings.api_batch_limit
    if len(payload.customers) > limit:
        raise HTTPException(
            status_code=413,
            detail=f"batch of {len(payload.customers)} exceeds the limit of {limit}",
        )
    predictions = [
        _predict_one(model, customer, endpoint="predict_batch")
        for customer in payload.customers
    ]
    return BatchResponse(
        model_version=model.version, count=len(predictions), predictions=predictions
    )


@router.get("/metrics", tags=["ops"])
def prometheus_metrics() -> Response:
    body, content_type = metrics.render()
    return Response(content=body, media_type=content_type)
