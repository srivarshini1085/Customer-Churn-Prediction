"""API endpoints."""

from __future__ import annotations

import time
from dataclasses import asdict

import numpy as np
from fastapi import APIRouter, Depends, HTTPException, Request, Response
from fastapi import status as http_status

from churn import config
from churn.counterfactual import what_if
from churn.logging_ import request_id_var
from churn.predict import ChurnModel, log_prediction
from churn.retention import STRATEGIES, plan_from_scores
from churn.schema import defaults as schema_defaults
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
    RetentionPlanRequest,
    RetentionPlanResponse,
    WhatIfRequest,
    WhatIfResponse,
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
            Factor(feature=name, contribution=value) for name, value in model.top_factors(features)
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
def predict_batch(payload: BatchRequest, model: ChurnModel = Depends(get_model)) -> BatchResponse:
    limit = config.settings.api_batch_limit
    if len(payload.customers) > limit:
        raise HTTPException(
            status_code=413,
            detail=f"batch of {len(payload.customers)} exceeds the limit of {limit}",
        )
    predictions = [
        _predict_one(model, customer, endpoint="predict_batch") for customer in payload.customers
    ]
    return BatchResponse(
        model_version=model.version, count=len(predictions), predictions=predictions
    )


@router.post("/explain/whatif", response_model=WhatIfResponse, tags=["explain"])
def explain_whatif(
    payload: WhatIfRequest, model: ChurnModel = Depends(get_model)
) -> WhatIfResponse:
    """Realistic changes (contract, payment, add-ons, billing) that lower churn risk most."""
    start = time.perf_counter()
    result = what_if(
        model,
        payload.customer.model_dump(exclude_none=True),
        max_changes=payload.max_changes,
        top_k=payload.top_k,
    )
    metrics.LATENCY.labels(endpoint="explain_whatif").observe(time.perf_counter() - start)
    # asdict turns the nested dataclasses into plain dicts; pydantic then
    # validates them (including the risk-tier literals) against the schema.
    return WhatIfResponse.model_validate(
        {
            **asdict(result),
            "model_version": model.version,
            "request_id": request_id_var.get() or "",
        }
    )


@router.post("/retention/plan", response_model=RetentionPlanResponse, tags=["retention"])
def retention_plan(
    payload: RetentionPlanRequest, model: ChurnModel = Depends(get_model)
) -> RetentionPlanResponse:
    """Choose which customers get which retention offer within ``budget``."""
    limit = config.settings.api_batch_limit
    if len(payload.customers) > limit:
        raise HTTPException(
            status_code=413,
            detail=f"plan for {len(payload.customers)} customers exceeds the limit of {limit}",
        )
    features = [c.model_dump(exclude_none=True) for c in payload.customers]
    probability = model.predict_proba_many(features)
    default_charges = float(schema_defaults()["MonthlyCharges"])
    charges = np.array([float(f.get("MonthlyCharges", default_charges)) for f in features])

    plans = {
        s: plan_from_scores(probability, charges, payload.budget, s, payload.customer_ids)
        for s in STRATEGIES
    }
    chosen = plans[payload.strategy]
    return RetentionPlanResponse.model_validate(
        {
            "model_version": model.version,
            "plan": chosen.summary(),
            "comparison": [plan.summary() for plan in plans.values()],
            "actions": chosen.action_table().to_dict(orient="records"),
            "assumptions": {
                "horizon_months": config.settings.retention_horizon_months,
                "offers": [vars(offer) for offer in config.RETENTION_OFFERS],
                "baseline_offer": config.BASELINE_OFFER,
            },
        }
    )


@router.get("/metrics", tags=["ops"])
def prometheus_metrics() -> Response:
    body, content_type = metrics.render()
    return Response(content=body, media_type=content_type)
