"""Pydantic request/response models.

``CustomerFeatures`` is generated from :data:`churn.schema.INPUT_FIELDS` so the
API contract, the CLI and the dashboard all stay in sync with one definition.
Every field is optional (an omitted field uses the schema default); provided
values are range- and enum-checked, and unknown fields are rejected (422).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, create_model, model_validator

from churn.schema import INPUT_FIELDS
from churn.schema import Field as SchemaField


def _annotation(field: SchemaField) -> Any:
    if field.name == "SeniorCitizen":
        return Literal[0, 1]
    if field.choices:
        return Literal[tuple(field.choices)]
    return {"number": float, "int": int}[field.kind]


def _field_info(field: SchemaField) -> Any:
    # Only hard lower bounds are enforced (a negative tenure/charge is impossible).
    # Unusually large but plausible values are accepted and surface as a
    # validation *warning* in the response rather than a 422.
    constraints: dict[str, Any] = {"default": None, "description": field.label}
    if field.min_value is not None:
        constraints["ge"] = field.min_value
    return Field(**constraints)


if TYPE_CHECKING:
    # mypy cannot see fields of a model built at runtime, so it sees a plain
    # BaseModel subclass instead. The real class is generated below.
    class CustomerFeatures(BaseModel):
        """One customer's attributes. All fields optional."""

else:
    CustomerFeatures = create_model(
        "CustomerFeatures",
        __config__=ConfigDict(extra="forbid"),
        **{field.name: (_annotation(field) | None, _field_info(field)) for field in INPUT_FIELDS},
    )
    CustomerFeatures.__doc__ = "One customer's attributes. All fields optional."


class BatchRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    customers: list[CustomerFeatures] = Field(..., min_length=1)


class Factor(BaseModel):
    feature: str
    contribution: float


class PredictionResponse(BaseModel):
    churn: bool
    probability: float
    risk_tier: Literal["High", "Moderate", "Low"]
    threshold: float
    model_version: str
    warnings: list[str]
    top_factors: list[Factor]
    request_id: str
    latency_ms: float


class BatchResponse(BaseModel):
    model_version: str
    count: int
    predictions: list[PredictionResponse]


class HealthResponse(BaseModel):
    status: Literal["ok", "degraded"]
    model_version: str | None
    uptime_seconds: float


class ModelInfoResponse(BaseModel):
    version: str
    model_name: str
    created_at: str | None
    threshold: float
    threshold_objective: str
    calibration: str | None
    metrics: dict[str, Any]
    features: list[str]


class ErrorResponse(BaseModel):
    detail: str
    request_id: str | None = None


class WhatIfRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    customer: CustomerFeatures
    max_changes: int = Field(default=2, ge=1, le=3, description="Most edits combined")
    top_k: int = Field(default=5, ge=1, le=20, description="Scenarios to return")


class ChangeOut(BaseModel):
    feature: str
    from_value: str
    to_value: str


class ScenarioOut(BaseModel):
    changes: list[ChangeOut]
    probability: float
    delta: float
    risk_tier: Literal["High", "Moderate", "Low"]


class WhatIfResponse(BaseModel):
    model_version: str
    baseline_probability: float
    baseline_risk_tier: Literal["High", "Moderate", "Low"]
    scenarios: list[ScenarioOut]
    request_id: str


class RetentionPlanRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    customers: list[CustomerFeatures] = Field(..., min_length=1)
    customer_ids: list[str] | None = Field(
        default=None, description="Optional ids, one per customer (default: row number)"
    )
    budget: float = Field(..., ge=0, description="Total amount that may be spent")
    strategy: Literal["optimized", "greedy", "top_risk"] = "optimized"

    @model_validator(mode="after")
    def _ids_match_customers(self) -> RetentionPlanRequest:
        if self.customer_ids is not None and len(self.customer_ids) != len(self.customers):
            raise ValueError("customer_ids must have one id per customer")
        return self


class RetentionAction(BaseModel):
    customer: str
    churn_probability: float
    customer_value: float
    offer: str
    offer_label: str
    cost: float
    expected_saving: float


class StrategySummary(BaseModel):
    strategy: str
    budget: float
    spent: float
    expected_saving: float
    roi: float
    targeted: int
    share_of_optimum: float


class RetentionPlanResponse(BaseModel):
    model_version: str
    plan: StrategySummary
    comparison: list[StrategySummary]
    actions: list[RetentionAction]
    assumptions: dict[str, Any]
