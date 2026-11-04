"""What-if analysis: which realistic changes would lower a customer's churn risk?

For one customer we enumerate every combination of up to ``max_changes``
actionable edits (longer contract, automatic payment, adding OnlineSecurity or
TechSupport, switching paperless billing), score them all in a single batch with
the calibrated model, and return the ones that lower the churn probability most.

Why exhaustive search instead of an optimiser: with five actionable features and
at most two target values each, even three simultaneous changes is only ~130
candidate rows, so brute force is fast, exact and easy to explain.

Rules that keep the suggestions realistic:

* Attributes in :data:`churn.config.IMMUTABLE_FEATURES` (gender, SeniorCitizen,
  tenure, ...) are never changed.
* Contracts may only get longer; add-on services can only be added when the
  customer has internet service and does not already have them.
* A combination is only reported if *every* change in it pays its way: dropping
  any one change must cost at least ``min_gain``. This hides combos that are just
  "the best single change plus something irrelevant".

Limitations (worth saying out loud): the model learned *associations*, not causal
effects, so "customers on two-year contracts churn less" does not prove that
moving this customer will make them stay. ``MonthlyCharges`` is held fixed, so
adding a service is evaluated as if it were free.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass
from typing import Any

from churn import config
from churn.predict import ChurnModel, risk_tier
from churn.schema import defaults as schema_defaults


@dataclass(frozen=True)
class Change:
    """One edit to a customer attribute."""

    feature: str
    from_value: str
    to_value: str


@dataclass(frozen=True)
class Scenario:
    """A set of changes and the churn probability the model predicts after them."""

    changes: tuple[Change, ...]
    probability: float
    delta: float  # probability - baseline probability (negative = lower risk)
    risk_tier: str


@dataclass(frozen=True)
class WhatIfResult:
    """Baseline score plus the best risk-reducing scenarios, lowest risk first."""

    baseline_probability: float
    baseline_risk_tier: str
    scenarios: list[Scenario]


def complete_features(features: dict[str, Any]) -> dict[str, Any]:
    """The customer's full attribute set, with omitted fields set to schema defaults."""
    row = schema_defaults()
    row.update({k: v for k, v in features.items() if v is not None})
    return row


def candidate_changes(customer: dict[str, Any]) -> dict[str, list[str]]:
    """For each actionable feature, the realistic values this customer could move to."""
    options: dict[str, list[str]] = {}
    for feature, targets in config.ACTIONABLE_CHANGES.items():
        if feature in config.IMMUTABLE_FEATURES:  # defensive: config must never overlap
            continue
        current = str(customer[feature])
        allowed = [t for t in targets if t != current and _is_realistic(customer, feature, t)]
        if allowed:
            options[feature] = allowed
    return options


def _is_realistic(customer: dict[str, Any], feature: str, target: str) -> bool:
    """Business rules on top of the allowed values in config."""
    current = str(customer[feature])
    if feature == "Contract":
        order = config.CONTRACT_ORDER
        return current in order and order.index(target) > order.index(current)
    if feature in config.ADDON_SERVICE_COLUMNS:
        # Add-ons can only be added, and only for customers with internet.
        return current == "No" and str(customer["InternetService"]) != "No"
    return True


def _enumerate(options: dict[str, list[str]], max_changes: int) -> list[frozenset[tuple[str, str]]]:
    """Every combination of 1..max_changes edits, at most one value per feature."""
    combos: list[frozenset[tuple[str, str]]] = []
    features = sorted(options)
    for size in range(1, max_changes + 1):
        for chosen in itertools.combinations(features, size):
            for targets in itertools.product(*(options[f] for f in chosen)):
                combos.append(frozenset(zip(chosen, targets, strict=True)))
    return combos


def _every_change_pays(
    combo: frozenset[tuple[str, str]],
    probabilities: dict[frozenset[tuple[str, str]], float],
    min_gain: float,
) -> bool:
    """True if removing any single change raises the probability by >= ``min_gain``."""
    if len(combo) == 1:
        return True
    p = probabilities[combo]
    return all(probabilities[combo - {edit}] - p >= min_gain for edit in combo)


def what_if(
    model: ChurnModel,
    features: dict[str, Any],
    max_changes: int = 2,
    top_k: int = 5,
    min_gain: float = 0.005,
) -> WhatIfResult:
    """Search realistic changes and return the ones that lower churn risk the most.

    Args:
        model: the fitted model; its calibrated probabilities are used.
        features: the customer (omitted fields use schema defaults).
        max_changes: the most edits combined in one scenario.
        top_k: how many scenarios to return.
        min_gain: smallest probability drop (absolute, 0.005 = 0.5 points) that
            counts as an improvement, both overall and for each change in a combo.
    """
    if max_changes < 1:
        raise ValueError("max_changes must be at least 1")
    customer = complete_features(features)
    baseline = model.predict_proba(customer)

    combos = _enumerate(candidate_changes(customer), max_changes)
    if not combos:
        return WhatIfResult(baseline, risk_tier(baseline), [])

    rows = [{**customer, **dict(combo)} for combo in combos]
    scored = model.predict_proba_many(rows)
    probabilities = dict(zip(combos, (float(p) for p in scored), strict=True))

    scenarios = [
        Scenario(
            changes=tuple(
                Change(feature, str(customer[feature]), target) for feature, target in sorted(combo)
            ),
            probability=p,
            delta=p - baseline,
            risk_tier=risk_tier(p),
        )
        for combo, p in probabilities.items()
        if baseline - p >= min_gain and _every_change_pays(combo, probabilities, min_gain)
    ]
    scenarios.sort(key=lambda s: (s.probability, len(s.changes)))
    return WhatIfResult(baseline, risk_tier(baseline), scenarios[:top_k])
