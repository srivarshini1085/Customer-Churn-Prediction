"""Retention optimiser: economics, budget feasibility, optimality and baselines."""

from __future__ import annotations

import itertools

import hypothesis.strategies as st
import numpy as np
import pandas as pd
import pytest
from hypothesis import given, settings

from churn import config
from churn.config import RetentionOffer
from churn.retention import (
    STRATEGIES,
    compare_strategies,
    customer_value,
    expected_savings,
    offer_costs,
    plan_from_scores,
    plan_retention,
    savings_curve,
)
from tests.conftest import HIGH_RISK, LOW_RISK

OFFERS = config.RETENTION_OFFERS


def _brute_force_optimum(probability, charges, budget) -> float:
    """Best total saving over every assignment (tiny instances only)."""
    costs = offer_costs(charges, OFFERS)
    savings = expected_savings(probability, customer_value(charges), costs, OFFERS)
    best = 0.0
    for combo in itertools.product(range(len(OFFERS) + 1), repeat=len(charges)):
        chosen = [(i, j - 1) for i, j in enumerate(combo) if j]
        if sum(costs[i, j] for i, j in chosen) <= budget + 1e-9:
            best = max(best, sum(savings[i, j] for i, j in chosen))
    return best


# --------------------------------------------------------------------------- #
# Assumptions and economics
# --------------------------------------------------------------------------- #


def test_offer_assumptions_are_sane():
    names = [o.name for o in OFFERS]
    assert len(names) == len(set(names))
    assert config.BASELINE_OFFER in names
    for offer in OFFERS:
        assert offer.fixed_cost > 0
        assert 0 < offer.success_rate <= 1
        assert 0 <= offer.discount_rate <= 1


def test_customer_value_is_monthly_charge_times_horizon():
    np.testing.assert_allclose(customer_value(np.array([50.0, 100.0]), 12), [600.0, 1200.0])
    horizon = config.settings.retention_horizon_months
    np.testing.assert_allclose(customer_value(np.array([10.0])), [10.0 * horizon])


def test_costs_and_savings_follow_the_documented_formulas():
    offers = (
        RetentionOffer("flat", "Flat", fixed_cost=10.0, success_rate=0.5),
        RetentionOffer(
            "disc", "Disc", fixed_cost=5.0, success_rate=0.2, discount_rate=0.1, discount_months=3
        ),
    )
    charges = np.array([100.0])
    costs = offer_costs(charges, offers)
    np.testing.assert_allclose(costs, [[10.0, 5.0 + 0.1 * 100 * 3]])
    savings = expected_savings(np.array([0.4]), np.array([1200.0]), costs, offers)
    np.testing.assert_allclose(savings, [[0.4 * 0.5 * 1200 - 10, 0.4 * 0.2 * 1200 - 35]])


# --------------------------------------------------------------------------- #
# Feasibility and structure — every strategy, random instances
# --------------------------------------------------------------------------- #


@settings(max_examples=60, deadline=None)
@given(
    data=st.lists(st.tuples(st.floats(0, 1), st.floats(18, 120)), min_size=1, max_size=40),
    budget=st.floats(0, 3000),
    strategy=st.sampled_from(STRATEGIES),
)
def test_plans_never_overspend_and_target_each_customer_once(data, budget, strategy):
    probability = np.array([p for p, _ in data])
    charges = np.array([c for _, c in data])
    plan = plan_from_scores(probability, charges, budget, strategy)
    assert plan.spent <= budget + 1e-6
    assert plan.actions["customer"].is_unique
    assert plan.expected_saving <= plan.upper_bound + 1e-6
    if strategy != "top_risk":  # the naive baseline may make loss-making offers
        assert (plan.actions["expected_saving"] > 0).all()


def test_upper_bound_is_valid_and_optimizer_is_close_on_small_problems():
    rng = np.random.default_rng(7)
    gaps = []
    for _ in range(40):
        probability, charges = rng.uniform(0, 1, 6), rng.uniform(20, 120, 6)
        budget = float(rng.uniform(20, 300))
        plan = plan_from_scores(probability, charges, budget)
        optimum = _brute_force_optimum(probability, charges, budget)
        assert optimum <= plan.upper_bound + 1e-6
        assert plan.expected_saving <= optimum + 1e-6
        if optimum > 0:
            gaps.append(1 - plan.expected_saving / optimum)
    assert np.mean(gaps) < 0.05  # small on average; see the module docstring


def test_optimizer_is_exact_when_the_budget_covers_every_worthwhile_offer():
    probability, charges = np.array([0.9, 0.5, 0.05]), np.array([90.0, 60.0, 30.0])
    plan = plan_from_scores(probability, charges, budget=10_000)
    assert plan.expected_saving == pytest.approx(_brute_force_optimum(probability, charges, 10_000))
    assert plan.share_of_optimum == pytest.approx(1.0)


def test_optimizer_is_near_optimal_and_beats_baselines_at_scale():
    rng = np.random.default_rng(0)
    probability = rng.beta(2, 5, 3000)
    charges = rng.uniform(18, 120, 3000)
    plans = {s: plan_from_scores(probability, charges, 20_000, s) for s in STRATEGIES}
    assert plans["optimized"].share_of_optimum > 0.99
    assert plans["optimized"].expected_saving >= plans["greedy"].expected_saving
    assert plans["optimized"].expected_saving > plans["top_risk"].expected_saving


def test_top_risk_sends_the_baseline_offer_to_the_riskiest_customers():
    probability = np.array([0.1, 0.9, 0.5, 0.7])
    charges = np.full(4, 70.0)
    plan = plan_from_scores(probability, charges, budget=100, strategy="top_risk")
    assert set(plan.actions["offer"]) == {config.BASELINE_OFFER}
    assert set(plan.actions["customer"]) == {"1", "3"}  # the two riskiest that fit


def test_zero_budget_targets_nobody():
    plan = plan_from_scores(np.array([0.9]), np.array([100.0]), budget=0)
    assert plan.targeted == 0 and plan.spent == 0 and plan.roi == 0.0


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"budget": -1}, "non-negative"),
        ({"budget": 10, "strategy": "magic"}, "unknown strategy"),
        (
            {"budget": 10, "offers": (RetentionOffer("free", "Free", 0.0, 0.5),)},
            "positive fixed cost",
        ),
    ],
)
def test_invalid_inputs_are_rejected(kwargs, message):
    with pytest.raises(ValueError, match=message):
        plan_from_scores(np.array([0.5]), np.array([50.0]), **kwargs)


def test_action_table_is_rounded_and_summary_is_consistent():
    plan = plan_from_scores(np.array([0.83456789, 0.6]), np.array([99.99, 70.0]), 500)
    table = plan.action_table()
    assert (table["churn_probability"] == table["churn_probability"].round(4)).all()
    assert (table["cost"] == table["cost"].round(2)).all()
    summary = plan.summary()
    assert summary["targeted"] == plan.targeted
    assert summary["spent"] == round(plan.spent, 2)


# --------------------------------------------------------------------------- #
# End to end with a real model
# --------------------------------------------------------------------------- #


def test_plan_retention_uses_customer_ids_and_scores_with_the_model(fitted_model):
    df = pd.DataFrame([{**HIGH_RISK, "customerID": "hi"}, {**LOW_RISK, "customerID": "lo"}])
    plan = plan_retention(df, fitted_model, budget=1000)
    assert set(plan.actions["customer"]) <= {"hi", "lo"}
    row = plan.actions.set_index("customer").loc["hi"]
    assert row["churn_probability"] == pytest.approx(fitted_model.predict_proba(HIGH_RISK))


def test_plan_retention_without_ids_or_monthly_charges(fitted_model):
    df = pd.DataFrame([{"tenure": 1, "Contract": "Month-to-month"}])
    plan = plan_retention(df, fitted_model, budget=1000, strategy="greedy")
    assert list(plan.actions["customer"]) in ([], ["0"])
    if plan.targeted:
        default = config.settings.retention_horizon_months * 70.0
        assert plan.actions["customer_value"].iloc[0] == pytest.approx(default)


def test_compare_strategies_and_savings_curve(fitted_model, sample_xy):
    x, _ = sample_xy
    plans = compare_strategies(x.head(300), fitted_model, budget=2000)
    assert list(plans) == list(STRATEGIES)
    bounds = {round(p.upper_bound, 6) for p in plans.values()}
    assert len(bounds) == 1  # the bound depends only on the customers and budget

    probability = np.array([0.8, 0.4, 0.6])
    curve = savings_curve(probability, np.array([90.0, 50.0, 70.0]), [0, 100, 500])
    assert len(curve) == 3 * len(STRATEGIES)
    optimized = curve[curve["strategy"] == "optimized"]["expected_saving"].to_numpy()
    assert (np.diff(optimized) >= -1e-9).all()  # more budget never hurts
