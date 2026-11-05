"""Retention budget optimiser: who should get which offer, given a fixed budget?

Economics (all inputs except P(churn) are business assumptions from config):

* **Customer value** = ``MonthlyCharges x settings.retention_horizon_months``.
  The horizon is "how many more months of revenue we keep if this customer
  stays". A flat 12 months is deliberately simple; a survival model could give
  each customer their own expected remaining lifetime instead.
* **Offer cost** = ``fixed_cost + discount_rate x MonthlyCharges x discount_months``
  (a discount is revenue given away, so it counts as cost).
* **Expected saving** of giving offer *o* to customer *i*::

      P(churn_i) x success_rate_o x value_i - cost_io

  i.e. the revenue we expect to rescue, minus what the offer costs for certain.

The optimisation problem: each customer gets **at most one** offer, the total
cost must fit the budget, and we maximise the total expected saving. That is a
*multiple-choice knapsack*. Three strategies are compared:

``optimized`` (default) — Lagrangian relaxation
    Put a price ``lam`` on every unit of budget. For a given ``lam`` the problem
    splits into one tiny problem per customer: take the offer with the highest
    ``saving - lam x cost``, or nothing if that is negative. A higher ``lam``
    makes offers "more expensive", so fewer customers are targeted. Binary-search
    the smallest ``lam`` whose plan fits the budget, then spend what is left with
    a greedy fill. No rounding is needed and it runs in milliseconds for
    thousands of customers.

    The same idea gives a *certificate*: for any ``lam >= 0``,
    ``sum_i max(0, max_o(saving_io - lam x cost_io)) + lam x budget`` is an upper
    bound on the best achievable saving. :attr:`RetentionPlan.upper_bound`
    reports it, so every plan can say how close to optimal it is.

    It is not exact: the shortfall is at most about one customer's saving. On the
    full Telco file that is invisible (plans reach ~100% of the bound), but on a
    toy problem with a handful of customers it can be a large share.

    (An exact dynamic programme over the budget was tried first. It needs costs
    rounded to fixed budget steps, and with $10 offers that rounding distorted
    costs by up to 40%, so it was dropped in favour of this method.)
``greedy``
    Rank every (customer, offer) pair by saving per unit of cost and take them in
    order while the budget lasts. Simple, but it commits each customer to their
    best-*ratio* offer and never upgrades them to a bigger, better offer.
``top_risk``
    The naive baseline: sort customers by churn probability and send everyone
    the same offer (``config.BASELINE_OFFER``) until the money runs out, whether
    or not the offer pays for itself.

Only pairs with a positive expected saving are used by ``optimized`` and
``greedy``, so an optimised plan may deliberately leave budget unspent.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd

from churn import config
from churn.batch import score_frame
from churn.config import RetentionOffer
from churn.predict import ChurnModel
from churn.schema import defaults as schema_defaults

Strategy = Literal["optimized", "greedy", "top_risk"]
STRATEGIES: tuple[Strategy, ...] = ("optimized", "greedy", "top_risk")

#: Columns of :attr:`RetentionPlan.actions`, in order.
ACTION_COLUMNS: list[str] = [
    "customer",
    "churn_probability",
    "customer_value",
    "offer",
    "offer_label",
    "cost",
    "expected_saving",
]

_NONE = -1  # "no offer" in an assignment array


@dataclass(frozen=True)
class RetentionPlan:
    """The chosen actions for one strategy and budget."""

    strategy: str
    budget: float
    actions: pd.DataFrame  # one row per targeted customer, see ACTION_COLUMNS
    upper_bound: float  # no plan for these customers and budget can save more

    @property
    def spent(self) -> float:
        return float(self.actions["cost"].sum())

    @property
    def expected_saving(self) -> float:
        """Total expected net saving (rescued revenue minus offer costs)."""
        return float(self.actions["expected_saving"].sum())

    @property
    def targeted(self) -> int:
        return len(self.actions)

    @property
    def roi(self) -> float:
        """Expected net saving per unit spent (0 when nothing is spent)."""
        return self.expected_saving / self.spent if self.spent > 0 else 0.0

    @property
    def share_of_optimum(self) -> float:
        """Expected saving as a fraction of the upper bound (1.0 = provably optimal)."""
        return self.expected_saving / self.upper_bound if self.upper_bound > 0 else 1.0

    def action_table(self) -> pd.DataFrame:
        """The action list rounded for people: 4 dp for probabilities, 2 dp for money."""
        return self.actions.round(
            {"churn_probability": 4, "customer_value": 2, "cost": 2, "expected_saving": 2}
        )

    def summary(self) -> dict[str, float | int | str]:
        return {
            "strategy": self.strategy,
            "budget": self.budget,
            "spent": round(self.spent, 2),
            "expected_saving": round(self.expected_saving, 2),
            "roi": round(self.roi, 3),
            "targeted": self.targeted,
            "share_of_optimum": round(self.share_of_optimum, 4),
        }


# --------------------------------------------------------------------------- #
# Economics
# --------------------------------------------------------------------------- #


def customer_value(monthly_charges: np.ndarray, horizon_months: int | None = None) -> np.ndarray:
    """Revenue kept if the customer stays: monthly charge x assumed horizon."""
    months = horizon_months or config.settings.retention_horizon_months
    return np.asarray(monthly_charges, dtype=float) * months


def offer_costs(monthly_charges: np.ndarray, offers: Sequence[RetentionOffer]) -> np.ndarray:
    """Cost matrix of shape ``(customers, offers)``."""
    charges = np.asarray(monthly_charges, dtype=float)[:, None]
    fixed = np.array([o.fixed_cost for o in offers])
    discount = np.array([o.discount_rate * o.discount_months for o in offers])
    return np.asarray(fixed + charges * discount, dtype=float)


def expected_savings(
    probability: np.ndarray,
    value: np.ndarray,
    costs: np.ndarray,
    offers: Sequence[RetentionOffer],
) -> np.ndarray:
    """``P(churn) x success_rate x value - cost`` for every (customer, offer) pair."""
    success = np.array([o.success_rate for o in offers])
    rescued = np.asarray(probability, dtype=float)[:, None] * success * value[:, None]
    return np.asarray(rescued - costs, dtype=float)


# --------------------------------------------------------------------------- #
# Strategies — each returns an assignment: offer column per customer, or _NONE
# --------------------------------------------------------------------------- #


def _best_response(costs: np.ndarray, savings: np.ndarray, lam: float) -> np.ndarray:
    """Each customer's best offer when every unit of budget is priced at ``lam``."""
    adjusted = savings - lam * costs
    best = np.argmax(adjusted, axis=1)
    worthwhile = adjusted[np.arange(len(best)), best] > 0
    return np.where(worthwhile, best, _NONE)


def _total(values: np.ndarray, assignment: np.ndarray) -> float:
    rows = np.flatnonzero(assignment != _NONE)
    return float(values[rows, assignment[rows]].sum())


def upper_bound(costs: np.ndarray, savings: np.ndarray, budget: float, lam: float) -> float:
    """Lagrangian bound: no feasible plan can save more than this (any ``lam >= 0``)."""
    adjusted = np.maximum((savings - lam * costs).max(axis=1, initial=0.0), 0.0)
    return float(adjusted.sum() + lam * budget)


def _fill(costs: np.ndarray, savings: np.ndarray, assignment: np.ndarray, left: float) -> None:
    """Spend leftover budget on the best saving-per-cost moves (in place).

    A move is giving an untargeted customer an offer, or upgrading a targeted
    customer to an offer that saves more. Each move must fit the leftover budget.
    """
    rows = np.arange(len(assignment))
    while True:
        has = assignment != _NONE
        current_cost = np.where(has, costs[rows, np.maximum(assignment, 0)], 0.0)
        current_saving = np.where(has, savings[rows, np.maximum(assignment, 0)], 0.0)
        extra_cost = costs - current_cost[:, None]
        extra_saving = savings - current_saving[:, None]
        valid = (extra_saving > 0) & (extra_cost <= left)
        if not valid.any():
            return
        ratio = np.where(
            valid, extra_saving / np.maximum(extra_cost, 1e-12), -np.inf
        )  # a free upgrade (extra cost <= 0) ranks first
        i, j = np.unravel_index(int(np.argmax(ratio)), ratio.shape)
        assignment[i] = j
        left -= float(extra_cost[i, j])


def _optimized(costs: np.ndarray, savings: np.ndarray, budget: float) -> tuple[np.ndarray, float]:
    """Lagrangian relaxation + greedy fill. Returns (assignment, upper bound)."""
    free = _best_response(costs, savings, 0.0)
    if _total(costs, free) <= budget:  # every worthwhile offer fits: optimal
        return free, _total(savings, free)

    # Above the best saving-per-cost ratio no offer is worthwhile, so the plan is
    # empty and always fits. Start safely above it: at exactly that ratio float
    # rounding can leave one offer marginally "worthwhile" and break the budget.
    positive = costs > 0
    lo, hi = 0.0, 2 * float((savings[positive] / costs[positive]).max(initial=0.0)) + 1.0
    bound = upper_bound(costs, savings, budget, hi)
    for _ in range(60):  # bisection: lo never fits, hi always fits
        mid = (lo + hi) / 2
        bound = min(bound, upper_bound(costs, savings, budget, mid))
        if _total(costs, _best_response(costs, savings, mid)) <= budget:
            hi = mid
        else:
            lo = mid
    assignment = _best_response(costs, savings, hi)
    _fill(costs, savings, assignment, budget - _total(costs, assignment))
    return assignment, bound


def _greedy(costs: np.ndarray, savings: np.ndarray, budget: float) -> np.ndarray:
    """Take (customer, offer) pairs by saving-per-cost until the budget is used."""
    assignment = np.full(len(costs), _NONE)
    rows, cols = np.nonzero(savings > 0)
    order = np.argsort(-(savings[rows, cols] / costs[rows, cols]), kind="stable")
    left = budget
    for k in order:
        i, j = rows[k], cols[k]
        if assignment[i] == _NONE and costs[i, j] <= left:
            assignment[i] = j
            left -= costs[i, j]
    return assignment


def _top_risk(
    costs: np.ndarray, probability: np.ndarray, budget: float, offer_index: int
) -> np.ndarray:
    """Highest-risk customers first, everyone gets the baseline offer."""
    assignment = np.full(len(costs), _NONE)
    left = budget
    for i in np.argsort(-probability, kind="stable"):
        if costs[i, offer_index] <= left:
            assignment[i] = offer_index
            left -= costs[i, offer_index]
    return assignment


# --------------------------------------------------------------------------- #
# Public API
# --------------------------------------------------------------------------- #


def plan_from_scores(
    probability: np.ndarray,
    monthly_charges: np.ndarray,
    budget: float,
    strategy: Strategy = "optimized",
    customers: Sequence[str] | None = None,
    offers: Sequence[RetentionOffer] = config.RETENTION_OFFERS,
) -> RetentionPlan:
    """Build a plan from already-computed churn probabilities."""
    if budget < 0:
        raise ValueError("budget must be non-negative")
    if strategy not in STRATEGIES:
        raise ValueError(f"unknown strategy {strategy!r}; choose from {STRATEGIES}")
    if any(o.fixed_cost <= 0 for o in offers):
        raise ValueError("every offer needs a positive fixed cost")
    probability = np.asarray(probability, dtype=float)
    charges = np.asarray(monthly_charges, dtype=float)
    ids = list(customers) if customers is not None else [str(i) for i in range(len(charges))]

    value = customer_value(charges)
    costs = offer_costs(charges, offers)
    savings = expected_savings(probability, value, costs, offers)

    assignment, bound = _optimized(costs, savings, budget)
    if strategy == "greedy":
        assignment = _greedy(costs, savings, budget)
    elif strategy == "top_risk":
        names = [o.name for o in offers]
        assignment = _top_risk(costs, probability, budget, names.index(config.BASELINE_OFFER))

    targeted = np.flatnonzero(assignment != _NONE)
    rows = [
        {
            "customer": ids[i],
            "churn_probability": probability[i],
            "customer_value": value[i],
            "offer": offers[assignment[i]].name,
            "offer_label": offers[assignment[i]].label,
            "cost": costs[i, assignment[i]],
            "expected_saving": savings[i, assignment[i]],
        }
        for i in targeted
    ]
    actions = pd.DataFrame(rows, columns=ACTION_COLUMNS)
    actions = actions.sort_values("expected_saving", ascending=False, ignore_index=True)
    return RetentionPlan(strategy, float(budget), actions, bound)


def score_customers(
    df: pd.DataFrame, model: ChurnModel
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Churn probability, monthly charges and an id per row of ``df``."""
    scored = score_frame(df, model)
    if "MonthlyCharges" in scored.columns:
        charges = pd.to_numeric(scored["MonthlyCharges"], errors="coerce")
    else:
        charges = pd.Series(np.nan, index=scored.index)
    charges = charges.fillna(float(schema_defaults()["MonthlyCharges"]))
    if config.ID_COLUMN in df.columns:
        ids = df[config.ID_COLUMN].astype(str).tolist()
    else:
        ids = [str(i) for i in range(len(df))]
    return scored["churn_probability"].to_numpy(dtype=float), charges.to_numpy(float), ids


def plan_retention(
    df: pd.DataFrame,
    model: ChurnModel,
    budget: float,
    strategy: Strategy = "optimized",
) -> RetentionPlan:
    """Score a customer table with ``model`` and plan the retention budget.

    ``customerID`` identifies customers when present, else the row number. A
    missing ``MonthlyCharges`` falls back to the schema default.
    """
    probability, charges, ids = score_customers(df, model)
    return plan_from_scores(probability, charges, budget, strategy, ids)


def compare_strategies(
    df: pd.DataFrame, model: ChurnModel, budget: float
) -> dict[str, RetentionPlan]:
    """One plan per strategy for the same customers and budget (scored once)."""
    probability, charges, ids = score_customers(df, model)
    return {s: plan_from_scores(probability, charges, budget, s, ids) for s in STRATEGIES}


def savings_curve(
    probability: np.ndarray, monthly_charges: np.ndarray, budgets: Sequence[float]
) -> pd.DataFrame:
    """Expected saving of every strategy at each budget, for an ROI chart."""
    rows = []
    for budget in budgets:
        for strategy in STRATEGIES:
            plan = plan_from_scores(probability, monthly_charges, budget, strategy)
            rows.append({"budget": budget, **plan.summary()})
    return pd.DataFrame(rows)
