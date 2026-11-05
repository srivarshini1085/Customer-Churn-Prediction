"""Retention budget planner page: budget slider, strategy comparison, ROI curve, download."""

from __future__ import annotations

import io

import numpy as np
import pandas as pd
import streamlit as st

from churn import config
from churn.retention import STRATEGIES, plan_from_scores, savings_curve, score_customers
from churn.ui.components import local_model, model_or_none, page_header

_STRATEGY_LABELS = {
    "optimized": "Optimised (Lagrangian)",
    "greedy": "Greedy (saving per cost)",
    "top_risk": "Top-N by risk (baseline)",
}


@st.cache_data(show_spinner="Scoring customers...")
def _score(csv_bytes: bytes) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Churn probability, monthly charges and ids for an uploaded / sample CSV."""
    return score_customers(pd.read_csv(io.BytesIO(csv_bytes)), local_model())


@st.cache_data(show_spinner="Computing the ROI curve...")
def _curve(probability: np.ndarray, charges: np.ndarray, max_budget: float) -> pd.DataFrame:
    budgets = np.linspace(0, max_budget, 11)
    return savings_curve(probability, charges, [float(b) for b in budgets])


def _load_csv() -> bytes | None:
    source = st.radio(
        "Customers to plan for",
        ["Sample dataset (Telco, 7,043 customers)", "Upload CSV"],
        horizontal=True,
    )
    if source.startswith("Sample"):
        path = config.settings.data_path
        if not path.exists():
            st.error(f"Sample dataset not found at {path}.")
            return None
        return path.read_bytes()
    uploaded = st.file_uploader("Upload customer CSV", type=["csv"], key="retention_upload")
    if uploaded is None:
        st.info("Upload a CSV with the 19 customer columns (customerID is optional).")
        return None
    return uploaded.getvalue()


def _assumptions() -> None:
    with st.expander("Business assumptions (edit in churn/config.py)"):
        st.write(
            f"Customer value = MonthlyCharges x **{config.settings.retention_horizon_months} "
            "months** of revenue kept if the customer stays."
        )
        st.dataframe(
            pd.DataFrame(
                [
                    {
                        "Offer": o.label,
                        "Fixed cost": o.fixed_cost,
                        "Discount": f"{o.discount_rate:.0%} x {o.discount_months} months"
                        if o.discount_months
                        else "-",
                        "Assumed success rate": f"{o.success_rate:.0%}",
                    }
                    for o in config.RETENTION_OFFERS
                ]
            ),
            use_container_width=True,
            hide_index=True,
        )
        st.caption(
            "Costs and success rates are illustrative placeholders, not measured "
            "results. Expected saving = P(churn) x success rate x value - cost."
        )


def retention_page() -> None:
    """Choose which customers get which offer within a budget."""
    page_header(
        "💰 Retention Budget Planner",
        "Pick a budget and see which customers should get which retention offer to "
        "maximise the expected saving, compared with simpler targeting rules.",
    )

    if model_or_none() is None:
        return
    csv_bytes = _load_csv()
    if csv_bytes is None:
        return
    try:
        probability, charges, ids = _score(csv_bytes)
    except Exception as exc:
        st.error(f"Could not score the customers: {exc}")
        return

    budget = st.slider("Retention budget", 0, 50_000, 5_000, step=500)
    plans = {s: plan_from_scores(probability, charges, budget, s, ids) for s in STRATEGIES}
    best, baseline = plans["optimized"], plans["top_risk"]

    cols = st.columns(4)
    cols[0].metric("Customers targeted", f"{best.targeted:,}")
    cols[1].metric("Budget spent", f"{best.spent:,.0f}")
    cols[2].metric(
        "Expected saving",
        f"{best.expected_saving:,.0f}",
        delta=f"{best.expected_saving - baseline.expected_saving:+,.0f} vs top-N by risk",
    )
    cols[3].metric("Expected ROI", f"{best.roi:.2f}x")

    st.subheader("Strategy comparison")
    st.dataframe(
        pd.DataFrame(
            [
                {
                    "Strategy": _STRATEGY_LABELS[p.strategy],
                    "Targeted": p.targeted,
                    "Spent": round(p.spent, 2),
                    "Expected saving": round(p.expected_saving, 2),
                    "ROI": round(p.roi, 2),
                    "% of best possible": f"{p.share_of_optimum:.1%}",
                }
                for p in plans.values()
            ]
        ),
        use_container_width=True,
        hide_index=True,
    )

    st.subheader("Expected saving by budget")
    curve = _curve(probability, charges, 50_000.0)
    chart = curve.pivot(index="budget", columns="strategy", values="expected_saving")
    st.line_chart(chart.rename(columns=_STRATEGY_LABELS))
    st.caption(
        "Expected net saving (rescued revenue minus offer cost) for each strategy as "
        "the budget grows. The optimised plan stops spending once no offer pays for itself."
    )

    st.subheader("Action list")
    choice = st.selectbox(
        "Strategy to export", list(STRATEGIES), format_func=_STRATEGY_LABELS.__getitem__
    )
    actions = plans[choice].action_table()
    if actions.empty:
        st.info("No customer is worth an offer at this budget.")
    else:
        st.bar_chart(actions["offer_label"].value_counts())
        st.dataframe(actions.head(200), use_container_width=True, hide_index=True)
        st.download_button(
            "⬇️ Download action list (CSV)",
            data=actions.to_csv(index=False),
            file_name=f"retention_plan_{choice}_{budget}.csv",
            mime="text/csv",
            use_container_width=True,
        )

    _assumptions()
