"""Streamlit UI package for Customer Churn Prediction."""

from __future__ import annotations

import streamlit as st

from churn.ui.analysis import churn_analysis_page
from churn.ui.batch import batch_prediction_page
from churn.ui.components import ensure_model
from churn.ui.monitoring import monitoring_page
from churn.ui.performance import performance_page
from churn.ui.prediction import prediction_page
from churn.ui.retention import retention_page
from churn.ui.survival import survival_page
from churn.ui.theme import init_page


def main() -> None:
    """Run the multi-page Streamlit application."""
    init_page()

    st.sidebar.title("Customer Churn Prediction")
    page = st.sidebar.radio(
        "Select Page",
        [
            "Prediction",
            "Churn Analysis",
            "Model Performance",
            "Batch Prediction",
            "Retention Planner",
            "Time to Churn",
            "Monitoring",
        ],
    )
    st.sidebar.markdown("---")
    status = ensure_model()
    if status is None:
        st.sidebar.warning("No model yet: run `python -m churn train`.")
    else:
        st.sidebar.caption(f"Model version: `{status}`")
    st.sidebar.info(
        "Use the navigation above to make individual predictions, analyze "
        "customer churn, inspect model performance, run batch predictions, "
        "plan a retention budget, see when customers tend to churn, or "
        "check new data for drift."
    )

    if page == "Prediction":
        prediction_page()
    elif page == "Churn Analysis":
        churn_analysis_page()
    elif page == "Model Performance":
        performance_page()
    elif page == "Batch Prediction":
        batch_prediction_page()
    elif page == "Retention Planner":
        retention_page()
    elif page == "Time to Churn":
        survival_page()
    elif page == "Monitoring":
        monitoring_page()


__all__ = [
    "main",
    "prediction_page",
    "churn_analysis_page",
    "performance_page",
    "batch_prediction_page",
    "retention_page",
    "survival_page",
    "monitoring_page",
    "init_page",
]
