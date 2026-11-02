"""Streamlit UI package for Customer Churn Prediction."""

from __future__ import annotations

import streamlit as st

from churn.ui.analysis import churn_analysis_page
from churn.ui.batch import batch_prediction_page
from churn.ui.performance import performance_page
from churn.ui.prediction import prediction_page
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
        ],
    )
    st.sidebar.markdown("---")
    st.sidebar.info(
        "Use the navigation above to make individual predictions, analyze "
        "customer churn, inspect model performance, or run batch predictions."
    )

    if page == "Prediction":
        prediction_page()
    elif page == "Churn Analysis":
        churn_analysis_page()
    elif page == "Model Performance":
        performance_page()
    elif page == "Batch Prediction":
        batch_prediction_page()


__all__ = [
    "main",
    "prediction_page",
    "churn_analysis_page",
    "performance_page",
    "batch_prediction_page",
    "init_page",
]
