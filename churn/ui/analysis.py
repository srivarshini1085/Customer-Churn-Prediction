"""Exploratory churn analysis and customer dataset visualization page."""

from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st

from churn.ui.components import kpi_row, page_header


def churn_analysis_page() -> None:
    """Analyze churn patterns in uploaded customer data."""
    page_header(
        "📈 Churn Analysis",
        "Explore a customer file: its size, missing values and how churn is distributed.",
    )

    uploaded_file = st.file_uploader("Upload customer CSV file", type=["csv"])
    if uploaded_file is None:
        st.info(
            "Upload a CSV to begin, for example `data/Telco-Customer-Churn.csv`. A column "
            "named churn, churned, exited or is_churn enables the churn charts."
        )
        return

    try:
        df = pd.read_csv(uploaded_file)
    except Exception as exc:
        st.error(f"Could not read the CSV file: {exc}")
        return

    st.subheader("Dataset Preview")
    st.dataframe(df.head(20), use_container_width=True)

    st.subheader("Dataset Summary")
    kpi_row(
        [
            ("Rows", f"{len(df):,}", None),
            ("Columns", str(len(df.columns)), None),
            ("Missing Values", f"{int(df.isna().sum().sum()):,}", None),
        ]
    )

    churn_columns = [
        column
        for column in df.columns
        if column.lower() in {"churn", "churned", "exited", "is_churn"}
    ]

    if not churn_columns:
        st.warning(
            "No churn column was detected. Expected names include `churn`, "
            "`churned`, `exited`, or `is_churn`."
        )
        return

    churn_column = churn_columns[0]
    st.subheader("Churn Distribution")
    churn_counts = df[churn_column].value_counts(dropna=False)
    st.bar_chart(churn_counts)

    st.subheader("Churn Statistics")
    total_customers = len(df)
    if total_customers > 0:
        churn_rate = (
            df[churn_column]
            .astype(str)
            .str.lower()
            .isin(["1", "true", "yes", "churn", "churned"])
            .mean()
        )
        st.metric("Estimated Churn Rate", f"{churn_rate * 100:.2f}%")

    numeric_columns = df.select_dtypes(include=np.number).columns.tolist()
    if numeric_columns:
        st.subheader("Numeric Feature Analysis")
        selected_column = st.selectbox("Select a numeric feature", numeric_columns)
        st.line_chart(df[selected_column].reset_index(drop=True))
