"""Batch churn scoring and CSV export page."""

from __future__ import annotations

import io
from typing import Any

import numpy as np
import pandas as pd
import streamlit as st

from churn import config
from churn.schema import INPUT_FIELDS
from churn.ui.components import kpi_row, model_or_none, page_header, score


def _score_via_api(df: pd.DataFrame) -> pd.DataFrame:
    """Row-by-row scoring through the REST API (used when CHURN_API_URL is set)."""
    predictions: list[dict[str, Any]] = []
    progress_bar = st.progress(0, text="Scoring customers through the API...")
    for row_number, (_, row) in enumerate(df.iterrows(), start=1):
        features: dict[str, Any] = {}
        for field in INPUT_FIELDS:
            raw_value = row[field.name]
            if pd.isna(raw_value):
                raw_value = field.default
            try:
                features[field.name] = field.coerce(raw_value)
            except (TypeError, ValueError):
                features[field.name] = field.default
        try:
            result = score(features)
            predictions.append(
                {
                    "churn_probability": result.get("probability", np.nan),
                    "churn_prediction": result.get("churn", False),
                    "risk_tier": result.get("risk_tier", "Unknown"),
                }
            )
        except Exception as exc:
            predictions.append(
                {
                    "churn_probability": np.nan,
                    "churn_prediction": False,
                    "risk_tier": f"Error: {exc}",
                }
            )
        progress_bar.progress(row_number / len(df))
    progress_bar.empty()
    return pd.concat([df.reset_index(drop=True), pd.DataFrame(predictions)], axis=1)


def _score_locally(df: pd.DataFrame) -> pd.DataFrame | None:
    """One vectorised pass with the local model (same code as `churn score`)."""
    from churn.batch import score_frame

    model = model_or_none()
    if model is None:
        return None
    with st.spinner(f"Scoring {len(df):,} customers..."):
        return score_frame(df, model)


def batch_prediction_page() -> None:
    """Predict churn for multiple customers from a CSV file."""
    page_header(
        "📦 Batch Prediction",
        "Upload a CSV of customers to score them all at once and download the results.",
    )

    uploaded_file = st.file_uploader("Upload CSV", type=["csv"], key="batch_upload")
    if uploaded_file is None:
        st.info(
            "Upload a CSV with the 19 customer columns, for example "
            "`data/Telco-Customer-Churn.csv`. Extra columns such as customerID are kept."
        )
        return

    try:
        df = pd.read_csv(uploaded_file)
    except Exception as exc:
        st.error(f"Could not read the CSV file: {exc}")
        return
    if df.empty:
        st.warning("The uploaded file has no rows.")
        return

    with st.expander(f"Uploaded data ({len(df):,} rows)"):
        st.dataframe(df.head(20), use_container_width=True)

    missing_fields = [field.name for field in INPUT_FIELDS if field.name not in df.columns]
    if missing_fields:
        st.error(f"The following required columns are missing: {', '.join(missing_fields)}")
        return

    if not st.button("🚀 Run Batch Prediction", type="primary", use_container_width=True):
        return

    result_df = _score_via_api(df) if config.settings.api_url else _score_locally(df)
    if result_df is None:
        return

    st.success("Batch prediction completed!")
    tiers = result_df["risk_tier"].value_counts()
    kpi_row(
        [
            ("Customers scored", f"{len(result_df):,}", None),
            ("Average churn probability", f"{result_df['churn_probability'].mean():.1%}", None),
            ("High risk", f"{int(tiers.get('High', 0)):,}", "Probability >= 70%"),
            ("Moderate risk", f"{int(tiers.get('Moderate', 0)):,}", "Probability 40-69%"),
        ]
    )
    st.bar_chart(tiers.reindex(["High", "Moderate", "Low"]).fillna(0))

    st.subheader("Prediction Results")
    st.dataframe(result_df, use_container_width=True)

    csv_buffer = io.StringIO()
    result_df.to_csv(csv_buffer, index=False)
    st.download_button(
        label="⬇️ Download Predictions CSV",
        data=csv_buffer.getvalue(),
        file_name="churn_predictions.csv",
        mime="text/csv",
        use_container_width=True,
    )
