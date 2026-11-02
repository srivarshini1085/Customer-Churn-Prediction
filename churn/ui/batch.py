"""Batch churn scoring and CSV export page."""

from __future__ import annotations

import io
from typing import Any

import numpy as np
import pandas as pd
import streamlit as st

from churn.schema import INPUT_FIELDS
from churn.ui.components import score


def batch_prediction_page() -> None:
    """Predict churn for multiple customers from a CSV file."""
    st.title("📦 Batch Prediction")
    st.write(
        "Upload a CSV containing customer information to generate predictions for multiple customers."
    )

    uploaded_file = st.file_uploader(
        "Upload CSV",
        type=["csv"],
        key="batch_upload",
    )

    if uploaded_file is None:
        st.info("Upload a CSV file to begin.")
        return

    try:
        df = pd.read_csv(uploaded_file)
    except Exception as exc:
        st.error(f"Could not read the CSV file: {exc}")
        return

    st.subheader("Uploaded Data")
    st.dataframe(df.head(20), use_container_width=True)

    required_fields = [field.name for field in INPUT_FIELDS]
    missing_fields = [field for field in required_fields if field not in df.columns]

    if missing_fields:
        st.error(f"The following required columns are missing: {', '.join(missing_fields)}")
        return

    if st.button("🚀 Run Batch Prediction", type="primary", use_container_width=True):
        predictions: list[dict[str, Any]] = []
        progress_bar = st.progress(0)
        total_rows = len(df)

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

            if total_rows > 0:
                progress_bar.progress(int(row_number / total_rows * 100))

        result_df = pd.concat([df.reset_index(drop=True), pd.DataFrame(predictions)], axis=1)

        st.success("Batch prediction completed!")
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
