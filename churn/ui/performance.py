"""Model performance evaluation, metrics, and confusion matrix display page."""

from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st

from churn.ui.components import local_model


def performance_page() -> None:
    """Display model performance information."""
    st.title("🎯 Model Performance")

    try:
        model = local_model()
    except Exception as exc:
        st.error(f"Could not load model: {exc}")
        return

    metadata = getattr(model, "metadata", {}) or {}
    test_metrics = metadata.get("test_metrics", {}) or {}

    if not test_metrics:
        st.warning("No saved test metrics were found in the model metadata.")
        return

    st.subheader("Test Metrics")
    accuracy = test_metrics.get("accuracy")
    precision = test_metrics.get("precision")
    recall = test_metrics.get("recall")
    f1 = test_metrics.get("f1")
    roc_auc = test_metrics.get("roc_auc")

    metric_columns = st.columns(5)
    with metric_columns[0]:
        st.metric("Accuracy", f"{accuracy:.3f}" if accuracy is not None else "N/A")
    with metric_columns[1]:
        st.metric("Precision", f"{precision:.3f}" if precision is not None else "N/A")
    with metric_columns[2]:
        st.metric("Recall", f"{recall:.3f}" if recall is not None else "N/A")
    with metric_columns[3]:
        st.metric("F1 Score", f"{f1:.3f}" if f1 is not None else "N/A")
    with metric_columns[4]:
        st.metric("ROC-AUC", f"{roc_auc:.3f}" if roc_auc is not None else "N/A")

    pr_auc = test_metrics.get("pr_auc")
    if pr_auc is not None:
        st.metric("PR-AUC", f"{pr_auc:.3f}")

    st.subheader("Model Details")
    model_columns = st.columns(2)
    with model_columns[0]:
        version = metadata.get("model_version") or getattr(model, "version", "N/A")
        st.write("**Model Version:**", version)
        threshold = getattr(model, "threshold", metadata.get("threshold", "N/A"))
        st.write("**Threshold:**", threshold)

    with model_columns[1]:
        model_type = metadata.get("model_type", type(model).__name__)
        st.write("**Model Type:**", model_type)
        features = metadata.get("features", [])
        st.write("**Features:**", len(features) if features else "N/A")

    st.subheader("Confusion Matrix")
    confusion_data = metadata.get("confusion_matrix")
    if confusion_data is not None:
        try:
            confusion_array = np.asarray(confusion_data)
            if confusion_array.shape == (2, 2):
                confusion_df = pd.DataFrame(
                    confusion_array,
                    index=["Actual 0", "Actual 1"],
                    columns=["Predicted 0", "Predicted 1"],
                )
                st.dataframe(confusion_df, use_container_width=True)
            else:
                st.info("Saved confusion matrix has an unexpected format.")
        except Exception:
            st.info("Could not display the saved confusion matrix.")
    else:
        st.info("No saved confusion matrix was found.")
