"""Model performance page: held-out metrics, CV vs test, confusion matrix, calibration, fairness."""

from __future__ import annotations

from typing import Any

import pandas as pd
import streamlit as st

from churn.fairness import interpret
from churn.ui.components import fmt, kpi_row, model_or_none, page_header


def performance_page() -> None:
    """Display model performance information saved at training time."""
    page_header(
        "🎯 Model Performance",
        "Every number here was computed on the held-out test set when this model "
        "version was trained, and is stored in its manifest.",
    )
    model = model_or_none()
    if model is None:
        return

    metadata = getattr(model, "metadata", {}) or {}
    test = metadata.get("test_metrics") or {}
    if not test:
        st.warning("No saved test metrics were found in the model metadata.")
        return
    tuned = test.get("tuned", {})

    kpi_row(
        [
            ("ROC-AUC", fmt(test.get("roc_auc")), "Ranking quality: 0.5 random, 1.0 perfect"),
            ("PR-AUC", fmt(test.get("pr_auc")), "Precision-recall area; baseline = churn rate"),
            (
                "Brier score",
                fmt(test.get("brier")),
                "Calibration error of probabilities; lower is better",
            ),
            (
                "Decision threshold",
                fmt(tuned.get("threshold"), 2),
                "Tuned on out-of-fold training predictions",
            ),
        ]
    )
    kpi_row(
        [
            ("Recall", fmt(tuned.get("recall")), "Share of churners the model flags"),
            ("Precision", fmt(tuned.get("precision")), "Share of flagged customers who churn"),
            ("F1", fmt(tuned.get("f1")), None),
            ("Accuracy", fmt(tuned.get("accuracy")), "Misleading alone: ~73% of customers stay"),
        ]
    )

    left, right = st.columns(2)
    with left:
        st.subheader("Confusion matrix (test, tuned threshold)")
        cm = tuned.get("confusion_matrix")
        if cm:
            st.dataframe(
                pd.DataFrame(
                    [[cm["tn"], cm["fp"]], [cm["fn"], cm["tp"]]],
                    index=["Actually stayed", "Actually churned"],
                    columns=["Predicted stay", "Predicted churn"],
                ),
                use_container_width=True,
            )
        else:
            st.info("No saved confusion matrix was found.")

        st.subheader("Cross-validation vs. test")
        cv = metadata.get("cv_scores") or {}
        rows = [
            {"Metric": "ROC-AUC", "CV (train)": cv.get("roc_auc"), "Test": test.get("roc_auc")},
            {"Metric": "PR-AUC", "CV (train)": cv.get("pr_auc"), "Test": test.get("pr_auc")},
        ]
        st.dataframe(pd.DataFrame(rows).round(4), use_container_width=True, hide_index=True)
        st.caption("Close CV and test scores mean the model generalises rather than memorises.")

    with right:
        st.subheader("Calibration (reliability)")
        reliability = test.get("reliability")
        if reliability:
            curve = pd.DataFrame(
                {
                    "Predicted probability": reliability["prob_pred"],
                    "Observed churn rate": reliability["prob_true"],
                    "Perfect calibration": reliability["prob_pred"],
                }
            ).set_index("Predicted probability")
            st.line_chart(curve)
            st.caption(
                "Each point is a tenth of the test customers. On the diagonal, a predicted "
                "30% really churns 30% of the time, which is what the budget optimiser needs."
            )
        else:
            st.info("No reliability curve was saved with this model.")

        st.subheader("Model details")
        st.dataframe(
            pd.DataFrame(
                [
                    ("Version", metadata.get("version") or getattr(model, "version", "n/a")),
                    ("Model", metadata.get("model_name", type(model).__name__)),
                    ("Calibration", metadata.get("calibration") or "none"),
                    ("Trained", metadata.get("created_at", "n/a")),
                    (
                        "Training / test rows",
                        f"{metadata.get('n_train')} / {metadata.get('n_test')}",
                    ),
                    ("Encoded features", len(getattr(model, "feature_names", []) or [])),
                ],
                columns=["Field", "Value"],
            ).astype(str),
            use_container_width=True,
            hide_index=True,
        )

    _fairness_section(metadata.get("fairness"))


def _fairness_section(report: dict[str, Any] | None) -> None:
    """Per-group recall / precision / FPR saved with the model at training time."""
    st.subheader("Fairness by group")
    if not report:
        st.info("This model version has no fairness report; retrain to add one.")
        return
    rows = [
        {
            "Attribute": attribute,
            "Group": group,
            "Customers": m["n"],
            "Churn rate": m["churn_rate"],
            "Recall": m["recall"],
            "Precision": m["precision"],
            "False-positive rate": m["false_positive_rate"],
        }
        for attribute, result in report["attributes"].items()
        for group, m in result["groups"].items()
    ]
    st.dataframe(pd.DataFrame(rows).round(3), use_container_width=True, hide_index=True)
    for finding in interpret(report):
        st.markdown(f"- {finding}")
    st.caption(
        f"Held-out test set at the tuned threshold ({report['threshold']:.2f}). Gaps above "
        f"{report['gap_flag']:.2f} are flagged for review. Recall matters most here: a "
        "missed churner gets no retention offer. When churn rates differ between groups, "
        "precision and FPR cannot all be equal at once."
    )
