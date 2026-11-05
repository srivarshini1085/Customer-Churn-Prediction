"""Monitoring page: PSI drift of an uploaded batch vs. the training data."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from churn import config
from churn.ui.components import model_or_none, page_header
from churn.validation import PSI_MAJOR, PSI_MINOR, drift_report, drift_summary

_ICONS = {"green": "🟢 Stable", "amber": "🟠 Moderate shift", "red": "🔴 Major shift"}


def _load_batch() -> pd.DataFrame | None:
    source = st.radio(
        "Batch to check",
        ["Upload CSV", "Sample: last 1,000 rows of the Telco data"],
        horizontal=True,
    )
    if source.startswith("Sample"):
        path = config.settings.data_path
        if not path.exists():
            st.error(f"Sample dataset not found at {path}.")
            return None
        st.caption(
            "The sample comes from the same data the model was trained on, so it should "
            "show little or no drift."
        )
        return pd.read_csv(path).tail(1000)

    uploaded = st.file_uploader("Upload a CSV of recent customers", type=["csv"], key="drift")
    if uploaded is None:
        st.info(
            "Upload a CSV with some or all of the 19 customer columns. Each column is "
            "compared with the distribution the model was trained on."
        )
        return None
    try:
        return pd.read_csv(uploaded)
    except Exception as exc:
        st.error(f"Could not read the CSV file: {exc}")
        return None


def monitoring_page() -> None:
    """Has the incoming data moved away from what the model learned from?"""
    page_header(
        "📡 Monitoring: Data Drift",
        "The Population Stability Index (PSI) measures how much a feature's distribution "
        "has moved between the training data and a new batch.",
    )
    cols = st.columns(3)
    cols[0].success(f"🟢 PSI < {PSI_MINOR}: stable")
    cols[1].warning(f"🟠 {PSI_MINOR} to {PSI_MAJOR}: moderate shift, keep an eye on it")
    cols[2].error(f"🔴 PSI >= {PSI_MAJOR}: major shift")

    model = model_or_none()
    if model is None:
        return
    if model.training_stats is None:
        st.warning("This model has no training-data snapshot; retrain to enable drift checks.")
        return

    batch = _load_batch()
    if batch is None:
        return
    if batch.empty:
        st.warning("The uploaded file has no rows.")
        return

    report = drift_report(model.training_stats, batch)
    if not report:
        st.error("None of the model's feature columns were found in this file.")
        return
    summary = drift_summary(report)

    if len(batch) < config.DRIFT_MIN_ROWS:
        st.warning(
            f"Only {len(batch)} rows. PSI is noisy below about {config.DRIFT_MIN_ROWS} "
            "rows, so treat the colours as indicative."
        )
    missing = sorted(set(config.FEATURE_COLUMNS) - set(report))
    if missing:
        st.info(f"Not checked (column missing from the file): {', '.join(missing)}")

    if summary["retrain_recommended"]:
        st.error(f"**Retrain recommended:** {summary['reason']}.")
    else:
        st.success(f"**No retraining needed:** {summary['reason']}.")
    st.caption(
        f"Rule: retrain when any feature is red, or when {config.DRIFT_AMBER_RETRAIN_COUNT} "
        "or more are amber. Drift says the inputs changed; check recent outcomes before "
        f"acting. Compared with the training data of model {model.version}."
    )

    counts = summary["counts"]
    kpis = st.columns(4)
    kpis[0].metric("Rows checked", f"{len(batch):,}")
    kpis[1].metric("🟢 Stable", counts["green"])
    kpis[2].metric("🟠 Moderate", counts["amber"])
    kpis[3].metric("🔴 Major", counts["red"])

    table = pd.DataFrame(
        [
            {"Feature": feature, "PSI": row["psi"], "Status": _ICONS[summary["statuses"][feature]]}
            for feature, row in report.items()
        ]
    ).sort_values("PSI", ascending=False, ignore_index=True)
    st.subheader("PSI by feature")
    st.bar_chart(table.set_index("Feature")["PSI"])
    st.dataframe(table, use_container_width=True, hide_index=True)
