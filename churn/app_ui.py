"""Customer Churn Prediction Streamlit Dashboard."""

from __future__ import annotations

from typing import Any

import io

import numpy as np
import pandas as pd
import streamlit as st

from churn import config
from churn.schema import GROUPS, INPUT_FIELDS, Field


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Customer Churn Prediction",
    page_icon="📊",
    layout="wide",
)


# ============================================================
# COMMON SETTINGS
# ============================================================

_BANNER = {
    "High": st.error,
    "Moderate": st.warning,
    "Low": st.success,
}


# ============================================================
# MODEL LOADING
# ============================================================

@st.cache_resource(show_spinner="Loading model...")
def _local_model():
    from churn.predict import load_model

    return load_model()


# ============================================================
# DATA LOADING
# ============================================================

@st.cache_data
def load_dataset() -> pd.DataFrame:
    """Load and clean the Telco Customer Churn dataset."""

    path = config.DATA_PATH

    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found at: {path}"
        )

    df = pd.read_csv(path)

    # Remove unwanted spaces from column names
    df.columns = df.columns.str.strip()

    # --------------------------------------------------------
    # IMPORTANT:
    # Telco dataset contains blank strings in TotalCharges.
    # Convert them to NaN and then numeric.
    # This fixes the:
    # "Cannot use median strategy with non-numeric data"
    # error.
    # --------------------------------------------------------

    if "TotalCharges" in df.columns:
        df["TotalCharges"] = (
            df["TotalCharges"]
            .replace(r"^\s*$", np.nan, regex=True)
        )

        df["TotalCharges"] = pd.to_numeric(
            df["TotalCharges"],
            errors="coerce",
        )

    if "tenure" in df.columns:
        df["tenure"] = pd.to_numeric(
            df["tenure"],
            errors="coerce",
        )

    if "MonthlyCharges" in df.columns:
        df["MonthlyCharges"] = pd.to_numeric(
            df["MonthlyCharges"],
            errors="coerce",
        )

    if "SeniorCitizen" in df.columns:
        df["SeniorCitizen"] = pd.to_numeric(
            df["SeniorCitizen"],
            errors="coerce",
        )

    return df


# ============================================================
# CLEAN FEATURES FOR MODEL
# ============================================================

def clean_features(df: pd.DataFrame) -> pd.DataFrame:
    """Prepare customer features before sending them to the model."""

    data = df.copy()

    # Make sure all expected columns exist
    missing = [
        col
        for col in config.FEATURE_COLUMNS
        if col not in data.columns
    ]

    if missing:
        raise ValueError(
            f"Missing required columns: {missing}"
        )

    # Numeric columns
    for col in config.NUMERIC_COLUMNS:
        data[col] = pd.to_numeric(
            data[col],
            errors="coerce",
        )

    # SeniorCitizen
    if "SeniorCitizen" in data.columns:
        data["SeniorCitizen"] = pd.to_numeric(
            data["SeniorCitizen"],
            errors="coerce",
        )

    # Do not manually fill categorical values here.
    # The trained pipeline handles categorical processing.

    return data[config.FEATURE_COLUMNS]


# ============================================================
# SINGLE CUSTOMER PREDICTION
# ============================================================

def _score(features: dict[str, Any]) -> dict[str, Any]:
    """Run prediction using API or local model."""

    api_url = config.settings.api_url

    # --------------------------------------------------------
    # API MODE
    # --------------------------------------------------------

    if api_url:

        import httpx

        resp = httpx.post(
            f"{api_url.rstrip('/')}/predict",
            json=features,
            timeout=10,
        )

        resp.raise_for_status()

        body = resp.json()

        meta = httpx.get(
            f"{api_url.rstrip('/')}/model",
            timeout=10,
        ).json()

        body["top_factors"] = [
            (
                f["feature"],
                f["contribution"],
            )
            for f in body.get("top_factors", [])
        ]

        body["model_meta"] = meta

        return body

    # --------------------------------------------------------
    # LOCAL MODEL
    # --------------------------------------------------------

    model = _local_model()

    result = model.predict(features)

    result["top_factors"] = model.top_factors(
        features
    )

    result["model_meta"] = {
        "model_name": model.metadata.get(
            "model_name"
        ),
        "created_at": model.metadata.get(
            "created_at"
        ),
        "metrics": {
            "roc_auc": model.metadata.get(
                "test_metrics",
                {},
            ).get("roc_auc"),
            "pr_auc": model.metadata.get(
                "test_metrics",
                {},
            ).get("pr_auc"),
            "brier": model.metadata.get(
                "test_metrics",
                {},
            ).get("brier"),
        },
    }

    return result


# ============================================================
# INPUT WIDGET
# ============================================================

def _input_widget(field: Field) -> Any:

    if field.name == "SeniorCitizen":

        choice = st.selectbox(
            field.label,
            ["No", "Yes"],
            index=int(field.default),
            help=field.help,
        )

        return 1 if choice == "Yes" else 0

    if field.kind == "choice":

        options = list(
            field.choices or ()
        )

        index = (
            options.index(field.default)
            if field.default in options
            else 0
        )

        return st.selectbox(
            field.label,
            options,
            index=index,
            help=field.help,
        )

    if field.kind == "int":

        return int(
            st.number_input(
                field.label,
                min_value=(
                    int(field.min_value)
                    if field.min_value is not None
                    else 0
                ),
                max_value=(
                    int(field.max_value)
                    if field.max_value is not None
                    else 1000
                ),
                value=int(field.default),
                step=1,
                help=field.help,
            )
        )

    return float(
        st.number_input(
            field.label,
            min_value=(
                float(field.min_value)
                if field.min_value is not None
                else 0.0
            ),
            value=float(field.default),
            help=field.help,
        )
    )


# ============================================================
# COLLECT INPUTS
# ============================================================

def _collect_inputs() -> dict[str, Any]:

    features: dict[str, Any] = {}

    tabs = st.tabs(GROUPS)

    for tab, group in zip(
        tabs,
        GROUPS,
        strict=True,
    ):

        with tab:

            columns = st.columns(2)

            fields = [
                f
                for f in INPUT_FIELDS
                if f.group == group
            ]

            for idx, field in enumerate(fields):

                with columns[idx % 2]:

                    features[field.name] = (
                        _input_widget(field)
                    )

    return features


# ============================================================
# SHOW PREDICTION RESULT
# ============================================================

def _show_result(
    result: dict[str, Any]
) -> None:

    tier = result["risk_tier"]

    st.subheader("🔮 Prediction")

    left, right = st.columns([1, 2])

    left.metric(
        "Churn Probability",
        f"{result['probability'] * 100:.1f}%",
    )

    right.progress(
        min(
            result["probability"],
            1.0,
        )
    )

    outcome = (
        "predicted to churn"
        if result["churn"]
        else "predicted to stay"
    )

    _BANNER[tier](
        f"**{tier} risk** — customer is "
        f"{outcome} "
        f"_(decision threshold "
        f"{result['threshold']:.2f})_"
    )

    st.caption(
        "Recommended action: "
        f"{config.RISK_ACTIONS[tier]}"
    )

    # Warnings
    for warning in result.get(
        "warnings",
        [],
    ):
        st.warning(
            f"⚠️ {warning}"
        )

    # --------------------------------------------------------
    # TOP FACTORS
    # --------------------------------------------------------

    factors = result.get(
        "top_factors",
        [],
    )

    if factors:

        st.subheader(
            "📌 Drivers of this Prediction"
        )

        for name, value in factors:

            verb = (
                "raises"
                if value > 0
                else "lowers"
            )

            st.write(
                f"- **{name}** "
                f"{verb} churn risk "
                f"({value:+.3f})"
            )

    # --------------------------------------------------------
    # MODEL INFO
    # --------------------------------------------------------

    meta = result.get(
        "model_meta",
        {},
    )

    metrics = meta.get(
        "metrics",
        {},
    )

    bits = [
        f"Version `{result.get('model_version', '?')}`",
        f"Model `{meta.get('model_name', '?')}`",
    ]

    if metrics.get("roc_auc") is not None:

        bits.append(
            f"ROC-AUC "
            f"{metrics['roc_auc']:.3f}"
        )

    if metrics.get("pr_auc") is not None:

        bits.append(
            f"PR-AUC "
            f"{metrics['pr_auc']:.3f}"
        )

    if metrics.get("brier") is not None:

        bits.append(
            f"Brier "
            f"{metrics['brier']:.3f}"
        )

    st.caption(
        "  ·  ".join(bits)
    )


# ============================================================
# PAGE 1 — PREDICTION
# ============================================================

def prediction_page() -> None:

    st.header(
        "🔮 Customer Churn Prediction"
    )

    mode = (
        "API"
        if config.settings.api_url
        else "local model"
    )

    st.write(
        "Enter the customer's details across "
        "the three tabs, then predict the "
        "probability that they churn."
    )

    st.caption(
        f"Scoring via {mode}"
    )

    if not config.settings.api_url:

        try:

            _local_model()

        except FileNotFoundError:

            st.error(
                "No trained model found. "
                "Run `python -m churn train` first."
            )

            st.stop()

    st.divider()

    with st.form(
        "customer"
    ):

        features = _collect_inputs()

        submitted = st.form_submit_button(
            "🔮 Predict Churn",
            use_container_width=True,
        )

    if submitted:

        st.divider()

        try:

            result = _score(
                features
            )

            _show_result(
                result
            )

        except Exception as exc:

            st.error(
                f"Scoring failed: {exc}"
            )


# ============================================================
# PAGE 2 — CHURN ANALYSIS
# ============================================================

def analysis_page() -> None:

    st.header(
        "📊 Churn Analysis"
    )

    st.write(
        "Explore customer churn patterns "
        "and understand which customer "
        "groups have higher churn rates."
    )

    try:

        df = load_dataset()

    except Exception as exc:

        st.error(
            f"Could not load dataset: {exc}"
        )

        return

    if config.TARGET_COLUMN not in df.columns:

        st.error(
            "Churn column was not found "
            "in the dataset."
        )

        return

    # --------------------------------------------------------
    # Convert Churn to numeric
    # --------------------------------------------------------

    churn_numeric = (
        df[config.TARGET_COLUMN]
        .astype(str)
        .str.strip()
        .str.lower()
        .map({
            "yes": 1,
            "no": 0,
            "1": 1,
            "0": 0,
        })
    )

    df["_churn_numeric"] = churn_numeric

    # --------------------------------------------------------
    # KPI CARDS
    # --------------------------------------------------------

    total_customers = len(df)

    churned = int(
        df["_churn_numeric"]
        .sum()
    )

    stayed = (
        total_customers
        - churned
    )

    churn_rate = (
        churned / total_customers * 100
        if total_customers
        else 0
    )

    c1, c2, c3, c4 = st.columns(4)

    c1.metric(
        "Total Customers",
        f"{total_customers:,}",
    )

    c2.metric(
        "Churned Customers",
        f"{churned:,}",
    )

    c3.metric(
        "Customers Stayed",
        f"{stayed:,}",
    )

    c4.metric(
        "Overall Churn Rate",
        f"{churn_rate:.1f}%",
    )

    st.divider()

    # --------------------------------------------------------
    # CHURN DISTRIBUTION
    # --------------------------------------------------------

    st.subheader(
        "📈 Churn Distribution"
    )

    distribution = (
        df[config.TARGET_COLUMN]
        .value_counts()
    )

    st.bar_chart(
        distribution
    )

    # --------------------------------------------------------
    # CHURN BY CONTRACT
    # --------------------------------------------------------

    if "Contract" in df.columns:

        st.subheader(
            "📄 Churn Rate by Contract"
        )

        contract_rate = (
            df.groupby("Contract")[
                "_churn_numeric"
            ]
            .mean()
            .mul(100)
            .sort_values(
                ascending=False
            )
        )

        st.bar_chart(
            contract_rate
        )

        st.dataframe(
            contract_rate
            .round(2)
            .rename(
                "Churn Rate (%)"
            )
        )

    # --------------------------------------------------------
    # CHURN BY INTERNET SERVICE
    # --------------------------------------------------------

    if "InternetService" in df.columns:

        st.subheader(
            "🌐 Churn Rate by Internet Service"
        )

        internet_rate = (
            df.groupby(
                "InternetService"
            )["_churn_numeric"]
            .mean()
            .mul(100)
            .sort_values(
                ascending=False
            )
        )

        st.bar_chart(
            internet_rate
        )

    # --------------------------------------------------------
    # CHURN BY PAYMENT METHOD
    # --------------------------------------------------------

    if "PaymentMethod" in df.columns:

        st.subheader(
            "💳 Churn Rate by Payment Method"
        )

        payment_rate = (
            df.groupby(
                "PaymentMethod"
            )["_churn_numeric"]
            .mean()
            .mul(100)
            .sort_values(
                ascending=False
            )
        )

        st.bar_chart(
            payment_rate
        )

    # --------------------------------------------------------
    # CHURN BY SENIOR CITIZEN
    # --------------------------------------------------------

    if "SeniorCitizen" in df.columns:

        st.subheader(
            "👥 Churn Rate by Senior Citizen"
        )

        senior_df = df.copy()

        senior_df["Senior Citizen"] = (
            senior_df["SeniorCitizen"]
            .map({
                0: "No",
                1: "Yes",
            })
        )

        senior_rate = (
            senior_df.groupby(
                "Senior Citizen"
            )["_churn_numeric"]
            .mean()
            .mul(100)
        )

        st.bar_chart(
            senior_rate
        )

    # --------------------------------------------------------
    # CHURN BY TENURE
    # --------------------------------------------------------

    if "tenure" in df.columns:

        st.subheader(
            "⏳ Churn Rate by Tenure Group"
        )

        tenure_bins = [
            -1,
            12,
            24,
            48,
            1000,
        ]

        tenure_labels = [
            "0-12 months",
            "12-24 months",
            "24-48 months",
            "48+ months",
        ]

        df["Tenure Group"] = pd.cut(
            df["tenure"],
            bins=tenure_bins,
            labels=tenure_labels,
        )

        tenure_rate = (
            df.groupby(
                "Tenure Group",
                observed=False,
            )["_churn_numeric"]
            .mean()
            .mul(100)
        )

        st.bar_chart(
            tenure_rate
        )

    # --------------------------------------------------------
    # DATA PREVIEW
    # --------------------------------------------------------

    with st.expander(
        "🔎 View Dataset"
    ):

        st.dataframe(
            df.drop(
                columns=[
                    "_churn_numeric"
                ],
                errors="ignore",
            ),
            use_container_width=True,
        )


# ============================================================
# PAGE 3 — MODEL PERFORMANCE
# ============================================================

def performance_page() -> None:

    st.header(
        "🤖 Model Performance"
    )

    st.write(
        "Evaluate the trained machine "
        "learning model using the "
        "available customer dataset."
    )

    # --------------------------------------------------------
    # LOAD MODEL
    # --------------------------------------------------------

    try:

        model = _local_model()

    except Exception as exc:

        st.error(
            f"Could not load model: {exc}"
        )

        return

    # --------------------------------------------------------
    # MODEL INFORMATION
    # --------------------------------------------------------

    st.subheader(
        "📋 Model Information"
    )

    model_name = model.metadata.get(
        "model_name",
        "Unknown",
    )

    version = model.metadata.get(
        "version",
        model.version,
    )

    threshold = model.threshold

    c1, c2, c3 = st.columns(3)

    c1.metric(
        "Model",
        str(model_name),
    )

    c2.metric(
        "Version",
        str(version),
    )

    c3.metric(
        "Decision Threshold",
        f"{threshold:.3f}",
    )

    st.divider()

    # --------------------------------------------------------
    # SAVED TEST METRICS
    # --------------------------------------------------------

    st.subheader(
        "📈 Saved Test Metrics"
    )

    saved_metrics = model.metadata.get(
        "test_metrics",
        {},
    )

    c1, c2, c3, c4 = st.columns(4)

    roc_saved = saved_metrics.get(
        "roc_auc"
    )

    pr_saved = saved_metrics.get(
        "pr_auc"
    )

    brier_saved = saved_metrics.get(
        "brier"
    )

    accuracy_saved = saved_metrics.get(
        "accuracy"
    )

    c1.metric(
        "ROC-AUC",
        f"{roc_saved:.3f}"
        if roc_saved is not None
        else "N/A",
    )

    c2.metric(
        "PR-AUC",
        f"{pr_saved:.3f}"
        if pr_saved is not None
        else "N/A",
    )

    c3.metric(
        "Brier",
        f"{brier_saved:.3f}"
        if brier_saved is not None
        else "N/A",
    )

    c4.metric(
        "Accuracy",
        f"{accuracy_saved:.3f}"
        if accuracy_saved is not None
        else "N/A",
    )

    st.divider()

    # --------------------------------------------------------
    # LOAD DATA
    # --------------------------------------------------------

    try:

        df = load_dataset()

    except Exception as exc:

        st.error(
            f"Could not load dataset: {exc}"
        )

        return

    if config.TARGET_COLUMN not in df.columns:

        st.error(
            "Target column 'Churn' "
            "was not found."
        )

        return

    # --------------------------------------------------------
    # CLEAN DATA
    # --------------------------------------------------------

    try:

        X = clean_features(
            df
        )

        y = (
            df[config.TARGET_COLUMN]
            .astype(str)
            .str.strip()
            .str.lower()
            .map({
                "yes": 1,
                "no": 0,
                "1": 1,
                "0": 0,
            })
        )

        valid = y.notna()

        X = X.loc[valid]
        y = y.loc[valid].astype(int)

    except Exception as exc:

        st.error(
            f"Could not prepare evaluation data: {exc}"
        )

        return

    # --------------------------------------------------------
    # PREDICT
    # --------------------------------------------------------

    try:

        probabilities = (
            model.pipeline
            .predict_proba(X)[:, 1]
        )

        predictions = (
            probabilities
            >= threshold
        ).astype(int)

    except Exception as exc:

        st.error(
            f"Could not evaluate model: {exc}"
        )

        return

    # --------------------------------------------------------
    # CALCULATE METRICS
    # --------------------------------------------------------

    from sklearn.metrics import (
        accuracy_score,
        average_precision_score,
        classification_report,
        confusion_matrix,
        f1_score,
        precision_score,
        recall_score,
        roc_auc_score,
        roc_curve,
    )

    accuracy = accuracy_score(
        y,
        predictions,
    )

    precision = precision_score(
        y,
        predictions,
        zero_division=0,
    )

    recall = recall_score(
        y,
        predictions,
        zero_division=0,
    )

    f1 = f1_score(
        y,
        predictions,
        zero_division=0,
    )

    try:

        roc_auc = roc_auc_score(
            y,
            probabilities,
        )

    except ValueError:

        roc_auc = 0.0

    try:

        pr_auc = average_precision_score(
            y,
            probabilities,
        )

    except ValueError:

        pr_auc = 0.0

    # --------------------------------------------------------
    # CURRENT EVALUATION METRICS
    # --------------------------------------------------------

    st.subheader(
        "📊 Evaluation Metrics"
    )

    c1, c2, c3, c4, c5 = st.columns(5)

    c1.metric(
        "Accuracy",
        f"{accuracy:.3f}",
    )

    c2.metric(
        "Precision",
        f"{precision:.3f}",
    )

    c3.metric(
        "Recall",
        f"{recall:.3f}",
    )

    c4.metric(
        "F1 Score",
        f"{f1:.3f}",
    )

    c5.metric(
        "ROC-AUC",
        f"{roc_auc:.3f}",
    )

    # --------------------------------------------------------
    # CONFUSION MATRIX
    # --------------------------------------------------------

    st.subheader(
        "🔢 Confusion Matrix"
    )

    cm = confusion_matrix(
        y,
        predictions,
    )

    cm_df = pd.DataFrame(
        cm,
        index=[
            "Actual: Stayed",
            "Actual: Churned",
        ],
        columns=[
            "Predicted: Stayed",
            "Predicted: Churned",
        ],
    )

    st.dataframe(
        cm_df,
        use_container_width=True,
    )

    # --------------------------------------------------------
    # CLASSIFICATION REPORT
    # --------------------------------------------------------

    st.subheader(
        "📋 Classification Report"
    )

    report = classification_report(
        y,
        predictions,
        target_names=[
            "Stayed",
            "Churned",
        ],
        output_dict=True,
        zero_division=0,
    )

    report_df = pd.DataFrame(
        report
    ).transpose()

    st.dataframe(
        report_df.round(3),
        use_container_width=True,
    )

    # --------------------------------------------------------
    # ROC CURVE
    # --------------------------------------------------------

    st.subheader(
        "📈 ROC Curve"
    )

    fpr, tpr, _ = roc_curve(
        y,
        probabilities,
    )

    roc_df = pd.DataFrame({
        "False Positive Rate": fpr,
        "True Positive Rate": tpr,
    })

    st.line_chart(
        roc_df,
        x="False Positive Rate",
        y="True Positive Rate",
    )

    st.success(
        f"ROC-AUC: {roc_auc:.3f}"
    )


# ============================================================
# PAGE 4 — BATCH PREDICTION
# ============================================================

def batch_prediction_page() -> None:

    st.header(
        "📁 Batch Prediction"
    )

    st.write(
        "Upload a CSV file containing "
        "multiple customers and generate "
        "churn predictions."
    )

    uploaded_file = st.file_uploader(
        "Upload Customer CSV",
        type=["csv"],
    )

    if uploaded_file is None:

        st.info(
            "Upload a CSV file to begin "
            "batch prediction."
        )

        return

    try:

        df = pd.read_csv(
            uploaded_file
        )

        df.columns = (
            df.columns
            .str.strip()
        )

    except Exception as exc:

        st.error(
            f"Could not read CSV: {exc}"
        )

        return

    st.success(
        f"Successfully loaded "
        f"{len(df):,} customer records."
    )

    # --------------------------------------------------------
    # CHECK COLUMNS
    # --------------------------------------------------------

    missing = [
        col
        for col in config.FEATURE_COLUMNS
        if col not in df.columns
    ]

    if missing:

        st.error(
            "The uploaded CSV is missing "
            "the following required columns:"
        )

        st.write(
            missing
        )

        st.info(
            "Please upload a CSV containing "
            "the same customer feature columns "
            "as the training dataset."
        )

        return

    # --------------------------------------------------------
    # PREVIEW
    # --------------------------------------------------------

    st.subheader(
        "👀 Customer Data Preview"
    )

    st.dataframe(
        df.head(10),
        use_container_width=True,
    )

    # --------------------------------------------------------
    # PREDICT
    # --------------------------------------------------------

    if st.button(
        "🔮 Generate Predictions",
        use_container_width=True,
    ):

        try:

            model = _local_model()

            X = clean_features(
                df
            )

            probabilities = (
                model.pipeline
                .predict_proba(X)[:, 1]
            )

            predictions = (
                probabilities
                >= model.threshold
            )

            results = df.copy()

            results[
                "Churn Probability"
            ] = (
                probabilities * 100
            ).round(2)

            results[
                "Predicted Churn"
            ] = np.where(
                predictions,
                "Yes",
                "No",
            )

            # Risk tier
            def get_risk(
                probability: float
            ) -> str:

                if probability >= 0.70:
                    return "High"

                if probability >= 0.40:
                    return "Moderate"

                return "Low"

            results[
                "Risk Tier"
            ] = [
                get_risk(p)
                for p in probabilities
            ]

            st.success(
                "Predictions generated successfully!"
            )

            # ------------------------------------------------
            # SUMMARY
            # ------------------------------------------------

            st.subheader(
                "📊 Prediction Summary"
            )

            total = len(results)

            predicted_churn = int(
                predictions.sum()
            )

            predicted_stay = (
                total
                - predicted_churn
            )

            c1, c2, c3 = st.columns(3)

            c1.metric(
                "Total Customers",
                f"{total:,}",
            )

            c2.metric(
                "Predicted Churn",
                f"{predicted_churn:,}",
            )

            c3.metric(
                "Predicted Stay",
                f"{predicted_stay:,}",
            )

            # ------------------------------------------------
            # RESULT TABLE
            # ------------------------------------------------

            st.subheader(
                "📋 Prediction Results"
            )

            st.dataframe(
                results,
                use_container_width=True,
            )

            # ------------------------------------------------
            # DOWNLOAD
            # ------------------------------------------------

            csv_buffer = io.StringIO()

            results.to_csv(
                csv_buffer,
                index=False,
            )

            st.download_button(
                label="⬇️ Download Predictions CSV",
                data=csv_buffer.getvalue(),
                file_name="churn_predictions.csv",
                mime="text/csv",
                use_container_width=True,
            )

        except Exception as exc:

            st.error(
                f"Batch prediction failed: {exc}"
            )


# ============================================================
# MAIN APPLICATION
# ============================================================

def main() -> None:

    # --------------------------------------------------------
    # TITLE
    # --------------------------------------------------------

    st.title(
        "📊 Customer Churn Prediction"
    )

    st.caption(
        "Machine Learning + Streamlit"
    )

    # --------------------------------------------------------
    # SIDEBAR
    # --------------------------------------------------------

    st.sidebar.title(
        "Navigation"
    )

    st.sidebar.write(
        "Select Page"
    )

    page = st.sidebar.radio(
        "Select Page",
        [
            "🏠 Prediction",
            "📊 Churn Analysis",
            "🤖 Model Performance",
            "📁 Batch Prediction",
        ],
        label_visibility="collapsed",
    )

    st.sidebar.divider()

    st.sidebar.caption(
        "Customer Churn Prediction System"
    )

    st.sidebar.caption(
        "Machine Learning + Streamlit"
    )

    # --------------------------------------------------------
    # PAGE SELECTION
    # --------------------------------------------------------

    if page == "🏠 Prediction":

        prediction_page()

    elif page == "📊 Churn Analysis":

        analysis_page()

    elif page == "🤖 Model Performance":

        performance_page()

    elif page == "📁 Batch Prediction":

        batch_prediction_page()


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":
    main()