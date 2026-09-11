from __future__ import annotations

import io
from typing import Any

import numpy as np
import pandas as pd
import streamlit as st

from churn import config
from churn.schema import GROUPS, INPUT_FIELDS

# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Customer Churn Prediction",
    page_icon="📊",
    layout="wide",
)


# ============================================================
# HELPER FUNCTIONS
# ============================================================


def _fields_for_group(group_name: str) -> list[Any]:
    """Return all input fields belonging to a group."""

    return [
        field
        for field in INPUT_FIELDS
        if field.group == group_name
    ]


@st.cache_resource(show_spinner="Loading model...")
def _local_model():
    """Load the local churn prediction model."""

    from churn.predict import load_model

    return load_model()


def _score(features: dict[str, Any]) -> dict[str, Any]:
    """Score a customer using the configured API or local model."""

    api_url = config.settings.api_url

    # --------------------------------------------------------
    # API MODEL
    # --------------------------------------------------------

    if api_url:
        import httpx

        try:
            response = httpx.post(
                f"{api_url.rstrip('/')}/predict",
                json=features,
                timeout=10,
            )

            response.raise_for_status()

            result = response.json()

            try:
                model_response = httpx.get(
                    f"{api_url.rstrip('/')}/model",
                    timeout=10,
                )

                if model_response.is_success:
                    model_info = model_response.json()
                    result.setdefault(
                        "model_meta",
                        model_info,
                    )

            except Exception:
                pass

            return result

        except Exception as exc:
            st.warning(
                "API scoring failed. "
                "Using local model instead. "
                f"Error: {exc}"
            )

    # --------------------------------------------------------
    # LOCAL MODEL
    # --------------------------------------------------------

    model = _local_model()

    if hasattr(model, "predict_with_details"):
        return model.predict_with_details(features)

    probability = float(
        model.predict_proba(features)[0][1]
    )

    threshold = getattr(
        model,
        "threshold",
        0.5,
    )

    churn = probability >= threshold

    if probability >= 0.7:
        risk_tier = "High"

    elif probability >= 0.4:
        risk_tier = "Moderate"

    else:
        risk_tier = "Low"

    return {
        "probability": probability,
        "churn": churn,
        "risk_tier": risk_tier,
        "threshold": threshold,
        "warnings": [],
        "top_factors": [],
        "model_version": getattr(
            model,
            "version",
            "local",
        ),
        "model_meta": getattr(
            model,
            "metadata",
            {},
        ),
    }


def _show_result(
    result: dict[str, Any],
) -> None:
    """Display prediction results."""

    probability = float(
        result.get(
            "probability",
            0.0,
        )
    )

    churn = result.get(
        "churn",
        False,
    )

    risk_tier = result.get(
        "risk_tier",
        "Low",
    )

    # --------------------------------------------------------
    # RESULT SUMMARY
    # --------------------------------------------------------

    st.subheader("Prediction Result")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            "Churn Probability",
            f"{probability * 100:.1f}%",
        )

    with col2:
        st.metric(
            "Prediction",
            (
                "Likely to Churn"
                if churn
                else "Likely to Stay"
            ),
        )

    with col3:
        st.metric(
            "Risk Tier",
            risk_tier,
        )

    # --------------------------------------------------------
    # RISK MESSAGE
    # --------------------------------------------------------

    if risk_tier == "High":
        st.error(
            "⚠️ High churn risk"
        )

    elif risk_tier == "Moderate":
        st.warning(
            "⚠️ Moderate churn risk"
        )

    else:
        st.success(
            "✅ Low churn risk"
        )

    # --------------------------------------------------------
    # WARNINGS
    # --------------------------------------------------------

    warnings = result.get(
        "warnings"
    ) or []

    if warnings:

        st.subheader("Warnings")

        for warning in warnings:
            st.warning(
                str(warning)
            )

    # --------------------------------------------------------
    # TOP CHURN FACTORS
    # --------------------------------------------------------

    top_factors = result.get(
        "top_factors"
    ) or []

    if top_factors:

        st.subheader(
            "Top Churn Factors"
        )

        if isinstance(
            top_factors,
            list,
        ):

            for factor in top_factors:

                if isinstance(
                    factor,
                    dict,
                ):

                    name = factor.get(
                        "feature",
                        factor.get(
                            "name",
                            "Feature",
                        ),
                    )

                    value = factor.get(
                        "value",
                        "",
                    )

                    impact = factor.get(
                        "impact",
                        "",
                    )

                    text = f"**{name}**"

                    if value != "":
                        text += f": {value}"

                    if impact != "":
                        text += f" — {impact}"

                    st.write(text)

                else:
                    st.write(
                        f"• {factor}"
                    )

    # --------------------------------------------------------
    # MODEL INFORMATION
    # --------------------------------------------------------

    model_meta = result.get(
        "model_meta"
    ) or {}

    if model_meta:

        with st.expander(
            "Model Information"
        ):

            st.write(
                "Model version:",
                (
                    result.get(
                        "model_version"
                    )
                    or model_meta.get(
                        "model_version"
                    )
                    or model_meta.get(
                        "version"
                    )
                    or "N/A"
                ),
            )

            saved_metrics = (
                model_meta.get(
                    "test_metrics",
                    {},
                )
            )

            if saved_metrics:

                st.write(
                    "Saved test metrics:"
                )

                metric_cols = st.columns(
                    5
                )

                with metric_cols[0]:

                    value = (
                        saved_metrics.get(
                            "accuracy"
                        )
                    )

                    st.metric(
                        "Accuracy",
                        (
                            f"{value:.3f}"
                            if value is not None
                            else "N/A"
                        ),
                    )

                with metric_cols[1]:

                    value = (
                        saved_metrics.get(
                            "precision"
                        )
                    )

                    st.metric(
                        "Precision",
                        (
                            f"{value:.3f}"
                            if value is not None
                            else "N/A"
                        ),
                    )

                with metric_cols[2]:

                    value = (
                        saved_metrics.get(
                            "recall"
                        )
                    )

                    st.metric(
                        "Recall",
                        (
                            f"{value:.3f}"
                            if value is not None
                            else "N/A"
                        ),
                    )

                with metric_cols[3]:

                    value = (
                        saved_metrics.get(
                            "f1"
                        )
                    )

                    st.metric(
                        "F1",
                        (
                            f"{value:.3f}"
                            if value is not None
                            else "N/A"
                        ),
                    )

                with metric_cols[4]:

                    value = (
                        saved_metrics.get(
                            "roc_auc"
                        )
                    )

                    st.metric(
                        "ROC-AUC",
                        (
                            f"{value:.3f}"
                            if value is not None
                            else "N/A"
                        ),
                    )

                pr_saved = (
                    saved_metrics.get(
                        "pr_auc"
                    )
                )

                if pr_saved is not None:

                    st.metric(
                        "PR-AUC",
                        f"{pr_saved:.3f}",
                    )


# ============================================================
# PREDICTION PAGE
# ============================================================


def prediction_page() -> None:
    """Customer churn prediction page."""

    st.title(
        "📊 Customer Churn Prediction"
    )

    st.write(
        "Enter customer information below "
        "to predict the probability of churn."
    )

    features: dict[str, Any] = {}

    # GROUPS is a list of group names.
    for group_name in GROUPS:

        st.subheader(
            group_name
        )

        group_fields = [
            field
            for field in INPUT_FIELDS
            if field.group == group_name
        ]

        columns = st.columns(2)

        for index, field in enumerate(
            group_fields
        ):

            with columns[index % 2]:

                # ------------------------------------------------
                # NUMBER
                # ------------------------------------------------

                if field.kind == "number":

                    min_value = (
                        float(
                            field.min_value
                        )
                        if field.min_value
                        is not None
                        else None
                    )

                    max_value = (
                        float(
                            field.max_value
                        )
                        if field.max_value
                        is not None
                        else None
                    )

                    default_value = float(
                        field.default
                        if field.default
                        is not None
                        else 0.0
                    )

                    if (
                        min_value
                        is not None
                    ):
                        default_value = max(
                            default_value,
                            min_value,
                        )

                    if (
                        max_value
                        is not None
                    ):
                        default_value = min(
                            default_value,
                            max_value,
                        )

                    value = st.number_input(
                        field.label,
                        min_value=min_value,
                        max_value=max_value,
                        value=default_value,
                        help=(
                            field.help
                            or None
                        ),
                        key=(
                            f"prediction_"
                            f"{field.name}"
                        ),
                    )

                # ------------------------------------------------
                # INTEGER
                # ------------------------------------------------

                elif field.kind == "int":

                    if field.choices:

                        options = list(
                            field.choices
                        )

                        default_index = 0

                        default_string = str(
                            field.default
                        )

                        if (
                            default_string
                            in options
                        ):
                            default_index = (
                                options.index(
                                    default_string
                                )
                            )

                        selected = (
                            st.selectbox(
                                field.label,
                                options,
                                index=default_index,
                                help=(
                                    field.help
                                    or None
                                ),
                                key=(
                                    f"prediction_"
                                    f"{field.name}"
                                ),
                            )
                        )

                        value = (
                            field.coerce(
                                selected
                            )
                        )

                    else:

                        min_value = (
                            int(
                                field.min_value
                            )
                            if field.min_value
                            is not None
                            else None
                        )

                        max_value = (
                            int(
                                field.max_value
                            )
                            if field.max_value
                            is not None
                            else None
                        )

                        value = st.number_input(
                            field.label,
                            min_value=min_value,
                            max_value=max_value,
                            value=int(
                                field.default
                            ),
                            step=1,
                            help=(
                                field.help
                                or None
                            ),
                            key=(
                                f"prediction_"
                                f"{field.name}"
                            ),
                        )

                # ------------------------------------------------
                # CHOICE
                # ------------------------------------------------

                elif field.kind == "choice":

                    options = list(
                        field.choices
                        or ()
                    )

                    if options:

                        default_index = 0

                        if (
                            field.default
                            in options
                        ):
                            default_index = (
                                options.index(
                                    field.default
                                )
                            )

                        value = (
                            st.selectbox(
                                field.label,
                                options,
                                index=default_index,
                                help=(
                                    field.help
                                    or None
                                ),
                                key=(
                                    f"prediction_"
                                    f"{field.name}"
                                ),
                            )
                        )

                    else:

                        value = st.text_input(
                            field.label,
                            value=str(
                                field.default
                                if field.default
                                is not None
                                else ""
                            ),
                            help=(
                                field.help
                                or None
                            ),
                            key=(
                                f"prediction_"
                                f"{field.name}"
                            ),
                        )

                # ------------------------------------------------
                # FALLBACK
                # ------------------------------------------------

                else:

                    value = st.text_input(
                        field.label,
                        value=str(
                            field.default
                            if field.default
                            is not None
                            else ""
                        ),
                        help=(
                            field.help
                            or None
                        ),
                        key=(
                            f"prediction_"
                            f"{field.name}"
                        ),
                    )

                features[field.name] = value

    # --------------------------------------------------------
    # PREDICTION BUTTON
    # --------------------------------------------------------

    if st.button(
        "🔮 Predict Churn",
        type="primary",
        use_container_width=True,
    ):

        try:

            with st.spinner(
                "Making prediction..."
            ):

                result = _score(
                    features
                )

            _show_result(
                result
            )

        except Exception as exc:

            st.error(
                f"Prediction failed: {exc}"
            )


# ============================================================
# CHURN ANALYSIS PAGE
# ============================================================


def churn_analysis_page() -> None:
    """Analyze churn patterns in uploaded customer data."""

    st.title(
        "📈 Churn Analysis"
    )

    uploaded_file = st.file_uploader(
        "Upload customer CSV file",
        type=["csv"],
    )

    if uploaded_file is None:

        st.info(
            "Upload a CSV file to begin "
            "the analysis."
        )

        return

    try:

        df = pd.read_csv(
            uploaded_file
        )

    except Exception as exc:

        st.error(
            f"Could not read the CSV file: {exc}"
        )

        return

    st.subheader(
        "Dataset Preview"
    )

    st.dataframe(
        df.head(20),
        use_container_width=True,
    )

    st.subheader(
        "Dataset Summary"
    )

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            "Rows",
            len(df),
        )

    with col2:
        st.metric(
            "Columns",
            len(df.columns),
        )

    with col3:
        st.metric(
            "Missing Values",
            int(
                df.isna()
                .sum()
                .sum()
            ),
        )

    churn_columns = [
        column
        for column in df.columns
        if column.lower()
        in {
            "churn",
            "churned",
            "exited",
            "is_churn",
        }
    ]

    if not churn_columns:

        st.warning(
            "No churn column was detected. "
            "Expected names include `churn`, "
            "`churned`, `exited`, or `is_churn`."
        )

        return

    churn_column = churn_columns[0]

    st.subheader(
        "Churn Distribution"
    )

    churn_counts = (
        df[churn_column]
        .value_counts(
            dropna=False
        )
    )

    st.bar_chart(
        churn_counts
    )

    st.subheader(
        "Churn Statistics"
    )

    total_customers = len(df)

    if total_customers > 0:

        churn_rate = (
            df[churn_column]
            .astype(str)
            .str.lower()
            .isin(
                [
                    "1",
                    "true",
                    "yes",
                    "churn",
                    "churned",
                ]
            )
            .mean()
        )

        st.metric(
            "Estimated Churn Rate",
            f"{churn_rate * 100:.2f}%",
        )

    numeric_columns = (
        df.select_dtypes(
            include=np.number
        )
        .columns
        .tolist()
    )

    if numeric_columns:

        st.subheader(
            "Numeric Feature Analysis"
        )

        selected_column = (
            st.selectbox(
                "Select a numeric feature",
                numeric_columns,
            )
        )

        st.line_chart(
            df[
                selected_column
            ].reset_index(
                drop=True
            )
        )


# ============================================================
# MODEL PERFORMANCE PAGE
# ============================================================


def performance_page() -> None:
    """Display model performance information."""

    st.title(
        "🎯 Model Performance"
    )

    try:

        model = _local_model()

    except Exception as exc:

        st.error(
            f"Could not load model: {exc}"
        )

        return

    metadata = (
        getattr(
            model,
            "metadata",
            {},
        )
        or {}
    )

    test_metrics = (
        metadata.get(
            "test_metrics",
            {},
        )
        or {}
    )

    if not test_metrics:

        st.warning(
            "No saved test metrics were "
            "found in the model metadata."
        )

        return

    st.subheader(
        "Test Metrics"
    )

    accuracy = test_metrics.get(
        "accuracy"
    )

    precision = test_metrics.get(
        "precision"
    )

    recall = test_metrics.get(
        "recall"
    )

    f1 = test_metrics.get(
        "f1"
    )

    roc_auc = test_metrics.get(
        "roc_auc"
    )

    metric_columns = st.columns(5)

    with metric_columns[0]:

        st.metric(
            "Accuracy",
            (
                f"{accuracy:.3f}"
                if accuracy is not None
                else "N/A"
            ),
        )

    with metric_columns[1]:

        st.metric(
            "Precision",
            (
                f"{precision:.3f}"
                if precision is not None
                else "N/A"
            ),
        )

    with metric_columns[2]:

        st.metric(
            "Recall",
            (
                f"{recall:.3f}"
                if recall is not None
                else "N/A"
            ),
        )

    with metric_columns[3]:

        st.metric(
            "F1 Score",
            (
                f"{f1:.3f}"
                if f1 is not None
                else "N/A"
            ),
        )

    with metric_columns[4]:

        st.metric(
            "ROC-AUC",
            (
                f"{roc_auc:.3f}"
                if roc_auc is not None
                else "N/A"
            ),
        )

    pr_auc = test_metrics.get(
        "pr_auc"
    )

    if pr_auc is not None:

        st.metric(
            "PR-AUC",
            f"{pr_auc:.3f}",
        )

    st.subheader(
        "Model Details"
    )

    model_columns = st.columns(2)

    with model_columns[0]:

        st.write(
            "**Model Version:**",
            (
                metadata.get(
                    "model_version"
                )
                or getattr(
                    model,
                    "version",
                    "N/A",
                )
            ),
        )

        st.write(
            "**Threshold:**",
            getattr(
                model,
                "threshold",
                metadata.get(
                    "threshold",
                    "N/A",
                ),
            ),
        )

    with model_columns[1]:

        st.write(
            "**Model Type:**",
            metadata.get(
                "model_type",
                type(model).__name__,
            ),
        )

        features = metadata.get(
            "features",
            [],
        )

        st.write(
            "**Features:**",
            len(features)
            if features
            else "N/A",
        )

    st.subheader(
        "Confusion Matrix"
    )

    confusion_data = metadata.get(
        "confusion_matrix"
    )

    if confusion_data is not None:

        try:

            confusion_array = np.asarray(
                confusion_data
            )

            if confusion_array.shape == (
                2,
                2,
            ):

                confusion_df = pd.DataFrame(
                    confusion_array,
                    index=[
                        "Actual 0",
                        "Actual 1",
                    ],
                    columns=[
                        "Predicted 0",
                        "Predicted 1",
                    ],
                )

                st.dataframe(
                    confusion_df,
                    use_container_width=True,
                )

            else:

                st.info(
                    "Saved confusion matrix "
                    "has an unexpected format."
                )

        except Exception:

            st.info(
                "Could not display the saved "
                "confusion matrix."
            )

    else:

        st.info(
            "No saved confusion matrix "
            "was found."
        )


# ============================================================
# BATCH PREDICTION PAGE
# ============================================================


def batch_prediction_page() -> None:
    """Predict churn for multiple customers from a CSV file."""

    st.title(
        "📦 Batch Prediction"
    )

    st.write(
        "Upload a CSV containing customer "
        "information to generate predictions "
        "for multiple customers."
    )

    uploaded_file = st.file_uploader(
        "Upload CSV",
        type=["csv"],
        key="batch_upload",
    )

    if uploaded_file is None:

        st.info(
            "Upload a CSV file to begin."
        )

        return

    try:

        df = pd.read_csv(
            uploaded_file
        )

    except Exception as exc:

        st.error(
            f"Could not read the CSV file: {exc}"
        )

        return

    st.subheader(
        "Uploaded Data"
    )

    st.dataframe(
        df.head(20),
        use_container_width=True,
    )

    # INPUT_FIELDS is a list of Field objects.
    required_fields = [
        field.name
        for field in INPUT_FIELDS
    ]

    missing_fields = [
        field
        for field in required_fields
        if field not in df.columns
    ]

    if missing_fields:

        st.error(
            "The following required columns "
            "are missing: "
            + ", ".join(
                missing_fields
            )
        )

        return

    if st.button(
        "🚀 Run Batch Prediction",
        type="primary",
        use_container_width=True,
    ):

        predictions: list[
            dict[str, Any]
        ] = []

        progress_bar = st.progress(
            0
        )

        total_rows = len(df)

        for row_number, (
            _,
            row,
        ) in enumerate(
            df.iterrows(),
            start=1,
        ):

            features: dict[
                str,
                Any,
            ] = {}

            for field in INPUT_FIELDS:

                raw_value = row[
                    field.name
                ]

                if pd.isna(
                    raw_value
                ):
                    raw_value = (
                        field.default
                    )

                try:

                    features[
                        field.name
                    ] = field.coerce(
                        raw_value
                    )

                except (
                    TypeError,
                    ValueError,
                ):

                    features[
                        field.name
                    ] = field.default

            try:

                result = _score(
                    features
                )

                predictions.append(
                    {
                        "churn_probability": (
                            result.get(
                                "probability",
                                np.nan,
                            )
                        ),
                        "churn_prediction": (
                            result.get(
                                "churn",
                                False,
                            )
                        ),
                        "risk_tier": (
                            result.get(
                                "risk_tier",
                                "Unknown",
                            )
                        ),
                    }
                )

            except Exception as exc:

                predictions.append(
                    {
                        "churn_probability": np.nan,
                        "churn_prediction": False,
                        "risk_tier": (
                            f"Error: {exc}"
                        ),
                    }
                )

            if total_rows > 0:

                progress_bar.progress(
                    int(
                        row_number
                        / total_rows
                        * 100
                    )
                )

        result_df = pd.concat(
            [
                df.reset_index(
                    drop=True
                ),
                pd.DataFrame(
                    predictions
                ),
            ],
            axis=1,
        )

        st.success(
            "Batch prediction completed!"
        )

        st.subheader(
            "Prediction Results"
        )

        st.dataframe(
            result_df,
            use_container_width=True,
        )

        csv_buffer = io.StringIO()

        result_df.to_csv(
            csv_buffer,
            index=False,
        )

        st.download_button(
            label=(
                "⬇️ Download Predictions CSV"
            ),
            data=csv_buffer.getvalue(),
            file_name=(
                "churn_predictions.csv"
            ),
            mime="text/csv",
            use_container_width=True,
        )


# ============================================================
# MAIN APPLICATION
# ============================================================


def main() -> None:
    """Run the Streamlit application."""

    st.sidebar.title(
        "Customer Churn Prediction"
    )

    page = st.sidebar.radio(
        "Select Page",
        [
            "Prediction",
            "Churn Analysis",
            "Model Performance",
            "Batch Prediction",
        ],
    )

    st.sidebar.markdown(
        "---"
    )

    st.sidebar.info(
        "Use the navigation above to make "
        "individual predictions, analyze "
        "customer churn, inspect model "
        "performance, or run batch predictions."
    )

    if page == "Prediction":

        prediction_page()

    elif page == "Churn Analysis":

        churn_analysis_page()

    elif page == "Model Performance":

        performance_page()

    elif page == "Batch Prediction":

        batch_prediction_page()