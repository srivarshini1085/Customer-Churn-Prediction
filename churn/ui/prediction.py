"""Customer churn single prediction page."""

from __future__ import annotations

from typing import Any

import streamlit as st

from churn.schema import GROUPS, INPUT_FIELDS
from churn.ui.components import score, show_result, show_what_if, what_if


def prediction_page() -> None:
    """Customer churn prediction page."""
    st.title("📊 Customer Churn Prediction")
    st.write("Enter customer information below to predict the probability of churn.")

    features: dict[str, Any] = {}

    for group_name in GROUPS:
        st.subheader(group_name)
        group_fields = [field for field in INPUT_FIELDS if field.group == group_name]
        columns = st.columns(2)

        for index, field in enumerate(group_fields):
            with columns[index % 2]:
                value: Any
                if field.kind == "number":
                    min_val = float(field.min_value) if field.min_value is not None else None
                    max_val = float(field.max_value) if field.max_value is not None else None
                    default_val = float(field.default if field.default is not None else 0.0)

                    if min_val is not None:
                        default_val = max(default_val, min_val)
                    if max_val is not None:
                        default_val = min(default_val, max_val)

                    value = st.number_input(
                        field.label,
                        min_value=min_val,
                        max_value=max_val,
                        value=default_val,
                        help=field.help or None,
                        key=f"prediction_{field.name}",
                    )

                elif field.kind == "int":
                    if field.choices:
                        options = list(field.choices)
                        default_index = 0
                        default_string = str(field.default)
                        if default_string in options:
                            default_index = options.index(default_string)

                        selected = st.selectbox(
                            field.label,
                            options,
                            index=default_index,
                            help=field.help or None,
                            key=f"prediction_{field.name}",
                        )
                        value = field.coerce(selected)
                    else:
                        min_int = int(field.min_value) if field.min_value is not None else None
                        max_int = int(field.max_value) if field.max_value is not None else None
                        default_int = int(field.default if field.default is not None else 0)

                        value = st.number_input(
                            field.label,
                            min_value=min_int,
                            max_value=max_int,
                            value=default_int,
                            step=1,
                            help=field.help or None,
                            key=f"prediction_{field.name}",
                        )

                elif field.kind == "choice":
                    options = list(field.choices or ())
                    if options:
                        default_index = 0
                        if field.default in options:
                            default_index = options.index(field.default)

                        value = st.selectbox(
                            field.label,
                            options,
                            index=default_index,
                            help=field.help or None,
                            key=f"prediction_{field.name}",
                        )
                    else:
                        value = st.text_input(
                            field.label,
                            value=str(field.default if field.default is not None else ""),
                            help=field.help or None,
                            key=f"prediction_{field.name}",
                        )

                else:
                    value = st.text_input(
                        field.label,
                        value=str(field.default if field.default is not None else ""),
                        help=field.help or None,
                        key=f"prediction_{field.name}",
                    )

                features[field.name] = value

    if st.button("🔮 Predict Churn", type="primary", use_container_width=True):
        try:
            with st.spinner("Making prediction..."):
                result = score(features)
            show_result(result)
        except Exception as exc:
            st.error(f"Prediction failed: {exc}")
            return

        try:
            with st.spinner("Searching for changes that lower the risk..."):
                scenarios = what_if(features)
            show_what_if(scenarios)
        except Exception as exc:
            st.error(f"What-if analysis failed: {exc}")
