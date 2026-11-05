"""Customer churn single prediction page."""

from __future__ import annotations

from typing import Any

import streamlit as st

from churn.schema import GROUPS, INPUT_FIELDS, Field
from churn.ui.components import (
    page_header,
    score,
    show_result,
    show_time_to_churn,
    show_what_if,
    what_if,
)


def _field_input(field: Field) -> Any:
    """One input widget for a schema field, pre-filled with its default."""
    key = f"prediction_{field.name}"
    help_text = field.help or None

    if field.kind == "number":
        min_val = float(field.min_value) if field.min_value is not None else None
        max_val = float(field.max_value) if field.max_value is not None else None
        default_val = float(field.default if field.default is not None else 0.0)
        if min_val is not None:
            default_val = max(default_val, min_val)
        if max_val is not None:
            default_val = min(default_val, max_val)
        return st.number_input(
            field.label,
            min_value=min_val,
            max_value=max_val,
            value=default_val,
            help=help_text,
            key=key,
        )

    if field.kind == "int" and not field.choices:
        return st.number_input(
            field.label,
            min_value=int(field.min_value) if field.min_value is not None else None,
            max_value=int(field.max_value) if field.max_value is not None else None,
            value=int(field.default if field.default is not None else 0),
            step=1,
            help=help_text,
            key=key,
        )

    options = list(field.choices or ())
    if not options:
        default = str(field.default if field.default is not None else "")
        return st.text_input(field.label, value=default, help=help_text, key=key)
    default_index = options.index(str(field.default)) if str(field.default) in options else 0
    selected = st.selectbox(field.label, options, index=default_index, help=help_text, key=key)
    return field.coerce(selected) if field.kind == "int" else selected


def prediction_page() -> None:
    """Customer churn prediction page."""
    page_header(
        "📊 Customer Churn Prediction",
        "Enter a customer's details to see their churn risk, what drives it, when they "
        "are likely to leave, and what would lower the risk.",
    )

    features: dict[str, Any] = {}
    with st.form("prediction_form"):
        for tab, group_name in zip(st.tabs(list(GROUPS)), GROUPS, strict=True):
            with tab:
                group_fields = [f for f in INPUT_FIELDS if f.group == group_name]
                columns = st.columns(3)
                for index, field in enumerate(group_fields):
                    with columns[index % 3]:
                        features[field.name] = _field_input(field)
        submitted = st.form_submit_button(
            "🔮 Predict Churn", type="primary", use_container_width=True
        )

    if not submitted:
        st.info("Fill in the form (every field has a sensible default) and press Predict.")
        return

    try:
        with st.spinner("Making prediction..."):
            result = score(features)
        show_result(result)
    except Exception as exc:
        st.error(f"Prediction failed: {exc}")
        return

    timing, actions = st.tabs(["⏳ Time to churn", "💡 What would reduce this risk?"])
    with timing:
        try:
            show_time_to_churn(features)
        except Exception as exc:
            st.error(f"Time-to-churn estimate failed: {exc}")
    with actions:
        try:
            with st.spinner("Searching for changes that lower the risk..."):
                scenarios = what_if(features)
            show_what_if(scenarios)
        except Exception as exc:
            st.error(f"What-if analysis failed: {exc}")
