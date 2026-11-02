"""UI theme, layout configuration, and styling helpers."""

from __future__ import annotations

import contextlib
from typing import Literal

import streamlit as st

PAGE_TITLE = "Customer Churn Prediction"
PAGE_ICON = "📊"
PAGE_LAYOUT: Literal["centered", "wide"] = "wide"


def init_page() -> None:
    """Initialize page configuration and global theme settings."""
    with contextlib.suppress(Exception):
        st.set_page_config(
            page_title=PAGE_TITLE,
            page_icon=PAGE_ICON,
            layout=PAGE_LAYOUT,
        )
