"""UI theme, layout configuration, and styling helpers.

Colours live in ``.streamlit/config.toml``; this module adds the page config and
a little CSS so metrics render as cards and wide pages stay readable on a laptop.
"""

from __future__ import annotations

import contextlib
from typing import Literal

import streamlit as st

PAGE_TITLE = "Customer Churn Prediction"
PAGE_ICON = "📊"
PAGE_LAYOUT: Literal["centered", "wide"] = "wide"

#: Colour per risk tier, used for banners and charts.
RISK_COLORS: dict[str, str] = {"High": "#dc2626", "Moderate": "#d97706", "Low": "#16a34a"}

_CSS = """
<style>
/* Keep wide pages readable on a 13-15" laptop. */
.block-container {max-width: 1280px; padding-top: 2rem; padding-bottom: 3rem;}
/* Metrics as KPI cards. */
div[data-testid="stMetric"] {
    background: var(--secondary-background-color, #f3f6fb);
    border: 1px solid rgba(37, 99, 235, 0.12);
    border-radius: 12px;
    padding: 0.75rem 1rem;
}
div[data-testid="stMetricLabel"] p {font-size: 0.85rem; opacity: 0.8;}
</style>
"""


def init_page() -> None:
    """Initialize page configuration and global theme settings."""
    with contextlib.suppress(Exception):
        st.set_page_config(
            page_title=PAGE_TITLE,
            page_icon=PAGE_ICON,
            layout=PAGE_LAYOUT,
        )
    st.markdown(_CSS, unsafe_allow_html=True)
