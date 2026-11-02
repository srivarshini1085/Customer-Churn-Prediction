"""Backward compatibility entry point for Streamlit UI.

All Streamlit UI components have been modularized under :mod:`churn.ui`.
This module re-exports :func:`main` and other functions for legacy callers.
"""

from __future__ import annotations

from churn.ui import (
    batch_prediction_page,
    churn_analysis_page,
    main,
    performance_page,
    prediction_page,
)
from churn.ui.components import (
    fields_for_group,
    local_model,
    score,
    show_result,
)

__all__ = [
    "main",
    "prediction_page",
    "churn_analysis_page",
    "performance_page",
    "batch_prediction_page",
    "local_model",
    "score",
    "show_result",
    "fields_for_group",
]

if __name__ == "__main__":
    main()
