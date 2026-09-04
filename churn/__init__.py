"""Customer churn prediction package.

A single scikit-learn ``Pipeline`` (feature engineering + preprocessing + model)
is trained, cross-validated, threshold-tuned and persisted as one artifact. The
CLI (:mod:`churn.cli`) and the Streamlit dashboard (:mod:`churn.app_ui`) both load
that same artifact, so training and serving can never drift apart.

Public API lives in the submodules — e.g. ``from churn.predict import predict_one``.
Nothing is re-exported here so that ``import churn`` (and ``churn --help``) stay
cheap and do not pull in pandas / scikit-learn.
"""

__version__ = "2.0.0"
