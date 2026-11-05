"""Runtime settings and the fixed dataset schema.

``settings`` (a ``pydantic-settings`` object) holds everything that may vary
between environments — paths, CV parameters, calibration mode, logging. Override
any field with an environment variable prefixed ``CHURN_`` (e.g.
``CHURN_MODEL_VERSION=20260905T101500Z``) or a ``.env`` file in the project root.

The dataset *schema* (which columns exist and what they mean) is not
configurable and stays as plain module constants.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    """Environment-overridable configuration (prefix ``CHURN_``)."""

    model_config = SettingsConfigDict(
        env_prefix="CHURN_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        protected_namespaces=(),
    )

    # -- paths ----------------------------------------------------------- #
    data_path: Path = PROJECT_ROOT / "data" / "Telco-Customer-Churn.csv"
    models_dir: Path = PROJECT_ROOT / "models"
    prediction_log_path: Path | None = None

    # -- model selection -------------------------------------------------- #
    model_version: str = "latest"
    random_state: int = 42
    test_size: float = Field(default=0.2, gt=0, lt=1)
    cv_folds: int = Field(default=5, ge=2)
    selection_metric: str = "roc_auc"
    calibration: Literal["auto", "sigmoid", "isotonic", "none"] = "auto"

    # -- service -------------------------------------------------------- #
    api_url: str | None = None
    api_batch_limit: int = Field(default=1000, ge=1)
    log_json: bool = False
    #: Train and register a model when the dashboard starts without a loadable one
    #: (set CHURN_AUTO_TRAIN=true on Streamlit Cloud / Hugging Face Spaces).
    auto_train: bool = False

    # -- retention planning (business assumption, see RETENTION_OFFERS) ----- #
    #: Months of revenue a retained customer is assumed to bring in.
    retention_horizon_months: int = Field(default=12, ge=1)

    @property
    def registry_dir(self) -> Path:
        return self.models_dir / "registry"


settings = Settings()

# --------------------------------------------------------------------------- #
# Convenience aliases (backwards compatible with the pre-settings constants)
# --------------------------------------------------------------------------- #

DATA_PATH: Path = settings.data_path
MODELS_DIR: Path = settings.models_dir
REGISTRY_DIR: Path = settings.registry_dir

RANDOM_STATE: int = settings.random_state
TEST_SIZE: float = settings.test_size
CV_FOLDS: int = settings.cv_folds
SELECTION_METRIC: str = settings.selection_metric

# --------------------------------------------------------------------------- #
# Dataset schema (not configurable)
# --------------------------------------------------------------------------- #

ID_COLUMN: str = "customerID"
TARGET_COLUMN: str = "Churn"

NUMERIC_COLUMNS: list[str] = ["tenure", "MonthlyCharges", "TotalCharges"]
PASSTHROUGH_COLUMNS: list[str] = ["SeniorCitizen"]
CATEGORICAL_COLUMNS: list[str] = [
    "gender",
    "Partner",
    "Dependents",
    "PhoneService",
    "MultipleLines",
    "InternetService",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
    "Contract",
    "PaperlessBilling",
    "PaymentMethod",
]

#: All 19 feature columns the model consumes (order is not significant — the
#: pipeline selects columns by name).
FEATURE_COLUMNS: list[str] = NUMERIC_COLUMNS + PASSTHROUGH_COLUMNS + CATEGORICAL_COLUMNS

#: Add-on services counted by the ``num_addons`` engineered feature.
ADDON_SERVICE_COLUMNS: list[str] = [
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
]

# Engineered feature names (produced by churn.features.add_engineered)
ENGINEERED_NUMERIC_COLUMNS: list[str] = ["num_addons"]
ENGINEERED_CATEGORICAL_COLUMNS: list[str] = ["tenure_group"]
ENGINEERED_PASSTHROUGH_COLUMNS: list[str] = ["has_internet"]

TENURE_BINS: list[float] = [-0.1, 12, 24, 48, 1e9]
TENURE_LABELS: list[str] = ["0-12m", "12-24m", "24-48m", "48m+"]

# --------------------------------------------------------------------------- #
# What-if (counterfactual) search — see churn.counterfactual
# --------------------------------------------------------------------------- #

#: Attributes a retention team cannot (or must not) change. The search never
#: touches these, even if changing them would lower the predicted risk.
IMMUTABLE_FEATURES: list[str] = [
    "gender",
    "SeniorCitizen",
    "Partner",
    "Dependents",
    "tenure",
    "TotalCharges",
]

#: Changes a retention offer could realistically make, as feature -> values the
#: customer could be moved to. Further rules (e.g. contracts only get longer,
#: add-ons need internet) are applied in churn.counterfactual.
ACTIONABLE_CHANGES: dict[str, tuple[str, ...]] = {
    "Contract": ("One year", "Two year"),
    "PaymentMethod": ("Bank transfer (automatic)", "Credit card (automatic)"),
    "OnlineSecurity": ("Yes",),
    "TechSupport": ("Yes",),
    "PaperlessBilling": ("Yes", "No"),
}

#: Contract lengths in order; a what-if may only move a customer to a longer one.
CONTRACT_ORDER: tuple[str, ...] = ("Month-to-month", "One year", "Two year")

# --------------------------------------------------------------------------- #
# Business risk bands (fixed) — independent of the tuned decision threshold
# --------------------------------------------------------------------------- #

RISK_BANDS: list[tuple[float, str]] = [
    (0.70, "High"),
    (0.40, "Moderate"),
    (0.0, "Low"),
]

RISK_ACTIONS: dict[str, str] = {
    "High": (
        "Immediate intervention: customer-success outreach, contract discounts, "
        "loyalty incentives or complimentary service upgrades."
    ),
    "Moderate": (
        "Targeted engagement: satisfaction check, follow-up survey, automated-"
        "payment incentive or a 1-year contract discount."
    ),
    "Low": (
        "Healthy account: routine communication, cross-sell / up-sell and regular loyalty rewards."
    ),
}

# --------------------------------------------------------------------------- #
# Retention offers — BUSINESS ASSUMPTIONS, not learned from data
# --------------------------------------------------------------------------- #
# These numbers are illustrative placeholders for a portfolio project. In a real
# deployment they would come from the finance team (costs) and from A/B tests of
# past campaigns (success rates). Every optimiser result scales with them.


@dataclass(frozen=True)
class RetentionOffer:
    """A retention action and its assumed cost and effect.

    Cost to the business for one customer:
    ``fixed_cost + discount_rate * MonthlyCharges * discount_months``.
    ``success_rate`` is the assumed probability that the offer keeps a customer
    who would otherwise have churned.
    """

    name: str
    label: str
    fixed_cost: float
    success_rate: float
    discount_rate: float = 0.0
    discount_months: int = 0


RETENTION_OFFERS: tuple[RetentionOffer, ...] = (
    RetentionOffer(
        name="retention_call",
        label="Customer-success call",
        fixed_cost=10.0,
        success_rate=0.10,
    ),
    RetentionOffer(
        name="discount_20pct_3m",
        label="20% discount for 3 months",
        fixed_cost=5.0,
        success_rate=0.25,
        discount_rate=0.20,
        discount_months=3,
    ),
    RetentionOffer(
        name="contract_bundle",
        label="1-year contract with 12 months free Tech Support",
        fixed_cost=60.0,
        success_rate=0.40,
    ),
)

#: The single offer the naive "top-N by risk" baseline sends to everyone.
BASELINE_OFFER: str = "discount_20pct_3m"

# --------------------------------------------------------------------------- #
# Survival analysis (time to churn) — see churn.survival
# --------------------------------------------------------------------------- #

#: Categorical covariates of the Cox model, one-hot encoded against the first
#: (baseline) level listed here.
SURVIVAL_CATEGORICAL: dict[str, tuple[str, ...]] = {
    "Contract": ("Month-to-month", "One year", "Two year"),
    "InternetService": ("DSL", "Fiber optic", "No"),
    "PaymentMethod": (
        "Electronic check",
        "Mailed check",
        "Bank transfer (automatic)",
        "Credit card (automatic)",
    ),
}
#: Yes/No covariates, encoded as 1 when the value is "Yes".
SURVIVAL_FLAGS: tuple[str, ...] = (
    "OnlineSecurity",
    "TechSupport",
    "PaperlessBilling",
    "Partner",
    "Dependents",
)
#: tenure is the duration and TotalCharges ~ tenure x MonthlyCharges, so both are
#: deliberately left out: using them would leak the answer into the covariates.
SURVIVAL_EXCLUDED: tuple[str, ...] = ("tenure", "TotalCharges")
#: Ridge penalty for the Cox fit (small; keeps estimates stable for rare levels).
SURVIVAL_PENALIZER: float = 0.01
#: Horizon for the "probability of churning within N months" estimate.
SURVIVAL_HORIZON_MONTHS: int = 12
#: Significance level used to flag proportional-hazards violations.
SURVIVAL_PH_ALPHA: float = 0.05

# --------------------------------------------------------------------------- #
# Responsible ML: fairness report and drift monitoring
# --------------------------------------------------------------------------- #

#: Attributes the fairness report breaks metrics down by.
FAIRNESS_ATTRIBUTES: tuple[str, ...] = ("gender", "SeniorCitizen", "Partner")
#: A between-group gap in recall / precision / FPR larger than this is flagged.
FAIRNESS_GAP_FLAG: float = 0.10

#: Retrain is recommended when any feature is red, or at least this many are amber.
DRIFT_AMBER_RETRAIN_COUNT: int = 3
#: Below this many rows PSI is too noisy to act on; the monitoring page warns.
DRIFT_MIN_ROWS: int = 100
