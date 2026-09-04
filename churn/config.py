"""Runtime settings and the fixed dataset schema.

``settings`` (a ``pydantic-settings`` object) holds everything that may vary
between environments — paths, CV parameters, calibration mode, logging. Override
any field with an environment variable prefixed ``CHURN_`` (e.g.
``CHURN_MODEL_VERSION=20260905T101500Z``) or a ``.env`` file in the project root.

The dataset *schema* (which columns exist and what they mean) is not
configurable and stays as plain module constants.
"""

from __future__ import annotations

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
        "Healthy account: routine communication, cross-sell / up-sell and regular "
        "loyalty rewards."
    ),
}
