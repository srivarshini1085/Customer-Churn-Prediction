"""Single source of truth for the 19 customer inputs.

Both the CLI (:mod:`churn.cli`) and the Streamlit dashboard (:mod:`churn.app_ui`)
build their forms from :data:`INPUT_FIELDS`, so the two interfaces always ask for
exactly the same things, with the same defaults, in the same order.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

_SERVICE = ("No", "Yes", "No internet service")
_YES_NO = ("No", "Yes")


@dataclass(frozen=True)
class Field:
    """One input: how to label it, prompt for it and coerce the raw value."""

    name: str
    label: str
    kind: Literal["number", "int", "choice"]
    default: Any
    group: Literal["Demographics", "Account", "Services"]
    choices: tuple[str, ...] | None = None
    min_value: float | None = None
    max_value: float | None = None
    help: str = ""

    def coerce(self, raw: Any) -> Any:
        """Convert a raw CLI/prompt string to the value type, or fall back."""
        if raw is None or raw == "":
            return self.default
        if self.kind == "number":
            return float(raw)
        if self.kind == "int":
            return int(raw)
        return str(raw)


def _choice(name: str, label: str, group: str, choices: tuple[str, ...], default: str) -> Field:
    return Field(name, label, "choice", default, group, choices=choices)


INPUT_FIELDS: list[Field] = [
    # -- Demographics --------------------------------------------------------- #
    _choice("gender", "Gender", "Demographics", ("Female", "Male"), "Female"),
    Field("SeniorCitizen", "Senior citizen", "int", 0, "Demographics",
          choices=("0", "1"), help="1 = yes, 0 = no"),
    _choice("Partner", "Has partner", "Demographics", _YES_NO, "No"),
    _choice("Dependents", "Has dependents", "Demographics", _YES_NO, "No"),
    # -- Account ------------------------------------------------------------ #
    Field("tenure", "Tenure (months)", "int", 12, "Account", min_value=0, max_value=100),
    _choice("Contract", "Contract", "Account",
            ("Month-to-month", "One year", "Two year"), "Month-to-month"),
    _choice("PaperlessBilling", "Paperless billing", "Account", _YES_NO, "Yes"),
    _choice("PaymentMethod", "Payment method", "Account",
            ("Electronic check", "Mailed check", "Bank transfer (automatic)",
             "Credit card (automatic)"), "Electronic check"),
    Field("MonthlyCharges", "Monthly charges", "number", 70.0, "Account", min_value=0.0),
    Field("TotalCharges", "Total charges", "number", 840.0, "Account", min_value=0.0),
    # -- Services --------------------------------------------------------- #
    _choice("PhoneService", "Phone service", "Services", _YES_NO, "Yes"),
    _choice("MultipleLines", "Multiple lines", "Services",
            ("No", "Yes", "No phone service"), "No"),
    _choice("InternetService", "Internet service", "Services",
            ("DSL", "Fiber optic", "No"), "Fiber optic"),
    _choice("OnlineSecurity", "Online security", "Services", _SERVICE, "No"),
    _choice("OnlineBackup", "Online backup", "Services", _SERVICE, "No"),
    _choice("DeviceProtection", "Device protection", "Services", _SERVICE, "No"),
    _choice("TechSupport", "Tech support", "Services", _SERVICE, "No"),
    _choice("StreamingTV", "Streaming TV", "Services", _SERVICE, "No"),
    _choice("StreamingMovies", "Streaming movies", "Services", _SERVICE, "No"),
]

GROUPS: list[str] = ["Demographics", "Account", "Services"]


def defaults() -> dict[str, Any]:
    """A complete, internally consistent customer record from every default."""
    return {f.name: f.default for f in INPUT_FIELDS}
