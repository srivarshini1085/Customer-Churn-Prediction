"""Shared UI components, model loaders, scoring dispatch, and result displays."""

from __future__ import annotations

from typing import Any, cast

import numpy as np
import pandas as pd
import streamlit as st

from churn import config
from churn.schema import INPUT_FIELDS


def fields_for_group(group_name: str) -> list[Any]:
    """Return all input fields belonging to a specific schema group."""
    return [field for field in INPUT_FIELDS if field.group == group_name]


@st.cache_resource(show_spinner="Loading model...")
def local_model() -> Any:
    """Load and cache the local churn prediction model from the registry."""
    from churn.predict import load_model

    return load_model()


def score(features: dict[str, Any]) -> dict[str, Any]:
    """Score a customer payload using the configured REST API or local model fallback."""
    api_url = config.settings.api_url

    # Attempt scoring via the REST API if configured
    if api_url:
        import httpx

        try:
            response = httpx.post(
                f"{api_url.rstrip('/')}/predict",
                json=features,
                timeout=10,
            )
            response.raise_for_status()
            result: dict[str, Any] = response.json()

            try:
                model_response = httpx.get(
                    f"{api_url.rstrip('/')}/model",
                    timeout=10,
                )
                if model_response.is_success:
                    model_info = model_response.json()
                    result.setdefault("model_meta", model_info)
            except Exception:
                pass

            return result
        except Exception as exc:
            st.warning(f"API scoring failed. Using local model instead. Error: {exc}")

    # Fallback to local model scoring
    model = local_model()

    if hasattr(model, "predict_with_details"):
        return cast(dict[str, Any], model.predict_with_details(features))

    probability = float(model.predict_proba(features))
    threshold = getattr(model, "threshold", 0.5)
    churn = bool(probability >= threshold)

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
        "top_factors": [
            {"feature": name, "contribution": value} for name, value in model.top_factors(features)
        ],
        "model_version": getattr(model, "version", "local"),
        "model_meta": getattr(model, "metadata", {}),
    }


def show_result(result: dict[str, Any]) -> None:
    """Prediction summary: KPI cards, risk banner with next action, drivers, model info."""
    probability = float(result.get("probability", 0.0))
    churn = bool(result.get("churn", False))
    risk_tier = str(result.get("risk_tier", "Low"))
    threshold = result.get("threshold")

    st.subheader("Prediction Result")
    kpi_row(
        [
            ("Churn Probability", f"{probability * 100:.1f}%", "Calibrated probability"),
            (
                "Prediction",
                "Likely to Churn" if churn else "Likely to Stay",
                f"Flagged when probability >= the tuned threshold ({fmt(threshold, 2)})",
            ),
            ("Risk Tier", risk_tier, "High >= 70%, Moderate 40-69%, Low < 40%"),
        ]
    )

    action = config.RISK_ACTIONS.get(risk_tier, "")
    banner = {"High": st.error, "Moderate": st.warning}.get(risk_tier, st.success)
    icon = "✅" if risk_tier == "Low" else "⚠️"
    banner(f"{icon} **{risk_tier} churn risk.** {action}")

    for warning in result.get("warnings") or []:
        st.warning(f"Input check: {warning}")

    factors = [f for f in (result.get("top_factors") or []) if isinstance(f, dict)]
    if factors:
        st.subheader("Top Churn Factors")
        table = pd.DataFrame(
            {
                "Factor": [f.get("feature", f.get("name", "Feature")) for f in factors],
                "Contribution": [float(f.get("contribution", 0.0)) for f in factors],
            }
        )
        st.altair_chart(_factor_chart(table), use_container_width=True)
        st.caption(
            "Signed contribution to this customer's score (log-odds): positive pushes "
            "churn risk up, negative pulls it down."
        )

    model_meta = result.get("model_meta") or {}
    if model_meta:
        with st.expander("Model Information"):
            version = (
                result.get("model_version")
                or model_meta.get("model_version")
                or model_meta.get("version")
                or "N/A"
            )
            st.write("Model version:", version)
            test = model_meta.get("test_metrics") or model_meta.get("metrics") or {}
            tuned = test.get("tuned", test)
            kpi_row(
                [
                    ("ROC-AUC", fmt(test.get("roc_auc")), None),
                    ("Recall", fmt(tuned.get("recall")), None),
                    ("Precision", fmt(tuned.get("precision")), None),
                    ("Brier", fmt(test.get("brier")), None),
                ]
            )


def _factor_chart(table: pd.DataFrame) -> Any:
    """Horizontal bars in the given (magnitude) order: red raises risk, green lowers it."""
    import altair as alt

    table = table.assign(Effect=np.where(table["Contribution"] > 0, "raises risk", "lowers risk"))
    return (
        alt.Chart(table)
        .mark_bar()
        .encode(
            x=alt.X("Contribution:Q", title="Contribution to churn risk (log-odds)"),
            y=alt.Y(
                "Factor:N", sort=None, title=None, axis=alt.Axis(labelLimit=260, labelOverlap=False)
            ),
            color=alt.Color(
                "Effect:N",
                scale=alt.Scale(
                    domain=["raises risk", "lowers risk"], range=["#dc2626", "#16a34a"]
                ),
                legend=alt.Legend(orient="bottom", title=None),
            ),
            tooltip=["Factor", alt.Tooltip("Contribution:Q", format="+.3f")],
        )
        .properties(height=44 * len(table) + 50)
    )


def what_if(features: dict[str, Any], max_changes: int = 2, top_k: int = 5) -> dict[str, Any]:
    """Risk-reducing scenarios from the REST API, or the local model as a fallback.

    Returns the same JSON shape as ``POST /explain/whatif``.
    """
    api_url = config.settings.api_url
    if api_url:
        import httpx

        try:
            response = httpx.post(
                f"{api_url.rstrip('/')}/explain/whatif",
                json={"customer": features, "max_changes": max_changes, "top_k": top_k},
                timeout=10,
            )
            response.raise_for_status()
            return cast(dict[str, Any], response.json())
        except Exception as exc:
            st.warning(f"API what-if failed. Using local model instead. Error: {exc}")

    from dataclasses import asdict

    from churn.counterfactual import what_if as local_what_if

    result = local_what_if(local_model(), features, max_changes=max_changes, top_k=top_k)
    return asdict(result)


def show_what_if(result: dict[str, Any]) -> None:
    """Render the "What would reduce this risk?" panel."""
    st.subheader("What would reduce this risk?")
    scenarios = result.get("scenarios") or []
    if not scenarios:
        st.info(
            "None of the realistic changes (longer contract, automatic payment, "
            "adding Online Security / Tech Support, paperless billing) lowers this "
            "customer's churn probability by at least 0.5 percentage points."
        )
        return

    baseline = float(result.get("baseline_probability", 0.0))
    rows = [
        {
            "Changes": "; ".join(
                f"{c['feature']}: {c['from_value']} → {c['to_value']}" for c in scenario["changes"]
            ),
            "New churn probability": f"{scenario['probability'] * 100:.1f}%",
            "Change": f"{scenario['delta'] * 100:+.1f} pts",
            "New risk tier": scenario["risk_tier"],
        }
        for scenario in scenarios
    ]
    st.write(f"Current churn probability: **{baseline * 100:.1f}%**")
    st.dataframe(rows, use_container_width=True, hide_index=True)
    st.caption(
        "Fixed attributes (gender, senior citizen, partner, dependents, tenure, total "
        "charges) are never changed. The model learned associations, not causal effects, "
        "and monthly charges are held constant, so treat these as hypotheses to test."
    )


@st.cache_resource(show_spinner="Loading survival model...")
def local_survival() -> Any:
    """The time-to-churn model saved with the configured model version (or None)."""
    from churn.registry import ModelRegistry

    try:
        return ModelRegistry().load_survival(config.settings.model_version)
    except FileNotFoundError:
        return None


def show_time_to_churn(features: dict[str, Any]) -> None:
    """Render the "likely months until churn" estimate for one customer."""
    st.subheader("Time to churn")
    survival = local_survival()
    if survival is None:
        st.info(
            "This model version has no survival model. Run `python -m churn train` to "
            "register a version that includes one."
        )
        return

    estimate = survival.predict([features])[0]
    col1, col2 = st.columns(2)
    with col1:
        months = estimate["months_until_churn"]
        if months is not None:
            st.metric("Likely months until churn", f"{months} months")
        elif estimate["beyond_months"]:
            st.metric("Likely months until churn", f"> {estimate['beyond_months']} months")
        else:
            st.metric("Likely months until churn", "Beyond observed data")
    with col2:
        horizon = estimate["horizon_months"]
        if horizon > 0:
            st.metric(
                f"Chance of churning in the next {horizon} months",
                f"{estimate['churn_within_horizon'] * 100:.1f}%",
            )
        else:
            st.metric("Chance of churning soon", "Beyond observed data")
    st.caption(
        "From a Cox proportional-hazards model, given that the customer has already "
        "stayed for their current tenure. 'Likely months' is the median: half of "
        "similar customers would have left by then. Estimates past the longest "
        "observed tenure are not extrapolated."
    )


# --------------------------------------------------------------------------- #
# Layout helpers shared by every page
# --------------------------------------------------------------------------- #


def page_header(title: str, subtitle: str) -> None:
    """Title plus a one-line explanation of what the page is for."""
    st.title(title)
    st.caption(subtitle)


def kpi_row(items: list[tuple[str, str, str | None]]) -> None:
    """A row of KPI cards: (label, value, help text or None)."""
    for column, (label, value, help_text) in zip(st.columns(len(items)), items, strict=True):
        column.metric(label, value, help=help_text)


def fmt(value: Any, digits: int = 3) -> str:
    """Format a saved metric, or "n/a" when it is missing."""
    return f"{value:.{digits}f}" if isinstance(value, int | float) else "n/a"


def model_or_none() -> Any:
    """The local model, or None after showing a helpful message when there is none."""
    try:
        return local_model()
    except FileNotFoundError:
        st.warning(
            "No trained model was found. Train one with `python -m churn train` "
            "(about a minute), then reload this page."
        )
    except Exception as exc:
        st.error(f"Could not load the model: {exc}")
    return None


def ensure_model() -> str | None:
    """Version of the loadable model, training one first if auto-train is on.

    For hosted demos (Streamlit Cloud, Hugging Face Spaces) the model binaries are
    not in git, so with ``CHURN_AUTO_TRAIN=true`` the first visitor waits for one
    training run and every later session reuses it. Returns None when there is no
    model and auto-train is off.
    """
    from churn.predict import load_model

    try:
        return load_model().version
    except FileNotFoundError:
        pass
    if not config.settings.auto_train:
        return None
    return _train_once()


@st.cache_resource(show_spinner=False)
def _train_once() -> str:
    """Train and register a model; cached so it runs once per server process."""
    from churn.predict import clear_cache, load_model
    from churn.train import train

    with st.spinner("First start: training the model. This takes a few minutes..."):
        train()
    clear_cache()
    local_model.clear()
    local_survival.clear()
    return load_model().version
