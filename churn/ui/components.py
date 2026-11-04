"""Shared UI components, model loaders, scoring dispatch, and result displays."""

from __future__ import annotations

from typing import Any, cast

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
    """Display structured prediction results including metrics, risk tier, warnings, and factors."""
    probability = float(result.get("probability", 0.0))
    churn = bool(result.get("churn", False))
    risk_tier = str(result.get("risk_tier", "Low"))

    st.subheader("Prediction Result")
    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Churn Probability", f"{probability * 100:.1f}%")
    with col2:
        st.metric("Prediction", "Likely to Churn" if churn else "Likely to Stay")
    with col3:
        st.metric("Risk Tier", risk_tier)

    # Risk message banner
    if risk_tier == "High":
        st.error("⚠️ High churn risk")
    elif risk_tier == "Moderate":
        st.warning("⚠️ Moderate churn risk")
    else:
        st.success("✅ Low churn risk")

    # Warnings section
    warnings = result.get("warnings") or []
    if warnings:
        st.subheader("Warnings")
        for warning in warnings:
            st.warning(str(warning))

    # Top churn factors / explanations
    top_factors = result.get("top_factors") or []
    if top_factors:
        st.subheader("Top Churn Factors")
        if isinstance(top_factors, list):
            for factor in top_factors:
                if isinstance(factor, dict):
                    name = factor.get("feature", factor.get("name", "Feature"))
                    value = factor.get("value", "")
                    impact = factor.get("impact", "")
                    contribution = factor.get("contribution")
                    text = f"**{name}**"
                    if value != "":
                        text += f": {value}"
                    if impact != "":
                        text += f" — {impact}"
                    if contribution is not None:
                        direction = "raises" if contribution > 0 else "lowers"
                        text += f" — {direction} risk ({contribution:+.3f})"
                    st.write(text)
                else:
                    st.write(f"• {factor}")

    # Model metadata section
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

            saved_metrics = model_meta.get("test_metrics", {})
            if saved_metrics:
                st.write("Saved test metrics:")
                metric_cols = st.columns(5)
                metric_keys = ["accuracy", "precision", "recall", "f1", "roc_auc"]
                labels = ["Accuracy", "Precision", "Recall", "F1", "ROC-AUC"]

                for col, key, label in zip(metric_cols, metric_keys, labels, strict=False):
                    with col:
                        val = saved_metrics.get(key)
                        st.metric(label, f"{val:.3f}" if val is not None else "N/A")

                pr_saved = saved_metrics.get("pr_auc")
                if pr_saved is not None:
                    st.metric("PR-AUC", f"{pr_saved:.3f}")


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
