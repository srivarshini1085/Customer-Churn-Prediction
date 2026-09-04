"""Streamlit dashboard. Launched by ``streamlit run app.py``.

Scores either **locally** (loads the model from the registry) or against a
running **API** when ``CHURN_API_URL`` is set — both paths return the same shape.
"""

from __future__ import annotations

from typing import Any

import streamlit as st

from churn import config
from churn.schema import GROUPS, INPUT_FIELDS, Field

_BANNER = {"High": st.error, "Moderate": st.warning, "Low": st.success}


@st.cache_resource(show_spinner="Loading model...")
def _local_model():
    from churn.predict import load_model

    return load_model()


def _score(features: dict[str, Any]) -> dict[str, Any]:
    """Unified result: probability / churn / risk_tier / threshold / warnings /
    top_factors / model_version / model_meta."""
    api_url = config.settings.api_url
    if api_url:
        import httpx

        resp = httpx.post(f"{api_url.rstrip('/')}/predict", json=features, timeout=10)
        resp.raise_for_status()
        body = resp.json()
        meta = httpx.get(f"{api_url.rstrip('/')}/model", timeout=10).json()
        body["top_factors"] = [(f["feature"], f["contribution"]) for f in body["top_factors"]]
        body["model_meta"] = meta
        return body

    model = _local_model()
    result = model.predict(features)
    result["top_factors"] = model.top_factors(features)
    result["model_meta"] = {
        "model_name": model.metadata.get("model_name"),
        "created_at": model.metadata.get("created_at"),
        "metrics": {
            "roc_auc": model.metadata.get("test_metrics", {}).get("roc_auc"),
            "brier": model.metadata.get("test_metrics", {}).get("brier"),
        },
    }
    return result


def _input_widget(field: Field) -> Any:
    if field.name == "SeniorCitizen":
        choice = st.selectbox(
            field.label, ["No", "Yes"], index=int(field.default), help=field.help
        )
        return 1 if choice == "Yes" else 0
    if field.kind == "choice":
        options = list(field.choices or ())
        index = options.index(field.default) if field.default in options else 0
        return st.selectbox(field.label, options, index=index, help=field.help)
    if field.kind == "int":
        return int(
            st.number_input(
                field.label,
                min_value=int(field.min_value) if field.min_value is not None else 0,
                max_value=int(field.max_value) if field.max_value is not None else 1000,
                value=int(field.default),
                step=1,
                help=field.help,
            )
        )
    return float(
        st.number_input(
            field.label,
            min_value=float(field.min_value) if field.min_value is not None else 0.0,
            value=float(field.default),
            help=field.help,
        )
    )


def _collect_inputs() -> dict[str, Any]:
    features: dict[str, Any] = {}
    for tab, group in zip(st.tabs(GROUPS), GROUPS, strict=True):
        with tab:
            columns = st.columns(2)
            for idx, field in enumerate(f for f in INPUT_FIELDS if f.group == group):
                with columns[idx % 2]:
                    features[field.name] = _input_widget(field)
    return features


def _show_result(result: dict[str, Any]) -> None:
    tier = result["risk_tier"]

    st.subheader("Prediction")
    left, right = st.columns([1, 2])
    left.metric("Churn probability", f"{result['probability'] * 100:.1f}%")
    right.progress(min(result["probability"], 1.0))

    outcome = "predicted to churn" if result["churn"] else "predicted to stay"
    _BANNER[tier](
        f"**{tier} risk** — customer is {outcome} "
        f"_(decision threshold {result['threshold']:.2f})_"
    )
    st.caption(f"Recommended action: {config.RISK_ACTIONS[tier]}")

    for warning in result.get("warnings", []):
        st.warning(f"⚠ {warning}")

    factors = result.get("top_factors", [])
    if factors:
        st.subheader("Drivers of this prediction")
        for name, value in factors:
            verb = "raises" if value > 0 else "lowers"
            st.write(f"- **{name}** {verb} churn risk ({value:+.3f})")

    meta = result.get("model_meta", {})
    metrics = meta.get("metrics", {})
    bits = [f"version `{result.get('model_version', '?')}`", f"model `{meta.get('model_name', '?')}`"]
    if metrics.get("roc_auc"):
        bits.append(f"test ROC-AUC {metrics['roc_auc']:.3f}")
    if metrics.get("brier"):
        bits.append(f"Brier {metrics['brier']:.3f}")
    st.caption("  ·  ".join(bits))


def main() -> None:
    st.set_page_config(
        page_title="Customer Churn Prediction", page_icon="📊", layout="wide"
    )
    st.title("📊 Customer Churn Prediction")
    mode = "API" if config.settings.api_url else "local model"
    st.write(
        f"Enter the customer's details across the three tabs, then predict the "
        f"probability that they churn.  _(scoring via {mode})_"
    )

    if not config.settings.api_url:
        try:
            _local_model()
        except FileNotFoundError:
            st.error("No trained model found. Run `python -m churn train` first.")
            st.stop()

    st.divider()
    with st.form("customer"):
        features = _collect_inputs()
        submitted = st.form_submit_button("🔮 Predict churn", use_container_width=True)

    if submitted:
        st.divider()
        try:
            _show_result(_score(features))
        except Exception as exc:  # noqa: BLE001 - surface any scoring error in the UI
            st.error(f"Scoring failed: {exc}")


if __name__ == "__main__":
    main()
