"""Time-to-churn page: Kaplan-Meier curves, Cox hazard ratios and assumption checks."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from churn import config
from churn.ui.components import local_survival, page_header


def _curves_chart(km_curves: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Months as rows, one survival column per contract type (for st.line_chart)."""
    frames = [
        curve.set_index("month")["survival"].rename(contract)
        for contract, curve in km_curves.items()
    ]
    return pd.concat(frames, axis=1).sort_index().ffill()


def survival_page() -> None:
    """Show when customers churn, not just whether."""
    page_header(
        "⏳ Time to Churn",
        "Survival analysis treats tenure as the time a customer has stayed and Churn as "
        "the event. Customers who have not left yet still count: we know they lasted at "
        "least their tenure.",
    )

    survival = local_survival()
    if survival is None:
        st.info(
            "The configured model version has no survival model. Run "
            "`python -m churn train` to register one."
        )
        return
    metrics = survival.metrics

    cols = st.columns(3)
    cols[0].metric(
        "Concordance (test)",
        f"{metrics['concordance_test']:.3f}" if "concordance_test" in metrics else "n/a",
    )
    cols[1].metric("Concordance (train)", f"{metrics['concordance_train']:.3f}")
    cols[2].metric("Churn events in training data", f"{metrics['events_train']:,}")
    st.caption(
        "Concordance: for pairs of customers, how often the model ranks the one who "
        "churned sooner as higher risk. 0.5 is random, 1.0 is perfect."
    )

    st.subheader("Share of customers still active, by contract (Kaplan-Meier)")
    st.line_chart(_curves_chart(survival.km_curves), x_label="Months", y_label="Survival")
    with st.expander("Survival at selected months (with 95% confidence interval)"):
        rows = []
        for contract, curve in survival.km_curves.items():
            indexed = curve.set_index("month")
            for month in (6, 12, 24, 48, int(survival.max_tenure)):
                at = indexed[indexed.index <= month].iloc[-1]
                rows.append(
                    {
                        "Contract": contract,
                        "Month": month,
                        "Still active": f"{at['survival']:.1%}",
                        "95% CI": f"{at['lower_95']:.1%} - {at['upper_95']:.1%}",
                    }
                )
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    st.subheader("What speeds up or slows down churn (Cox model)")
    hr = survival.hazard_ratios().sort_values("hazard_ratio", ascending=False)
    violations = set(metrics["ph_violations"])
    hr["Assumption check"] = [
        "⚠️ effect changes over time" if c in violations else "✅ ok" for c in hr["covariate"]
    ]
    st.dataframe(
        hr.rename(
            columns={
                "covariate": "Covariate",
                "hazard_ratio": "Hazard ratio",
                "lower_95": "Lower 95%",
                "upper_95": "Upper 95%",
                "p": "p-value",
            }
        ).round({"Hazard ratio": 3, "Lower 95%": 3, "Upper 95%": 3, "p-value": 4}),
        use_container_width=True,
        hide_index=True,
    )
    st.caption(
        "Hazard ratio above 1: customers with this attribute tend to churn sooner (2.0 = "
        "twice the monthly risk); below 1: they stay longer. Baselines: month-to-month "
        "contract, DSL internet, electronic check. MonthlyCharges is per $10."
    )
    if violations:
        st.warning(
            f"The proportional-hazards test (p < {config.SURVIVAL_PH_ALPHA}) flags "
            f"{len(violations)} covariate(s). Their effect is not constant over tenure, "
            "so the hazard ratio shown is an average over time. The ranking of customers "
            "(concordance) is still valid; the exact ratios should be read with care."
        )

    with st.expander("Limitations"):
        st.markdown(
            f"""
- **One snapshot.** The data is a single extract: `tenure` is months since sign-up and
  `Churn` is whether the customer left recently. We never observe a customer over
  time, so durations are reconstructed rather than tracked.
- **Tenure stops at {survival.max_tenure:.0f} months.** Nothing is known beyond that, so
  estimates past the data are shown as "more than N months", not extrapolated.
- **Current values only.** Contract, services and charges are today's values, not
  the ones the customer had throughout their tenure.
- **Risk is U-shaped in tenure.** Churn is highest in the first year, lower in years
  2-3, and rises again past ~60 months. The Kaplan-Meier curves show the same, so it
  is in the data; it may reflect how the snapshot was built, so treat long-tenure
  estimates with extra caution.
- **`tenure` and `TotalCharges` are not covariates**, because they encode the
  duration itself and would leak the answer.
"""
        )
