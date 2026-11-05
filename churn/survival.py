"""Time to churn: Kaplan-Meier curves and a Cox proportional-hazards model.

The churn classifier answers *whether* a customer is likely to leave; survival
analysis answers *when*. We treat ``tenure`` (months as a customer) as the
duration and ``Churn`` as the event. Customers who have not churned are
*right-censored*: we know they lasted at least ``tenure`` months, not when they
will leave. Survival methods use those rows correctly instead of dropping them.

* **Kaplan-Meier** (non-parametric): the share of customers still active after
  *t* months, estimated separately per contract type.
* **Cox proportional hazards** (semi-parametric): each covariate multiplies the
  monthly risk of churning by a constant factor (its *hazard ratio*). Quality is
  measured with the **concordance index** on the held-out test split: the share
  of comparable customer pairs where the model ranks the one who churned sooner
  as higher risk (0.5 = random, 1.0 = perfect).
* The **proportional-hazards assumption** (those factors stay constant over
  time) is tested per covariate with Schoenfeld residuals.

Per-customer estimates are *conditional on the current tenure*: a customer who
has already stayed ``t`` months has survival ``S(t + u) / S(t)`` for the next
``u`` months. The "likely months until churn" is the median of that conditional
distribution, i.e. the ``u`` where it drops to 0.5.

Limitations (worth stating in any write-up):

* The data is a single snapshot. ``tenure`` is time since sign-up and ``Churn``
  is whether they left in the last month, so we only see one exit window rather
  than a full history; durations are reconstructed, not observed live.
* The observed tenure stops at 72 months. Beyond that the baseline hazard is
  unknown, so estimates past the data are reported as "more than N months"
  instead of being extrapolated.
* Covariates are today's values (e.g. the current contract), not the values the
  customer had over their whole lifetime.
* The churn rate is U-shaped in tenure: highest in the first year, lower in years
  2-3, then rising again for customers past ~60 months (Kaplan-Meier shows this
  too, so it is in the data, not a model artefact). It may reflect how the
  snapshot was assembled, so long-tenure estimates deserve extra caution.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from lifelines import CoxPHFitter, KaplanMeierFitter
from lifelines.statistics import proportional_hazard_test

from churn import config
from churn.schema import defaults as schema_defaults

logger = logging.getLogger(__name__)

DURATION, EVENT = "tenure", "churned"


def covariate_names() -> list[str]:
    """Columns of the Cox design matrix, in a fixed order."""
    names = [
        f"{feature}={level}"
        for feature, levels in config.SURVIVAL_CATEGORICAL.items()
        for level in levels[1:]  # first level is the baseline
    ]
    names += [f"{flag}=Yes" for flag in config.SURVIVAL_FLAGS]
    return [*names, "SeniorCitizen", "MonthlyCharges_per_10"]


def design_matrix(x: pd.DataFrame) -> pd.DataFrame:
    """Encode customer attributes as Cox covariates (missing columns use defaults)."""
    fallback = schema_defaults()
    get = {
        col: (x[col] if col in x.columns else pd.Series(fallback[col], index=x.index))
        for col in (
            *config.SURVIVAL_CATEGORICAL,
            *config.SURVIVAL_FLAGS,
            "SeniorCitizen",
            "MonthlyCharges",
        )
    }
    out = pd.DataFrame(index=x.index)
    for feature, levels in config.SURVIVAL_CATEGORICAL.items():
        values = get[feature].astype(str)
        for level in levels[1:]:
            out[f"{feature}={level}"] = (values == level).astype(float)
    for flag in config.SURVIVAL_FLAGS:
        out[f"{flag}=Yes"] = (get[flag].astype(str) == "Yes").astype(float)
    out["SeniorCitizen"] = pd.to_numeric(get["SeniorCitizen"], errors="coerce").fillna(0)
    charges = pd.to_numeric(get["MonthlyCharges"], errors="coerce")
    out["MonthlyCharges_per_10"] = charges.fillna(float(fallback["MonthlyCharges"])) / 10
    return out[covariate_names()]


def kaplan_meier_by_contract(x: pd.DataFrame, y: pd.Series) -> dict[str, pd.DataFrame]:
    """Survival table (month, survival, 95% CI) per contract type."""
    curves: dict[str, pd.DataFrame] = {}
    for contract in config.SURVIVAL_CATEGORICAL["Contract"]:
        mask = x["Contract"].astype(str) == contract
        if not mask.any():
            continue
        km = KaplanMeierFitter().fit(x.loc[mask, DURATION], y[mask], label=contract)
        table = pd.concat([km.survival_function_, km.confidence_interval_], axis=1)
        table.columns = ["survival", "lower_95", "upper_95"]
        table.index.name = "month"
        curves[contract] = table.reset_index()
    return curves


@dataclass
class SurvivalModel:
    """A fitted Cox model plus Kaplan-Meier curves and evaluation results."""

    cox: CoxPHFitter
    km_curves: dict[str, pd.DataFrame]
    max_tenure: float
    metrics: dict[str, Any] = field(default_factory=dict)

    # -- persistence --------------------------------------------------- #

    def save(self, path: str | Path) -> Path:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, target)
        return target

    @classmethod
    def load(cls, path: str | Path) -> SurvivalModel:
        model = joblib.load(Path(path))
        if not isinstance(model, cls):
            raise TypeError(f"{path} does not contain a SurvivalModel")
        return model

    # -- interpretation ------------------------------------------------ #

    def hazard_ratios(self) -> pd.DataFrame:
        """Hazard ratio, 95% CI and p-value per covariate (HR > 1 = churns sooner)."""
        s = self.cox.summary
        table = pd.DataFrame(
            {
                "hazard_ratio": s["exp(coef)"],
                "lower_95": s["exp(coef) lower 95%"],
                "upper_95": s["exp(coef) upper 95%"],
                "p": s["p"],
            }
        )
        table.index.name = "covariate"
        return table.reset_index()

    # -- per-customer estimates ---------------------------------------- #

    def predict(
        self, customers: list[dict[str, Any]], horizon: int | None = None
    ) -> list[dict[str, Any]]:
        """Months-until-churn estimate and churn-within-horizon probability per customer.

        ``months_until_churn`` is ``None`` when the median falls beyond the observed
        data; ``beyond_months`` then says how far the data reaches for that customer.
        """
        horizon = horizon or config.SURVIVAL_HORIZON_MONTHS
        rows = [
            {**schema_defaults(), **{k: v for k, v in c.items() if v is not None}}
            for c in customers
        ]
        frame = pd.DataFrame(rows)
        tenure = pd.to_numeric(frame[DURATION], errors="coerce").fillna(0).to_numpy(float)
        covariates = design_matrix(frame)

        times = np.arange(0, self.max_tenure + 1)
        # Rows = times, columns = customers. S(t) for each customer's own covariates.
        surv = self.cox.predict_survival_function(covariates, times=times).to_numpy()

        results = []
        for k, t in enumerate(tenure):
            start = int(min(max(t, 0), self.max_tenure))
            s_now = surv[start, k]
            remaining = surv[start:, k] / s_now if s_now > 0 else np.zeros(1)
            below = np.flatnonzero(remaining <= 0.5)
            reach = len(remaining) - 1  # months of data beyond the current tenure
            end = min(horizon, reach)
            results.append(
                {
                    "months_until_churn": int(below[0]) if below.size else None,
                    "beyond_months": None if below.size else reach,
                    "churn_within_horizon": float(1 - remaining[end]),
                    "horizon_months": int(end),
                }
            )
        return results


def fit_survival(
    x_train: pd.DataFrame,
    y_train: pd.Series,
    x_test: pd.DataFrame | None = None,
    y_test: pd.Series | None = None,
) -> SurvivalModel:
    """Fit Kaplan-Meier curves and a Cox model; evaluate on the test split if given."""
    train = design_matrix(x_train)
    train[DURATION] = pd.to_numeric(x_train[DURATION]).astype(float)
    train[EVENT] = np.asarray(y_train, dtype=int)

    cox = CoxPHFitter(penalizer=config.SURVIVAL_PENALIZER)
    cox.fit(train, duration_col=DURATION, event_col=EVENT)

    ph = proportional_hazard_test(cox, train, time_transform="rank").summary["p"]
    violations = sorted(name for name, p in ph.items() if p < config.SURVIVAL_PH_ALPHA)

    metrics: dict[str, Any] = {
        "n_train": len(train),
        "events_train": int(train[EVENT].sum()),
        "concordance_train": float(cox.concordance_index_),
        "ph_test_p": {str(name): float(p) for name, p in ph.items()},
        "ph_violations": violations,
        "penalizer": config.SURVIVAL_PENALIZER,
        "excluded_covariates": list(config.SURVIVAL_EXCLUDED),
    }
    if x_test is not None and y_test is not None:
        test = design_matrix(x_test)
        test[DURATION] = pd.to_numeric(x_test[DURATION]).astype(float)
        test[EVENT] = np.asarray(y_test, dtype=int)
        metrics["concordance_test"] = float(cox.score(test, scoring_method="concordance_index"))
        metrics["n_test"] = len(test)

    logger.info(
        "Cox model: C-index train %.3f, test %s; PH violations: %s",
        metrics["concordance_train"],
        f"{metrics['concordance_test']:.3f}" if "concordance_test" in metrics else "n/a",
        violations or "none",
    )
    return SurvivalModel(
        cox=cox,
        km_curves=kaplan_meier_by_contract(x_train, y_train),
        max_tenure=float(x_train[DURATION].max()),
        metrics=metrics,
    )


def model_card_section(model: SurvivalModel) -> str:
    """Markdown summary of the survival model for the model card."""
    m = model.metrics
    hr = model.hazard_ratios().sort_values("hazard_ratio", ascending=False)
    test = f"{m['concordance_test']:.3f}" if "concordance_test" in m else "n/a"
    violations = ", ".join(f"`{v}`" for v in m["ph_violations"]) or "none"
    lines = [
        "## Time to churn (survival analysis)",
        "",
        "Cox proportional-hazards model with `tenure` as the duration and `Churn` as the "
        f"event; `{'`, `'.join(m['excluded_covariates'])}` are excluded because they "
        "encode the duration itself.",
        "",
        f"- Concordance index: **{m['concordance_train']:.3f}** (train), **{test}** (test)",
        f"- Proportional-hazards violations (p < {config.SURVIVAL_PH_ALPHA}): {violations}",
        "",
        "| covariate | hazard ratio | 95% CI | p |",
        "| --- | --- | --- | --- |",
        *(
            f"| {r.covariate} | {r.hazard_ratio:.2f} | {r.lower_95:.2f}-{r.upper_95:.2f} "
            f"| {r.p:.3g} |"
            for r in hr.itertuples()
        ),
        "",
        "Hazard ratio > 1 means customers with that attribute tend to churn sooner. A "
        "violated assumption means the effect changes with tenure, so its single hazard "
        "ratio is an average over time rather than a constant.",
        "",
        "Limitations: one snapshot of data (tenure reconstructed from sign-up, one exit "
        "window), tenure observed only up to "
        f"{model.max_tenure:.0f} months, covariates are current values, and churn "
        "rises again for customers past ~60 months, so long-tenure estimates deserve "
        "extra caution.",
    ]
    return "\n".join(lines) + "\n"
