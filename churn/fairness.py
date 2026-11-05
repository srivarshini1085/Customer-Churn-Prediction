"""Fairness report: does the model work equally well across customer groups?

For each attribute in :data:`churn.config.FAIRNESS_ATTRIBUTES` (gender,
SeniorCitizen, Partner) and each of its groups we compute, on the held-out test
set at the model's tuned decision threshold:

* **recall** (true-positive rate): of the customers who did churn, the share we
  flagged. A low recall for a group means its churners are missed and never get
  a retention offer, which is the harm that matters most for this use case.
* **precision**: of the customers we flagged, the share who churned.
* **false-positive rate**: of the customers who stayed, the share we flagged
  (they receive an offer they did not need).

The group's size and actual churn rate are reported alongside, because when base
rates differ between groups, precision and FPR differ too even for a perfectly
calibrated model; it is mathematically impossible to equalise every metric at
once. Gaps larger than :data:`churn.config.FAIRNESS_GAP_FLAG` are flagged for
review, not treated as proof of unfairness.

Note that gender and SeniorCitizen are themselves model inputs. Dropping them
would not by itself remove a gap, because correlated features remain.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from churn import config

METRICS: tuple[str, ...] = ("recall", "precision", "false_positive_rate")
_LABELS = {
    "recall": "Recall",
    "precision": "Precision",
    "false_positive_rate": "False-positive rate",
}


def _ratio(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def group_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, Any]:
    """Size, base rate and recall / precision / FPR for one group (None if undefined)."""
    y_true, y_pred = np.asarray(y_true, dtype=int), np.asarray(y_pred, dtype=int)
    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())
    tn = int(((y_true == 0) & (y_pred == 0)).sum())
    return {
        "n": len(y_true),
        "churners": tp + fn,
        "churn_rate": _ratio(tp + fn, len(y_true)),
        "flagged_rate": _ratio(tp + fp, len(y_true)),
        "recall": _ratio(tp, tp + fn),
        "precision": _ratio(tp, tp + fp),
        "false_positive_rate": _ratio(fp, fp + tn),
    }


def fairness_report(
    x: pd.DataFrame,
    y_true: pd.Series | np.ndarray,
    probability: np.ndarray,
    threshold: float,
    attributes: tuple[str, ...] = config.FAIRNESS_ATTRIBUTES,
) -> dict[str, Any]:
    """Per-group metrics, the largest gap per metric, and flagged gaps, per attribute."""
    y = np.asarray(y_true, dtype=int)
    predicted = (np.asarray(probability, dtype=float) >= threshold).astype(int)
    report: dict[str, Any] = {
        "threshold": float(threshold),
        "gap_flag": config.FAIRNESS_GAP_FLAG,
        "attributes": {},
    }
    for attribute in attributes:
        values = x[attribute].astype(str).to_numpy()
        groups = {
            group: group_metrics(y[values == group], predicted[values == group])
            for group in sorted(set(values))
        }
        gaps: dict[str, float | None] = {}
        for metric in METRICS:
            defined = [g[metric] for g in groups.values() if g[metric] is not None]
            gaps[metric] = max(defined) - min(defined) if len(defined) > 1 else None
        report["attributes"][attribute] = {
            "groups": groups,
            "gaps": gaps,
            "flagged": [
                m for m, gap in gaps.items() if gap is not None and gap > report["gap_flag"]
            ],
        }
    return report


def interpret(report: dict[str, Any]) -> list[str]:
    """Plain-language findings generated from the numbers in ``report``."""
    findings: list[str] = []
    for attribute, result in report["attributes"].items():
        groups = result["groups"]
        recall = {g: m["recall"] for g, m in groups.items() if m["recall"] is not None}
        base = {g: m["churn_rate"] for g, m in groups.items() if m["churn_rate"] is not None}
        if len(recall) < 2:
            findings.append(f"{attribute}: not enough groups with churners to compare.")
            continue
        low, high = min(recall, key=recall.__getitem__), max(recall, key=recall.__getitem__)
        sentence = (
            f"{attribute}: recall ranges from {recall[low]:.2f} ({attribute}={low}) to "
            f"{recall[high]:.2f} ({attribute}={high})"
        )
        if result["flagged"]:
            names = ", ".join(_LABELS[m].lower() for m in result["flagged"])
            sentence += (
                f"; the gap in {names} exceeds {report['gap_flag']:.2f} and is worth reviewing."
            )
        else:
            sentence += f"; all gaps are within {report['gap_flag']:.2f}."
        if len(base) > 1 and max(base.values()) - min(base.values()) > report["gap_flag"]:
            sentence += (
                f" Actual churn rates differ ({', '.join(f'{g}: {r:.0%}' for g, r in base.items())}),"
                " which by itself moves precision and FPR."
            )
        smallest = min(recall, key=lambda g: groups[g]["churners"])
        churners, r = groups[smallest]["churners"], recall[smallest]
        margin = 1.96 * (r * (1 - r) / churners) ** 0.5  # normal approximation
        sentence += (
            f" The smallest group ({attribute}={smallest}) has {churners} churners in the "
            f"test set, so its recall is uncertain by about ±{margin:.2f} (95%)."
        )
        findings.append(sentence)
    return findings


def _fmt(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.3f}"


def model_card_section(report: dict[str, Any]) -> str:
    """Markdown fairness section for the model card."""
    lines = [
        "## Fairness",
        "",
        f"Held-out test set, tuned threshold {report['threshold']:.2f}. Gaps above "
        f"{report['gap_flag']:.2f} are flagged for review.",
        "",
        "| attribute | group | n | churn rate | recall | precision | FPR |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for attribute, result in report["attributes"].items():
        for group, m in result["groups"].items():
            lines.append(
                f"| {attribute} | {group} | {m['n']} | {_fmt(m['churn_rate'])} "
                f"| {_fmt(m['recall'])} | {_fmt(m['precision'])} "
                f"| {_fmt(m['false_positive_rate'])} |"
            )
    lines += ["", "**Interpretation**", "", *(f"- {line}" for line in interpret(report))]
    lines += [
        "",
        "Recall is the most important metric here: a missed churner never receives a "
        "retention offer. When churn rates differ between groups, equal precision and "
        "equal FPR cannot both hold, so gaps there are expected and need context. Gender "
        "and SeniorCitizen are model inputs; removing them would not remove gaps on its "
        "own, because correlated features remain.",
    ]
    return "\n".join(lines) + "\n"
