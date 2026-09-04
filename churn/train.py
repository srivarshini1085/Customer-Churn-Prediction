"""Training pipeline: compare candidates, calibrate the winner, tune the
threshold, snapshot the training distribution and register a new model version.

Run with ``python -m churn train`` (or the legacy ``python src/train.py``).
Registers ``models/registry/<version>/`` with ``model.joblib`` + ``manifest.json``
+ ``model_card.md`` and promotes it to ``latest``.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
from sklearn.pipeline import Pipeline

from churn import config
from churn.data import load_clean_split
from churn.evaluate import (
    CandidateResult,
    optimal_threshold,
    select_best,
    test_report,
    tune_and_score,
)
from churn.features import pretty_feature_name
from churn.model import calibrate, candidates, choose_calibration_method
from churn.predict import ChurnModel
from churn.registry import ModelRegistry, provenance
from churn.validation import TrainingStats

logger = logging.getLogger(__name__)

_OBJECTIVE_BLURB = {
    "f1": "maximises churn-class F1 on out-of-fold predictions",
    "recall_at_precision": "maximises churn recall at precision >= 0.5 (out-of-fold)",
}


def _global_importances(pipeline: Pipeline, names: list[str]) -> dict[str, float]:
    """Signed coefficients (linear) or feature importances (trees), keyed by name."""
    clf = pipeline.named_steps["clf"]
    if hasattr(clf, "coef_"):
        values = np.ravel(clf.coef_)
    elif hasattr(clf, "feature_importances_"):
        values = np.asarray(clf.feature_importances_)
    else:
        return {}
    return {name: float(v) for name, v in zip(names, values, strict=True)}


def _print_table(results: list[CandidateResult], winner: CandidateResult) -> None:
    header = f"{'model':<24}{'roc_auc':>10}{'pr_auc':>10}{'f1':>8}{'recall':>9}{'precision':>11}"
    rule = "=" * len(header)
    print(f"\n{rule}\nCROSS-VALIDATED MODEL COMPARISON\n{rule}\n{header}\n{'-' * len(header)}")
    for r in sorted(results, key=lambda x: x.cv_scores["roc_auc"], reverse=True):
        mark = "  <-- selected" if r.name == winner.name else ""
        s = r.cv_scores
        print(
            f"{r.name:<24}{s['roc_auc']:>10.4f}{s['pr_auc']:>10.4f}"
            f"{s['f1']:>8.4f}{s['recall']:>9.4f}{s['precision']:>11.4f}{mark}"
        )
    print(rule)
    leader = max(results, key=lambda r: r.cv_scores["roc_auc"])
    if leader.name != winner.name:
        print(
            f"note: {winner.name} chosen over {leader.name} - ROC-AUC is tied to 3 dp, "
            f"so higher recall ({winner.cv_scores['recall']:.3f} vs "
            f"{leader.cv_scores['recall']:.3f}) decides."
        )
    print()


def _resolve_calibration_method(base: Pipeline, x, y) -> str | None:
    mode = config.settings.calibration
    if mode == "none":
        return None
    if mode in ("sigmoid", "isotonic"):
        return mode
    return choose_calibration_method(base, x, y)


def _model_card(manifest: dict[str, Any]) -> str:
    m = manifest
    tstats, test = m["threshold_stats"], m["test_metrics"]
    tuned, default = test["tuned"], test["default_0.5"]
    top = sorted(
        m["global_importances"].items(), key=lambda kv: abs(kv[1]), reverse=True
    )[:15]
    cal = m["calibration"] or "none"

    lines = [
        f"# Churn model card — `{m['version']}`",
        "",
        f"- **Created:** {m['created_at']}",
        f"- **Selected model:** `{m['model_name']}`  (`{m['best_params']}`)",
        f"- **Selection metric:** {config.SELECTION_METRIC} "
        f"(stratified {config.CV_FOLDS}-fold CV)",
        f"- **Calibration:** {cal}",
        f"- **git:** `{m.get('git_sha')}`  **python:** {m.get('python')}  "
        f"**sklearn:** {m.get('sklearn')}",
        f"- **Training rows:** {m['n_train']}  **Test rows:** {m['n_test']}  "
        f"**data sha256:** `{str(m.get('data_sha256'))[:12]}…`",
        "",
        "## Decision threshold",
        "",
        f"Tuned to **{m['threshold']:.2f}** "
        f"({_OBJECTIVE_BLURB[m['threshold_objective']]}).",
        "",
        "| metric | CV @ tuned threshold |",
        "| --- | --- |",
        f"| precision | {tstats['cv_precision']:.3f} |",
        f"| recall | {tstats['cv_recall']:.3f} |",
        f"| f1 | {tstats['cv_f1']:.3f} |",
        "",
        "The fixed business risk bands (High >= 0.70, Moderate 0.40-0.69, Low < 0.40) "
        "are separate from this threshold and drive the recommended retention action.",
        "",
        "## Held-out test performance",
        "",
        f"- ROC-AUC: **{test['roc_auc']:.4f}**   PR-AUC: **{test['pr_auc']:.4f}**   "
        f"Brier: **{test['brier']:.4f}** (lower is better; calibration quality)",
        "",
        "| threshold | precision | recall | f1 | accuracy |",
        "| --- | --- | --- | --- | --- |",
        f"| tuned ({tuned['threshold']:.2f}) | {tuned['precision']:.3f} "
        f"| {tuned['recall']:.3f} | {tuned['f1']:.3f} | {tuned['accuracy']:.3f} |",
        f"| default (0.50) | {default['precision']:.3f} | {default['recall']:.3f} "
        f"| {default['f1']:.3f} | {default['accuracy']:.3f} |",
        "",
        "## Top 15 feature weights",
        "",
        "| feature | weight |",
        "| --- | --- |",
        *(f"| {pretty_feature_name(name)} | {value:+.4f} |" for name, value in top),
    ]
    return "\n".join(lines) + "\n"


def train(threshold_objective: str = "f1") -> ChurnModel:
    """Run the full training pipeline and register a new model version."""
    x_train, x_test, y_train, y_test = load_clean_split()

    results: list[CandidateResult] = []
    for candidate in candidates():
        logger.info("Tuning %s ...", candidate.name)
        results.append(tune_and_score(candidate, x_train, y_train))

    best = select_best(results)
    _print_table(results, best)
    logger.info("Selected %s", best.name)

    base_pipeline: Pipeline = best.estimator  # already refit on all of x_train

    method = _resolve_calibration_method(base_pipeline, x_train, y_train)
    if method is None:
        scoring_pipeline = base_pipeline
        logger.info("Calibration disabled")
    else:
        scoring_pipeline = calibrate(base_pipeline, x_train, y_train, method)
        logger.info("Calibrated with method=%s", method)

    threshold, threshold_stats = optimal_threshold(
        scoring_pipeline, x_train, y_train, objective=threshold_objective
    )
    test_metrics = test_report(scoring_pipeline, x_test, y_test, threshold)

    names = list(base_pipeline.named_steps["prep"].get_feature_names_out())
    training_stats = TrainingStats.from_frame(x_train)

    registry = ModelRegistry()
    version = registry.next_version()

    manifest: dict[str, Any] = {
        "version": version,
        **provenance(),
        "model_name": best.name,
        "best_params": best.best_params,
        "threshold": threshold,
        "threshold_objective": threshold_objective,
        "calibration": method,
        "n_train": int(len(x_train)),
        "n_test": int(len(x_test)),
        "cv_scores": best.cv_scores,
        "test_metrics": test_metrics,
        "threshold_stats": threshold_stats,
        "training_stats": training_stats.summary(),
        "global_importances": _global_importances(base_pipeline, names),
    }

    model = ChurnModel(
        pipeline=scoring_pipeline,
        base_pipeline=base_pipeline,
        threshold=threshold,
        feature_names=names,
        training_stats=training_stats,
        metadata=manifest,
    )

    registry.save(model, manifest, _model_card(manifest), promote=True)

    print(
        f"Registered '{version}'  model={best.name}  "
        f"test ROC-AUC={test_metrics['roc_auc']:.4f}  "
        f"recall={test_metrics['tuned']['recall']:.3f} @ threshold {threshold:.2f}  "
        f"Brier={test_metrics['brier']:.4f}"
    )
    return model


def main() -> None:
    from churn.logging_ import configure_logging

    configure_logging()
    train()


if __name__ == "__main__":
    main()
