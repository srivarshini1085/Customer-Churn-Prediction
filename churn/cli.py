"""Command-line interface: ``python -m churn {train,predict,score,drift,serve,registry}``.

Heavy imports are deferred into the subcommand handlers so ``churn --help`` and
argument errors stay fast.
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
from typing import Any

from churn.logging_ import configure_logging
from churn.schema import INPUT_FIELDS, Field, defaults


def _arg_name(field_name: str) -> str:
    """``MonthlyCharges`` -> ``--monthly-charges``; ``tenure`` -> ``--tenure``."""
    kebab = re.sub(r"(?<=[a-z])(?=[A-Z])", "-", field_name).replace("_", "-")
    return "--" + kebab.lower()


def _build_parser() -> argparse.ArgumentParser:
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("-v", "--verbose", action="store_true", help="show info-level logs")

    parser = argparse.ArgumentParser(
        prog="churn", description="Customer churn prediction toolkit.", parents=[common]
    )
    parser.add_argument(
        "--version", dest="show_version", action="store_true", help="print version and exit"
    )
    sub = parser.add_subparsers(dest="command")

    train_p = sub.add_parser("train", parents=[common], help="train, calibrate and register a model")
    train_p.add_argument(
        "--threshold-objective",
        choices=["f1", "recall_at_precision"],
        default="f1",
        help="how the decision threshold is chosen (default: f1)",
    )

    predict_p = sub.add_parser("predict", parents=[common], help="score a single customer")
    predict_p.add_argument("-i", "--interactive", action="store_true", help="prompt for each field")
    predict_p.add_argument("--json", action="store_true", help="print the raw result as JSON")
    predict_p.add_argument("--model-version", default=None, help="pin a registry version")
    for field in INPUT_FIELDS:
        kwargs: dict[str, Any] = {"dest": field.name, "help": f"{field.label} (default: {field.default})"}
        if field.choices:
            kwargs["choices"] = list(field.choices)
        predict_p.add_argument(_arg_name(field.name), **kwargs)

    score_p = sub.add_parser("score", parents=[common], help="score a CSV of customers")
    score_p.add_argument("--input", required=True, help="input CSV path")
    score_p.add_argument("--output", required=True, help="output CSV path")
    score_p.add_argument("--model-version", default=None, help="pin a registry version")

    drift_p = sub.add_parser("drift", parents=[common], help="PSI drift report vs. the training data")
    drift_p.add_argument("--current", required=True, help="CSV to compare against training")
    drift_p.add_argument("--model-version", default=None, help="pin a registry version")

    serve_p = sub.add_parser("serve", parents=[common], help="run the REST API (uvicorn)")
    serve_p.add_argument("--host", default="127.0.0.1")
    serve_p.add_argument("--port", type=int, default=8000)
    serve_p.add_argument("--reload", action="store_true", help="auto-reload on code changes")

    reg_p = sub.add_parser("registry", parents=[common], help="inspect / manage model versions")
    reg_sub = reg_p.add_subparsers(dest="registry_command", required=True)
    reg_sub.add_parser("list", parents=[common], help="list registered versions")
    show_p = reg_sub.add_parser("show", parents=[common], help="print a manifest")
    show_p.add_argument("version", nargs="?", default="latest")
    promote_p = reg_sub.add_parser("promote", parents=[common], help="set 'latest' (rollback)")
    promote_p.add_argument("version")

    return parser


# --------------------------------------------------------------------------- #
# handlers
# --------------------------------------------------------------------------- #


def _run_train(args: argparse.Namespace) -> int:
    from churn.train import train

    train(threshold_objective=args.threshold_objective)
    return 0


def _prompt(field: Field) -> Any:
    suffix = f" {list(field.choices)}" if field.choices else ""
    raw = input(f"{field.label}{suffix} [{field.default}]: ").strip()
    return field.coerce(raw or None)


def _collect_features(args: argparse.Namespace) -> dict[str, Any]:
    features = defaults()
    if args.interactive:
        print("=" * 40, "\n     CUSTOMER CHURN PREDICTION\n", "=" * 40, sep="")
        for field in INPUT_FIELDS:
            features[field.name] = _prompt(field)
        return features
    for field in INPUT_FIELDS:
        supplied = getattr(args, field.name, None)
        if supplied is not None:
            features[field.name] = field.coerce(supplied)
    return features


def _run_predict(args: argparse.Namespace) -> int:
    from churn.predict import load_model, log_prediction

    features = _collect_features(args)
    model = load_model(args.model_version)
    result = model.predict(features)
    log_prediction(features, result)

    if args.json:
        print(json.dumps(result, indent=2, default=str))
        return 0

    verdict = "LIKELY TO CHURN" if result["churn"] else "NOT likely to churn"
    print("\n" + "=" * 44)
    print(f"  Customer is {verdict}")
    print(f"  Churn probability : {result['probability'] * 100:.2f}%")
    print(f"  Risk tier         : {result['risk_tier']}")
    print(f"  Decision threshold: {result['threshold']:.2f}")
    print(f"  Model version     : {result['model_version']}")
    print("=" * 44)

    for warning in result["warnings"]:
        print(f"  ⚠ {warning}")

    factors = model.top_factors(features)
    if factors:
        header = "Drivers (signed)" if model.is_linear else "Influential factors"
        print(f"\n  {header}:")
        for name, value in factors:
            if model.is_linear:
                marker = "raises" if value > 0 else "lowers"
                print(f"    {marker} risk  {name}  ({value:+.3f})")
            else:
                print(f"    - {name}  (importance {value:.3f})")
    print()
    return 0


def _run_score(args: argparse.Namespace) -> int:
    from churn.batch import score_csv

    out = score_csv(args.input, args.output, version=args.model_version)
    print(f"Wrote {out}")
    return 0


def _run_drift(args: argparse.Namespace) -> int:
    import pandas as pd

    from churn.predict import load_model
    from churn.validation import PSI_MAJOR, drift_report

    model = load_model(args.model_version)
    if model.training_stats is None:
        print("This model has no training-stats snapshot; retrain to enable drift checks.")
        return 1

    df = pd.read_csv(args.current)
    report = drift_report(model.training_stats, df)
    print(f"\nDrift vs. training data of model {model.version}  (PSI > {PSI_MAJOR} = drift)\n")
    print(f"{'feature':<22}{'psi':>10}   status")
    print("-" * 44)
    for feature, row in sorted(report.items(), key=lambda kv: kv[1]["psi"], reverse=True):
        flag = "DRIFT" if row["drifted"] else "ok"
        print(f"{feature:<22}{row['psi']:>10.4f}   {flag}")
    drifted = [f for f, r in report.items() if r["drifted"]]
    print("\n" + (f"{len(drifted)} feature(s) drifted: {', '.join(drifted)}" if drifted else "No drift detected."))
    return 1 if drifted else 0


def _run_serve(args: argparse.Namespace) -> int:
    import uvicorn

    uvicorn.run("api:app", host=args.host, port=args.port, reload=args.reload)
    return 0


def _run_registry(args: argparse.Namespace) -> int:
    from churn.registry import ModelRegistry

    registry = ModelRegistry()
    if args.registry_command == "list":
        rows = registry.list()
        if not rows:
            print("Registry is empty. Run `python -m churn train`.")
            return 0
        print(f"{'version':<20}{'model':<22}{'roc_auc':>9}{'recall':>9}{'thr':>7}   latest")
        print("-" * 76)
        for r in rows:
            print(
                f"{r['version']:<20}{str(r['model']):<22}"
                f"{(r['roc_auc'] or 0):>9.4f}{(r['recall'] or 0):>9.3f}"
                f"{(r['threshold'] or 0):>7.2f}   {'*' if r['is_latest'] else ''}"
            )
        return 0
    if args.registry_command == "show":
        print(json.dumps(registry.manifest(args.version), indent=2, default=str))
        return 0
    if args.registry_command == "promote":
        registry.promote(args.version)
        print(f"Promoted {args.version} to latest")
        return 0
    return 1


_HANDLERS = {
    "train": _run_train,
    "predict": _run_predict,
    "score": _run_score,
    "drift": _run_drift,
    "serve": _run_serve,
    "registry": _run_registry,
}


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    if getattr(args, "show_version", False):
        from churn import __version__

        print(__version__)
        return 0
    if not args.command:
        _build_parser().print_help()
        return 1
    configure_logging(logging.INFO if args.verbose else logging.WARNING)
    return _HANDLERS[args.command](args)


if __name__ == "__main__":
    sys.exit(main())
