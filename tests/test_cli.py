"""Unit and integration tests for the CLI subcommands."""

from __future__ import annotations

from churn.cli import main


def test_cli_version(capsys) -> None:
    code = main(["--version"])
    assert code == 0
    captured = capsys.readouterr()
    assert "2.0.0" in captured.out


def test_cli_no_args(capsys) -> None:
    code = main([])
    assert code == 1
    captured = capsys.readouterr()
    assert "Customer churn prediction toolkit" in captured.out


def test_cli_predict(capsys, registered_model: tuple[str, object]) -> None:
    code = main(
        [
            "predict",
            "--tenure",
            "12",
            "--contract",
            "Month-to-month",
            "--monthly-charges",
            "65.5",
            "--json",
        ]
    )
    assert code == 0
    captured = capsys.readouterr()
    assert "probability" in captured.out
    assert "risk_tier" in captured.out


def test_cli_registry_list_and_show(capsys, registered_model: tuple[str, object]) -> None:
    code = main(["registry", "list"])
    assert code == 0
    captured = capsys.readouterr()
    version, _ = registered_model
    assert version in captured.out

    code_show = main(["registry", "show", "latest"])
    assert code_show == 0
    captured_show = capsys.readouterr()
    assert "model_name" in captured_show.out


def test_cli_drift_and_score_help(capsys) -> None:
    try:
        main(["drift", "--help"])
    except SystemExit as exc:
        assert exc.code == 0

    try:
        main(["score", "--help"])
    except SystemExit as exc:
        assert exc.code == 0


def test_cli_plan_writes_action_list(capsys, tmp_path, registered_model, raw_df) -> None:
    source = tmp_path / "customers.csv"
    raw_df.head(200).to_csv(source, index=False)
    output = tmp_path / "out" / "plan.csv"

    code = main(["plan", "--budget", "500", "--input", str(source), "--output", str(output)])
    assert code == 0
    printed = capsys.readouterr().out
    for strategy in ("optimized", "greedy", "top_risk"):
        assert strategy in printed
    assert "business assumptions" in printed

    import pandas as pd

    actions = pd.read_csv(output)
    assert list(actions.columns) == [
        "customer",
        "churn_probability",
        "customer_value",
        "offer",
        "offer_label",
        "cost",
        "expected_saving",
    ]
    assert actions["cost"].sum() <= 500 + 1e-6
    assert set(actions["customer"]) <= set(raw_df["customerID"])


def test_cli_plan_strategy_choice_and_negative_budget(capsys, tmp_path, registered_model, raw_df):
    source = tmp_path / "customers.csv"
    raw_df.head(50).to_csv(source, index=False)
    output = tmp_path / "baseline.csv"
    args = ["plan", "--input", str(source), "--output", str(output)]

    assert main([*args, "--budget", "200", "--strategy", "top_risk"]) == 0
    import pandas as pd

    from churn import config

    assert set(pd.read_csv(output)["offer"]) <= {config.BASELINE_OFFER}
    assert main([*args, "--budget", "-5"]) == 2
    assert "non-negative" in capsys.readouterr().out


def test_cli_drift_reports_status_and_retrain_signal(capsys, tmp_path, registered_model, raw_df):
    shifted = raw_df.sample(400, random_state=0).copy()
    shifted["MonthlyCharges"] = shifted["MonthlyCharges"] * 1.5
    path = tmp_path / "batch.csv"
    shifted.to_csv(path, index=False)

    assert main(["drift", "--current", str(path)]) == 1
    out = capsys.readouterr().out
    assert "RETRAIN RECOMMENDED" in out
    assert "MonthlyCharges" in out and "red" in out

    same = tmp_path / "same.csv"
    raw_df.to_csv(same, index=False)
    assert main(["drift", "--current", str(same)]) == 0
    assert "No retraining needed" in capsys.readouterr().out
