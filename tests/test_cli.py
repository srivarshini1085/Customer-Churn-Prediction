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
