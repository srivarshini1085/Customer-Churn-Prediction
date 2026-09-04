"""Tests for churn.registry.ModelRegistry."""

from __future__ import annotations

import pytest

from churn.predict import ChurnModel
from churn.registry import ModelRegistry, new_version


def test_new_version_is_sortable():
    assert new_version() < new_version().replace("2026", "2027")


def test_save_load_roundtrip(fitted_model, tmp_path):
    registry = ModelRegistry(root=tmp_path / "reg")
    version = registry.save(fitted_model, {"model_name": "lr"}, "# card\n")

    loaded = registry.load(version)
    assert isinstance(loaded, ChurnModel)
    assert loaded.feature_names == fitted_model.feature_names
    assert registry.resolve("latest") == version


def test_list_and_promote_rollback(fitted_model, tmp_path):
    registry = ModelRegistry(root=tmp_path / "reg")
    v1 = registry.save(fitted_model, {"model_name": "a"}, "c")
    v2 = registry.save(fitted_model, {"model_name": "b"}, "c")

    assert registry.resolve("latest") == v2
    rows = {r["version"]: r for r in registry.list()}
    assert rows[v2]["is_latest"] is True and rows[v1]["is_latest"] is False

    registry.promote(v1)  # rollback
    assert registry.resolve("latest") == v1


def test_empty_registry_errors(tmp_path):
    registry = ModelRegistry(root=tmp_path / "empty")
    with pytest.raises(FileNotFoundError, match="empty"):
        registry.load("latest")


def test_promote_unknown_version_errors(fitted_model, tmp_path):
    registry = ModelRegistry(root=tmp_path / "reg")
    registry.save(fitted_model, {}, "c")
    with pytest.raises(FileNotFoundError):
        registry.promote("does-not-exist")


def test_manifest_contains_version(fitted_model, tmp_path):
    registry = ModelRegistry(root=tmp_path / "reg")
    version = registry.save(fitted_model, {"model_name": "lr", "threshold": 0.4}, "c")
    manifest = registry.manifest(version)
    assert manifest["version"] == version
    assert manifest["threshold"] == 0.4
