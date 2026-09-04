"""File-system model registry with versioning, promotion and rollback.

Layout::

    models/registry/
        20260905T101500Z/
            model.joblib      # the ChurnModel
            manifest.json     # version, provenance, metrics, training-stats summary
            model_card.md     # human-readable
        20260905T140322Z/
            ...
        latest.json           # {"version": "20260905T140322Z"}  ← the promoted one

``latest.json`` is a pointer file rather than a symlink so it works on Windows.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import logging
import platform
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import sklearn

from churn import config
from churn.predict import ChurnModel

logger = logging.getLogger(__name__)

_LATEST = "latest.json"


def new_version(now: dt.datetime | None = None) -> str:
    """A sortable UTC version id, e.g. ``20260905T140322Z``."""
    now = now or dt.datetime.now(dt.timezone.utc)
    return now.strftime("%Y%m%dT%H%M%SZ")


def git_sha() -> str | None:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=config.PROJECT_ROOT,
            capture_output=True,
            text=True,
            timeout=5,
        )
        return out.stdout.strip() or None if out.returncode == 0 else None
    except (OSError, subprocess.SubprocessError):
        return None


def data_sha256(path: Path) -> str | None:
    if not path.exists():
        return None
    h = hashlib.sha256()
    h.update(path.read_bytes())
    return h.hexdigest()


def provenance() -> dict[str, Any]:
    """Environment / data fingerprint stored on every model."""
    return {
        "created_at": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "git_sha": git_sha(),
        "python": platform.python_version(),
        "sklearn": sklearn.__version__,
        "numpy": np.__version__,
        "data_sha256": data_sha256(config.DATA_PATH),
    }


@dataclass
class ModelRegistry:
    """CRUD over the on-disk registry. Defaults to ``config.REGISTRY_DIR``."""

    root: Path | None = None

    def __post_init__(self) -> None:
        self.root = (
            Path(self.root)
            if self.root is not None
            else config.settings.registry_dir
        )

    # -- internals ------------------------------------------------------- #

    def _dir(self, version: str) -> Path:
        return self.root / version

    def _pointer(self) -> Path:
        return self.root / _LATEST

    def next_version(self) -> str:
        """A fresh, collision-free version id."""
        base = new_version()
        version, n = base, 1
        while self._dir(version).exists():
            version = f"{base[:-1]}-{n}Z"
            n += 1
        return version

    # -- write --------------------------------------------------------- #

    def save(
        self,
        model: ChurnModel,
        manifest: dict[str, Any],
        model_card: str,
        *,
        promote: bool = True,
    ) -> str:
        """Persist a model + its manifest + card under a fresh version."""
        version = manifest.get("version") or self.next_version()
        manifest = {**manifest, "version": version}

        target = self._dir(version)
        target.mkdir(parents=True, exist_ok=True)
        model.save(target / "model.joblib")
        (target / "manifest.json").write_text(
            json.dumps(manifest, indent=2, default=str), encoding="utf-8"
        )
        (target / "model_card.md").write_text(model_card, encoding="utf-8")
        logger.info("Registered model %s", version)

        if promote:
            self.promote(version)
        return version

    def promote(self, version: str) -> None:
        """Point ``latest`` at ``version`` (rollback = promote an older one)."""
        if not self._dir(version).exists():
            raise FileNotFoundError(f"No model version {version!r} in {self.root}")
        self.root.mkdir(parents=True, exist_ok=True)
        self._pointer().write_text(
            json.dumps({"version": version}), encoding="utf-8"
        )
        logger.info("Promoted %s to latest", version)

    # -- read ---------------------------------------------------------- #

    def list_versions(self) -> list[str]:
        if not self.root.exists():
            return []
        return sorted(
            p.name for p in self.root.iterdir() if p.is_dir() and (p / "manifest.json").exists()
        )

    def resolve(self, version: str = "latest") -> str:
        if version != "latest":
            return version
        pointer = self._pointer()
        if pointer.exists():
            return json.loads(pointer.read_text())["version"]
        versions = self.list_versions()
        if not versions:
            raise FileNotFoundError(
                f"Model registry {self.root} is empty. Run `python -m churn train` first."
            )
        return versions[-1]

    def manifest(self, version: str = "latest") -> dict[str, Any]:
        version = self.resolve(version)
        path = self._dir(version) / "manifest.json"
        if not path.exists():
            raise FileNotFoundError(f"No manifest for version {version!r}")
        return json.loads(path.read_text(encoding="utf-8"))

    def load(self, version: str = "latest") -> ChurnModel:
        version = self.resolve(version)
        path = self._dir(version) / "model.joblib"
        if not path.exists():
            raise FileNotFoundError(
                f"No model artifact for version {version!r} in {self.root}"
            )
        return ChurnModel.load(path)

    def list(self) -> list[dict[str, Any]]:
        """Summary of every registered version, newest last."""
        latest = None
        if self._pointer().exists():
            latest = json.loads(self._pointer().read_text())["version"]
        rows: list[dict[str, Any]] = []
        for version in self.list_versions():
            m = json.loads((self._dir(version) / "manifest.json").read_text())
            test = m.get("test_metrics", {})
            rows.append(
                {
                    "version": version,
                    "is_latest": version == latest,
                    "model": m.get("model_name"),
                    "created_at": m.get("created_at"),
                    "roc_auc": test.get("roc_auc"),
                    "recall": test.get("tuned", {}).get("recall"),
                    "threshold": m.get("threshold"),
                }
            )
        return rows
