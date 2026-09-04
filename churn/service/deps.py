"""Model lifecycle for the API: load once at startup, expose via a dependency."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass

from fastapi import HTTPException, Request, status

from churn.predict import ChurnModel, clear_cache, load_model

logger = logging.getLogger(__name__)


@dataclass
class ModelState:
    model: ChurnModel | None = None
    error: str | None = None
    started_at: float = 0.0

    @property
    def uptime_seconds(self) -> float:
        return time.time() - self.started_at

    @property
    def healthy(self) -> bool:
        return self.model is not None


def load_model_state() -> ModelState:
    """Attempt to load the configured model; never raises."""
    state = ModelState(started_at=time.time())
    try:
        clear_cache()
        state.model = load_model()
        logger.info("Loaded model %s", state.model.version)
    except Exception as exc:  # noqa: BLE001 - startup must not crash the process
        state.error = str(exc)
        logger.error("Model load failed: %s", exc)
    return state


def get_model(request: Request) -> ChurnModel:
    """FastAPI dependency — 503 until a model is loaded."""
    state: ModelState = request.app.state.model_state
    if state.model is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Model not available: {state.error or 'not loaded'}",
        )
    return state.model
