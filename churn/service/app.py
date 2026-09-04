"""FastAPI application factory."""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from churn import __version__, config
from churn.logging_ import configure_logging, request_id_var
from churn.service.deps import load_model_state
from churn.service.metrics import PREDICTION_ERRORS
from churn.service.middleware import RequestContextMiddleware
from churn.service.models import ErrorResponse
from churn.service.routes import router

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    configure_logging(json_output=config.settings.log_json)
    app.state.model_state = load_model_state()
    if not app.state.model_state.healthy:
        logger.warning("Starting API with no model loaded — /predict will return 503")
    yield


def create_app() -> FastAPI:
    app = FastAPI(
        title="Customer Churn Prediction API",
        version=__version__,
        summary="Real-time churn probability, risk tier and drivers for a customer.",
        lifespan=lifespan,
    )
    app.add_middleware(RequestContextMiddleware)
    app.include_router(router)

    @app.exception_handler(RequestValidationError)
    async def _validation_error(request: Request, exc: RequestValidationError):
        PREDICTION_ERRORS.labels(endpoint=request.url.path.lstrip("/") or "root").inc()
        return JSONResponse(
            status_code=422,
            content=ErrorResponse(
                detail="; ".join(
                    f"{'.'.join(map(str, e['loc']))}: {e['msg']}" for e in exc.errors()
                ),
                request_id=request_id_var.get(),
            ).model_dump(),
        )

    @app.exception_handler(Exception)
    async def _unhandled(request: Request, exc: Exception):
        PREDICTION_ERRORS.labels(endpoint=request.url.path.lstrip("/") or "root").inc()
        logger.exception("unhandled error")
        return JSONResponse(
            status_code=500,
            content=ErrorResponse(
                detail="internal error", request_id=request_id_var.get()
            ).model_dump(),
        )

    return app
