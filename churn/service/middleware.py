"""Request-id + access-logging middleware."""

from __future__ import annotations

import logging
import time
import uuid

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response

from churn.logging_ import request_id_var

logger = logging.getLogger("churn.service.access")

_HEADER = "x-request-id"


class RequestContextMiddleware(BaseHTTPMiddleware):
    """Assign every request an id, expose it on the response, log the round-trip."""

    async def dispatch(self, request: Request, call_next) -> Response:
        request_id = request.headers.get(_HEADER) or uuid.uuid4().hex
        token = request_id_var.set(request_id)
        start = time.perf_counter()
        try:
            response = await call_next(request)
        except Exception:
            logger.exception(
                "unhandled error", extra={"path": request.url.path, "method": request.method}
            )
            raise
        finally:
            request_id_var.reset(token)

        elapsed_ms = (time.perf_counter() - start) * 1000
        response.headers[_HEADER] = request_id
        logger.info(
            "request",
            extra={
                "method": request.method,
                "path": request.url.path,
                "status": response.status_code,
                "duration_ms": round(elapsed_ms, 2),
                "request_id": request_id,
            },
        )
        return response
