"""Middleware: request-id stamping + privacy-safe request logging (no bodies, no PII)."""

from __future__ import annotations

import logging
import time

from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.requests import Request
from starlette.responses import Response

from app.utils.logging import get_request_id, log_event, set_request_id


class RequestIDMiddleware(BaseHTTPMiddleware):
    """Every request gets a request_id: honoured from X-Request-ID, else generated.

    The id is stamped into the logging context (every log line carries it) and
    echoed back in the X-Request-ID response header so clients can quote it.
    """

    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        incoming = request.headers.get("X-Request-ID")
        request_id = set_request_id(incoming)[:64]
        request.state.request_id = request_id
        started = time.perf_counter()
        try:
            response = await call_next(request)
        except Exception:
            log_event(
                logging.getLogger("app.request"),
                logging.ERROR,
                "request crashed",
                method=request.method,
                path=request.url.path,
                duration_ms=int((time.perf_counter() - started) * 1000),
                request_id=request_id,
            )
            raise
        duration_ms = int((time.perf_counter() - started) * 1000)
        response.headers["X-Request-ID"] = request_id
        # method/path/status/duration only - request bodies are never logged
        log_event(
            logging.getLogger("app.request"),
            logging.INFO,
            "request",
            method=request.method,
            path=request.url.path,
            status=response.status_code,
            duration_ms=duration_ms,
            request_id=request_id,
        )
        return response


def current_request_id() -> str | None:
    return get_request_id()


def register_middleware(app) -> None:
    app.add_middleware(RequestIDMiddleware)
