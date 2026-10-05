"""Structured JSON logging with a request_id on every record.

Privacy rules enforced here:
  * keys matching SENSITIVE_KEY_PATTERNS are replaced with "***" recursively;
  * e-mail addresses are masked unless the caller opts out;
  * we never log request bodies from the middleware (only method/path/status/duration).
"""

from __future__ import annotations

import json
import logging
import re
import sys
from contextvars import ContextVar
from datetime import datetime, timezone
from typing import Any

from app.utils.ids import new_request_id

_request_id: ContextVar[str | None] = ContextVar("request_id", default=None)

SENSITIVE_KEY_PATTERNS = (
    "password",
    "password_hash",
    "token",
    "authorization",
    "api_key",
    "apikey",
    "secret",
    "jwt",
    "cookie",
)

_EMAIL_RE = re.compile(r"([A-Za-z0-9._%+-])[A-Za-z0-9._%+-]*@([A-Za-z0-9.-]+\.[A-Za-z]{2,})")


def set_request_id(request_id: str | None) -> str:
    rid = request_id or new_request_id()
    _request_id.set(rid)
    return rid


def get_request_id() -> str | None:
    return _request_id.get()


def mask_email(email: str | None) -> str | None:
    """jane.doe@example.com -> j***e@example.com (never log raw addresses)."""
    if not email:
        return email
    return _EMAIL_RE.sub(lambda m: f"{m.group(1)}***@{m.group(2)}", email)


def _is_sensitive(key: str) -> bool:
    lowered = key.lower()
    return any(p in lowered for p in SENSITIVE_KEY_PATTERNS)


def redact(value: Any, _depth: int = 0) -> Any:
    """Recursively redact sensitive keys and mask e-mails in string values."""
    if _depth > 6:
        return "***"
    if isinstance(value, dict):
        return {
            k: ("***" if _is_sensitive(str(k)) else redact(v, _depth + 1)) for k, v in value.items()
        }
    if isinstance(value, (list, tuple, set)):
        return [redact(v, _depth + 1) for v in value]
    if isinstance(value, str):
        return mask_email(value)
    if isinstance(value, (int, float, bool)) or value is None:
        return value
    return str(value)


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "ts": datetime.now(timezone.utc).isoformat(timespec="milliseconds"),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
        }
        rid = get_request_id()
        if rid:
            payload["request_id"] = rid
        extra = getattr(record, "extra_fields", None)
        if isinstance(extra, dict):
            payload.update(redact(extra))
        if record.exc_info:
            payload["exc"] = self.formatException(record.exc_info)[-4000:]
        return json.dumps(payload, default=str, ensure_ascii=False)


class PlainFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        rid = get_request_id() or "-"
        base = f"{record.levelname:<7} [{rid[:8]}] {record.name}: {record.getMessage()}"
        extra = getattr(record, "extra_fields", None)
        if isinstance(extra, dict) and extra:
            base += " " + json.dumps(redact(extra), default=str)
        if record.exc_info:
            base += "\n" + self.formatException(record.exc_info)
        return base


def configure_logging(level: str = "INFO", json_logs: bool = True) -> None:
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(JsonFormatter() if json_logs else PlainFormatter())
    root = logging.getLogger()
    root.handlers = [handler]
    root.setLevel(level.upper())
    # keep noisy third-party loggers readable but quiet
    for noisy in ("httpx", "httpcore", "motor", "pymongo", "urllib3", "asyncio"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


def log_event(logger: logging.Logger, level: int, msg: str, **fields: Any) -> None:
    logger.log(level, msg, extra={"extra_fields": fields})


def log_mock(logger: logging.Logger, component: str, msg: str, **fields: Any) -> None:
    """Every mock-mode code path logs through here so '[MOCK MODE]' is greppable."""
    log_event(logger, logging.WARNING, f"[MOCK MODE] {msg}", component=component, mock_mode=True, **fields)
