"""The single tool gateway.

Every tool call made by the agentic layer passes through `ToolGateway.call`, which adds:
  * Pydantic input validation + Pydantic output validation;
  * a hard timeout (asyncio.wait_for);
  * try/except -> structured error envelopes (never a raw traceback to the agents);
  * a request-id-stamped log line and a ToolCallRecord for the response trace.

Input VALUES are never logged (they may contain applicant details) - only the key names.
"""

from __future__ import annotations

import asyncio
import logging
import time
from datetime import datetime, timezone
from typing import Any, Awaitable, Callable

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from app.utils.logging import get_request_id, log_event

logger = logging.getLogger(__name__)


class ToolError(BaseModel):
    error_type: str
    message: str
    retryable: bool = False


class ToolCallRecord(BaseModel):
    tool: str
    request_id: str | None = None
    ok: bool = False
    duration_ms: int = 0
    mock_mode: bool = False
    input_keys: list[str] = Field(default_factory=list)
    error: ToolError | None = None
    started_at: str = ""


class ToolResult(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    ok: bool
    tool: str
    request_id: str | None = None
    duration_ms: int = 0
    mock_mode: bool = False
    output: Any | None = None
    error: ToolError | None = None
    record: ToolCallRecord

    def trace_entry(self) -> dict[str, Any]:
        return {
            "tool": self.tool,
            "ok": self.ok,
            "duration_ms": self.duration_ms,
            "mock_mode": self.mock_mode,
            "error": self.error.model_dump() if self.error else None,
        }


ToolHandler = Callable[[BaseModel], Awaitable[Any]]


class ToolDefinition:
    def __init__(
        self,
        *,
        name: str,
        description: str,
        input_model: type[BaseModel],
        output_model: type[BaseModel] | None,
        handler: ToolHandler,
        timeout_seconds: float,
        mock_mode: bool = False,
        never_llm: bool = False,
    ):
        self.name = name
        self.description = description
        self.input_model = input_model
        self.output_model = output_model
        self.handler = handler
        self.timeout_seconds = timeout_seconds
        self.mock_mode = mock_mode
        self.never_llm = never_llm  # marks deterministic tools for documentation/UI

    def descriptor(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "input_schema": self.input_model.model_json_schema()["title"],
            "timeout_seconds": self.timeout_seconds,
            "mock_mode": self.mock_mode,
            "deterministic": self.never_llm,
        }


class ToolGateway:
    def __init__(self, *, default_timeout_seconds: float = 15.0):
        self._tools: dict[str, ToolDefinition] = {}
        self.default_timeout_seconds = default_timeout_seconds

    def register(
        self,
        *,
        name: str,
        description: str,
        input_model: type[BaseModel],
        handler: ToolHandler,
        output_model: type[BaseModel] | None = None,
        timeout_seconds: float | None = None,
        mock_mode: bool = False,
        never_llm: bool = False,
    ) -> None:
        if name in self._tools:
            raise ValueError(f"tool {name!r} is already registered")
        self._tools[name] = ToolDefinition(
            name=name,
            description=description,
            input_model=input_model,
            output_model=output_model,
            handler=handler,
            timeout_seconds=timeout_seconds or self.default_timeout_seconds,
            mock_mode=mock_mode,
            never_llm=never_llm,
        )

    def tool_names(self) -> list[str]:
        return sorted(self._tools)

    def descriptors(self) -> list[dict[str, Any]]:
        return [self._tools[name].descriptor() for name in self.tool_names()]

    async def call(
        self, name: str, payload: BaseModel | dict[str, Any] | None = None, *, request_id: str | None = None
    ) -> ToolResult:
        request_id = request_id or get_request_id()
        started_at = datetime.now(timezone.utc).isoformat(timespec="milliseconds")
        started = time.perf_counter()

        def finish(
            *,
            ok: bool,
            output: Any = None,
            error: ToolError | None = None,
            input_keys: list[str] | None = None,
            mock_mode: bool = False,
        ) -> ToolResult:
            duration_ms = int((time.perf_counter() - started) * 1000)
            record = ToolCallRecord(
                tool=name,
                request_id=request_id,
                ok=ok,
                duration_ms=duration_ms,
                mock_mode=mock_mode,
                input_keys=input_keys or [],
                error=error,
                started_at=started_at,
            )
            log_event(
                logger,
                logging.INFO if ok else logging.WARNING,
                "tool call",
                tool=name,
                request_id=request_id,
                ok=ok,
                duration_ms=duration_ms,
                mock_mode=mock_mode,
                input_keys=record.input_keys,
                error_type=error.error_type if error else None,
            )
            return ToolResult(
                ok=ok,
                tool=name,
                request_id=request_id,
                duration_ms=duration_ms,
                mock_mode=mock_mode,
                output=output,
                error=error,
                record=record,
            )

        definition = self._tools.get(name)
        if definition is None:
            return finish(
                ok=False,
                error=ToolError(error_type="unknown_tool", message=f"tool {name!r} is not registered"),
            )

        if isinstance(payload, BaseModel):
            if isinstance(payload, definition.input_model):
                validated = payload
                input_keys = list(payload.model_dump().keys())
            else:
                try:
                    data = payload.model_dump()
                except Exception:  # noqa: BLE001
                    data = {}
                try:
                    validated = definition.input_model.model_validate(data)
                    input_keys = list(data.keys())
                except ValidationError as exc:
                    return finish(
                        ok=False,
                        error=ToolError(error_type="invalid_input", message=str(exc)[:400]),
                        input_keys=list(data.keys()),
                    )
        else:
            data = payload or {}
            try:
                validated = definition.input_model.model_validate(data)
                input_keys = list(data.keys())
            except ValidationError as exc:
                return finish(ok=False, error=ToolError(error_type="invalid_input", message=str(exc)[:400]), input_keys=list(data.keys()))

        try:
            raw_output = await asyncio.wait_for(definition.handler(validated), timeout=definition.timeout_seconds)
        except asyncio.TimeoutError:
            return finish(
                ok=False,
                error=ToolError(
                    error_type="timeout", message=f"tool {name!r} exceeded {definition.timeout_seconds}s", retryable=True
                ),
                input_keys=input_keys,
                mock_mode=definition.mock_mode,
            )
        except Exception as exc:  # noqa: BLE001 - structured envelope instead of a traceback
            return finish(
                ok=False,
                error=ToolError(error_type=type(exc).__name__, message=str(exc)[:400], retryable=False),
                input_keys=input_keys,
                mock_mode=definition.mock_mode,
            )

        output: Any = raw_output
        if definition.output_model is not None:
            try:
                if isinstance(raw_output, definition.output_model):
                    output = raw_output
                else:
                    output = definition.output_model.model_validate(raw_output)
            except ValidationError as exc:
                return finish(
                    ok=False,
                    error=ToolError(error_type="invalid_output", message=str(exc)[:400]),
                    input_keys=input_keys,
                    mock_mode=definition.mock_mode,
                )
        return finish(ok=True, output=output, input_keys=input_keys, mock_mode=definition.mock_mode)
