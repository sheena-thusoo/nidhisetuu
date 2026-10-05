"""ToolGateway behaviour: validation, timeouts, error envelopes, unknown tools, key-only logging."""

from __future__ import annotations

import asyncio

import pytest
from pydantic import BaseModel, Field

from app.tools.gateway import ToolGateway


class _EchoInput(BaseModel):
    text: str
    count: int = Field(default=1, ge=1, le=5)


class _EchoOutput(BaseModel):
    text: str
    count: int


def _gateway(timeout: float = 15.0) -> ToolGateway:
    gw = ToolGateway(default_timeout_seconds=timeout)
    gw.register(
        name="echo",
        description="echo tool for tests",
        input_model=_EchoInput,
        output_model=_EchoOutput,
        handler=lambda payload: _echo(payload),
    )
    return gw


async def _echo(payload: _EchoInput) -> _EchoOutput:
    return _EchoOutput(text=payload.text, count=payload.count)


async def test_successful_call_validates_and_returns_output():
    result = await _gateway().call("echo", {"text": "hi", "count": 2}, request_id="t1")
    assert result.ok is True
    assert isinstance(result.output, _EchoOutput)
    assert result.output.text == "hi"
    assert result.output.count == 2
    assert result.mock_mode is False
    assert result.error is None


async def test_invalid_input_returns_envelope_not_exception():
    result = await _gateway().call("echo", {"text": "hi", "count": 99}, request_id="t2")
    assert result.ok is False
    assert result.output is None
    assert result.error is not None
    assert result.error.error_type == "invalid_input"
    assert "count" in result.error.message


async def test_missing_required_field_is_invalid_input():
    result = await _gateway().call("echo", {"count": 1}, request_id="t3")
    assert result.ok is False
    assert result.error.error_type == "invalid_input"
    assert "text" in result.error.message


async def test_unknown_tool_returns_typed_error():
    result = await _gateway().call("nope", {}, request_id="t4")
    assert result.ok is False
    assert result.error.error_type == "unknown_tool"
    assert "nope" in result.error.message


async def test_handler_exception_becomes_structured_envelope():
    gw = ToolGateway(default_timeout_seconds=5)

    class _In(BaseModel):
        x: int = 0

    async def boom(_payload):
        raise RuntimeError("kaboom")

    gw.register(name="boom", description="d", input_model=_In, handler=boom)
    result = await gw.call("boom", {"x": 1}, request_id="t5")
    assert result.ok is False
    assert result.error.error_type == "RuntimeError"
    assert "kaboom" in result.error.message
    # the raw exception never escapes the gateway
    assert result.output is None


async def test_timeout_is_returned_as_retryable_error():
    gw = ToolGateway(default_timeout_seconds=5)

    class _In(BaseModel):
        x: int = 0

    async def slow(_payload):
        await asyncio.sleep(0.2)

    gw.register(name="slow", description="d", input_model=_In, handler=slow, timeout_seconds=0.05)
    result = await gw.call("slow", {}, request_id="t6")
    assert result.ok is False
    assert result.error.error_type == "timeout"
    assert result.error.retryable is True


async def test_trace_record_contains_keys_only_and_no_values():
    result = await _gateway().call("echo", {"text": "SECRET-INCOME-VALUE", "count": 1}, request_id="t7")
    record = result.record
    assert record.tool == "echo"
    assert record.ok is True
    assert record.request_id == "t7"
    assert set(record.input_keys) == {"text", "count"}
    # the recorded trace must never contain the input VALUES
    dumped = record.model_dump_json()
    assert "SECRET-INCOME-VALUE" not in dumped


async def test_duplicate_registration_rejected():
    gw = _gateway()
    with pytest.raises(ValueError, match="already registered"):
        gw.register(
            name="echo",
            description="dup",
            input_model=_EchoInput,
            handler=_echo,
        )


async def test_tool_names_sorted_and_descriptor_shape():
    gw = _gateway()
    gw.register(
        name="aaa",
        description="first alphabetically",
        input_model=_EchoInput,
        handler=_echo,
        never_llm=True,
        mock_mode=True,
    )
    assert gw.tool_names() == ["aaa", "echo"]
    descriptor = {d["name"]: d for d in gw.descriptors()}["aaa"]
    assert descriptor["deterministic"] is True
    assert descriptor["mock_mode"] is True
    assert descriptor["timeout_seconds"] == 15.0


async def test_accepts_already_validated_pydantic_payload():
    result = await _gateway().call("echo", _EchoInput(text="obj", count=3), request_id="t8")
    assert result.ok is True
    assert result.output.count == 3
