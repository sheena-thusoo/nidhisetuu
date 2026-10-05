"""LLM client - Groq (Llama 3.3 70B) when configured, otherwise callers use deterministic
fallbacks and log [MOCK MODE].

This module ONLY produces text/classifications. It never computes numbers.
"""

from __future__ import annotations

import logging
import time
from typing import Protocol

from pydantic import BaseModel

from app.config import get_settings
from app.utils.logging import log_mock

logger = logging.getLogger(__name__)


class LLMError(RuntimeError):
    def __init__(self, message: str, *, provider: str = "groq"):
        super().__init__(message)
        self.provider = provider


class LLMResult(BaseModel):
    text: str
    provider: str
    model: str
    mock_mode: bool = False
    latency_ms: int = 0


class LLMClient(Protocol):
    provider: str
    model: str
    mock_mode: bool

    async def complete(
        self, system: str, user: str, *, json_mode: bool = False, temperature: float = 0.0
    ) -> LLMResult: ...


class GroqLLMClient:
    """Live Groq client (Llama 3.3 70B by default) through LangChain's ChatGroq."""

    provider = "groq"
    mock_mode = False

    def __init__(self, api_key: str, model: str, temperature: float = 0.0):
        # imported lazily so the package import stays cheap and testable
        from langchain_groq import ChatGroq

        self.model = model
        self._temperature = temperature
        self._chat = ChatGroq(api_key=api_key, model=model, temperature=temperature, timeout=30, max_retries=1)

    async def complete(
        self, system: str, user: str, *, json_mode: bool = False, temperature: float = 0.0
    ) -> LLMResult:
        from langchain_core.messages import HumanMessage, SystemMessage

        started = time.perf_counter()
        messages = [SystemMessage(content=system), HumanMessage(content=user)]
        kwargs: dict = {}
        if json_mode:
            kwargs["response_format"] = {"type": "json_object"}
        if temperature != self._temperature:
            self._chat.temperature = temperature
        try:
            response = await self._chat.ainvoke(messages, **kwargs)
        except Exception as exc:  # noqa: BLE001 - converted into a typed error for the caller
            raise LLMError(f"groq call failed: {exc}", provider=self.provider) from exc
        text = response.content if isinstance(response.content, str) else str(response.content)
        return LLMResult(
            text=text,
            provider=self.provider,
            model=self.model,
            mock_mode=False,
            latency_ms=int((time.perf_counter() - started) * 1000),
        )


_warned_mock = False


def get_llm_client() -> LLMClient | None:
    """Return a live client, or None to tell callers to use deterministic fallbacks."""
    global _warned_mock
    settings = get_settings()
    if settings.groq_enabled:
        return GroqLLMClient(settings.groq_api_key or "", settings.groq_model, settings.groq_temperature)
    if not _warned_mock:
        log_mock(
            logger,
            "groq",
            "GROQ_API_KEY not set - LLM nodes use deterministic fallbacks (regex intake, template explanation, lexical verification).",
        )
        _warned_mock = True
    return None
