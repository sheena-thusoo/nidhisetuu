"""Tavily client tests (Phase 5): mock mode + live-path mapping with fakes."""

from __future__ import annotations

import pytest

from app.config import get_settings
from app.research.tavily_client import TavilyError, tavily_search


async def test_mock_mode_returns_deterministic_labelled_items():
    bundle = await tavily_search("nsfdc education loan interest rate")
    assert bundle.mock_mode is True
    assert bundle.provider == "tavily-mock"
    assert len(bundle.items) == 2
    first = bundle.items[0]
    assert first.label == "external_research"
    assert first.mock_mode is True
    assert "[MOCK RESULT]" in first.snippet
    assert first.caveat  # external results always carry the caveat
    # determinism: identical query -> identical urls
    again = await tavily_search("nsfdc education loan interest rate")
    assert [i.url for i in again.items] == [i.url for i in bundle.items]
    # different query -> different urls
    other = await tavily_search("different query entirely")
    assert [i.url for i in other.items] != [i.url for i in bundle.items]
    # the query is recorded and the warning is explicit about mock status
    assert bundle.query == "nsfdc education loan interest rate"
    assert any("TAVILY_API_KEY not set" in w for w in bundle.warnings)


async def test_mock_mode_clamps_max_results():
    bundle = await tavily_search("test query", max_results=7)
    assert len(bundle.items) == 3  # hard cap for mock mode


async def test_empty_query_raises():
    with pytest.raises(TavilyError):
        await tavily_search("   ")


class _FakeResponse:
    def __init__(self, payload, status_code=200):
        self._payload = payload
        self.status_code = status_code

    def raise_for_status(self):
        if self.status_code >= 400:
            import httpx

            raise httpx.HTTPStatusError("boom", request=None, response=None)

    def json(self):
        return self._payload


class _FakeAsyncClient:
    def __init__(self, payload=None, error: Exception | None = None, **kwargs):
        self._payload = payload
        self._error = error

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc_info):
        return False

    async def post(self, url, json=None):
        if self._error:
            raise self._error
        assert json["api_key"], "live path must send the configured API key"
        assert url.endswith("/search")
        return _FakeResponse(self._payload)


async def test_live_path_maps_tavily_results(monkeypatch):
    import app.research.tavily_client as mod

    payload = {
        "results": [
            {"title": "NSFDC official page", "url": "https://www.nsfdc.nic.in/", "content": "Loan details...",
             "score": 0.91, "published_date": "2026-08-01"},
            {"title": "Second", "url": "https://example.org/2", "content": "...", "score": 0.5},
        ]
    }
    monkeypatch.setattr(mod.httpx, "AsyncClient", lambda **kw: _FakeAsyncClient(payload=payload))
    settings = get_settings().model_copy(update={"tavily_api_key": "tvly-fake-key"})
    bundle = await tavily_search("nsfdc loan", max_results=2, settings=settings)
    assert bundle.mock_mode is False
    assert bundle.provider == "tavily"
    assert [i.title for i in bundle.items] == ["NSFDC official page", "Second"]
    assert bundle.items[0].score == 0.91
    assert bundle.items[0].label == "external_research"


async def test_live_path_http_error_raises_tavily_error(monkeypatch):
    import httpx

    import app.research.tavily_client as mod

    monkeypatch.setattr(
        mod.httpx, "AsyncClient", lambda **kw: _FakeAsyncClient(error=httpx.ConnectError("network down"))
    )
    settings = get_settings().model_copy(update={"tavily_api_key": "tvly-fake-key"})
    with pytest.raises(TavilyError, match="network down"):
        await tavily_search("nsfdc loan", settings=settings)


@pytest.mark.live_tavily
async def test_live_tavily_real_api():
    settings = get_settings()
    if not settings.tavily_enabled:
        pytest.skip("TAVILY_API_KEY not configured (mock mode: skipped, never counted as a pass)")
    bundle = await tavily_search("NSFDC education loan eligibility", max_results=2)
    assert bundle.mock_mode is False
    assert bundle.items, "live Tavily should return at least one result for a real query"
    assert all(item.label == "external_research" for item in bundle.items)
