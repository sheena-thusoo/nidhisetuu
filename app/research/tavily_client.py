"""External research via Tavily (search API).

MOCK MODE (no TAVILY_API_KEY): returns deterministic, clearly-labelled placeholder
items so the agentic flow, the API and the UI can be exercised offline. Mock items
NEVER look like real citations and are always flagged `mock_mode=True`.

LIVE MODE: POST {base_url}/search with the API key; maps Tavily results 1:1 onto
ExternalResearchItem. All failures raise TavilyError so the gateway turns them into
a structured error envelope and the graph continues with a warning.
"""

from __future__ import annotations

import hashlib
import logging

import httpx

from app.config import Settings, get_settings
from app.schemas.research import ExternalResearchBundle, ExternalResearchItem
from app.utils.logging import log_event, log_mock

logger = logging.getLogger(__name__)


class TavilyError(RuntimeError):
    pass


_MOCK_SNIPPET = (
    "[MOCK RESULT] TAVILY_API_KEY is not configured, so this is a placeholder web-search item. "
    "Configure TAVILY_API_KEY to retrieve real external sources. "
    "NidhiSetu never merges external results into its verified internal evidence."
)


def _mock_items(query: str, max_results: int) -> list[ExternalResearchItem]:
    items: list[ExternalResearchItem] = []
    for index in range(max(1, min(max_results, 3))):
        digest = hashlib.sha256(f"{query}|{index}".encode("utf-8")).hexdigest()[:10]
        items.append(
            ExternalResearchItem(
                title=f"Mock external result {index + 1} for: {query[:60]}",
                url=f"https://mock.example.gov.in/{digest}",
                snippet=_MOCK_SNIPPET,
                provider="tavily-mock",
                mock_mode=True,
                score=None,
            )
        )
    return items


async def tavily_search(
    query: str,
    *,
    max_results: int | None = None,
    settings: Settings | None = None,
) -> ExternalResearchBundle:
    settings = settings or get_settings()
    query = (query or "").strip()
    if not query:
        raise TavilyError("empty external research query")
    limit = max_results or settings.tavily_max_results

    if not settings.tavily_enabled:
        log_mock(logger, "tavily", "TAVILY_API_KEY not set - returning deterministic mock research items")
        return ExternalResearchBundle(
            query=query,
            provider="tavily-mock",
            mock_mode=True,
            items=_mock_items(query, limit),
            warnings=["TAVILY_API_KEY not set; external results are placeholders and NOT real web citations."],
        )

    try:
        async with httpx.AsyncClient(timeout=15.0) as client:
            response = await client.post(
                f"{settings.tavily_base_url.rstrip('/')}/search",
                json={
                    "api_key": settings.tavily_api_key,
                    "query": query,
                    "search_depth": "basic",
                    "max_results": max(1, min(limit, 10)),
                },
            )
            response.raise_for_status()
            payload = response.json()
    except httpx.HTTPError as exc:
        log_event(logger, logging.WARNING, "tavily request failed", error=str(exc))
        raise TavilyError(f"tavily request failed: {exc}") from exc

    items: list[ExternalResearchItem] = []
    for result in (payload.get("results") or [])[: max(1, min(limit, 10))]:
        items.append(
            ExternalResearchItem(
                title=str(result.get("title") or "(untitled result)"),
                url=str(result.get("url") or ""),
                snippet=str(result.get("content") or "")[:800],
                provider="tavily",
                mock_mode=False,
                score=float(result["score"]) if result.get("score") is not None else None,
                published_date=result.get("published_date"),
            )
        )
    return ExternalResearchBundle(query=query, provider="tavily", mock_mode=False, items=items)
