"""External research (Tavily) schemas - always labelled as external, never merged with internal evidence."""

from __future__ import annotations

from pydantic import BaseModel, Field

from app.schemas.rag import EvidenceVerdict


class ExternalResearchItem(BaseModel):
    title: str
    url: str
    snippet: str
    provider: str = "tavily"
    mock_mode: bool = False
    score: float | None = None
    published_date: str | None = None
    verification: EvidenceVerdict | None = None
    label: str = "external_research"  # never "internal_verified"
    caveat: str = (
        "External web result - not an authoritative internal source. Verify with the concerned "
        "department/agency before acting on it."
    )


class ExternalResearchBundle(BaseModel):
    query: str
    provider: str = "tavily"
    mock_mode: bool = False
    items: list[ExternalResearchItem] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
