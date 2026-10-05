"""Document parsing for ingestion: plain text and HTML (table-aware).

Why not `BeautifulSoup.get_text()`:
  plain get_text() flattens <table> markup into a stream of cell strings with no row
  structure, so "slab A | 13.5 | 24-36" and "slab B | 14.0 | 60-84" lose their row
  grouping and a rate can end up adjacent to the wrong slab in the chunk text.
  This parser renders every <tr> as ONE pipe-delimited line ("| cell | cell | cell |"),
  which app/rag/chunking.py then keeps together as a table block (is_table_row=True).

Headings are emitted as standalone lines so the chunker's heading detector can use
them as section boundaries.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from pathlib import Path

from app.utils.logging import log_event

logger = logging.getLogger(__name__)

_WHITESPACE_RE = re.compile(r"\s+")
_HEADINGS = {"h1", "h2", "h3", "h4", "h5", "h6"}
_SKIP_TAGS = {"script", "style", "noscript", "template"}


@dataclass
class ParsedDocument:
    text: str
    parser: str = "text"  # text | html+tables
    title: str | None = None
    scheme_id: str | None = None
    source_url: str | None = None
    meta: dict[str, str] = field(default_factory=dict)


def _clean(text: str | None) -> str:
    return _WHITESPACE_RE.sub(" ", text or "").strip()


def parse_text(path: Path) -> ParsedDocument:
    return ParsedDocument(text=path.read_text(encoding="utf-8", errors="replace"), parser="text")


def parse_html(html: str) -> ParsedDocument:
    from bs4 import BeautifulSoup  # lazy import keeps module import cheap

    soup = BeautifulSoup(html, "html.parser")
    for tag in soup.find_all(_SKIP_TAGS):
        tag.decompose()

    meta: dict[str, str] = {}
    for element in soup.find_all("meta"):
        name = element.get("name") or element.get("property")
        content = element.get("content")
        if name and content:
            meta[str(name)] = str(content)

    title = _clean(soup.title.string) if soup.title and soup.title.string else None

    body = soup.body or soup
    lines: list[str] = []

    def render(tag) -> None:
        name = (tag.name or "").lower()
        if name in _SKIP_TAGS:
            return
        if name in _HEADINGS:
            text = _clean(tag.get_text(" "))
            if text:
                lines.append(text)
            return
        if name == "table":
            lines.extend(_render_table(tag))
            return
        if name in {"p", "li", "tr", "td", "th", "pre"}:
            text = _clean(tag.get_text(" "))
            if text:
                lines.append(text)
            return
        # container elements: recurse in document order
        for child in tag.children:
            if getattr(child, "name", None):
                render(child)

    for child in body.children:
        if getattr(child, "name", None):
            render(child)

    return ParsedDocument(
        text="\n".join(line for line in lines if line),
        parser="html+tables",
        title=title,
        scheme_id=meta.get("scheme_id"),
        source_url=meta.get("source_url"),
        meta=meta,
    )


def _render_table(table) -> list[str]:
    """Render one table as consecutive pipe-delimited rows (header row included)."""
    rows: list[str] = []
    for tr in table.find_all("tr"):
        cells = [_clean(cell.get_text(" ")) for cell in tr.find_all(["th", "td"])]
        cells = [cell for cell in cells if cell]
        if cells:
            rows.append("| " + " | ".join(cells) + " |")
    return rows


def parse_document(path: Path) -> ParsedDocument:
    """Dispatch by extension; unknown extensions are treated as plain text."""
    suffix = path.suffix.lower()
    if suffix in {".html", ".htm"}:
        parsed = parse_html(path.read_text(encoding="utf-8", errors="replace"))
        log_event(
            logger,
            logging.INFO,
            "parsed html document",
            file=path.name,
            parser=parsed.parser,
            chars=len(parsed.text),
        )
        return parsed
    if suffix == ".txt":
        return parse_text(path)
    # pdf/docx etc. are intentionally not supported at this stage; Phase 4 records
    # the unsupported extension in the ingestion job instead of guessing.
    raise UnsupportedDocumentType(f"unsupported document type: {path.suffix!r}")


class UnsupportedDocumentType(ValueError):
    pass
