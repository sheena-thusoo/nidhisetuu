"""Section-aware chunking with stable, content-addressed chunk IDs.

Design notes:
  * headings are detected but NEVER dropped - they are kept as the first line of the
    following chunk, so a chunk always carries its own section context;
  * pipe-delimited table rows (produced by the HTML table parser) stay in one block,
    and any chunk containing table rows is flagged is_table_row=True;
  * chunk IDs are sha256(scheme_id|version|content_hash|chunk_index) so re-ingesting the
    same document version overwrites instead of duplicating.
"""

from __future__ import annotations

import hashlib
import re

from app.schemas.rag import RawChunk

_NUMBERED_HEADING = re.compile(r"^(\d+(?:\.\d+)*)[.)]?\s+(.{2,80})$")


def _is_heading(line: str) -> str | None:
    stripped = line.strip()
    if not stripped or len(stripped) > 90:
        return None
    if stripped.endswith((".", ",", ";")):
        return None
    numbered = _NUMBERED_HEADING.match(stripped)
    if numbered:
        title = numbered.group(2).strip().rstrip(":").strip()
        if re.search(r"\d", title) or len(title.split()) > 8:
            return None
        return f"{numbered.group(1)} {title}"
    if stripped.endswith(":") and len(stripped.split()) <= 12 and not re.search(r"\d", stripped):
        return stripped[:-1].strip()
    words = stripped.split()
    if 2 <= len(words) <= 8 and stripped.isupper():
        return stripped.title()
    return None


def _is_table_row(line: str) -> bool:
    stripped = line.strip()
    return stripped.startswith("|") and stripped.endswith("|") and stripped.count("|") >= 3


def chunk_text(
    text: str,
    *,
    chunk_size: int = 900,
    overlap: int = 150,
    max_chunks: int = 400,
) -> list[RawChunk]:
    normalized = text.replace("\r\n", "\n").replace("\r", "\n")
    lines = normalized.split("\n")

    blocks: list[tuple[str, bool, str | None]] = []
    current_section: str | None = None
    buffer: list[str] = []
    buffer_table = False
    buffer_section: str | None = None

    def flush() -> None:
        nonlocal buffer
        text_block = "\n".join(buffer).strip()
        if text_block:
            section = buffer_section
            if section is None and text_block.lstrip().startswith("scheme_id"):
                section = "Document metadata"
            blocks.append((text_block, buffer_table, section))
        buffer = []

    for line in lines:
        heading = _is_heading(line)
        if heading:
            flush()
            current_section = heading
            buffer = [line.strip()]  # heading stays in the content
            buffer_table = False
            buffer_section = current_section
            continue
        table = _is_table_row(line)
        if buffer and table != buffer_table:
            flush()
            buffer_section = current_section
        if not buffer:
            buffer_table = table
            buffer_section = current_section
        if not line.strip():
            if buffer and not buffer_table:
                flush()
            continue
        buffer.append(line)
    flush()

    chunks: list[RawChunk] = []
    index = 0
    cursor = 0
    current_text = ""
    current_section_name: str | None = None
    current_is_table = False
    current_start = 0

    def emit() -> None:
        nonlocal index, current_text, current_section_name, current_is_table, current_start
        if not current_text.strip():
            current_text = ""
            return
        chunks.append(
            RawChunk(
                chunk_index=index,
                text=current_text.strip(),
                section=current_section_name,
                is_table_row=current_is_table,
                char_start=current_start,
                char_end=current_start + len(current_text),
            )
        )
        index += 1
        consumed = len(current_text)
        tail = current_text[-overlap:] if overlap > 0 else ""
        # never start the next chunk in the middle of a line (e.g. half a table row):
        # snap the overlap tail forward to the next line boundary
        newline_pos = tail.find("\n")
        if newline_pos != -1:
            tail = tail[newline_pos + 1 :]
        current_start = current_start + consumed - len(tail)
        current_text = tail

    for block_text, is_table, section in blocks:
        if index >= max_chunks:
            break
        if not current_text.strip():
            current_section_name = section
            current_is_table = is_table
            current_start = cursor
        else:
            if current_section_name is None and section is not None:
                current_section_name = section
            section_changed = section is not None and section != current_section_name
            over_limit = len(current_text) + len(block_text) + 1 > chunk_size
            will_emit = over_limit or (section_changed and len(current_text) >= chunk_size // 2)
            if will_emit:
                # emit BEFORE switching the section label, otherwise this chunk is
                # mislabelled with the NEXT section's heading
                emit()
                current_section_name = section
                current_is_table = is_table
            elif section_changed:
                # small block merged in: the chunk now contains this section too
                current_section_name = section
            if is_table:
                # one meta-flag for the whole chunk: it CONTAINS pipe-table rows
                # (heading + table mixed chunks are still table-bearing citations)
                current_is_table = True
        current_text += ("\n" if current_text and not current_text.endswith("\n") else "") + block_text
        cursor += len(block_text) + 1
    emit()
    return chunks[:max_chunks]


def stable_chunk_id(scheme_id: str, scheme_version: int, content_hash: str, chunk_index: int) -> str:
    digest = hashlib.sha256(
        f"{scheme_id}|v{scheme_version}|{content_hash}|{chunk_index}".encode("utf-8")
    ).hexdigest()
    return digest[:32]
