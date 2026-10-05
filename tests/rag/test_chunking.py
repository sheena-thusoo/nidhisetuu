"""Chunker + parser tests: heading preservation, table row grouping, stable IDs, determinism."""

from __future__ import annotations

from pathlib import Path

import pytest

from app.rag.chunking import chunk_text, stable_chunk_id
from app.rag.parsing import (
    ParsedDocument,
    UnsupportedDocumentType,
    parse_document,
    parse_html,
)

DATA_DIR = Path("data/sample_documents")


def test_known_text_produces_expected_chunk_structure():
    text = "\n".join(
        [
            "Eligibility Criteria:",
            "The applicant must belong to the Scheduled Caste category.",
            "Annual family income must not exceed the notified ceiling for the applicant's location.",
            "",
            "Financial Parameters:",
            "The interest rate for the loan is ten percent per annum on a reducing balance basis.",
        ]
    )
    chunks = chunk_text(text, chunk_size=200, overlap=50)
    assert len(chunks) >= 2
    joined = " ".join(c.text for c in chunks)
    assert "Eligibility Criteria" in joined, "headings must never be dropped"
    assert "Financial Parameters" in joined
    # sections recorded on the chunk that contains the heading block
    assert any(c.section == "Eligibility Criteria" for c in chunks)


def test_chunks_are_deterministic():
    text = "Line one of the document.\nLine two of the document.\nLine three."
    a = chunk_text(text, chunk_size=50, overlap=10)
    b = chunk_text(text, chunk_size=50, overlap=10)
    assert [c.text for c in a] == [c.text for c in b]
    assert [c.chunk_index for c in a] == [c.chunk_index for c in b]


def test_pipe_table_rows_stay_together_in_one_block():
    text = "\n".join(
        [
            "1. Rate Table",
            "| Loan slab | Interest rate | Tenure |",
            "| Up to 200000 | 13.5 | 24 - 36 |",
            "| 200001 to 500000 | 13.5 | 36 - 60 |",
            "| 500001 to 1000000 | 14.0 | 60 - 84 |",
        ]
    )
    chunks = chunk_text(text, chunk_size=900, overlap=150)
    table_chunks = [c for c in chunks if c.is_table_row]
    assert table_chunks, "pipe rows must be recognised as table blocks"
    full_row = "| 200001 to 500000 | 13.5 | 36 - 60 |"
    assert any(full_row in c.text for c in table_chunks), "each pipe row must survive intact"


def test_overlap_never_starts_mid_line():
    text = "\n".join(
        f"Paragraph line {i} with some filler content to give the document length." for i in range(40)
    )
    chunks = chunk_text(text, chunk_size=300, overlap=80)
    for chunk in chunks:
        assert not chunk.text.startswith(" "), "chunks start at a line boundary"
    for chunk in chunks[1:]:
        first_line = chunk.text.split("\n", 1)[0]
        assert first_line.endswith("."), "overlap tail snapped to a sentence/line end"


def test_stable_chunk_id_is_content_addressed_and_32_hex():
    id1 = stable_chunk_id("demo_x", 1, "a" * 64, 0)
    id2 = stable_chunk_id("demo_x", 1, "a" * 64, 0)
    id3 = stable_chunk_id("demo_x", 1, "b" * 64, 0)
    id4 = stable_chunk_id("demo_x", 2, "a" * 64, 0)
    assert id1 == id2
    assert id1 != id3  # content change -> new id
    assert id1 != id4  # version change -> new id
    assert len(id1) == 32
    int(id1, 16)  # hex


def test_max_chunks_cap():
    text = "\n\n".join(f"Block {i} with enough content to be its own chunk." for i in range(50))
    chunks = chunk_text(text, chunk_size=40, overlap=5, max_chunks=5)
    assert len(chunks) <= 5


def test_empty_text_yields_no_chunks():
    assert chunk_text("") == []
    assert chunk_text("\n \n  \n") == []


# ------------------------------------------------------------------ parsing


def test_parse_html_renders_table_rows_as_pipe_lines():
    parsed = parse_html(
        """
        <html><head><meta name="scheme_id" content="demo_x"/>
        <title>Doc X</title></head><body>
        <h2>3. Rate table</h2>
        <table>
          <tr><th>Slab</th><th>Rate</th></tr>
          <tr><td>Up to 200000</td><td>13.5</td></tr>
          <tr><td>Above 200000</td><td>14.0</td></tr>
        </table>
        <script>evil()</script><style>.x{}</style>
        <p>Processing fee: 1.5%.</p>
        </body></html>
        """
    )
    lines = parsed.text.split("\n")
    assert "| Slab | Rate |" in lines
    assert "| Up to 200000 | 13.5 |" in lines
    assert "| Above 200000 | 14.0 |" in lines
    assert parsed.scheme_id == "demo_x"
    assert parsed.title == "Doc X"
    assert parsed.parser == "html+tables"
    assert "evil()" not in parsed.text and ".x{}" not in parsed.text
    assert "3. Rate table" in lines  # heading preserved as its own line


def test_parse_html_nested_lists_and_paragraphs():
    parsed = parse_html("<body><ul><li>First item</li><li>Second item</li></ul><div><p>Para</p></div></body>")
    lines = parsed.text.split("\n")
    assert lines == ["First item", "Second item", "Para"]


def test_demo_term_loan_html_fixture_end_to_end():
    parsed = parse_document(DATA_DIR / "demo_term_loan.html")
    assert parsed.scheme_id == "demo_term_loan"
    assert parsed.source_url and "nbcfdc" in parsed.source_url
    table_lines = [l for l in parsed.text.split("\n") if l.startswith("|")]
    assert len(table_lines) == 4  # header + 3 slabs
    assert any("14.0" in l and "60 - 84" in l for l in table_lines)
    # and the chunker keeps those rows intact
    chunks = chunk_text(parsed.text, chunk_size=900, overlap=150)
    assert any("13.5" in c.text and "24 - 36" in c.text for c in chunks)


def test_parse_text_passthrough_and_unsupported_type(tmp_path):
    parsed = parse_document(DATA_DIR / "demo_nfsdc_education_loan.txt")
    assert parsed.parser == "text"
    assert "81,000" in parsed.text or "81000" in parsed.text
    unsupported = tmp_path / "guideline.rtf"
    unsupported.write_text("{\\rtf1 fake}", encoding="utf-8")
    with pytest.raises(UnsupportedDocumentType):
        parse_document(unsupported)  # pdf/docx etc. not supported at this stage


def test_parsed_document_defaults():
    doc = ParsedDocument(text="hello")
    assert doc.parser == "text"
    assert doc.scheme_id is None and doc.title is None and doc.meta == {}
