"""Tests for turning PDF page indices into printed page numbers.

This is what decides whether *"bạn mở sách trang ... để kiểm tra lại nhé"*
sends a student to the right page, so the fallback behaviour matters as much
as the happy path: an unmeasured book must keep its old numbering rather
than be guessed at.
"""

from __future__ import annotations

import json

import pytest

from dora_edu import page_numbering
from dora_edu.models import RetrievedChunk


@pytest.fixture(autouse=True)
def _clear_offset_cache():
    """Offsets are cached process-wide; each test needs its own."""
    page_numbering.load_offsets.cache_clear()
    yield
    page_numbering.load_offsets.cache_clear()


def _write_offsets(tmp_path, offsets: dict[str, int]):
    path = tmp_path / "page_offsets.json"
    path.write_text(json.dumps({"offsets": offsets}), encoding="utf-8")
    return path


def test_a_measured_offset_is_subtracted(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(
        page_numbering, "OFFSETS_PATH", _write_offsets(tmp_path, {"SGKToan6tapmot.pdf": 1})
    )

    assert page_numbering.printed_page("SGKToan6tapmot.pdf", 72) == 71


def test_an_unmeasured_book_keeps_its_pdf_numbering(tmp_path, monkeypatch) -> None:
    # Guessing would turn citations that are currently right into wrong ones.
    monkeypatch.setattr(page_numbering, "OFFSETS_PATH", _write_offsets(tmp_path, {}))

    assert page_numbering.printed_page("SGKChuaDo.pdf", 72) == 72


def test_a_missing_offsets_file_is_not_an_error(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(page_numbering, "OFFSETS_PATH", tmp_path / "khong-co.json")

    assert page_numbering.load_offsets(tmp_path / "khong-co.json") == {}
    assert page_numbering.printed_page("SGKToan6tapmot.pdf", 10) == 10


def test_an_offset_that_would_produce_page_zero_is_ignored(tmp_path, monkeypatch) -> None:
    # Page 1 minus a 2-page offset is not a page -- the measurement is wrong,
    # so the raw index is the safer answer.
    monkeypatch.setattr(
        page_numbering, "OFFSETS_PATH", _write_offsets(tmp_path, {"A.pdf": 2})
    )

    assert page_numbering.printed_page("A.pdf", 1) == 1


def test_a_corrupt_offsets_file_falls_back_silently(tmp_path, monkeypatch) -> None:
    path = tmp_path / "page_offsets.json"
    path.write_text("{khong phai json", encoding="utf-8")
    monkeypatch.setattr(page_numbering, "OFFSETS_PATH", path)

    assert page_numbering.load_offsets(path) == {}


def test_a_negative_offset_is_rejected(tmp_path, monkeypatch) -> None:
    path = _write_offsets(tmp_path, {"A.pdf": -3})
    monkeypatch.setattr(page_numbering, "OFFSETS_PATH", path)

    assert page_numbering.load_offsets(path) == {}


# --- how it reaches the student -------------------------------------------


def _chunk(**metadata) -> RetrievedChunk:
    return RetrievedChunk(
        text="Phép nhân các số nguyên có các tính chất: Giao hoán: a . b = b . a",
        distance=0.2,
        metadata={
            "book_title": "Toán 6 - tập 1",
            "source_file": "SGKToan6tapmot.pdf",
            "page_start": 72,
            "page_end": 72,
            **metadata,
        },
    )


def test_a_citation_names_the_page_printed_in_the_book(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(
        page_numbering, "OFFSETS_PATH", _write_offsets(tmp_path, {"SGKToan6tapmot.pdf": 1})
    )

    assert _chunk().citation == "Toán 6 - tập 1, trang 71"


def test_a_citation_spanning_two_pages_shifts_both(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(
        page_numbering, "OFFSETS_PATH", _write_offsets(tmp_path, {"SGKToan6tapmot.pdf": 1})
    )

    assert _chunk(page_end=73).citation == "Toán 6 - tập 1, trang 71-72"


def test_a_chunk_without_a_source_file_is_cited_unchanged(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(
        page_numbering, "OFFSETS_PATH", _write_offsets(tmp_path, {"SGKToan6tapmot.pdf": 1})
    )
    chunk = RetrievedChunk(
        text="...", distance=0.1, metadata={"book_title": "SGK", "page_start": 5, "page_end": 5}
    )

    assert chunk.citation == "SGK, trang 5"
