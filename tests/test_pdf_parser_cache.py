"""Tests for OCR result caching in pdf_parser.py.

Re-chunking the same corpus with a different chunk size should never re-run
OCR: these tests exercise the cache round-trip and the cache-hit short
circuit in ``parse_pdf`` directly, without needing Tesseract installed.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from dora_edu.data_pipeline import pdf_parser as pdf_parser_module
from dora_edu.data_pipeline.pdf_parser import (
    _cache_path_for,
    _load_cached_pages,
    _save_cached_pages,
    parse_pdf,
)
from dora_edu.models import ParsedPage


def test_cache_path_mirrors_the_grade_folder_layout(tmp_path: Path) -> None:
    pdf_path = tmp_path / "raw_pdfs" / "9" / "SGKToan9tapmot.pdf"
    cache_dir = tmp_path / "processed"

    assert _cache_path_for(pdf_path, cache_dir) == cache_dir / "9" / "SGKToan9tapmot.json"


def test_save_then_load_round_trips_pages(tmp_path: Path) -> None:
    cache_path = tmp_path / "9" / "SGKToan9tapmot.json"
    pages = [ParsedPage(page_number=1, text="Trang mot"), ParsedPage(page_number=2, text="Trang hai")]

    _save_cached_pages(cache_path, pages)
    loaded = _load_cached_pages(cache_path)

    assert loaded == pages


def test_load_returns_none_when_the_cache_file_is_missing(tmp_path: Path) -> None:
    assert _load_cached_pages(tmp_path / "missing.json") is None


def test_load_returns_none_for_a_corrupt_cache_file_instead_of_raising(tmp_path: Path) -> None:
    cache_path = tmp_path / "bad.json"
    cache_path.write_text("{not valid json", encoding="utf-8")

    assert _load_cached_pages(cache_path) is None


class _FakePage:
    """Stands in for a pypdf page with a real (non-scanned) text layer."""

    def __init__(self, text: str) -> None:
        self._text = text

    def extract_text(self) -> str:
        return self._text


class _FakeReader:
    def __init__(self, _path: str) -> None:
        self.pages = [_FakePage("Noi dung trang mot"), _FakePage("Noi dung trang hai")]


def _fake_pdf(tmp_path: Path) -> Path:
    pdf_path = tmp_path / "raw_pdfs" / "6" / "SGKToan6tapmot.pdf"
    pdf_path.parent.mkdir(parents=True)
    pdf_path.write_bytes(b"%PDF-fake")
    return pdf_path


def test_parse_pdf_writes_a_cache_entry_on_a_cold_run(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(pdf_parser_module, "PdfReader", _FakeReader)
    pdf_path = _fake_pdf(tmp_path)
    cache_dir = tmp_path / "processed"

    pages = parse_pdf(pdf_path, cache_dir=cache_dir)

    assert [p.text for p in pages] == ["Noi dung trang mot", "Noi dung trang hai"]
    assert (cache_dir / "6" / "SGKToan6tapmot.json").is_file()


def test_parse_pdf_skips_re_reading_the_pdf_on_a_cache_hit(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(pdf_parser_module, "PdfReader", _FakeReader)
    pdf_path = _fake_pdf(tmp_path)
    cache_dir = tmp_path / "processed"

    first = parse_pdf(pdf_path, cache_dir=cache_dir)

    def _boom(_path: str) -> None:
        raise AssertionError("PdfReader must not be called again on a cache hit")

    monkeypatch.setattr(pdf_parser_module, "PdfReader", _boom)
    second = parse_pdf(pdf_path, cache_dir=cache_dir)

    assert second == first


def test_parse_pdf_without_a_cache_dir_never_touches_disk_cache(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(pdf_parser_module, "PdfReader", _FakeReader)
    pdf_path = _fake_pdf(tmp_path)

    parse_pdf(pdf_path)

    assert not (tmp_path / "processed").exists()


def test_a_corrupt_cache_file_is_treated_as_a_miss_and_overwritten(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(pdf_parser_module, "PdfReader", _FakeReader)
    pdf_path = _fake_pdf(tmp_path)
    cache_dir = tmp_path / "processed"
    cache_path = cache_dir / "6" / "SGKToan6tapmot.json"
    cache_path.parent.mkdir(parents=True)
    cache_path.write_text("not json", encoding="utf-8")

    pages = parse_pdf(pdf_path, cache_dir=cache_dir)

    assert [p.text for p in pages] == ["Noi dung trang mot", "Noi dung trang hai"]
    assert _load_cached_pages(cache_path) == pages
