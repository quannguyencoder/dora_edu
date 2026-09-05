"""Tests for PDF text cleaning and boilerplate removal."""

from __future__ import annotations

import unicodedata

from dora_edu.data_pipeline.pdf_parser import clean_text, find_repeated_lines


def test_clean_text_rejoins_words_split_across_a_line_break() -> None:
    assert "chuyển" in clean_text("Bai hoc noi ve chuy-\nển dong")


def test_clean_text_drops_standalone_page_numbers() -> None:
    cleaned = clean_text("Noi dung bai hoc\n42\nTiep theo")

    assert "42" not in cleaned.split("\n")
    assert "Noi dung bai hoc" in cleaned


def test_clean_text_collapses_repeated_spaces_and_blank_lines() -> None:
    cleaned = clean_text("Mot    hai\n\n\n\nBa")

    assert "Mot hai" in cleaned
    assert "\n\n\n" not in cleaned


def test_clean_text_normalises_vietnamese_to_a_single_unicode_form() -> None:
    # PDF extractors emit decomposed (NFD) Vietnamese; equality against text
    # the bot composes itself only holds once both sides are NFC.
    composed = 'Tiếng Việt'
    decomposed = unicodedata.normalize("NFD", composed)
    assert decomposed != composed
    assert clean_text(decomposed) == composed


def test_clean_text_handles_empty_input() -> None:
    assert clean_text("") == ""


def test_find_repeated_lines_detects_running_headers() -> None:
    pages = [f"SGK TOAN 6\nNoi dung rieng cua trang {n}" for n in range(6)]

    assert "SGK TOAN 6" in find_repeated_lines(pages)
    assert "Noi dung rieng cua trang 0" not in find_repeated_lines(pages)


def test_find_repeated_lines_ignores_very_short_books() -> None:
    assert find_repeated_lines(["SGK TOAN 6\nA", "SGK TOAN 6\nB"]) == set()


def test_find_repeated_lines_ignores_long_body_paragraphs() -> None:
    body = "Day la mot doan van ban rat dai " * 4
    pages = [f"{body}\nTrang {n}" for n in range(6)]

    assert body not in find_repeated_lines(pages)
