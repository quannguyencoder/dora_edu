"""Tests for sentence-aligned overlap chunking."""

from __future__ import annotations

import pytest

from dora_edu.data_pipeline.text_chunker import chunk_pages, split_into_sections, split_sentences
from dora_edu.models import ParsedPage


def _page(number: int, sentence_count: int) -> ParsedPage:
    text = " ".join(f"Day la cau {i} cua trang {number}." for i in range(1, sentence_count + 1))
    return ParsedPage(page_number=number, text=text)


def test_split_sentences_splits_on_terminal_punctuation() -> None:
    assert split_sentences("Cau mot. Cau hai! Cau ba?") == ["Cau mot.", "Cau hai!", "Cau ba?"]


def test_split_sentences_treats_blank_lines_as_paragraph_breaks() -> None:
    assert split_sentences("Doan mot\n\nDoan hai") == ["Doan mot", "Doan hai"]


def test_chunks_stay_within_the_target_size() -> None:
    chunks = chunk_pages([_page(1, 40)], chunk_size=200, chunk_overlap=50)

    assert chunks
    # Only a single oversized sentence may exceed the target.
    assert all(len(chunk.text) <= 200 for chunk in chunks)


def test_consecutive_chunks_share_overlapping_text() -> None:
    chunks = chunk_pages([_page(1, 40)], chunk_size=200, chunk_overlap=60)

    assert len(chunks) > 1
    first_sentences = set(split_sentences(chunks[0].text))
    second_sentences = set(split_sentences(chunks[1].text))
    assert first_sentences & second_sentences


def test_chunk_records_the_page_range_it_spans() -> None:
    chunks = chunk_pages([_page(5, 12), _page(6, 12)], chunk_size=400, chunk_overlap=50)

    assert min(chunk.page_start for chunk in chunks) == 5
    assert max(chunk.page_end for chunk in chunks) == 6
    assert any(chunk.page_start < chunk.page_end for chunk in chunks)


def test_chunk_indices_are_contiguous_and_ordered() -> None:
    chunks = chunk_pages([_page(1, 60)], chunk_size=250, chunk_overlap=40)

    assert [chunk.chunk_index for chunk in chunks] == list(range(len(chunks)))


def test_oversized_sentence_becomes_its_own_chunk_instead_of_being_cut() -> None:
    long_sentence = "a" * 900 + "."
    chunks = chunk_pages(
        [ParsedPage(page_number=1, text=long_sentence)], chunk_size=200, chunk_overlap=50
    )

    assert len(chunks) == 1
    assert chunks[0].text == long_sentence


def test_zero_overlap_produces_disjoint_chunks() -> None:
    chunks = chunk_pages([_page(1, 40)], chunk_size=200, chunk_overlap=0)

    assert len(chunks) > 1
    assert not set(split_sentences(chunks[0].text)) & set(split_sentences(chunks[1].text))


def test_empty_input_produces_no_chunks() -> None:
    assert chunk_pages([], chunk_size=200, chunk_overlap=50) == []


def test_overlap_must_be_smaller_than_chunk_size() -> None:
    with pytest.raises(ValueError):
        chunk_pages([_page(1, 5)], chunk_size=100, chunk_overlap=100)


# --- Heading-aware section boundaries ---------------------------------------


def test_split_into_sections_starts_a_new_section_at_a_lettered_heading() -> None:
    text = "b. Tim y\nNoi dung cua muc b.\nc. Lap dan y\nNoi dung cua muc c."

    assert split_into_sections(text) == [
        "b. Tim y\nNoi dung cua muc b.",
        "c. Lap dan y\nNoi dung cua muc c.",
    ]


def test_split_into_sections_starts_a_new_section_at_a_numbered_heading() -> None:
    text = "1. TRUOC KHI VIET\nNoi dung a.\n2. VIET BAI\nNoi dung b."

    assert split_into_sections(text) == [
        "1. TRUOC KHI VIET\nNoi dung a.",
        "2. VIET BAI\nNoi dung b.",
    ]


def test_split_into_sections_recognises_an_all_caps_bai_heading() -> None:
    text = "Truoc do.\nBÀI 1\nNoi dung bai 1."

    assert split_into_sections(text) == ["Truoc do.", "BÀI 1\nNoi dung bai 1."]


def test_split_into_sections_is_a_single_section_when_there_is_no_heading() -> None:
    text = "Cau mot khong co tieu de.\nCau hai cung vay."

    assert split_into_sections(text) == [text]


def test_split_into_sections_ignores_a_long_line_that_merely_starts_with_a_number() -> None:
    long_line = "1. " + "Mot cau dai khong phai tieu de " * 5

    assert split_into_sections(long_line + "\nCau tiep theo.") == [
        long_line + "\nCau tiep theo."
    ]


def test_a_labelled_subsection_never_shares_a_chunk_with_the_one_before_it() -> None:
    # Reproduces the real bug: "c. Lap dan y" (short) used to get merged with
    # unrelated "b. Tim y" filler text ahead of it because chunking ignored
    # section headings and only cut on a raw character budget.
    page_text = (
        "b. Tim y\n"
        + "Cau tim y so mot. " * 30
        + "\nc. Lap dan y\n"
        "Mo bai: Gioi thieu cau chuyen.\n"
        "Than bai: Ke lai dien bien cua cau chuyen."
    )
    chunks = chunk_pages(
        [ParsedPage(page_number=81, text=page_text)], chunk_size=450, chunk_overlap=80
    )

    dan_y_chunks = [c for c in chunks if "Lap dan y" in c.text]
    assert len(dan_y_chunks) == 1
    assert "Mo bai" in dan_y_chunks[0].text
    assert "Than bai" in dan_y_chunks[0].text
    assert "Cau tim y so mot" not in dan_y_chunks[0].text


def test_an_oversized_section_still_gets_split_by_the_size_budget() -> None:
    page_text = "c. Lap dan y\n" + "Cau dai. " * 100
    chunks = chunk_pages(
        [ParsedPage(page_number=1, text=page_text)], chunk_size=200, chunk_overlap=40
    )

    assert len(chunks) > 1
    assert all(len(chunk.text) <= 200 for chunk in chunks)
