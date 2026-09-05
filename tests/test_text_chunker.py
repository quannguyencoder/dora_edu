"""Tests for sentence-aligned overlap chunking."""

from __future__ import annotations

import pytest

from dora_edu.data_pipeline.text_chunker import chunk_pages, split_sentences
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
