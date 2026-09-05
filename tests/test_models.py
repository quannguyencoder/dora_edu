"""Tests for subject/grade normalisation and the core domain models."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from dora_edu.models import (
    RetrievedChunk,
    StudentProfile,
    TextbookMetadata,
    normalize_subject,
    strip_diacritics,
    validate_grade,
)


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("toan", "Toán"),
        (" TOAN ", "Toán"),
        ("Toán", "Toán"),
        ("lich su", "Lịch sử"),
        ("LICH SU", "Lịch sử"),
        ("Lịch Sử", "Lịch sử"),
        ("ngu van", "Ngữ văn"),
        ("van", "Ngữ văn"),
        ("dia ly", "Địa lí"),
        ("gdcd", "Giáo dục công dân"),
    ],
)
def test_normalize_subject_maps_aliases_to_one_canonical_name(raw: str, expected: str) -> None:
    assert normalize_subject(raw) == expected


def test_normalize_subject_keeps_unknown_subjects_but_collapses_whitespace() -> None:
    assert normalize_subject("  mon  hoc  moi  ") == "Mon hoc moi"


def test_normalize_subject_rejects_blank_input() -> None:
    with pytest.raises(ValueError):
        normalize_subject("   ")


def test_strip_diacritics_handles_vietnamese_d() -> None:
    assert strip_diacritics("Địa lí") == "Dia li"


@pytest.mark.parametrize("raw", [1, 12, "6", " 9 "])
def test_validate_grade_accepts_valid_levels(raw: int | str) -> None:
    assert 1 <= validate_grade(raw) <= 12


@pytest.mark.parametrize("raw", [0, 13, -1, "lop 6", "", None])
def test_validate_grade_rejects_out_of_range_values(raw: object) -> None:
    with pytest.raises(ValueError):
        validate_grade(raw)  # type: ignore[arg-type]


def test_student_profile_canonicalises_subject_on_construction() -> None:
    assert StudentProfile(grade=6, subject="  toan ").subject == "Toán"


@pytest.mark.parametrize("grade", [0, 13])
def test_student_profile_rejects_invalid_grade(grade: int) -> None:
    with pytest.raises(ValidationError):
        StudentProfile(grade=grade, subject="Toán")


def test_textbook_metadata_flattens_to_scalar_chroma_metadata() -> None:
    metadata = TextbookMetadata(
        grade=6, subject="toan", book_title="SGK Toán 6", source_file="toan6.pdf"
    )
    flat = metadata.as_chroma_metadata(page_start=10, page_end=11, chunk_index=3)

    assert flat["grade"] == 6
    assert flat["subject"] == "Toán"
    assert flat["page_start"] == 10
    assert all(isinstance(value, (str, int, float, bool)) for value in flat.values())


def test_retrieved_chunk_citation_renders_page_ranges() -> None:
    single = RetrievedChunk(
        text="x", distance=0.1, metadata={"book_title": "SGK Toán 6", "page_start": 12, "page_end": 12}
    )
    spanning = RetrievedChunk(
        text="x", distance=0.1, metadata={"book_title": "SGK Toán 6", "page_start": 12, "page_end": 14}
    )

    assert single.citation == "SGK Toán 6, trang 12"
    assert spanning.citation == "SGK Toán 6, trang 12-14"
