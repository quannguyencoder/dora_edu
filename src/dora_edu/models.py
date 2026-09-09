"""Core domain models shared across the data pipeline, RAG engine and bot layers.

These models are deliberately free of any messaging-platform or vendor type so
that they can travel unchanged between layers (rule: channel-agnostic core).
"""

from __future__ import annotations

import unicodedata
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator

MIN_GRADE = 1
MAX_GRADE = 12

#: Canonical MOET subject names keyed by their diacritic-free, lowercase form.
#: Students type subjects in many ways ("toan", "Toán", "TOAN"), but the value
#: stored in ChromaDB metadata must be byte-identical for equality filtering to
#: work, so every entry point normalises through :func:`normalize_subject`.
_SUBJECT_ALIASES: dict[str, str] = {
    "toan": "Toán",
    "nguvan": "Ngữ văn",
    "van": "Ngữ văn",
    "tiengviet": "Tiếng Việt",
    "vatli": "Vật lí",
    "vatly": "Vật lí",
    "hoahoc": "Hoá học",
    "hoa": "Hoá học",
    "sinhhoc": "Sinh học",
    "sinh": "Sinh học",
    "lichsu": "Lịch sử",
    "su": "Lịch sử",
    "diali": "Địa lí",
    "dialy": "Địa lí",
    "dia": "Địa lí",
    "tienganh": "Tiếng Anh",
    "anh": "Tiếng Anh",
    "tinhoc": "Tin học",
    "congnghe": "Công nghệ",
    "gdcd": "Giáo dục công dân",
    "giaoduccongdan": "Giáo dục công dân",
    "khoahoctunhien": "Khoa học tự nhiên",
    "kntn": "Khoa học tự nhiên",
    "lichsuvadiali": "Lịch sử và Địa lí",
    "khoahoc": "Khoa học",
    "daoduc": "Đạo đức",
    "tunhienvaxahoi": "Tự nhiên và Xã hội",
    "giaoducthechat": "Giáo dục thể chất",
    "amnhac": "Âm nhạc",
    "mithuat": "Mĩ thuật",
    "hoatdongtrainghiem": "Hoạt động trải nghiệm",
    "hoatdongtrainghiemhuongnghiep": "Hoạt động trải nghiệm, hướng nghiệp",
    "giaoduckinhtevaphapluat": "Giáo dục Kinh tế và Pháp luật",
}


def strip_diacritics(text: str) -> str:
    """Return ``text`` with Vietnamese diacritics and ``đ``/``Đ`` removed."""
    decomposed = unicodedata.normalize("NFD", text)
    without_marks = "".join(ch for ch in decomposed if not unicodedata.combining(ch))
    return without_marks.replace("đ", "d").replace("Đ", "D")


def normalize_subject(raw: str) -> str:
    """Map a free-form subject name to its canonical MOET spelling.

    Unknown subjects are not rejected — they are returned title-cased and
    whitespace-collapsed so that ingestion and retrieval still agree on a single
    stored value.

    Args:
        raw: Subject as typed by an admin or a student, e.g. ``" toan "``.

    Returns:
        The canonical subject name, e.g. ``"Toán"``.

    Raises:
        ValueError: If ``raw`` is empty or whitespace only.
    """
    collapsed = " ".join(raw.split())
    if not collapsed:
        raise ValueError("subject must not be empty")
    key = strip_diacritics(collapsed).lower().replace(" ", "")
    if key in _SUBJECT_ALIASES:
        return _SUBJECT_ALIASES[key]
    return collapsed[0].upper() + collapsed[1:]


def validate_grade(raw: int | str) -> int:
    """Coerce ``raw`` to a valid MOET grade level (1-12).

    Raises:
        ValueError: If the value is not an integer within the 1-12 range.
    """
    try:
        grade = int(raw)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"grade must be an integer, got {raw!r}") from exc
    if not MIN_GRADE <= grade <= MAX_GRADE:
        raise ValueError(f"grade must be between {MIN_GRADE} and {MAX_GRADE}, got {grade}")
    return grade


class StudentProfile(BaseModel):
    """The grade/subject pair that scopes every retrieval for one student.

    Both fields are mandatory: the RAG engine refuses to query ChromaDB without
    them, which is what keeps a 6th-grade student from ever seeing 9th-grade
    content.
    """

    model_config = ConfigDict(frozen=True)

    grade: int = Field(ge=MIN_GRADE, le=MAX_GRADE)
    subject: str = Field(min_length=1)

    @field_validator("subject")
    @classmethod
    def _canonicalise_subject(cls, value: str) -> str:
        return normalize_subject(value)


class TextbookMetadata(BaseModel):
    """Provenance attached to every chunk stored in ChromaDB."""

    model_config = ConfigDict(frozen=True)

    grade: int = Field(ge=MIN_GRADE, le=MAX_GRADE)
    subject: str = Field(min_length=1)
    book_title: str = Field(min_length=1)
    source_file: str = Field(min_length=1)

    @field_validator("subject")
    @classmethod
    def _canonicalise_subject(cls, value: str) -> str:
        return normalize_subject(value)

    def as_chroma_metadata(self, page_start: int, page_end: int, chunk_index: int) -> dict[str, Any]:
        """Build the flat metadata dict ChromaDB stores next to a chunk.

        ChromaDB only accepts scalar metadata values, so this returns a flat
        mapping of primitives.
        """
        return {
            "grade": self.grade,
            "subject": self.subject,
            "book_title": self.book_title,
            "source_file": self.source_file,
            "page_start": page_start,
            "page_end": page_end,
            "chunk_index": chunk_index,
        }


class ParsedPage(BaseModel):
    """One cleaned page extracted from a textbook PDF."""

    model_config = ConfigDict(frozen=True)

    page_number: int = Field(ge=1)
    text: str


class TextChunk(BaseModel):
    """A retrieval-sized slice of a textbook, kept aligned to page boundaries."""

    model_config = ConfigDict(frozen=True)

    text: str = Field(min_length=1)
    chunk_index: int = Field(ge=0)
    page_start: int = Field(ge=1)
    page_end: int = Field(ge=1)


class RetrievedChunk(BaseModel):
    """A chunk returned by the retriever, with its similarity distance."""

    model_config = ConfigDict(frozen=True)

    text: str
    distance: float
    metadata: dict[str, Any]

    @property
    def citation(self) -> str:
        """Return a short Vietnamese source label shown to the student."""
        book = self.metadata.get("book_title", "SGK")
        page_start = self.metadata.get("page_start")
        page_end = self.metadata.get("page_end")
        if page_start is None:
            return str(book)
        if page_end is None or page_end == page_start:
            return f"{book}, trang {page_start}"
        return f"{book}, trang {page_start}-{page_end}"
