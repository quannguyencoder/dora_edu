"""Shared test doubles for the DoraEdu test suite."""

from __future__ import annotations

from typing import Any

import pytest

from dora_edu.config import Settings
from dora_edu.llm.generator import AnswerGenerator, GeneratedAnswer
from dora_edu.models import RetrievedChunk, StudentProfile


@pytest.fixture
def settings() -> Settings:
    """Settings built from defaults only, ignoring any developer ``.env``."""
    return Settings(_env_file=None)


class FakeCollection:
    """Stands in for a ChromaDB collection and records the filters it receives."""

    def __init__(self, rows: list[dict[str, Any]] | None = None) -> None:
        self.rows = rows or []
        self.queries: list[dict[str, Any]] = []

    def query(self, **kwargs: Any) -> dict[str, Any]:
        """Return the rows whose metadata satisfies the recorded ``where`` clause."""
        self.queries.append(kwargs)
        matched = [row for row in self.rows if _matches(row["metadata"], kwargs.get("where"))]
        matched = matched[: kwargs.get("n_results", len(matched))]
        return {
            "documents": [[row["text"] for row in matched]],
            "metadatas": [[row["metadata"] for row in matched]],
            "distances": [[row["distance"] for row in matched]],
        }


def _matches(metadata: dict[str, Any], where: dict[str, Any] | None) -> bool:
    """Evaluate the subset of ChromaDB's ``where`` grammar the retriever emits."""
    if not where:
        return True
    if "$and" in where:
        return all(_matches(metadata, clause) for clause in where["$and"])
    for field, condition in where.items():
        expected = condition["$eq"] if isinstance(condition, dict) else condition
        if metadata.get(field) != expected:
            return False
    return True


class FakeGenerator(AnswerGenerator):
    """Records what it was asked to generate and returns a canned answer."""

    def __init__(self, answer: str = "Cau tra loi mau") -> None:
        self.answer = answer
        self.calls: list[dict[str, Any]] = []

    def generate(
        self,
        question: str,
        chunks: list[RetrievedChunk],
        profile: StudentProfile,
        history: list[dict[str, str]] | None = None,
    ) -> GeneratedAnswer:
        self.calls.append(
            {
                "question": question,
                "chunks": chunks,
                "profile": profile,
                "history": list(history or []),
            }
        )
        return GeneratedAnswer(answer=self.answer, grounded=bool(chunks))


@pytest.fixture
def textbook_rows() -> list[dict[str, Any]]:
    """Two same-topic passages that differ only by grade and subject."""
    return [
        {
            "text": "Phan so lop 6: tu so va mau so.",
            "distance": 0.10,
            "metadata": {
                "grade": 6,
                "subject": "Toán",
                "book_title": "SGK Toán 6",
                "page_start": 12,
                "page_end": 12,
            },
        },
        {
            "text": "Ham so bac nhat lop 9.",
            "distance": 0.11,
            "metadata": {
                "grade": 9,
                "subject": "Toán",
                "book_title": "SGK Toán 9",
                "page_start": 30,
                "page_end": 31,
            },
        },
        {
            "text": "Chien thang Bach Dang nam 938.",
            "distance": 0.12,
            "metadata": {
                "grade": 6,
                "subject": "Lịch sử",
                "book_title": "SGK Lịch sử 6",
                "page_start": 44,
                "page_end": 44,
            },
        },
    ]
