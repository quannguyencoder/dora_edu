"""Tests for the mandatory grade/subject isolation rule."""

from __future__ import annotations

import pytest

from dora_edu.models import StudentProfile
from dora_edu.rag_engine import retriever as retriever_module
from dora_edu.rag_engine.retriever import Retriever, build_metadata_filter
from tests.conftest import FakeCollection


@pytest.fixture
def collection(textbook_rows) -> FakeCollection:
    return FakeCollection(textbook_rows)


@pytest.fixture
def retriever(monkeypatch, collection, settings) -> Retriever:
    monkeypatch.setattr(retriever_module, "get_collection", lambda *a, **k: collection)
    return Retriever(settings)


def test_metadata_filter_always_constrains_both_grade_and_subject() -> None:
    where = build_metadata_filter(StudentProfile(grade=6, subject="Toán"))

    clauses = {list(clause)[0]: clause for clause in where["$and"]}
    assert clauses["grade"] == {"grade": {"$eq": 6}}
    assert clauses["subject"] == {"subject": {"$eq": "Toán"}}


def test_every_query_sent_to_chromadb_carries_the_filter(retriever, collection) -> None:
    retriever.retrieve("Phan so la gi?", StudentProfile(grade=6, subject="Toán"))

    where = collection.queries[0]["where"]
    fields = {list(clause)[0] for clause in where["$and"]}
    assert fields == {"grade", "subject"}


def test_grade_six_student_never_receives_grade_nine_content(retriever) -> None:
    results = retriever.retrieve("Toan hoc", StudentProfile(grade=6, subject="Toán"))

    assert results
    assert all(chunk.metadata["grade"] == 6 for chunk in results)
    assert all("lop 9" not in chunk.text for chunk in results)


def test_subject_is_isolated_as_strictly_as_grade(retriever) -> None:
    results = retriever.retrieve("Bach Dang", StudentProfile(grade=6, subject="Toán"))

    assert all(chunk.metadata["subject"] == "Toán" for chunk in results)


def test_subject_alias_still_matches_the_canonical_stored_value(retriever) -> None:
    results = retriever.retrieve("Phan so", StudentProfile(grade=6, subject="toan"))

    assert [chunk.metadata["subject"] for chunk in results] == ["Toán"]


def test_passages_beyond_the_distance_threshold_are_dropped(monkeypatch, settings) -> None:
    far = [
        {
            "text": "Khong lien quan.",
            "distance": 1.9,
            "metadata": {"grade": 6, "subject": "Toán", "book_title": "SGK", "page_start": 1},
        }
    ]
    collection = FakeCollection(far)
    monkeypatch.setattr(retriever_module, "get_collection", lambda *a, **k: collection)

    results = Retriever(settings).retrieve("abc", StudentProfile(grade=6, subject="Toán"))

    assert results == []


def test_empty_question_is_rejected_before_touching_the_database(retriever, collection) -> None:
    with pytest.raises(ValueError):
        retriever.retrieve("   ", StudentProfile(grade=6, subject="Toán"))

    assert collection.queries == []


def test_top_k_is_forwarded_to_chromadb(retriever, collection) -> None:
    retriever.retrieve("Phan so", StudentProfile(grade=6, subject="Toán"), top_k=3)

    assert collection.queries[0]["n_results"] == 3
