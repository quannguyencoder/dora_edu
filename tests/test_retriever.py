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
    assert clauses["grade"] == {"grade": {"$lte": 6}}
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


def test_a_higher_grade_student_can_still_review_lower_grade_content(retriever) -> None:
    # The grade filter is a ceiling: a grade-9 student reviewing earlier
    # material must still be able to reach grade-6 content of the same
    # subject, while never reaching a grade above their own.
    results = retriever.retrieve("Toan hoc", StudentProfile(grade=9, subject="Toán"))

    grades = {chunk.metadata["grade"] for chunk in results}
    assert 6 in grades
    assert 9 in grades
    assert all(grade <= 9 for grade in grades)


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


# --- Subject auto-detection (retrieve_best_subject / list_subjects) --------


def test_list_subjects_returns_every_subject_indexed_for_that_grade(retriever) -> None:
    assert retriever.list_subjects(6) == ["Lịch sử", "Toán"]


def test_list_subjects_is_empty_below_every_indexed_grade(retriever) -> None:
    # textbook_rows only has grade 6 and 9 content; grade 1 sees neither
    # under the "at or below" ceiling.
    assert retriever.list_subjects(1) == []


def test_list_subjects_includes_subjects_only_taught_in_lower_grades(monkeypatch, settings) -> None:
    rows = [
        {
            "text": "Dao duc lop 1.",
            "distance": 0.1,
            "metadata": {"grade": 1, "subject": "Đạo đức", "book_title": "SGK", "page_start": 1},
        },
        {
            "text": "Toan lop 9.",
            "distance": 0.1,
            "metadata": {"grade": 9, "subject": "Toán", "book_title": "SGK", "page_start": 1},
        },
    ]
    collection = FakeCollection(rows)
    monkeypatch.setattr(retriever_module, "get_collection", lambda *a, **k: collection)

    assert Retriever(settings).list_subjects(9) == ["Toán", "Đạo đức"]


def test_list_subjects_is_cached_after_the_first_call(retriever, collection) -> None:
    retriever.list_subjects(6)
    calls_after_first = len(collection.get_calls)

    retriever.list_subjects(6)

    assert len(collection.get_calls) == calls_after_first


def test_list_subjects_queries_one_grade_at_a_time_not_one_big_lte_query(
    retriever, collection
) -> None:
    # A single $lte query has to enumerate metadata for every chunk at or
    # below the grade, which blows past SQLite's bound-parameter limit for a
    # large corpus (confirmed live at grade 12, ~40k chunks) -- list_subjects
    # must query one grade at a time instead.
    retriever.list_subjects(6)

    assert len(collection.get_calls) == 6
    for call in collection.get_calls:
        assert "$eq" in call["where"]["grade"]


def test_retrieve_best_subject_picks_the_closest_matching_subject(retriever) -> None:
    chunks, subject = retriever.retrieve_best_subject("Phan so la gi?", grade=6)

    assert subject == "Toán"
    assert chunks
    assert all(chunk.metadata["subject"] == "Toán" for chunk in chunks)


def test_retrieve_best_subject_only_ever_sends_fully_filtered_queries(retriever, collection) -> None:
    retriever.retrieve_best_subject("Phan so la gi?", grade=6)

    assert collection.queries  # at least one subject was tried
    for call in collection.queries:
        fields = {list(clause)[0] for clause in call["where"]["$and"]}
        assert fields == {"grade", "subject"}


def test_retrieve_best_subject_returns_none_when_nothing_is_close_enough(
    monkeypatch, settings
) -> None:
    far = [
        {
            "text": "Khong lien quan.",
            "distance": 1.9,
            "metadata": {"grade": 6, "subject": "Toán", "book_title": "SGK", "page_start": 1},
        }
    ]
    collection = FakeCollection(far)
    monkeypatch.setattr(retriever_module, "get_collection", lambda *a, **k: collection)

    chunks, subject = Retriever(settings).retrieve_best_subject("abc", grade=6)

    assert chunks == []
    assert subject is None


def test_retrieve_best_subject_on_an_unknown_grade_returns_none(retriever) -> None:
    chunks, subject = retriever.retrieve_best_subject("Phan so la gi?", grade=1)

    assert chunks == []
    assert subject is None
