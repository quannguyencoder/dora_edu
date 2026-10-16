"""Tests for the mandatory grade/subject isolation rule."""

from __future__ import annotations

import pytest

from dora_edu.models import StudentProfile
from dora_edu.rag_engine import retriever as retriever_module
from dora_edu.rag_engine.retriever import Retriever, build_metadata_filter, keyword_phrase
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


# --- literal (full-text) matching alongside vector search ------------------


@pytest.mark.parametrize(
    ("question", "expected"),
    [
        ("Quang hợp là gì?", "quang hợp"),
        ("Thế nào là tam giác cân?", "tam giác cân"),
        ("Định luật Ôm phát biểu thế nào?", "định luật ôm"),
        ("Nguyên tử gồm những hạt nào?", "nguyên tử"),
        ("Khái niệm số hữu tỉ", "số hữu tỉ"),
    ],
)
def test_keyword_phrase_strips_the_question_leaving_the_term(question, expected) -> None:
    assert keyword_phrase(question) == expected


@pytest.mark.parametrize(
    "question",
    [
        # Not definition-shaped: a literal search for the whole sentence
        # would match nothing at all, so there is no term to extract.
        "Giúp mình bài 3 trang 20 với",
        "Tại sao lá cây có màu xanh?",
        # Too short to be a useful substring filter on its own.
        "Số là gì?",
    ],
)
def test_keyword_phrase_declines_when_there_is_no_usable_term(question) -> None:
    assert keyword_phrase(question) is None


def test_a_passage_containing_the_term_is_retrieved_even_when_it_ranks_badly(
    monkeypatch, settings
) -> None:
    # The real failure this fixes: "tam giác cân" embeds close to grade-2
    # pages about weighing things ("cân nặng"), which pushed the actual
    # tam-giác-cân page out of the results entirely.
    rows = [
        {
            "text": "Mai can nang khoang 31 kg. Viet can nang khoang 32 kg.",
            "distance": 0.32,
            "metadata": {
                "grade": 2,
                "subject": "Toán",
                "book_title": "Toán 2",
                "source_file": "Toan2.pdf",
                "chunk_index": 1,
                "page_start": 61,
                "page_end": 61,
            },
        },
        {
            "text": "Tam giác cân là tam giác có hai cạnh bên bằng nhau.",
            "distance": 0.80,
            "metadata": {
                "grade": 7,
                "subject": "Toán",
                "book_title": "Toán 7 - tập 1",
                "source_file": "Toan7.pdf",
                "chunk_index": 2,
                "page_start": 81,
                "page_end": 81,
            },
        },
    ]
    collection = FakeCollection(rows)
    monkeypatch.setattr(retriever_module, "get_collection", lambda *a, **k: collection)

    chunks = Retriever(settings).retrieve(
        "Thế nào là tam giác cân?", StudentProfile(grade=7, subject="Toán"), top_k=2
    )

    # The literal match comes first despite its larger vector distance: the
    # prompt leans on the earlier passages.
    assert "Tam giác cân là" in chunks[0].text
    assert len(chunks) == 2


def test_the_literal_pass_looks_for_the_definition_sentence_first(
    monkeypatch, settings
) -> None:
    collection = FakeCollection([])
    monkeypatch.setattr(retriever_module, "get_collection", lambda *a, **k: collection)

    Retriever(settings).retrieve("Quang hợp là gì?", StudentProfile(grade=7, subject="Toán"))

    contains = [
        (call.get("where_document") or {}).get("$contains") for call in collection.queries
    ]
    assert "quang hợp là" in contains
    assert "quang hợp" in contains


def test_a_question_with_no_usable_term_runs_one_plain_query(monkeypatch, settings) -> None:
    collection = FakeCollection([])
    monkeypatch.setattr(retriever_module, "get_collection", lambda *a, **k: collection)

    Retriever(settings).retrieve(
        "Tại sao lá cây có màu xanh?", StudentProfile(grade=7, subject="Toán")
    )

    assert len(collection.queries) == 1
    assert "where_document" not in collection.queries[0]


def test_every_literal_query_still_carries_the_grade_and_subject_filter(
    monkeypatch, settings
) -> None:
    # The safety rule is structural: no query may reach ChromaDB unscoped,
    # and the extra passes are queries like any other.
    collection = FakeCollection([])
    monkeypatch.setattr(retriever_module, "get_collection", lambda *a, **k: collection)

    Retriever(settings).retrieve("Quang hợp là gì?", StudentProfile(grade=7, subject="Toán"))

    assert len(collection.queries) > 1
    for call in collection.queries:
        fields = {list(clause)[0] for clause in call["where"]["$and"]}
        assert fields == {"grade", "subject"}


def test_a_passage_is_never_returned_twice_by_two_passes(monkeypatch, settings) -> None:
    rows = [
        {
            "text": "Quang hợp là quá trình lá cây dùng ánh sáng.",
            "distance": 0.40,
            "metadata": {
                "grade": 7,
                "subject": "Khoa học tự nhiên",
                "book_title": "Khoa học tự nhiên 7",
                "source_file": "KHTN7.pdf",
                "chunk_index": 5,
                "page_start": 104,
                "page_end": 104,
            },
        }
    ]
    collection = FakeCollection(rows)
    monkeypatch.setattr(retriever_module, "get_collection", lambda *a, **k: collection)

    chunks = Retriever(settings).retrieve(
        "Quang hợp là gì?", StudentProfile(grade=7, subject="Khoa học tự nhiên")
    )

    assert len(chunks) == 1
