"""Tests for the mandatory grade/subject isolation rule."""

from __future__ import annotations

import pytest

from dora_edu.models import StudentProfile
from dora_edu.rag_engine import retriever as retriever_module
from dora_edu.rag_engine.retriever import (
    Retriever,
    build_metadata_filter,
    keyword_phrase,
    name_variants,
)
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


def test_top_k_decides_how_many_passages_come_back(retriever, collection) -> None:
    chunks = retriever.retrieve("Phan so", StudentProfile(grade=6, subject="Toán"), top_k=3)

    assert len(chunks) <= 3
    # More than top_k is fetched on purpose: the grade re-ranking can only
    # promote a passage that was fetched, so the pool has to be wider than
    # the result.
    assert collection.queries[0]["n_results"] > 3


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


# --- preferring the student's own grade ------------------------------------


def _history_rows() -> list[dict]:
    """A topic both a primary and a lower-secondary book cover."""
    return [
        {
            "text": "Chiến dịch Điện Biên Phủ toàn thắng, kể cho học sinh nhỏ.",
            "distance": 0.40,
            "metadata": {
                "grade": 5,
                "subject": "Lịch sử và Địa lí",
                "book_title": "Lịch sử và Địa lí 5",
                "source_file": "LSDL5.pdf",
                "chunk_index": 1,
                "page_start": 66,
                "page_end": 66,
            },
        },
        {
            "text": "Chiến thắng Điện Biên Phủ năm 1954 buộc Pháp phải ký Hiệp định Giơ-ne-vơ.",
            "distance": 0.45,
            "metadata": {
                "grade": 9,
                "subject": "Lịch sử và Địa lí",
                "book_title": "Lịch sử và Địa lí 9",
                "source_file": "LSDL9.pdf",
                "chunk_index": 2,
                "page_start": 40,
                "page_end": 40,
            },
        },
    ]


def test_the_students_own_grade_wins_a_close_call(monkeypatch, settings) -> None:
    # Found live: a grade-9 student asking about Điện Biên Phủ was answered
    # from the grade-5 book, which the grade ceiling allows but which is not
    # what they are studying.
    collection = FakeCollection(_history_rows())
    monkeypatch.setattr(retriever_module, "get_collection", lambda *a, **k: collection)

    chunks = Retriever(settings).retrieve(
        "Chiến thắng Điện Biên Phủ diễn ra năm nào?",
        StudentProfile(grade=9, subject="Lịch sử và Địa lí"),
    )

    assert chunks[0].metadata["grade"] == 9


def test_a_lower_grade_passage_still_wins_when_it_is_clearly_closer(
    monkeypatch, settings
) -> None:
    # Revising earlier material has to keep working, so the adjustment is
    # small: a much better match from a lower grade still comes first.
    rows = _history_rows()
    rows[0]["distance"] = 0.20  # the grade-5 passage is now far closer
    collection = FakeCollection(rows)
    monkeypatch.setattr(retriever_module, "get_collection", lambda *a, **k: collection)

    chunks = Retriever(settings).retrieve(
        "Chiến thắng Điện Biên Phủ diễn ra năm nào?",
        StudentProfile(grade=9, subject="Lịch sử và Địa lí"),
    )

    assert chunks[0].metadata["grade"] == 5


def test_the_reported_distance_is_the_real_one_not_the_adjusted_one(
    monkeypatch, settings
) -> None:
    # The adjustment only orders the results; a logged or reported distance
    # must still be what ChromaDB measured.
    collection = FakeCollection(_history_rows())
    monkeypatch.setattr(retriever_module, "get_collection", lambda *a, **k: collection)

    chunks = Retriever(settings).retrieve(
        "Chiến thắng Điện Biên Phủ diễn ra năm nào?",
        StudentProfile(grade=9, subject="Lịch sử và Địa lí"),
    )

    assert {round(c.distance, 2) for c in chunks} == {0.40, 0.45}


@pytest.mark.parametrize(
    ("question", "expected"),
    [
        # Already just the term: nothing to strip, and searching for it
        # literally is exactly right.
        ("hằng đẳng thức đáng nhớ", "hằng đẳng thức đáng nhớ"),
        ("định lí Pythagore", "định lí pythagore"),
        # Question forms that are not "X là gì?" but still name a term.
        ("Bình phương của một tổng viết thế nào?", "bình phương của một tổng"),
        ("Khởi nghĩa Tây Sơn do ai lãnh đạo?", "khởi nghĩa tây sơn"),
        ("Nhà Lý được thành lập năm nào?", "nhà lý được thành lập"),
    ],
)
def test_a_term_is_found_in_questions_that_are_not_definitions(question, expected) -> None:
    # Found live: "Bình phương của một tổng viết thế nào?" ran no literal
    # pass at all, and the top three vector hits were the copyright page at
    # the back of the book.
    assert keyword_phrase(question) == expected


def test_a_long_sentence_with_nothing_to_strip_is_still_declined() -> None:
    # Searching for a whole sentence literally matches nothing, so it is
    # not worth a query.
    assert keyword_phrase("Giúp mình giải bài tập số 3 trang 20 sách giáo khoa với") is None


def test_two_passages_without_identifying_metadata_are_not_merged_into_one(
    monkeypatch, settings
) -> None:
    # Keyed only on (source_file, chunk_index), two chunks missing both
    # would share the key (None, None) and one would vanish as a duplicate.
    rows = [
        {
            "text": "Phan so lop 6.",
            "distance": 0.10,
            "metadata": {"grade": 6, "subject": "Toán", "book_title": "SGK", "page_start": 1},
        },
        {
            "text": "Ham so bac nhat lop 9.",
            "distance": 0.11,
            "metadata": {"grade": 9, "subject": "Toán", "book_title": "SGK", "page_start": 2},
        },
    ]
    collection = FakeCollection(rows)
    monkeypatch.setattr(retriever_module, "get_collection", lambda *a, **k: collection)

    chunks = Retriever(settings).retrieve(
        "hàm số bậc nhất", StudentProfile(grade=9, subject="Toán")
    )

    assert len(chunks) == 2


# --- foreign names the book spells differently -----------------------------


@pytest.mark.parametrize(
    ("typed", "expected"),
    [
        ("toạ độ decat", "toạ độ Descartes"),
        ("định lí py-ta-go", "định lí Pythagore"),
        ("định lí ta-lét", "định lí Thalès"),
    ],
)
def test_a_name_is_re_spelled_the_way_the_book_prints_it(typed, expected) -> None:
    # Found live: a grade-8 student asked "toạ độ decat là gì" and was told
    # the textbook does not cover it, while Toán 8 tập 2 discusses Descartes
    # on page 46. Capitalised, because the book capitalises a person's name
    # and ChromaDB's substring filter is case-sensitive.
    assert expected in name_variants(typed)


def test_the_bare_name_is_tried_as_well_as_the_whole_phrase() -> None:
    # The book names Descartes without ever writing "toạ độ Descartes", so
    # the phrase alone would still find nothing.
    assert "Descartes" in name_variants("toạ độ decat")


@pytest.mark.parametrize(
    "typed",
    [
        # Found live: a grade-9 student typed a final t, which matched none of
        # the "-các" spellings, so the lookup never ran and the bot refused a
        # question Toán 8 tập 2 trang 46 answers.
        "tọa độ đề-cát",
        # Accents, hyphens and case all even out before the comparison.
        "toa do de cac",
        "tọa độ đề các",
    ],
)
def test_a_name_is_matched_however_the_student_accents_it(typed) -> None:
    assert "Descartes" in name_variants(typed)


def test_a_phrase_with_no_known_name_yields_no_variants() -> None:
    assert name_variants("hằng đẳng thức đáng nhớ") == []


def test_normalising_cannot_match_a_name_across_two_words() -> None:
    # Dropping diacritics turns "đọc lít" into "doc lit"; hyphens become
    # spaces rather than vanishing so it cannot be read as "ơ-clít".
    assert name_variants("cách đọc lít nước") == []


def test_an_alternative_spelling_is_only_tried_when_the_first_search_fails(
    monkeypatch, settings
) -> None:
    # Re-spelling costs extra queries, so it must not run when the term as
    # typed already matches something.
    rows = [
        {
            "text": "Định lí Pythagore: trong tam giác vuông, bình phương cạnh huyền...",
            "distance": 0.2,
            "metadata": {
                "grade": 8, "subject": "Toán", "book_title": "Toán 8 - tập 2",
                "source_file": "T8b.pdf", "chunk_index": 1,
                "page_start": 93, "page_end": 93,
            },
        }
    ]
    collection = FakeCollection(rows)
    monkeypatch.setattr(retriever_module, "get_collection", lambda *a, **k: collection)

    Retriever(settings).retrieve(
        "Định lí Pythagore là gì?", StudentProfile(grade=8, subject="Toán")
    )

    tried = [(c.get("where_document") or {}).get("$contains") for c in collection.queries]
    assert not any(t and "py-ta-go" in t for t in tried)
