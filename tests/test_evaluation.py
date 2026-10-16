"""Tests for the benchmark runner's scoring.

The benchmark itself needs the real index and real API calls; what is tested
here is the part that must not be wrong about them -- how an answer is
turned into a verdict.
"""

from __future__ import annotations

import pytest

from dora_edu.evaluation import question_set
from dora_edu.evaluation.runner import (
    EvalReport,
    ask_one,
    format_markdown,
    run_benchmark,
    write_csv,
)
from dora_edu.evaluation.question_set import EvalQuestion
from dora_edu.llm.generator import GeneratedAnswer
from dora_edu.llm.prompts import NO_CONTEXT_ANSWER
from dora_edu.models import RetrievedChunk


class FakeRetriever:
    """Returns a fixed passage list, whatever is asked."""

    def __init__(self, chunks: list[RetrievedChunk]) -> None:
        self._chunks = chunks

    def retrieve(self, question: str, profile, *, top_k=None) -> list[RetrievedChunk]:
        return list(self._chunks)


class FakeGenerator:
    """Returns a fixed answer, whatever it is given."""

    def __init__(self, answer: str) -> None:
        self._answer = answer

    def generate(self, question, chunks, profile, history=None) -> GeneratedAnswer:
        return GeneratedAnswer(
            answer=self._answer,
            grounded=bool(chunks),
            citations=[c.citation for c in chunks],
        )


def _chunk(*, grade: int = 6, subject: str = "Toán", page: int = 12) -> RetrievedChunk:
    return RetrievedChunk(
        text="Số nguyên tố là số tự nhiên lớn hơn 1, chỉ có hai ước là 1 và chính nó.",
        distance=0.2,
        metadata={
            "grade": grade,
            "subject": subject,
            "book_title": f"Toán {grade} - tập 1",
            "page_start": page,
            "page_end": page,
        },
    )


_IN_BOOK = EvalQuestion("Số nguyên tố là gì?", 6, "Toán", True, ("ước",))
_OUT_OF_BOOK = EvalQuestion("Tích phân là gì?", 7, "Toán", False, note="Lớp 12")


def test_a_correct_in_textbook_answer_passes() -> None:
    outcome = ask_one(
        _IN_BOOK,
        FakeRetriever([_chunk()]),
        FakeGenerator("Theo sách Toán 6, tập 1, trang 12: số nguyên tố chỉ có hai ước."),
    )

    assert outcome.verdict == "ĐẠT"
    assert outcome.handled_correctly
    assert outcome.cites_a_page
    assert outcome.invented_pages == ()
    assert outcome.keywords_found == ("ước",)


def test_refusing_a_question_the_textbook_covers_is_a_wrong_refusal() -> None:
    outcome = ask_one(_IN_BOOK, FakeRetriever([]), FakeGenerator(NO_CONTEXT_ANSWER))

    assert outcome.refused
    assert outcome.verdict == "TỪ CHỐI NHẦM"
    assert not outcome.handled_correctly


def test_refusing_an_out_of_textbook_question_is_the_correct_outcome() -> None:
    outcome = ask_one(_OUT_OF_BOOK, FakeRetriever([]), FakeGenerator(NO_CONTEXT_ANSWER))

    assert outcome.handled_correctly
    assert outcome.verdict.startswith("ĐẠT")


def test_answering_an_out_of_textbook_question_is_counted_as_hallucination() -> None:
    outcome = ask_one(
        _OUT_OF_BOOK,
        FakeRetriever([_chunk(grade=7)]),
        FakeGenerator("Tích phân là phép tính ngược của đạo hàm, theo trang 12."),
    )

    assert not outcome.handled_correctly
    assert outcome.verdict == "TRẢ LỜI NGOÀI SÁCH"


def test_a_page_number_that_was_never_retrieved_is_flagged() -> None:
    # The most dangerous failure: an invented citation reads exactly as
    # authoritative as a real one.
    outcome = ask_one(
        _IN_BOOK,
        FakeRetriever([_chunk(page=12)]),
        FakeGenerator("Theo sách Toán 6, tập 1, trang 87: số nguyên tố có hai ước."),
    )

    assert outcome.invented_pages == (87,)
    assert outcome.verdict == "TRÍCH SAI TRANG"


def test_a_cited_page_range_counts_every_page_in_it() -> None:
    outcome = ask_one(
        _IN_BOOK,
        FakeRetriever([_chunk(page=12)]),
        FakeGenerator("Theo sách Toán 6, tập 1, trang 12-13: ... hai ước ..."),
    )

    # Page 13 was not retrieved, so it is still an invented citation.
    assert outcome.invented_pages == (13,)


def test_a_passage_from_the_wrong_subject_is_out_of_scope() -> None:
    outcome = ask_one(
        _IN_BOOK,
        FakeRetriever([_chunk(subject="Ngữ văn")]),
        FakeGenerator("Theo sách Toán 6, tập 1, trang 12: ... ước ..."),
    )

    assert not outcome.in_scope


def test_a_passage_from_a_higher_grade_is_out_of_scope() -> None:
    # The safety-critical direction: a grade-6 student must never be shown
    # grade-9 content.
    outcome = ask_one(
        _IN_BOOK,
        FakeRetriever([_chunk(grade=9)]),
        FakeGenerator("Theo sách Toán 9, trang 12: ... ước ..."),
    )

    assert not outcome.in_scope


def test_a_lower_grade_passage_stays_in_scope() -> None:
    # Revising earlier material is allowed by design.
    outcome = ask_one(
        EvalQuestion("Số nguyên tố là gì?", 8, "Toán", True, ("ước",)),
        FakeRetriever([_chunk(grade=6)]),
        FakeGenerator("Đây là kiến thức lớp 6: theo trang 12, ... hai ước ..."),
    )

    assert outcome.in_scope


def test_an_answer_missing_every_keyword_is_sent_to_a_human() -> None:
    outcome = ask_one(
        _IN_BOOK,
        FakeRetriever([_chunk()]),
        FakeGenerator("Theo sách Toán 6, tập 1, trang 12: bạn thử nghĩ xem nhé!"),
    )

    assert outcome.verdict == "CẦN NGƯỜI CHẤM"


def test_the_report_separates_wrong_refusals_from_hallucinations() -> None:
    report = run_benchmark(
        (_IN_BOOK, _OUT_OF_BOOK),
        delay_seconds=0,
        retriever=FakeRetriever([]),
        generator=FakeGenerator(NO_CONTEXT_ANSWER),
    )

    assert len(report.wrong_refusals) == 1
    assert report.hallucinations == ()
    assert len(report.in_textbook) == 1 and len(report.out_of_textbook) == 1


def test_the_markdown_report_states_the_headline_numbers() -> None:
    report = run_benchmark(
        (_IN_BOOK, _OUT_OF_BOOK),
        delay_seconds=0,
        retriever=FakeRetriever([_chunk()]),
        generator=FakeGenerator("Theo sách Toán 6, tập 1, trang 12: có hai ước."),
    )

    markdown = format_markdown(report)

    assert "Kết quả kiểm thử DoraEdu" in markdown
    # One in-textbook question answered, and the out-of-textbook one answered
    # too -- which must be reported as a hallucination, not hidden.
    assert "Trả lời câu ngoài sách (bịa đặt) | 1/1" in markdown
    assert _IN_BOOK.question in markdown


def test_the_csv_leaves_the_human_verdict_column_empty(tmp_path) -> None:
    report = run_benchmark(
        (_IN_BOOK,),
        delay_seconds=0,
        retriever=FakeRetriever([_chunk()]),
        generator=FakeGenerator("Theo sách Toán 6, tập 1, trang 12: có hai ước."),
    )
    path = tmp_path / "ket-qua.csv"

    write_csv(report, path)

    rows = path.read_text(encoding="utf-8").splitlines()
    assert "dung_sai_nguoi_cham" in rows[0]
    # The column a teacher fills in is present and empty.
    assert rows[1].count(",,") >= 1


def test_an_empty_report_does_not_divide_by_zero() -> None:
    assert "Kết quả kiểm thử" in format_markdown(EvalReport(outcomes=()))


# --- the benchmark set itself ----------------------------------------------


def test_the_set_has_fifty_questions_with_ten_outside_the_textbook() -> None:
    assert len(question_set.QUESTION_SET) == 50
    assert len(question_set.OUT_OF_TEXTBOOK_QUESTIONS) == 10
    assert len(question_set.IN_TEXTBOOK_QUESTIONS) == 40


def test_every_out_of_textbook_question_explains_why_it_is_outside() -> None:
    # Otherwise a reader of the report has to take that judgement on trust.
    assert all(q.note for q in question_set.OUT_OF_TEXTBOOK_QUESTIONS)


def test_the_set_spans_several_subjects_and_grades() -> None:
    assert len({q.subject for q in question_set.QUESTION_SET}) >= 5
    assert {q.grade for q in question_set.QUESTION_SET} == {6, 7, 8, 9}


@pytest.mark.parametrize(
    "topic",
    ["đa thức", "định lí thalès", "hằng đẳng thức", "dàn ý", "đạo hàm của"],
)
def test_no_question_asks_about_a_hand_patched_passage(topic) -> None:
    # The whole point of the set: measuring the pipeline, not the patches.
    asked = " ".join(q.question.lower() for q in question_set.IN_TEXTBOOK_QUESTIONS)
    assert topic not in asked


def test_a_provider_outage_is_not_counted_as_an_answer_or_a_refusal() -> None:
    # A rate-limited or unreachable provider never saw the question. Counted
    # as an answer, it would show up as a hallucination; counted as a
    # refusal, as a wrong refusal. Either way the run would be misread.
    from dora_edu.llm.generator import _SERVICE_ERROR_ANSWER

    outcome = ask_one(
        _OUT_OF_BOOK, FakeRetriever([_chunk(grade=7)]), FakeGenerator(_SERVICE_ERROR_ANSWER)
    )

    assert outcome.provider_error
    assert outcome.verdict.startswith("LỖI KỸ THUẬT")
    assert not outcome.handled_correctly


def test_provider_outages_are_reported_separately() -> None:
    from dora_edu.llm.generator import _RATE_LIMIT_ANSWER

    report = run_benchmark(
        (_IN_BOOK, _OUT_OF_BOOK),
        delay_seconds=0,
        retriever=FakeRetriever([_chunk()]),
        generator=FakeGenerator(_RATE_LIMIT_ANSWER),
    )

    assert len(report.provider_errors) == 2
    assert report.hallucinations == ()
    assert report.wrong_refusals == ()
    assert "Lỗi kỹ thuật, chưa hỏi được | 2" in format_markdown(report)
