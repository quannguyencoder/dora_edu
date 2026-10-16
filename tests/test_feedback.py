"""Tests for the answer log and the "Báo sai" / rating records."""

from __future__ import annotations

import pytest

from dora_edu.bot.feedback import FeedbackStore


def _log_one(store: FeedbackStore, *, grounded: bool = True) -> int:
    return store.record_answer(
        channel="discord",
        user_id="7",
        grade=6,
        subject="Toán",
        question="Phân số là gì?",
        answer="Phân số gồm tử số và mẫu số...",
        citations=["SGK Toán 6, trang 12"],
        grounded=grounded,
    )


@pytest.fixture
def store() -> FeedbackStore:
    """An in-memory store, so no test ever writes a database file."""
    return FeedbackStore()


def test_a_logged_answer_can_be_read_back_with_its_pages(store) -> None:
    answer_id = _log_one(store)

    record = store.get_answer(answer_id)

    assert record is not None
    assert record.question == "Phân số là gì?"
    assert record.citations == ["SGK Toán 6, trang 12"]
    assert record.grade == 6 and record.subject == "Toán"
    assert record.grounded is True


def test_reporting_an_answer_returns_it_so_the_student_can_be_told_the_page(store) -> None:
    answer_id = _log_one(store)

    record = store.record_report(answer_id, reporter_id="8")

    assert record is not None
    assert record.citations == ["SGK Toán 6, trang 12"]
    assert [r.answer.answer_id for r in store.recent_reports()] == [answer_id]
    assert store.recent_reports()[0].reporter_id == "8"


def test_reporting_an_unknown_answer_is_reported_rather_than_crashing(store) -> None:
    assert store.record_report(999, reporter_id="8") is None
    assert store.recent_reports() == []


def test_a_refusal_is_logged_too_so_a_wrong_refusal_can_be_flagged(store) -> None:
    # The judges' point: refusing when the textbook *does* cover something is
    # a bug, and students must be able to report it like any wrong answer.
    answer_id = _log_one(store, grounded=False)

    record = store.record_report(answer_id, reporter_id="8")

    assert record is not None
    assert record.grounded is False
    assert store.summary().refusals == 1


def test_ratings_are_averaged_and_one_student_cannot_vote_twice(store) -> None:
    answer_id = _log_one(store)

    store.record_rating(answer_id, "a", clarity=5, guidance=4)
    store.record_rating(answer_id, "b", clarity=3, guidance=2)
    store.record_rating(answer_id, "a", clarity=1, guidance=1)  # replaces a's first

    summary = store.summary()
    assert summary.ratings == 2
    assert summary.raters == 2
    assert summary.average_clarity == 2.0
    assert summary.average_guidance == 1.5


def test_a_score_outside_the_1_to_5_scale_is_rejected(store) -> None:
    answer_id = _log_one(store)

    with pytest.raises(ValueError, match="clarity"):
        store.record_rating(answer_id, "a", clarity=6)


def test_rating_an_unknown_answer_is_reported_rather_than_crashing(store) -> None:
    assert store.record_rating(999, "a", clarity=5) is None


def test_summary_counts_answers_reports_and_the_report_rate(store) -> None:
    first = _log_one(store)
    _log_one(store)
    _log_one(store)
    _log_one(store)
    store.record_report(first, reporter_id="8")

    summary = store.summary()
    assert summary.answers == 4
    assert summary.grounded_answers == 4
    assert summary.reports == 1
    assert summary.report_rate == 0.25


def test_reports_survive_a_restart_of_the_bot(tmp_path) -> None:
    db_path = tmp_path / "nested" / "feedback.db"
    store = FeedbackStore(db_path=db_path)
    answer_id = _log_one(store)
    store.record_report(answer_id, reporter_id="8")
    store.close()

    reopened = FeedbackStore(db_path=db_path)
    assert [r.answer.question for r in reopened.recent_reports()] == ["Phân số là gì?"]
    reopened.close()


def test_recent_reports_are_newest_first_and_capped(store) -> None:
    for _ in range(5):
        store.record_report(_log_one(store), reporter_id="8")

    reports = store.recent_reports(limit=3)

    assert len(reports) == 3
    assert [r.report_id for r in reports] == sorted(
        (r.report_id for r in reports), reverse=True
    )


# --- the teacher-facing review sheet ---------------------------------------


def test_the_review_sheet_states_what_was_collected(store) -> None:
    from dora_edu.evaluation.feedback_report import format_markdown

    answer_id = _log_one(store)
    store.record_report(answer_id, "11")
    store.record_rating(answer_id, "11", clarity=5, guidance=4)

    sheet = format_markdown(store)

    assert "Phiếu rà soát câu trả lời DoraEdu" in sheet
    assert "Phân số là gì?" in sheet
    assert "SGK Toán 6, trang 12" in sheet
    assert "5.00/5" in sheet and "4.00/5" in sheet


def test_the_review_sheet_says_so_when_nothing_was_flagged(store) -> None:
    from dora_edu.evaluation.feedback_report import format_markdown

    _log_one(store)

    assert "Chưa có câu nào bị báo sai" in format_markdown(store)


def test_a_missing_rating_average_is_not_shown_as_zero(store) -> None:
    # "0.00/5" would read as "students rated it terrible" rather than
    # "nobody has rated it yet".
    from dora_edu.evaluation.feedback_report import format_markdown

    _log_one(store)

    assert "chưa có" in format_markdown(store)


def test_the_review_csv_leaves_the_teacher_columns_empty(store, tmp_path) -> None:
    from dora_edu.evaluation.feedback_report import write_csv

    answer_id = _log_one(store)
    store.record_report(answer_id, "11")
    path = tmp_path / "phieu.csv"

    write_csv(store, path)

    rows = path.read_text(encoding="utf-8").splitlines()
    assert "ket_luan_cua_thay_co" in rows[0]
    assert rows[1].rstrip().endswith(",,")


def test_a_database_created_before_the_note_column_is_migrated(tmp_path) -> None:
    # The bot's own feedback.db already held real students' reports when the
    # reason field was added, so it has to be migrated in place: a
    # CREATE TABLE IF NOT EXISTS does nothing to an existing table, and
    # every write to it would then fail on the missing column.
    import sqlite3

    path = tmp_path / "old.db"
    conn = sqlite3.connect(path)
    conn.executescript(
        """
        CREATE TABLE answers (
            answer_id INTEGER PRIMARY KEY AUTOINCREMENT, created_at TEXT NOT NULL,
            channel TEXT NOT NULL, user_id TEXT NOT NULL, grade INTEGER, subject TEXT,
            question TEXT NOT NULL, answer TEXT NOT NULL, citations TEXT NOT NULL,
            grounded INTEGER NOT NULL);
        CREATE TABLE reports (
            report_id INTEGER PRIMARY KEY AUTOINCREMENT,
            answer_id INTEGER NOT NULL REFERENCES answers(answer_id),
            created_at TEXT NOT NULL, reporter_id TEXT NOT NULL);
        INSERT INTO answers VALUES
            (1,'2026-10-01','discord','7',6,'Toán','Câu cũ','Đáp án cũ','Toán 6, trang 1',1);
        INSERT INTO reports VALUES (1,1,'2026-10-01','7');
        """
    )
    conn.commit()
    conn.close()

    store = FeedbackStore(db_path=path)

    # The old report survives, with no reason recorded...
    old = store.recent_reports()
    assert len(old) == 1 and old[0].note is None
    # ...and a new one can carry a reason.
    answer_id = _log_one(store)
    store.record_report(answer_id, "8", "lý do mới")
    assert store.recent_reports()[0].note == "lý do mới"
