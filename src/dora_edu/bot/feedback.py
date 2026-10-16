"""Answer logging and student feedback ("Báo sai", 1-5 ratings).

No RAG system is right 100% of the time, and a wrong answer a young student
believes is more harmful than a visible refusal. Two mechanisms limit that
risk, and both need the same thing: a durable record of what was actually
answered, from which textbook page.

* Every grounded answer is logged here before it is sent, so a student's
  one-tap "Báo sai" can attach the question, the answer and the cited pages
  without the student having to retype anything.
* The same record carries the 1-5 ratings collected during classroom
  testing, so "is it easy to understand?" and "does the hinting actually
  make me think?" become numbers a teacher can review afterwards.

This module is channel-agnostic on purpose: Discord renders the report as a
button, Telegram/Zalo as the ``/baosai`` command, and both end up in the
same SQLite file.
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

#: Ratings are a 1-5 school-style scale, the same one the classroom test
#: sheets use, so a score outside it is a bug rather than a user mistake.
MIN_RATING = 1
MAX_RATING = 5

#: Citations are stored newline-joined rather than as JSON: a report is read
#: by a person (a teacher reviewing answers), so the stored form should stay
#: readable in a plain SQLite browser or CSV export.
_CITATION_SEPARATOR = "\n"


@dataclass(frozen=True)
class AnswerRecord:
    """One answer the bot sent, as stored for later verification."""

    answer_id: int
    created_at: str
    channel: str
    user_id: str
    #: The student's name as the chat app shows it, when the channel
    #: exposes one -- so a review sheet names a person, not an id number.
    display_name: str | None
    grade: int | None
    subject: str | None
    question: str
    answer: str
    citations: list[str]
    grounded: bool


@dataclass(frozen=True)
class ReportRecord:
    """One "Báo sai" report, joined with the answer it is about."""

    report_id: int
    created_at: str
    reporter_id: str
    answer: AnswerRecord
    #: What the student said was wrong, when they chose to say anything.
    note: str | None = None
    #: Who reported it, by name.
    reporter_name: str | None = None

    @property
    def reporter(self) -> str:
        """The reporter as a person: their name, falling back to their id."""
        return self.reporter_name or f"(mã {self.reporter_id})"


@dataclass(frozen=True)
class RatingRecord:
    """One student's 1-5 scores for an answer."""

    answer_id: int
    created_at: str
    user_id: str
    user_name: str | None
    clarity: int | None
    guidance: int | None
    comment: str | None

    @property
    def rater(self) -> str:
        """The rater as a person: their name, falling back to their id."""
        return self.user_name or f"(mã {self.user_id})"


@dataclass(frozen=True)
class FeedbackSummary:
    """Aggregate counts for the ``/thongke`` command and the CLI report."""

    answers: int
    grounded_answers: int
    refusals: int
    reports: int
    reporters: int
    ratings: int
    raters: int
    average_clarity: float | None
    average_guidance: float | None

    @property
    def report_rate(self) -> float:
        """Share of answers a student flagged as wrong, as a fraction."""
        return self.reports / self.answers if self.answers else 0.0


class FeedbackStore:
    """SQLite-backed log of answers, wrong-answer reports and 1-5 ratings.

    Unlike :class:`~dora_edu.bot.session.SessionStore`, there is no
    "persistence off" mode: a report is only useful if it survives the
    restart that follows the fix, so omitting ``db_path`` falls back to an
    in-memory SQLite database (used by the tests) rather than to dropping
    writes silently.
    """

    def __init__(self, db_path: Path | None = None) -> None:
        """Open (and create, if needed) the feedback database.

        Args:
            db_path: Where to keep the database; an in-memory one when
                omitted, which the test suite relies on.
        """
        if db_path is not None:
            db_path.parent.mkdir(parents=True, exist_ok=True)
        # check_same_thread=False: the Discord adapter runs the tutoring
        # pipeline in a worker thread (asyncio.to_thread) but handles button
        # interactions on the event loop thread, so both touch this handle.
        self._conn = sqlite3.connect(
            str(db_path) if db_path is not None else ":memory:", check_same_thread=False
        )
        self._conn.execute("PRAGMA foreign_keys = ON")
        self._create_schema()

    def _create_schema(self) -> None:
        """Create the three tables if this is a fresh database."""
        self._conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS answers (
                answer_id  INTEGER PRIMARY KEY AUTOINCREMENT,
                created_at TEXT    NOT NULL,
                channel    TEXT    NOT NULL,
                user_id    TEXT    NOT NULL,
                grade      INTEGER,
                subject    TEXT,
                question   TEXT    NOT NULL,
                answer     TEXT    NOT NULL,
                citations  TEXT    NOT NULL,
                grounded   INTEGER NOT NULL
            );
            CREATE TABLE IF NOT EXISTS reports (
                report_id   INTEGER PRIMARY KEY AUTOINCREMENT,
                answer_id   INTEGER NOT NULL REFERENCES answers(answer_id),
                created_at  TEXT    NOT NULL,
                reporter_id TEXT    NOT NULL,
                -- Why the student thinks it is wrong, in their own words.
                -- Optional: a report must stay one tap, because a required
                -- field is a reason not to report at all.
                note        TEXT
            );
            CREATE TABLE IF NOT EXISTS ratings (
                answer_id  INTEGER NOT NULL REFERENCES answers(answer_id),
                user_id    TEXT    NOT NULL,
                created_at TEXT    NOT NULL,
                clarity    INTEGER,
                guidance   INTEGER,
                comment    TEXT,
                PRIMARY KEY (answer_id, user_id)
            );
            """
        )
        self._add_missing_columns()
        self._conn.commit()

    #: Columns added after the first version of the schema shipped, as
    #: ``table -> column``. Everything here must be nullable, since existing
    #: rows cannot be given a value retrospectively.
    _LATER_COLUMNS: tuple[tuple[str, str], ...] = (
        ("reports", "note"),
        # Who gave the feedback, as the chat app shows them. Without this a
        # review sheet only identifies people by their channel id number,
        # which tells a teacher nothing about which student to go and ask.
        ("answers", "display_name"),
        ("reports", "reporter_name"),
        ("ratings", "user_name"),
    )

    def _add_missing_columns(self) -> None:
        """Bring an older database up to the current schema.

        ``CREATE TABLE IF NOT EXISTS`` does nothing to a table that already
        exists, so a database created before a column was added would keep
        failing on every write to it. The bot's own feedback.db is exactly
        such a file -- it already holds real students' reports, so it is
        migrated rather than recreated.
        """
        for table, column in self._LATER_COLUMNS:
            existing = {row[1] for row in self._conn.execute(f"PRAGMA table_info({table})")}
            if column not in existing:
                self._conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} TEXT")

    @staticmethod
    def _now() -> str:
        """Return the current UTC timestamp, second precision, ISO 8601."""
        return datetime.now(timezone.utc).replace(microsecond=0).isoformat()

    def record_answer(
        self,
        *,
        channel: str,
        user_id: str,
        grade: int | None,
        display_name: str | None = None,
        subject: str | None,
        question: str,
        answer: str,
        citations: list[str],
        grounded: bool,
    ) -> int:
        """Log one answer before it is sent and return its id.

        The id is what a later "Báo sai" tap or rating refers to, so it is
        written synchronously, on the same thread that produced the answer.

        Returns:
            The new answer's id.
        """
        cursor = self._conn.execute(
            "INSERT INTO answers "
            "(created_at, channel, user_id, display_name, grade, subject, question, "
            "answer, citations, grounded) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                self._now(),
                channel,
                user_id,
                display_name,
                grade,
                subject,
                question,
                answer,
                _CITATION_SEPARATOR.join(citations),
                int(grounded),
            ),
        )
        self._conn.commit()
        answer_id = cursor.lastrowid
        assert answer_id is not None  # SQLite always assigns one on INSERT
        return answer_id

    def get_answer(self, answer_id: int) -> AnswerRecord | None:
        """Return a logged answer, or ``None`` when the id is unknown."""
        row = self._conn.execute(
            "SELECT answer_id, created_at, channel, user_id, display_name, grade, "
            "subject, question, answer, citations, grounded FROM answers WHERE answer_id = ?",
            (answer_id,),
        ).fetchone()
        return self._as_answer_record(row) if row else None

    @staticmethod
    def _as_answer_record(row: tuple) -> AnswerRecord:
        """Build an :class:`AnswerRecord` from a raw ``answers`` row."""
        return AnswerRecord(
            answer_id=row[0],
            created_at=row[1],
            channel=row[2],
            user_id=row[3],
            display_name=row[4],
            grade=row[5],
            subject=row[6],
            question=row[7],
            answer=row[8],
            citations=[c for c in row[9].split(_CITATION_SEPARATOR) if c],
            grounded=bool(row[10]),
        )

    def record_report(
        self,
        answer_id: int,
        reporter_id: str,
        note: str | None = None,
        reporter_name: str | None = None,
    ) -> AnswerRecord | None:
        """File a "Báo sai" report against ``answer_id``.

        Args:
            answer_id: The answer the student is flagging.
            reporter_id: Who reported it, in the channel's own id space.
            note: Why they think it is wrong, in their own words. Optional
                by design -- "what exactly is wrong?" is a hard question for
                a 12-year-old who only knows the answer looks off, and
                making it mandatory would mostly stop them reporting.

        Returns:
            The reported answer, so the caller can tell the student which
            textbook page to go and check; ``None`` when ``answer_id`` is
            unknown (e.g. a button from a database that has since been
            reset), which the caller must handle rather than crash on.
        """
        record = self.get_answer(answer_id)
        if record is None:
            return None
        self._conn.execute(
            "INSERT INTO reports (answer_id, created_at, reporter_id, note, reporter_name) "
            "VALUES (?, ?, ?, ?, ?)",
            (answer_id, self._now(), reporter_id, note or None, reporter_name),
        )
        self._conn.commit()
        return record

    def record_rating(
        self,
        answer_id: int,
        user_id: str,
        *,
        clarity: int | None = None,
        guidance: int | None = None,
        comment: str | None = None,
        user_name: str | None = None,
    ) -> AnswerRecord | None:
        """Store (or replace) one student's 1-5 rating of ``answer_id``.

        Re-rating the same answer overwrites the previous score instead of
        adding a second row, so one tester cannot skew an average by tapping
        twice.

        Args:
            answer_id: The answer being rated.
            user_id: Who rated it.
            clarity: "Câu trả lời có dễ hiểu không?", 1-5.
            guidance: "Cách gợi mở có giúp bạn tự nghĩ ra không?", 1-5.
            comment: Optional free-text note from the tester.

        Returns:
            The rated answer, or ``None`` when ``answer_id`` is unknown.

        Raises:
            ValueError: If a score is outside the 1-5 scale.
        """
        for label, score in (("clarity", clarity), ("guidance", guidance)):
            if score is not None and not MIN_RATING <= score <= MAX_RATING:
                raise ValueError(f"{label} must be between {MIN_RATING} and {MAX_RATING}")

        record = self.get_answer(answer_id)
        if record is None:
            return None
        self._conn.execute(
            "INSERT INTO ratings "
            "(answer_id, user_id, created_at, clarity, guidance, comment, user_name) "
            "VALUES (?, ?, ?, ?, ?, ?, ?) "
            "ON CONFLICT (answer_id, user_id) DO UPDATE SET "
            "created_at = excluded.created_at, clarity = excluded.clarity, "
            "guidance = excluded.guidance, comment = excluded.comment, "
            "user_name = excluded.user_name",
            (answer_id, user_id, self._now(), clarity, guidance, comment, user_name),
        )
        self._conn.commit()
        return record

    def ratings_for(self, answer_id: int) -> list[RatingRecord]:
        """Return every student's rating of one answer, oldest first."""
        rows = self._conn.execute(
            "SELECT answer_id, created_at, user_id, user_name, clarity, guidance, comment "
            "FROM ratings WHERE answer_id = ? ORDER BY created_at, user_id",
            (answer_id,),
        ).fetchall()
        return [
            RatingRecord(
                answer_id=row[0],
                created_at=row[1],
                user_id=row[2],
                user_name=row[3],
                clarity=row[4],
                guidance=row[5],
                comment=row[6],
            )
            for row in rows
        ]

    def summary(self) -> FeedbackSummary:
        """Aggregate everything collected so far."""
        answers, grounded = self._conn.execute(
            "SELECT COUNT(*), COALESCE(SUM(grounded), 0) FROM answers"
        ).fetchone()
        reports, reporters = self._conn.execute(
            "SELECT COUNT(*), COUNT(DISTINCT reporter_id) FROM reports"
        ).fetchone()
        ratings, raters, clarity, guidance = self._conn.execute(
            "SELECT COUNT(*), COUNT(DISTINCT user_id), AVG(clarity), AVG(guidance) FROM ratings"
        ).fetchone()
        return FeedbackSummary(
            answers=answers,
            grounded_answers=grounded,
            refusals=answers - grounded,
            reports=reports,
            reporters=reporters,
            ratings=ratings,
            raters=raters,
            average_clarity=clarity,
            average_guidance=guidance,
        )

    def recent_reports(self, limit: int = 20) -> list[ReportRecord]:
        """Return the newest "Báo sai" reports, each with its full answer.

        Args:
            limit: Maximum reports to return, newest first.
        """
        rows = self._conn.execute(
            "SELECT r.report_id, r.created_at, r.reporter_id, r.note, r.reporter_name, "
            "a.answer_id, a.created_at, a.channel, a.user_id, a.display_name, a.grade, "
            "a.subject, a.question, a.answer, a.citations, a.grounded "
            "FROM reports r JOIN answers a ON a.answer_id = r.answer_id "
            "ORDER BY r.report_id DESC LIMIT ?",
            (limit,),
        ).fetchall()
        return [
            ReportRecord(
                report_id=row[0],
                created_at=row[1],
                reporter_id=row[2],
                note=row[3],
                reporter_name=row[4],
                answer=self._as_answer_record(row[5:]),
            )
            for row in rows
        ]

    def close(self) -> None:
        """Close the database handle."""
        self._conn.close()
