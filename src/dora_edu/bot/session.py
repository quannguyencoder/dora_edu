"""Sliding window chat history management.

Sessions are keyed by ``(channel, user_id)`` so the same numeric id on Telegram
and Zalo can never collide, and they hold the grade/subject profile that scopes
every retrieval for that student.
"""

from __future__ import annotations

import sqlite3
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path

from dora_edu.models import StudentProfile

#: One conversational turn contributes a student message and a tutor reply.
_MESSAGES_PER_TURN = 2


@dataclass
class Session:
    """Per-student conversational state.

    ``grade`` and ``subject`` are stored separately because students set them in
    two steps; :attr:`profile` only materialises once both are known, which is
    what the retriever requires.
    """

    max_turns: int
    grade: int | None = None
    subject: str | None = None
    _messages: deque[dict[str, str]] = field(default_factory=deque, repr=False)

    def __post_init__(self) -> None:
        """Size the sliding window to ``max_turns`` question/answer pairs."""
        self._messages = deque(self._messages, maxlen=self.max_turns * _MESSAGES_PER_TURN)

    @property
    def profile(self) -> StudentProfile | None:
        """Return the retrieval scope, or ``None`` while it is incomplete."""
        if self.grade is None or self.subject is None:
            return None
        return StudentProfile(grade=self.grade, subject=self.subject)

    @property
    def has_profile(self) -> bool:
        """Whether the student has set both a grade and a subject."""
        return self.grade is not None and self.subject is not None

    def record_turn(self, question: str, answer: str) -> None:
        """Append one question/answer pair, evicting the oldest when full."""
        self._messages.append({"role": "user", "content": question})
        self._messages.append({"role": "assistant", "content": answer})

    def history(self) -> list[dict[str, str]]:
        """Return the retained turns, oldest first, ready for the LLM."""
        return list(self._messages)

    def clear_history(self) -> None:
        """Drop the conversation history but keep the grade/subject profile."""
        self._messages.clear()


class SessionStore:
    """In-memory session registry, optionally backed by an on-disk SQLite file.

    Conversation history is cheap to rebuild and not worth persisting, but
    grade/subject are saved to ``db_path`` whenever set, so a bot restart --
    routine during development, or after a deploy -- does not force every
    student to redo ``/lop``. Suitable for a single-process bot; swapping in a
    different backend later only requires reimplementing this class, since
    callers depend on nothing beyond :meth:`get`, :meth:`save_profile` and
    :meth:`reset`.
    """

    def __init__(self, max_turns: int, db_path: Path | None = None) -> None:
        """Create a store whose sessions each retain ``max_turns`` turns.

        Args:
            max_turns: Question/answer pairs kept per session's history.
            db_path: Where to persist grade/subject across restarts. Purely
                in-memory (nothing survives a restart) when omitted, which is
                what every existing test relies on.
        """
        self._max_turns = max_turns
        self._sessions: dict[tuple[str, str], Session] = {}
        self._conn: sqlite3.Connection | None = None
        if db_path is not None:
            db_path.parent.mkdir(parents=True, exist_ok=True)
            self._conn = sqlite3.connect(db_path, check_same_thread=False)
            self._conn.execute(
                "CREATE TABLE IF NOT EXISTS profiles ("
                "channel TEXT NOT NULL, user_id TEXT NOT NULL, "
                "grade INTEGER, subject TEXT, "
                "PRIMARY KEY (channel, user_id))"
            )
            self._conn.commit()

    def get(self, key: tuple[str, str]) -> Session:
        """Return the session for ``key``, creating one when unknown.

        A newly-created session's grade/subject are loaded from disk when a
        persistence backend is configured, so a returning student picks up
        exactly where they left off even across a restart.
        """
        session = self._sessions.get(key)
        if session is None:
            grade, subject = self._load_profile(key)
            session = Session(max_turns=self._max_turns, grade=grade, subject=subject)
            self._sessions[key] = session
        return session

    def save_profile(self, key: tuple[str, str]) -> None:
        """Persist ``key``'s current grade/subject so they survive a restart.

        A no-op when no persistence backend is configured, or when ``key``
        has no session yet (nothing to save).
        """
        if self._conn is None:
            return
        session = self._sessions.get(key)
        if session is None:
            return
        channel, user_id = key
        self._conn.execute(
            "INSERT INTO profiles (channel, user_id, grade, subject) VALUES (?, ?, ?, ?) "
            "ON CONFLICT (channel, user_id) DO UPDATE SET "
            "grade = excluded.grade, subject = excluded.subject",
            (channel, user_id, session.grade, session.subject),
        )
        self._conn.commit()

    def _load_profile(self, key: tuple[str, str]) -> tuple[int | None, str | None]:
        """Read a previously-saved grade/subject for ``key``, if any."""
        if self._conn is None:
            return None, None
        channel, user_id = key
        row = self._conn.execute(
            "SELECT grade, subject FROM profiles WHERE channel = ? AND user_id = ?",
            (channel, user_id),
        ).fetchone()
        return (row[0], row[1]) if row else (None, None)

    def reset(self, key: tuple[str, str]) -> None:
        """Forget everything about one student, profile included."""
        self._sessions.pop(key, None)
        if self._conn is not None:
            channel, user_id = key
            self._conn.execute(
                "DELETE FROM profiles WHERE channel = ? AND user_id = ?", (channel, user_id)
            )
            self._conn.commit()

    def __len__(self) -> int:
        """Return the number of active (in-memory) sessions."""
        return len(self._sessions)
