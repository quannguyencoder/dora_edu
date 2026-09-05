"""Sliding window chat history management.

Sessions are keyed by ``(channel, user_id)`` so the same numeric id on Telegram
and Zalo can never collide, and they hold the grade/subject profile that scopes
every retrieval for that student.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

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
    """In-memory session registry.

    Suitable for a single-process bot. Swapping in a Redis-backed store later
    only requires reimplementing this class, since callers depend on nothing
    beyond :meth:`get` and :meth:`reset`.
    """

    def __init__(self, max_turns: int) -> None:
        """Create a store whose sessions each retain ``max_turns`` turns."""
        self._max_turns = max_turns
        self._sessions: dict[tuple[str, str], Session] = {}

    def get(self, key: tuple[str, str]) -> Session:
        """Return the session for ``key``, creating an empty one when unknown."""
        session = self._sessions.get(key)
        if session is None:
            session = Session(max_turns=self._max_turns)
            self._sessions[key] = session
        return session

    def reset(self, key: tuple[str, str]) -> None:
        """Forget everything about one student, profile included."""
        self._sessions.pop(key, None)

    def __len__(self) -> int:
        """Return the number of active sessions."""
        return len(self._sessions)
