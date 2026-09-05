"""Channel-independent tutoring logic.

This is the shared brain behind every adapter: it owns command handling,
session state, retrieval scoping and answer generation. It never imports a
messaging library, so Telegram and Zalo behave identically by construction.
"""

from __future__ import annotations

import logging

from dora_edu.bot.adapter import IncomingMessage, OutgoingMessage
from dora_edu.bot.session import Session, SessionStore
from dora_edu.config import Settings, get_settings
from dora_edu.llm import prompts
from dora_edu.llm.generator import AnswerGenerator
from dora_edu.models import StudentProfile, normalize_subject, validate_grade
from dora_edu.rag_engine.retriever import Retriever

logger = logging.getLogger(__name__)

#: Commands are ASCII so every messaging platform accepts them unchanged.
_COMMAND_PREFIX = "/"


def parse_command(text: str) -> tuple[str, str] | None:
    """Split a slash command into its name and argument.

    Telegram appends ``@botname`` to commands in group chats; that suffix is
    stripped here so the same parsing works on every channel.

    Args:
        text: Raw message text.

    Returns:
        A ``(command, argument)`` pair with the command lowercased and without
        its slash, or ``None`` when the message is not a command.
    """
    stripped = text.strip()
    if not stripped.startswith(_COMMAND_PREFIX):
        return None
    head, _, argument = stripped[1:].partition(" ")
    command = head.split("@", 1)[0].lower()
    if not command:
        return None
    return command, argument.strip()


class TutorService:
    """Turns a student message into a textbook-grounded tutoring reply."""

    def __init__(
        self,
        retriever: Retriever,
        generator: AnswerGenerator,
        sessions: SessionStore | None = None,
        settings: Settings | None = None,
    ) -> None:
        """Wire the service to its retrieval, generation and session backends."""
        self._settings = settings or get_settings()
        self._retriever = retriever
        self._generator = generator
        self._sessions = sessions or SessionStore(max_turns=self._settings.session_max_turns)

    def __call__(self, message: IncomingMessage) -> OutgoingMessage:
        """Satisfy the :class:`~dora_edu.bot.adapter.MessageHandler` protocol."""
        return self.handle(message)

    def handle(self, message: IncomingMessage) -> OutgoingMessage:
        """Handle one student message end to end.

        Args:
            message: The normalised incoming message.

        Returns:
            The reply to send back on the originating channel.
        """
        session = self._sessions.get(message.session_key)
        command = parse_command(message.text)
        if command is not None:
            return self._handle_command(command, message, session)
        return self._handle_question(message, session)

    # --- Commands -----------------------------------------------------------

    def _handle_command(
        self,
        command: tuple[str, str],
        message: IncomingMessage,
        session: Session,
    ) -> OutgoingMessage:
        """Dispatch a slash command."""
        name, argument = command

        if name in {"start", "batdau"}:
            return OutgoingMessage(text=prompts.WELCOME_MESSAGE)
        if name in {"help", "trogiup"}:
            return OutgoingMessage(text=prompts.HELP_MESSAGE)
        if name in {"lop", "grade"}:
            return self._set_grade(argument, session)
        if name in {"mon", "subject"}:
            return self._set_subject(argument, session)
        if name in {"toi", "me"}:
            return self._describe_profile(session)
        if name in {"xoa", "reset"}:
            session.clear_history()
            return OutgoingMessage(text="Cô đã xoá lịch sử trò chuyện rồi nhé! Em hỏi tiếp đi. 😊")

        logger.info("Unknown command %r from %s", name, message.channel)
        return OutgoingMessage(
            text=f"Cô chưa hiểu lệnh /{name}. Em xem lại các lệnh bằng /trogiup nhé!"
        )

    def _set_grade(self, argument: str, session: Session) -> OutgoingMessage:
        """Set or update the student's grade, keeping any subject already chosen."""
        if not argument:
            return OutgoingMessage(text="Em nhập lớp giúp cô nhé, ví dụ: /lop 6")
        try:
            grade = validate_grade(argument)
        except ValueError:
            return OutgoingMessage(text="Lớp phải là số từ 1 đến 12 em nhé. Ví dụ: /lop 8")

        # Changing grade invalidates the conversation: the retrieval scope moved.
        if session.grade != grade:
            session.clear_history()
        session.grade = grade

        if session.subject is None:
            return OutgoingMessage(
                text=f"Cô ghi nhận em học lớp {grade}. Giờ em chọn môn nhé, ví dụ: /mon Toán"
            )
        return OutgoingMessage(
            text=f"Đã chọn lớp {grade}, môn {session.subject}. Em hỏi cô đi nào! 😊"
        )

    def _set_subject(self, argument: str, session: Session) -> OutgoingMessage:
        """Set or update the student's subject, keeping any grade already chosen."""
        if not argument:
            return OutgoingMessage(text="Em nhập môn giúp cô nhé, ví dụ: /mon Lịch sử")
        try:
            subject = normalize_subject(argument)
        except ValueError:
            return OutgoingMessage(text="Em nhập tên môn giúp cô nhé, ví dụ: /mon Ngữ văn")

        if session.subject != subject:
            session.clear_history()
        session.subject = subject

        if session.grade is None:
            return OutgoingMessage(
                text=f"Cô ghi nhận môn {subject}. Giờ em cho cô biết em học lớp mấy nhé, "
                "ví dụ: /lop 6"
            )
        return OutgoingMessage(
            text=f"Đã chọn lớp {session.grade}, môn {subject}. Em cứ hỏi cô thoải mái nhé! 😊"
        )

    def _describe_profile(self, session: Session) -> OutgoingMessage:
        """Report the student's current grade and subject."""
        if not session.has_profile:
            return OutgoingMessage(text=prompts.PROFILE_REQUIRED_MESSAGE)
        return OutgoingMessage(
            text=f"Hiện em đang học lớp {session.grade}, môn {session.subject}. 📘"
        )

    # --- Questions ----------------------------------------------------------

    def _handle_question(self, message: IncomingMessage, session: Session) -> OutgoingMessage:
        """Answer a free-form question, scoped to the student's grade and subject."""
        question = message.text.strip()
        if not question:
            return OutgoingMessage(text="Em nhắn câu hỏi cho cô nhé! 😊")

        # Rule: never query the vector store without a grade and a subject.
        if not session.has_profile:
            return OutgoingMessage(text=prompts.PROFILE_REQUIRED_MESSAGE)

        profile = session.profile
        try:
            chunks = self._retriever.retrieve(question, profile)
        except (ValueError, RuntimeError) as exc:
            logger.error("Retrieval failed for %s: %s", message.session_key, exc)
            return OutgoingMessage(
                text="Cô đang gặp trục trặc khi tra cứu sách. Em thử lại sau ít phút nhé! 😔"
            )

        result = self._generator.generate(question, chunks, profile, session.history())
        session.record_turn(question, result.answer)
        return OutgoingMessage(text=result.answer)
