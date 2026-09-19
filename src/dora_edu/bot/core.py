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
from dora_edu.llm.generator import NOT_SUBJECT_SPECIFIC, AnswerGenerator
from dora_edu.models import KNOWN_SUBJECTS, StudentProfile, normalize_subject, validate_grade
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
    # A student easily types a stray space right after the slash ("/ lop 8");
    # without stripping it here, `head` comes out empty and the whole message
    # silently falls through to _handle_question instead of running the command.
    head, _, argument = stripped[1:].lstrip().partition(" ")
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
        # `sessions or SessionStore(...)` would look right but is wrong: SessionStore
        # defines __len__, and a freshly-constructed (or currently empty) store has
        # len() == 0, which Python treats as falsy in the absence of __bool__ -- so
        # `or` would silently discard a real, caller-supplied empty store and build
        # a fresh default one instead, dropping any configured `db_path` with it.
        self._sessions = sessions if sessions is not None else SessionStore(
            max_turns=self._settings.session_max_turns
        )

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
            reply = self._set_grade(argument, session)
            self._sessions.save_profile(message.session_key)
            return reply
        if name in {"mon", "subject"}:
            reply = self._set_subject(argument, session)
            self._sessions.save_profile(message.session_key)
            return reply
        if name in {"toi", "me"}:
            return self._describe_profile(session)
        if name in {"xoa", "reset"}:
            session.clear_history()
            return OutgoingMessage(text="Mình đã xoá lịch sử trò chuyện rồi nhé! Bạn hỏi tiếp đi. 😊")

        logger.info("Unknown command %r from %s", name, message.channel)
        return OutgoingMessage(
            text=f"Mình chưa hiểu lệnh /{name}. Bạn xem lại các lệnh bằng /trogiup nhé!"
        )

    def _set_grade(self, argument: str, session: Session) -> OutgoingMessage:
        """Set or update the student's grade, keeping any subject already chosen."""
        if not argument:
            return OutgoingMessage(text="Bạn nhập lớp giúp mình nhé, ví dụ: /lop 6")
        try:
            grade = validate_grade(argument)
        except ValueError:
            return OutgoingMessage(text="Lớp phải là số từ 1 đến 12 bạn nhé. Ví dụ: /lop 8")

        # Changing grade invalidates the conversation: the retrieval scope moved.
        if session.grade != grade:
            session.clear_history()
        session.grade = grade

        if session.subject is None:
            return OutgoingMessage(
                text=f"Mình ghi nhận bạn học lớp {grade}. Bạn hỏi luôn được rồi, mình sẽ tự "
                "nhận diện môn theo câu hỏi nhé! Muốn cố định 1 môn thì gõ /mon, ví dụ: /mon Toán 😊"
            )
        return OutgoingMessage(
            text=f"Đã chọn lớp {grade}, môn {session.subject}. Bạn hỏi mình đi nào! 😊"
        )

    def _set_subject(self, argument: str, session: Session) -> OutgoingMessage:
        """Set or update the student's subject, keeping any grade already chosen."""
        if not argument:
            return OutgoingMessage(text="Bạn nhập môn giúp mình nhé, ví dụ: /mon Lịch sử")
        try:
            subject = normalize_subject(argument)
        except ValueError:
            return OutgoingMessage(text="Bạn nhập tên môn giúp mình nhé, ví dụ: /mon Ngữ văn")

        # subject flows straight into the LLM system prompt (build_system_prompt),
        # so only a real, indexed subject may pass -- never arbitrary student text.
        if subject not in KNOWN_SUBJECTS:
            return OutgoingMessage(
                text="Mình chưa có môn này trong sách. Bạn nhập đúng tên môn học nhé, "
                "ví dụ: /mon Toán, /mon Ngữ văn, /mon Lịch sử..."
            )

        if session.subject != subject:
            session.clear_history()
        session.subject = subject

        if session.grade is None:
            return OutgoingMessage(
                text=f"Mình ghi nhận môn {subject}. Giờ bạn cho mình biết bạn học lớp mấy nhé, "
                "ví dụ: /lop 6"
            )
        return OutgoingMessage(
            text=f"Đã chọn lớp {session.grade}, môn {subject}. Bạn cứ hỏi mình thoải mái nhé! 😊"
        )

    def _describe_profile(self, session: Session) -> OutgoingMessage:
        """Report the student's current grade and subject."""
        if session.grade is None:
            return OutgoingMessage(text=prompts.PROFILE_REQUIRED_MESSAGE)
        if session.subject is None:
            return OutgoingMessage(
                text=f"Hiện bạn đang học lớp {session.grade}. Môn thì mình tự nhận diện theo "
                "từng câu hỏi của bạn. 📘"
            )
        return OutgoingMessage(
            text=f"Hiện bạn đang học lớp {session.grade}, môn {session.subject}. 📘"
        )

    # --- Questions ----------------------------------------------------------

    def _handle_question(self, message: IncomingMessage, session: Session) -> OutgoingMessage:
        """Answer a free-form question, scoped to the student's grade and subject.

        The subject only needs to be set explicitly (``/mon``) when the student
        wants to pin one; otherwise it is detected per question. Either way,
        every retrieval call that actually reaches ChromaDB still carries both
        a grade and a subject filter.
        """
        question = message.text.strip()
        if not question:
            return OutgoingMessage(text="Bạn nhắn câu hỏi cho mình nhé! 😊")

        # Rule: never query the vector store without at least a grade.
        if session.grade is None:
            return OutgoingMessage(text=prompts.PROFILE_REQUIRED_MESSAGE)

        try:
            if session.subject is not None:
                subject = session.subject
            else:
                subject = self._detect_subject(question, session.grade)
                if subject == NOT_SUBJECT_SPECIFIC:
                    subjects = self._retriever.list_subjects(session.grade)
                    return OutgoingMessage(
                        text=prompts.build_capability_message(session.grade, subjects)
                    )
                if subject is None:
                    return OutgoingMessage(text=prompts.SUBJECT_NOT_DETECTED_MESSAGE)
            profile = StudentProfile(grade=session.grade, subject=subject)
            chunks = self._retriever.retrieve(question, profile)
        except (ValueError, RuntimeError) as exc:
            logger.error("Retrieval failed for %s: %s", message.session_key, exc)
            return OutgoingMessage(
                text="Mình đang gặp trục trặc khi tra cứu sách. Bạn thử lại sau ít phút nhé! 😔"
            )

        result = self._generator.generate(question, chunks, profile, session.history())
        session.record_turn(question, result.answer)
        return OutgoingMessage(text=result.answer)

    def _detect_subject(self, question: str, grade: int) -> str | None:
        """Work out which subject of ``grade`` a subject-less question is about.

        Vector distance alone is not reliable enough to tell subjects apart
        (a history question can embed closer to a math passage than to the
        right history one), so this asks the LLM to classify the question
        first. If the LLM is unreachable, it falls back to the nearest-match
        heuristic rather than failing the question outright.

        A question that is not about any subject's content at all (e.g. "bạn
        hỗ trợ những môn nào") is reported as :data:`NOT_SUBJECT_SPECIFIC`
        rather than falling back to nearest-distance matching -- that
        fallback is only meant for genuinely ambiguous subject-content
        questions, and would otherwise anchor a non-content question to
        whichever textbook's wording happens to embed closest to it.

        Returns:
            One of the grade's indexed subjects, :data:`NOT_SUBJECT_SPECIFIC`,
            or ``None`` when neither approach finds a plausible match.
        """
        subjects = self._retriever.list_subjects(grade)
        detected = self._generator.classify_subject(question, subjects)
        if detected is not None:
            return detected

        logger.info("LLM subject classification inconclusive; falling back to nearest match")
        _chunks, detected = self._retriever.retrieve_best_subject(question, grade)
        return detected
