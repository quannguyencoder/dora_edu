"""Channel-independent tutoring logic.

This is the shared brain behind every adapter: it owns command handling,
session state, retrieval scoping and answer generation. It never imports a
messaging library, so Telegram and Zalo behave identically by construction.
"""

from __future__ import annotations

import logging
import re
import secrets

from dora_edu.bot.adapter import IncomingMessage, OutgoingMessage
from dora_edu.bot.feedback import FeedbackStore
from dora_edu.bot.session import Session, SessionStore
from dora_edu.config import Settings, get_settings, parse_model_list
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


#: How a channel's own control passes the answer it refers to: "#12". The "#"
#: is what keeps it apart from a score -- a student typing "/chamdiem 5 4"
#: means "clarity 5, guidance 4 for the last answer", and without the marker
#: a leading 5 would be indistinguishable from the id of answer 5.
_FEEDBACK_ID_TOKEN = re.compile(r"^#(\d+)$")

#: ``/baosai`` carries no scores, so a bare number there is unambiguous and
#: is accepted as the id too -- which keeps "/baosai 12" working for anyone
#: (or any older button) that sends the id without the marker.
_REPORT_ID_TOKEN = re.compile(r"^#?(\d+)$")


def _parse_report(
    argument: str, *, fallback_id: int | None
) -> tuple[int | None, str | None]:
    """Split a ``/baosai`` argument into an answer id and the student's reason.

    Accepts ``"#12 công thức sai"`` (what the Discord report form sends),
    ``"#12"`` or ``"12"`` (just the id), and ``"câu này sai rồi"`` (a
    student typing a reason for the last answer, no id at all).

    Args:
        argument: Everything after the command name.
        fallback_id: The answer to flag when the argument names none.

    Returns:
        An ``(answer_id, note)`` pair; ``note`` is ``None`` when the student
        gave no reason, which is the common case and perfectly fine.
    """
    tokens = argument.split()
    answer_id = fallback_id
    if tokens:
        match = _REPORT_ID_TOKEN.match(tokens[0])
        if match:
            answer_id = int(match.group(1))
            tokens = tokens[1:]
    return answer_id, " ".join(tokens).strip() or None


def _parse_rating(
    argument: str, *, fallback_id: int | None
) -> tuple[int | None, int | None, int | None, str | None]:
    """Parse a ``/chamdiem`` argument into an answer id, two scores and a note.

    Accepts what a student plausibly types (``"5 4"``, ``"5"``, ``"5 4 hơi
    dài"``) as well as what the Discord rating form sends (``"#12 5 4"``).

    Args:
        argument: Everything after the command name.
        fallback_id: The answer to rate when the argument names none.

    Returns:
        An ``(answer_id, clarity, guidance, comment)`` tuple; any element
        may be ``None`` when the argument did not carry it.
    """
    tokens = argument.split()
    answer_id = fallback_id
    if tokens:
        match = _FEEDBACK_ID_TOKEN.match(tokens[0])
        if match:
            answer_id = int(match.group(1))
            tokens = tokens[1:]

    scores: list[int] = []
    while tokens and tokens[0].isdigit() and len(scores) < 2:
        scores.append(int(tokens.pop(0)))

    return (
        answer_id,
        scores[0] if scores else None,
        scores[1] if len(scores) > 1 else None,
        " ".join(tokens).strip() or None,
    )


class TutorService:
    """Turns a student message into a textbook-grounded tutoring reply."""

    def __init__(
        self,
        retriever: Retriever,
        generator: AnswerGenerator,
        sessions: SessionStore | None = None,
        settings: Settings | None = None,
        feedback: FeedbackStore | None = None,
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
        # Defaulting to an in-memory store (rather than one pointed at
        # settings.feedback_db_path) keeps tests from writing a database
        # file; the bot entry points pass a persistent one explicitly.
        self._feedback = feedback if feedback is not None else FeedbackStore()

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
        try:
            session = self._sessions.get(message.session_key)
            command = parse_command(message.text)
            if command is not None:
                return self._handle_command(command, message, session)
            return self._handle_question(message, session)
        except Exception as exc:
            # This is the outermost boundary for one student's message: every
            # layer below already catches the specific errors it knows how
            # to handle, but a genuinely unexpected failure here (a new SDK
            # exception type, a bug) must never propagate past this point.
            # A crash here reaches the adapter as an uncaught exception,
            # which every channel logs and silently drops -- the student
            # gets no reply at all, not even an apology, which is worse
            # than any of the specific error messages below.
            logger.exception(
                "Unexpected error handling message from %s: %s", message.session_key, exc
            )
            return OutgoingMessage(
                text="Mình đang gặp trục trặc kỹ thuật. Bạn thử lại sau ít phút nhé! 😔"
            )

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
        if name in {"baosai", "report"}:
            return self._report_wrong_answer(argument, message, session)
        if name in {"chamdiem", "rate"}:
            return self._rate_answer(argument, message, session)
        if name in {"phanhoi", "thongke"}:
            return self._show_feedback(argument, message)
        if name in {"doimatkhau", "password"}:
            return self._change_password(argument, message)
        if name in {"xoa", "reset"}:
            session.clear_history()
            return OutgoingMessage(text="Mình đã xoá lịch sử trò chuyện rồi nhé! Bạn hỏi tiếp đi. 😊")

        logger.info("Unknown command %r from %s", name, message.channel)
        return OutgoingMessage(
            text=f"Mình chưa hiểu lệnh /{name}. Bạn xem lại các lệnh bằng /trogiup nhé!"
        )

    def _report_wrong_answer(
        self, argument: str, message: IncomingMessage, session: Session
    ) -> OutgoingMessage:
        """Record that a student flagged an answer as wrong.

        The id comes either from the channel's report control (the Discord
        form passes it as the command argument) or, when the channel has no
        buttons, from the last answer in this student's session. Anything
        after the id is kept as the student's own account of what is wrong,
        which is optional: a report has to stay one tap, or the students who
        notice a mistake but cannot articulate it simply will not report.

        Args:
            argument: ``"[#id] [lý do]"`` -- either part may be absent.
            message: The incoming message, used for the reporter's identity.
            session: The student's session.

        Returns:
            A thank-you naming the page to check, or an explanation of why
            the answer could no longer be flagged.
        """
        answer_id, note = _parse_report(argument, fallback_id=session.last_feedback_id)
        if answer_id is None:
            return OutgoingMessage(text=prompts.FEEDBACK_NOTHING_TO_REPORT_MESSAGE)

        record = self._feedback.record_report(
            answer_id, message.user_id, note, message.display_name
        )
        if record is None:
            return OutgoingMessage(text=prompts.FEEDBACK_UNKNOWN_MESSAGE)

        logger.info(
            "Answer %s flagged by %s on %s (grade=%s subject=%s grounded=%s note=%r)",
            answer_id,
            message.user_id,
            message.channel,
            record.grade,
            record.subject,
            record.grounded,
            note,
        )
        return OutgoingMessage(
            text=prompts.build_feedback_thanks(record.citations, gave_reason=note is not None),
            citations=record.citations,
        )

    def _is_allowed_feedback_reader(self, password: str, user_id: str) -> bool:
        """Whether this person may read the feedback reports.

        Args:
            password: What they typed after the command, possibly empty.
            user_id: Their id on the channel they are writing from.

        Returns:
            ``True`` when they are on the admin list, or gave the right
            password. ``False`` when neither is configured, so a deployment
            nobody has set up cannot be opened by guessing.
        """
        if user_id in parse_model_list(self._settings.admin_user_ids):
            return True
        if not password:
            return False
        if self._feedback.has_password():
            # A password set from the chat app wins over the one in .env, so
            # changing it actually changes it -- otherwise the old one from
            # the config file would keep working forever.
            return self._feedback.check_password(password)
        expected = self._settings.feedback_password
        if not expected:
            return False
        # Constant-time comparison: a plain `==` returns faster the earlier
        # the first wrong character is, which leaks the password one
        # character at a time to anyone patient enough to measure.
        return secrets.compare_digest(password, expected)

    def _change_password(self, argument: str, message: IncomingMessage) -> OutgoingMessage:
        """Change the feedback password from the chat app.

        Whoever can already read the feedback may change the password:
        either an id on the admin list, or anyone who types the current
        password. That is what makes rotating it possible after it leaks,
        without editing a file on the machine running the bot.

        Args:
            argument: ``"<mật khẩu cũ> <mật khẩu mới>"``, or just the new
                password for an admin who needs no old one.
            message: The incoming message, for identity and privacy.

        Returns:
            Confirmation, or what is missing.
        """
        if not message.is_private:
            logger.warning("Password change attempted in a shared channel")
            return OutgoingMessage(text=prompts.FEEDBACK_PASSWORD_LEAKED_MESSAGE)

        parts = argument.split()
        is_admin = message.user_id in parse_model_list(self._settings.admin_user_ids)

        if is_admin and len(parts) == 1:
            old, new = "", parts[0]
        elif len(parts) == 2:
            old, new = parts
        else:
            return OutgoingMessage(text=prompts.PASSWORD_USAGE_MESSAGE)

        if not is_admin and not self._is_allowed_feedback_reader(old, message.user_id):
            logger.info("Refused password change for %s", message.user_id)
            return OutgoingMessage(text=prompts.FEEDBACK_WRONG_PASSWORD_MESSAGE)

        if len(new) < prompts.MIN_PASSWORD_LENGTH:
            return OutgoingMessage(text=prompts.PASSWORD_TOO_SHORT_MESSAGE)

        self._feedback.set_password(new)
        logger.info("Feedback password changed by %s", message.user_id)
        return OutgoingMessage(text=prompts.PASSWORD_CHANGED_MESSAGE)

    def _show_feedback(self, argument: str, message: IncomingMessage) -> OutgoingMessage:
        """Show the collected feedback, for whoever runs the bot.

        Reading the review sheet means running ``dora-feedback`` in a
        terminal, which is a barrier when the person who wants to see "did
        anyone report anything?" is on their phone. This answers the same
        question in the chat app, with the reporters' names, so the team
        can act on feedback the day it arrives.

        Two ways in, because the reports name other students and are not for
        classmates to browse: being listed in ``ADMIN_USER_IDS``, or typing
        the password in ``FEEDBACK_PASSWORD``. With neither configured,
        nobody can read it.

        Args:
            argument: Everything after the command -- the password, if given.
            message: The incoming message, for the requester's identity.

        Returns:
            The summary, or a refusal explaining what is missing.
        """
        password = argument.strip()
        if password and not message.is_private:
            # The password was just typed into a channel everyone can read,
            # so it is no longer a secret. Say so plainly: a password the
            # owner believes is private but isn't is worse than none.
            logger.warning("Feedback password was typed in a shared channel")
            return OutgoingMessage(text=prompts.FEEDBACK_PASSWORD_LEAKED_MESSAGE)

        if not self._is_allowed_feedback_reader(password, message.user_id):
            # Deliberately never logs the attempted password.
            logger.info("Refused feedback summary for %s", message.user_id)
            return OutgoingMessage(
                text=prompts.FEEDBACK_WRONG_PASSWORD_MESSAGE
                if password
                else prompts.ADMIN_ONLY_MESSAGE
            )

        if not message.is_private:
            # A reply in a shared study channel is visible to everyone in it,
            # and this one names the students who reported and what they said.
            # Being allowed to read it is not permission to show it to the class.
            logger.info("Refusing to print the feedback summary in a shared channel")
            return OutgoingMessage(text=prompts.FEEDBACK_PRIVATE_ONLY_MESSAGE)

        return OutgoingMessage(
            text=prompts.build_feedback_summary_message(
                self._feedback.summary(), self._feedback.recent_reports(limit=5)
            )
        )

    def _rate_answer(
        self, argument: str, message: IncomingMessage, session: Session
    ) -> OutgoingMessage:
        """Record one student's 1-5 scores for an answer.

        This is how the classroom trial collects its numbers: each tester
        scores how easy the answer was to understand, and whether the
        Socratic hinting actually helped them work it out themselves, on the
        same 1-5 scale a teacher would use.

        Args:
            argument: ``"<clarity> <guidance> [ghi chú]"`` as a student
                types it, or ``"<id> <clarity> <guidance> [ghi chú]"`` as
                the Discord rating form sends it.
            message: The incoming message, for the rater's identity.
            session: The student's session, holding the answer to rate when
                no id is given.

        Returns:
            A thank-you, or an explanation of what to type instead.
        """
        answer_id, clarity, guidance, comment = _parse_rating(
            argument, fallback_id=session.last_feedback_id
        )
        if answer_id is None:
            return OutgoingMessage(text=prompts.FEEDBACK_NOTHING_TO_REPORT_MESSAGE)
        if clarity is None and guidance is None:
            return OutgoingMessage(text=prompts.RATING_USAGE_MESSAGE)

        try:
            record = self._feedback.record_rating(
                answer_id,
                message.user_id,
                clarity=clarity,
                guidance=guidance,
                comment=comment,
                user_name=message.display_name,
            )
        except ValueError:
            # A score outside 1-5: a typo in a hand-typed command, not a bug.
            return OutgoingMessage(text=prompts.RATING_USAGE_MESSAGE)
        if record is None:
            return OutgoingMessage(text=prompts.FEEDBACK_UNKNOWN_MESSAGE)

        logger.info(
            "Answer %s rated by %s (clarity=%s guidance=%s)",
            answer_id,
            message.user_id,
            clarity,
            guidance,
        )
        return OutgoingMessage(text=prompts.RATING_THANKS_MESSAGE)

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

        # A student casually dropping Vietnamese diacritics while typing fast
        # ("la gi" instead of "là gì") can shift the retrieval embedding far
        # enough to miss real content entirely, even though a person -- or
        # the LLM itself -- reads both the same way. Restoring diacritics
        # here only changes what gets searched, never the retrieved
        # textbook content, so it carries no hallucination risk.
        question = self._generator.restore_diacritics(question)

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
        answer = prompts.hide_internal_wording(result.answer)
        session.record_turn(question, answer)

        # Of the ten passages retrieved, an answer normally leans on one or
        # two; pointing the student at all ten would send them leafing
        # through pages unrelated to their question. The answer names its
        # own pages (rule 8), so those are matched back against what was
        # actually retrieved -- which also drops a page the LLM invented.
        # Empty citations mean the answer is not grounded in any passage (a
        # refusal), and a refusal has no page to check.
        cited = prompts.select_cited_sources(answer, chunks) if result.citations else []

        # Logged before sending, and for refusals too: a refusal the student
        # believes is wrong ("the book *does* cover this!") is exactly as
        # worth collecting as a wrong answer.
        feedback_id = self._feedback.record_answer(
            channel=message.channel,
            user_id=message.user_id,
            display_name=message.display_name,
            grade=profile.grade,
            subject=profile.subject,
            question=question,
            answer=answer,
            citations=cited,
            grounded=result.grounded,
        )
        session.last_feedback_id = feedback_id

        # Built from retrieval metadata, not from the LLM's prose, so it can
        # only ever name a page that really was retrieved.
        reminder = prompts.build_page_check_reminder(cited)
        text = f"{answer}\n\n{reminder}" if reminder else answer
        return OutgoingMessage(text=text, citations=cited, feedback_id=feedback_id)

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
