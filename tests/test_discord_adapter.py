"""Tests for the Discord channel adapter."""

from __future__ import annotations

import asyncio

import pytest

from dora_edu.bot.core import TutorService
from dora_edu.bot.discord.adapter import (
    _RATE_BUTTON_LABEL,
    _RATE_CUSTOM_ID_PREFIX,
    _REPORT_BUTTON_LABEL,
    _REPORT_CUSTOM_ID_PREFIX,
    DiscordAdapter,
    _TutorDiscordClient,
    split_for_discord,
)
from dora_edu.bot.session import SessionStore
from dora_edu.llm import prompts
from dora_edu.rag_engine import retriever as retriever_module
from dora_edu.rag_engine.retriever import Retriever
from tests.conftest import FakeCollection, FakeGenerator


class FakeAuthor:
    def __init__(self, user_id: str, display_name: str = "Hoc sinh", bot: bool = False) -> None:
        self.id = user_id
        self.display_name = display_name
        self.bot = bot


class FakeChannel:
    def __init__(self) -> None:
        self.sent: list[str] = []
        self.views: list[object] = []

    async def send(self, text: str, view: object = None) -> None:
        self.sent.append(text)
        self.views.append(view)


class FakeMessage:
    def __init__(self, user_id: str, content: str, *, bot: bool = False) -> None:
        self.author = FakeAuthor(user_id, bot=bot)
        self.content = content
        self.channel = FakeChannel()


def _run(coro):
    return asyncio.run(coro)


@pytest.fixture
def tutor(monkeypatch, textbook_rows, settings) -> TutorService:
    monkeypatch.setattr(
        retriever_module, "get_collection", lambda *a, **k: FakeCollection(textbook_rows)
    )
    return TutorService(
        retriever=Retriever(settings),
        generator=FakeGenerator(),
        sessions=SessionStore(max_turns=2),
        settings=settings,
    )


def _build_client(tutor) -> tuple[DiscordAdapter, _TutorDiscordClient]:
    import discord

    adapter = DiscordAdapter(tutor, token="fake-token")
    # Avoid opening a real gateway connection: build the client directly.
    client = _TutorDiscordClient(adapter, intents=discord.Intents.none())
    return adapter, client


def test_constructor_rejects_an_empty_token(tutor) -> None:
    with pytest.raises(ValueError, match="DISCORD_BOT_TOKEN"):
        DiscordAdapter(tutor, token="")


def test_a_message_from_another_bot_is_ignored(tutor) -> None:
    _, client = _build_client(tutor)
    message = FakeMessage("9", "hello", bot=True)

    _run(client.on_message(message))

    assert message.channel.sent == []


def test_an_empty_message_is_ignored(tutor) -> None:
    _, client = _build_client(tutor)
    message = FakeMessage("9", "")

    _run(client.on_message(message))

    assert message.channel.sent == []


def test_a_command_gets_answered_in_the_channel(tutor) -> None:
    _, client = _build_client(tutor)
    message = FakeMessage("9", "/start")

    _run(client.on_message(message))

    assert message.channel.sent == [prompts.WELCOME_MESSAGE]


def test_a_question_without_a_profile_gets_the_profile_prompt(tutor) -> None:
    _, client = _build_client(tutor)
    message = FakeMessage("9", "Xin chao co")

    _run(client.on_message(message))

    assert message.channel.sent == [prompts.PROFILE_REQUIRED_MESSAGE]


def test_a_full_conversation_flows_through_the_shared_tutor_service(tutor) -> None:
    _, client = _build_client(tutor)

    _run(client.on_message(FakeMessage("9", "/lop 6")))
    _run(client.on_message(FakeMessage("9", "/mon toan")))
    message = FakeMessage("9", "Phan so la gi?")
    _run(client.on_message(message))

    assert message.channel.sent == ["Cau tra loi mau"]


def test_discord_and_telegram_sessions_for_the_same_user_id_stay_isolated(tutor) -> None:
    _, client = _build_client(tutor)
    _run(client.on_message(FakeMessage("9", "/lop 6")))
    _run(client.on_message(FakeMessage("9", "/mon toan")))

    # dispatch() (used directly, as the Telegram/Zalo adapters do) stamps a
    # different channel name, so the same numeric id starts a blank profile.
    from dora_edu.bot.adapter import ChannelAdapter

    class OtherChannel(ChannelAdapter):
        channel_name = "telegram"

        def run(self) -> None:  # pragma: no cover
            raise NotImplementedError

    other = OtherChannel(tutor)
    reply = other.dispatch("9", "Phan so la gi?")
    assert reply.text == prompts.PROFILE_REQUIRED_MESSAGE


def test_short_replies_are_sent_as_one_discord_message() -> None:
    assert split_for_discord("Chao em!") == ["Chao em!"]


def test_long_replies_are_split_within_the_discord_limit() -> None:
    text = "\n".join(f"Dong so {i}" for i in range(500))
    pieces = split_for_discord(text)

    assert len(pieces) > 1
    assert all(len(piece) <= 2000 for piece in pieces)
    assert "".join(piece.replace("\n", "") for piece in pieces).startswith("Dong so 0")


def test_splitting_prefers_line_boundaries() -> None:
    text = "a" * 100 + "\n" + "b" * 100
    assert split_for_discord(text, limit=150) == ["a" * 100, "b" * 100]


def test_splitting_falls_back_to_hard_cuts_for_unbroken_text() -> None:
    pieces = split_for_discord("x" * 300, limit=100)

    assert len(pieces) == 3
    assert all(len(piece) == 100 for piece in pieces)


# --- the "Báo sai" and "Chấm điểm" buttons ---------------------------------


class FakeInteractionResponse:
    def __init__(self) -> None:
        self.sent: list[tuple[str, bool]] = []
        self.modals: list[object] = []

    async def send_message(self, text: str, ephemeral: bool = False) -> None:
        self.sent.append((text, ephemeral))

    async def send_modal(self, modal: object) -> None:
        self.modals.append(modal)


class FakeInteraction:
    def __init__(self, custom_id: str, user_id: str = "9") -> None:
        self.data = {"custom_id": custom_id}
        self.user = FakeAuthor(user_id)
        self.response = FakeInteractionResponse()


@pytest.fixture
def citing_tutor(monkeypatch, textbook_rows, settings) -> TutorService:
    """A tutor that cites its pages, so answers carry a feedback id."""
    monkeypatch.setattr(
        retriever_module, "get_collection", lambda *a, **k: FakeCollection(textbook_rows)
    )
    return TutorService(
        retriever=Retriever(settings),
        generator=FakeGenerator(cite_chunks=True),
        sessions=SessionStore(max_turns=2),
        settings=settings,
    )


def _answer_one_question(client) -> FakeMessage:
    _run(client.on_message(FakeMessage("9", "/lop 6")))
    _run(client.on_message(FakeMessage("9", "/mon toan")))
    message = FakeMessage("9", "Phan so la gi?")
    _run(client.on_message(message))
    return message


def test_an_answer_carries_the_feedback_buttons(citing_tutor) -> None:
    _, client = _build_client(citing_tutor)

    message = _answer_one_question(client)

    view = message.channel.views[-1]
    assert view is not None
    assert [item.label for item in view.children] == [
        _REPORT_BUTTON_LABEL,
        _RATE_BUTTON_LABEL,
    ]
    assert view.children[0].custom_id.startswith(_REPORT_CUSTOM_ID_PREFIX)
    assert view.children[1].custom_id.startswith(_RATE_CUSTOM_ID_PREFIX)


def test_a_command_reply_carries_no_report_button(citing_tutor) -> None:
    # /start is not an answer about the textbook, so there is nothing to flag.
    _, client = _build_client(citing_tutor)
    message = FakeMessage("9", "/start")

    _run(client.on_message(message))

    assert message.channel.views == [None]


def test_only_the_last_piece_of_a_split_answer_carries_the_button(
    monkeypatch, textbook_rows, settings
) -> None:
    monkeypatch.setattr(
        retriever_module, "get_collection", lambda *a, **k: FakeCollection(textbook_rows)
    )
    tutor = TutorService(
        retriever=Retriever(settings),
        generator=FakeGenerator(answer="x" * 5000, cite_chunks=True),
        sessions=SessionStore(max_turns=2),
        settings=settings,
    )
    _, client = _build_client(tutor)

    message = _answer_one_question(client)

    assert len(message.channel.views) > 1
    assert all(view is None for view in message.channel.views[:-1])
    assert message.channel.views[-1] is not None


def _submit_report(client, custom_id, *, reason: str = "", user_id: str = "9"):
    """Press "Báo sai" and submit the form it opens, as a student would."""
    press = FakeInteraction(custom_id, user_id=user_id)
    _run(client.on_interaction(press))
    modal = press.response.modals[0]
    # discord.py exposes TextInput.value read-only, so seed the backing field.
    modal.reason._value = reason
    submit = FakeInteraction(custom_id, user_id=user_id)
    _run(modal.on_submit(submit))
    return submit


def test_pressing_the_button_files_the_report_and_replies_privately(citing_tutor) -> None:
    _, client = _build_client(citing_tutor)
    message = _answer_one_question(client)
    custom_id = message.channel.views[-1].children[0].custom_id

    interaction = _submit_report(client, custom_id)

    text, ephemeral = interaction.response.sent[0]
    assert "Cảm ơn bạn đã báo" in text
    assert "SGK Toán 6, trang 12" in text
    # Ephemeral: the acknowledgement is for the reporter, not the whole
    # study channel.
    assert ephemeral is True


def test_a_button_from_a_previous_database_is_answered_not_crashed(citing_tutor) -> None:
    _, client = _build_client(citing_tutor)

    interaction = _submit_report(client, f"{_REPORT_CUSTOM_ID_PREFIX}4242")

    assert interaction.response.sent[0][0] == prompts.FEEDBACK_UNKNOWN_MESSAGE


def test_an_interaction_from_another_component_is_ignored(citing_tutor) -> None:
    _, client = _build_client(citing_tutor)
    interaction = FakeInteraction("some:other:button")

    _run(client.on_interaction(interaction))

    assert interaction.response.sent == []


def test_a_classmate_may_flag_an_answer_somebody_else_asked_for(citing_tutor) -> None:
    # In a shared study channel the person who spots the mistake is often
    # not the one who asked.
    _, client = _build_client(citing_tutor)
    message = _answer_one_question(client)
    custom_id = message.channel.views[-1].children[0].custom_id

    interaction = _submit_report(client, custom_id, user_id="77")

    assert "Cảm ơn bạn đã báo" in interaction.response.sent[0][0]


def test_pressing_cham_diem_opens_the_rating_form(citing_tutor) -> None:
    _, client = _build_client(citing_tutor)
    message = _answer_one_question(client)
    custom_id = message.channel.views[-1].children[1].custom_id
    interaction = FakeInteraction(custom_id)

    _run(client.on_interaction(interaction))

    # The form asks for two numbers, so the press opens it instead of
    # recording anything on its own.
    assert len(interaction.response.modals) == 1
    assert interaction.response.sent == []


def test_submitting_the_rating_form_records_both_scores(citing_tutor) -> None:
    tutor_feedback = citing_tutor._feedback
    _, client = _build_client(citing_tutor)
    message = _answer_one_question(client)
    custom_id = message.channel.views[-1].children[1].custom_id

    press = FakeInteraction(custom_id)
    _run(client.on_interaction(press))
    modal = press.response.modals[0]
    # discord.py exposes TextInput.value read-only (Discord fills it in from
    # the submitted form), so a test has to seed the backing field.
    modal.clarity._value = "5"
    modal.guidance._value = "4"
    modal.comment._value = "Giải thích rõ nhưng hơi dài"

    submit = FakeInteraction(custom_id)
    _run(modal.on_submit(submit))

    assert submit.response.sent[0] == (prompts.RATING_THANKS_MESSAGE, True)
    summary = tutor_feedback.summary()
    assert summary.ratings == 1
    assert summary.average_clarity == 5
    assert summary.average_guidance == 4


def test_a_rating_outside_the_scale_is_explained_not_stored(citing_tutor) -> None:
    tutor_feedback = citing_tutor._feedback
    _, client = _build_client(citing_tutor)
    message = _answer_one_question(client)
    custom_id = message.channel.views[-1].children[1].custom_id

    press = FakeInteraction(custom_id)
    _run(client.on_interaction(press))
    modal = press.response.modals[0]
    modal.clarity._value = "9"
    modal.guidance._value = "4"

    submit = FakeInteraction(custom_id)
    _run(modal.on_submit(submit))

    assert submit.response.sent[0][0] == prompts.RATING_USAGE_MESSAGE
    assert tutor_feedback.summary().ratings == 0


def test_the_report_form_keeps_the_reason_the_student_typed(citing_tutor) -> None:
    tutor_feedback = citing_tutor._feedback
    _, client = _build_client(citing_tutor)
    message = _answer_one_question(client)
    custom_id = message.channel.views[-1].children[0].custom_id

    interaction = _submit_report(client, custom_id, reason="Công thức bị sai dấu")

    assert "Cảm ơn bạn đã báo" in interaction.response.sent[0][0]
    assert "lý do bạn ghi" in interaction.response.sent[0][0]
    assert tutor_feedback.recent_reports()[0].note == "Công thức bị sai dấu"


def test_the_reason_field_may_be_left_empty(citing_tutor) -> None:
    # The whole point: a student who only senses the answer is off must be
    # able to report in one tap, without having to explain.
    tutor_feedback = citing_tutor._feedback
    _, client = _build_client(citing_tutor)
    message = _answer_one_question(client)
    custom_id = message.channel.views[-1].children[0].custom_id

    interaction = _submit_report(client, custom_id, reason="")

    assert "Cảm ơn bạn đã báo" in interaction.response.sent[0][0]
    assert "lý do bạn ghi" not in interaction.response.sent[0][0]
    reports = tutor_feedback.recent_reports()
    assert len(reports) == 1 and reports[0].note is None


def test_the_reason_field_is_not_required_by_discord(citing_tutor) -> None:
    _, client = _build_client(citing_tutor)
    message = _answer_one_question(client)
    custom_id = message.channel.views[-1].children[0].custom_id

    press = FakeInteraction(custom_id)
    _run(client.on_interaction(press))

    assert press.response.modals[0].reason.required is False
