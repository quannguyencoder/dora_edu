"""Tests for the Discord channel adapter."""

from __future__ import annotations

import asyncio

import pytest

from dora_edu.bot.core import TutorService
from dora_edu.bot.discord.adapter import DiscordAdapter, _TutorDiscordClient, split_for_discord
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

    async def send(self, text: str) -> None:
        self.sent.append(text)


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
