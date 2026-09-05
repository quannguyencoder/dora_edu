"""Tests proving the bot layer stays channel-agnostic.

The point of these tests is that a brand new channel can be added by writing an
adapter alone: no retrieval, prompt or session code changes.
"""

from __future__ import annotations

import pytest

from dora_edu.bot.adapter import ChannelAdapter, IncomingMessage, OutgoingMessage
from dora_edu.bot.core import TutorService
from dora_edu.bot.session import SessionStore
from dora_edu.bot.telegram.adapter import split_for_telegram
from dora_edu.llm import prompts
from dora_edu.rag_engine import retriever as retriever_module
from dora_edu.rag_engine.retriever import Retriever
from tests.conftest import FakeCollection, FakeGenerator


class FakeZaloAdapter(ChannelAdapter):
    """A stand-in for the planned Zalo channel, written against the interface only."""

    channel_name = "zalo"

    def __init__(self, handler) -> None:
        super().__init__(handler)
        self.sent: list[str] = []

    def run(self) -> None:  # pragma: no cover - nothing to poll in a test
        raise NotImplementedError

    def receive(self, user_id: str, text: str) -> str:
        """Simulate one inbound Zalo webhook message."""
        reply = self.dispatch(user_id, text, display_name="Hoc sinh Zalo")
        self.sent.append(reply.text)
        return reply.text


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


def test_incoming_message_session_key_pairs_channel_and_user() -> None:
    message = IncomingMessage(channel="zalo", user_id="42", text="xin chao")
    assert message.session_key == ("zalo", "42")


def test_outgoing_message_rejects_an_empty_reply() -> None:
    with pytest.raises(ValueError):
        OutgoingMessage(text="")


def test_a_new_channel_works_against_the_unchanged_tutoring_core(tutor) -> None:
    zalo = FakeZaloAdapter(tutor)

    assert zalo.receive("9", "/start") == prompts.WELCOME_MESSAGE
    zalo.receive("9", "/lop 6")
    zalo.receive("9", "/mon toan")
    answer = zalo.receive("9", "Phan so la gi?")

    assert answer == "Cau tra loi mau"


def test_adapter_stamps_its_own_channel_name_on_every_message(tutor) -> None:
    zalo = FakeZaloAdapter(tutor)
    zalo.receive("9", "/lop 6")
    zalo.receive("9", "/mon toan")

    # The same numeric id on another channel must start from a blank profile.
    other = FakeZaloAdapter(tutor)
    other.channel_name = "telegram"
    assert other.receive("9", "Phan so la gi?") == prompts.PROFILE_REQUIRED_MESSAGE


def test_short_replies_are_sent_as_one_telegram_message() -> None:
    assert split_for_telegram("Chao em!") == ["Chao em!"]


def test_long_replies_are_split_within_the_telegram_limit() -> None:
    text = "\n".join(f"Dong so {i}" for i in range(2000))
    pieces = split_for_telegram(text)

    assert len(pieces) > 1
    assert all(len(piece) <= 4096 for piece in pieces)
    assert "".join(piece.replace("\n", "") for piece in pieces).startswith("Dong so 0")


def test_splitting_prefers_line_boundaries() -> None:
    text = "a" * 100 + "\n" + "b" * 100
    assert split_for_telegram(text, limit=150) == ["a" * 100, "b" * 100]


def test_splitting_falls_back_to_hard_cuts_for_unbroken_text() -> None:
    pieces = split_for_telegram("x" * 300, limit=100)

    assert len(pieces) == 3
    assert all(len(piece) == 100 for piece in pieces)
