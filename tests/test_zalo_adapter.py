"""Tests for the real Zalo Official Account adapter.

Complements ``test_adapter.py`` (which proves the *interface* is
channel-agnostic using a throwaway fake): these tests exercise the actual
``ZaloAdapter`` -- signature verification, payload handling and the Send API
call -- against a fake ``httpx.Client`` instead of a real socket or network.
"""

from __future__ import annotations

import hashlib
import hmac
import json
from typing import Any

import httpx
import pytest

from dora_edu.bot.core import TutorService
from dora_edu.bot.session import SessionStore
from dora_edu.bot.zalo.adapter import ZaloAdapter
from dora_edu.llm import prompts
from dora_edu.rag_engine import retriever as retriever_module
from dora_edu.rag_engine.retriever import Retriever
from tests.conftest import FakeCollection, FakeGenerator

_SECRET = "test-oa-secret"
_TOKEN = "test-access-token"


def _sign(raw_body: bytes) -> str:
    return hmac.new(_SECRET.encode("utf-8"), raw_body, hashlib.sha256).hexdigest()


class FakeHttpxClient:
    """Stands in for ``httpx.Client``, recording every POST it is asked to make."""

    def __init__(self, raise_exc: Exception | None = None) -> None:
        self.calls: list[dict[str, Any]] = []
        self._raise_exc = raise_exc

    def post(self, url: str, *, headers: dict[str, str], json: dict[str, Any]) -> "_FakeResponse":
        self.calls.append({"url": url, "headers": headers, "json": json})
        return _FakeResponse(self._raise_exc)


class _FakeResponse:
    def __init__(self, raise_exc: Exception | None) -> None:
        self._raise_exc = raise_exc

    def raise_for_status(self) -> None:
        if self._raise_exc is not None:
            raise self._raise_exc


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


def test_constructor_rejects_a_missing_access_token(tutor) -> None:
    with pytest.raises(ValueError, match="ZALO_ACCESS_TOKEN"):
        ZaloAdapter(tutor, access_token="", oa_secret=_SECRET)


def test_constructor_rejects_a_missing_oa_secret(tutor) -> None:
    with pytest.raises(ValueError, match="ZALO_OA_SECRET"):
        ZaloAdapter(tutor, access_token=_TOKEN, oa_secret="")


def _build_adapter(tutor, client: FakeHttpxClient) -> ZaloAdapter:
    return ZaloAdapter(tutor, access_token=_TOKEN, oa_secret=_SECRET, client=client)


def _body(**fields: Any) -> bytes:
    return json.dumps(fields).encode("utf-8")


def test_a_correctly_signed_text_message_gets_answered_and_sent_back(tutor) -> None:
    client = FakeHttpxClient()
    adapter = _build_adapter(tutor, client)
    adapter.dispatch("9", "/lop 6")
    adapter.dispatch("9", "/mon toan")

    body = _body(event_name="user_send_text", sender={"id": "9"}, message={"text": "Phan so la gi?"})
    adapter.handle_delivery(body, _sign(body))

    assert len(client.calls) == 1
    call = client.calls[0]
    assert call["headers"]["access_token"] == _TOKEN
    assert call["json"]["recipient"]["user_id"] == "9"
    assert call["json"]["message"]["text"] == "Cau tra loi mau"


def test_a_message_without_a_profile_gets_the_profile_prompt(tutor) -> None:
    client = FakeHttpxClient()
    adapter = _build_adapter(tutor, client)

    body = _body(event_name="user_send_text", sender={"id": "1"}, message={"text": "Xin chao"})
    adapter.handle_delivery(body, _sign(body))

    assert client.calls[0]["json"]["message"]["text"] == prompts.PROFILE_REQUIRED_MESSAGE


def test_an_invalid_signature_is_rejected_without_answering(tutor) -> None:
    client = FakeHttpxClient()
    adapter = _build_adapter(tutor, client)

    body = _body(event_name="user_send_text", sender={"id": "9"}, message={"text": "hi"})
    adapter.handle_delivery(body, "not-the-real-signature")

    assert client.calls == []


def test_a_missing_signature_header_is_rejected(tutor) -> None:
    client = FakeHttpxClient()
    adapter = _build_adapter(tutor, client)

    body = _body(event_name="user_send_text", sender={"id": "9"}, message={"text": "hi"})
    adapter.handle_delivery(body, None)

    assert client.calls == []


def test_malformed_json_is_rejected_without_raising(tutor) -> None:
    client = FakeHttpxClient()
    adapter = _build_adapter(tutor, client)

    body = b"{not valid json"
    adapter.handle_delivery(body, _sign(body))

    assert client.calls == []


def test_non_text_events_are_acknowledged_but_not_answered(tutor) -> None:
    client = FakeHttpxClient()
    adapter = _build_adapter(tutor, client)

    body = _body(event_name="follow", sender={"id": "9"})
    adapter.handle_delivery(body, _sign(body))

    assert client.calls == []


def test_a_send_api_failure_is_logged_not_raised(tutor) -> None:
    client = FakeHttpxClient(raise_exc=httpx.HTTPStatusError("boom", request=None, response=None))
    adapter = _build_adapter(tutor, client)
    adapter.dispatch("9", "/lop 6")
    adapter.dispatch("9", "/mon toan")

    body = _body(event_name="user_send_text", sender={"id": "9"}, message={"text": "hoi gi do"})
    adapter.handle_delivery(body, _sign(body))  # must not raise
