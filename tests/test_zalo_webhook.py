"""Tests for Zalo webhook payload validation and signature verification."""

from __future__ import annotations

import hashlib
import hmac
import json

import pytest

from dora_edu.bot.zalo.webhook import ZaloWebhookEvent, verify_signature

_SECRET = "test-oa-secret"


def _sign(raw_body: bytes, secret: str = _SECRET) -> str:
    return hmac.new(secret.encode("utf-8"), raw_body, hashlib.sha256).hexdigest()


def test_verify_signature_accepts_a_correctly_signed_body() -> None:
    body = b'{"event_name": "user_send_text"}'
    assert verify_signature(body, _sign(body), _SECRET) is True


def test_verify_signature_rejects_a_tampered_body() -> None:
    body = b'{"event_name": "user_send_text"}'
    signature = _sign(body)
    tampered = body.replace(b"user_send_text", b"user_send_image")
    assert verify_signature(tampered, signature, _SECRET) is False


def test_verify_signature_rejects_the_wrong_secret() -> None:
    body = b'{"event_name": "user_send_text"}'
    assert verify_signature(body, _sign(body, secret="wrong-secret"), _SECRET) is False


def test_verify_signature_rejects_a_missing_header() -> None:
    assert verify_signature(b"{}", None, _SECRET) is False


def test_zalo_webhook_event_parses_a_real_shaped_text_message() -> None:
    payload = {
        "app_id": "123",
        "event_name": "user_send_text",
        "sender": {"id": "user-1"},
        "recipient": {"id": "oa-1"},
        "message": {"text": "Phan so la gi?", "msg_id": "abc"},
        "timestamp": "1700000000000",
    }
    event = ZaloWebhookEvent.model_validate(payload)

    assert event.sender.id == "user-1"
    assert event.message is not None
    assert event.message.text == "Phan so la gi?"
    assert event.is_text_message is True


def test_zalo_webhook_event_ignores_unknown_extra_fields() -> None:
    payload = {
        "event_name": "user_send_text",
        "sender": {"id": "user-1"},
        "message": {"text": "hi", "some_future_field": {"nested": True}},
        "another_future_top_level_field": 42,
    }
    ZaloWebhookEvent.model_validate(payload)  # must not raise


@pytest.mark.parametrize(
    "payload",
    [
        {"event_name": "follow", "sender": {"id": "user-1"}},
        {"event_name": "user_send_image", "sender": {"id": "user-1"}, "message": {"text": None}},
        {"event_name": "user_send_text", "sender": {"id": "user-1"}, "message": {"text": ""}},
        {"event_name": "user_send_text", "sender": {"id": "user-1"}},
    ],
)
def test_non_text_or_empty_events_are_not_text_messages(payload: dict) -> None:
    event = ZaloWebhookEvent.model_validate(payload)
    assert event.is_text_message is False


def test_zalo_webhook_event_rejects_a_missing_sender() -> None:
    with pytest.raises(ValueError):
        ZaloWebhookEvent.model_validate({"event_name": "user_send_text"})


def test_payload_round_trips_through_json_like_a_real_request_body() -> None:
    body = json.dumps(
        {"event_name": "user_send_text", "sender": {"id": "u1"}, "message": {"text": "chao co"}}
    ).encode("utf-8")
    event = ZaloWebhookEvent.model_validate(json.loads(body))
    assert verify_signature(body, _sign(body), _SECRET) is True
    assert event.message.text == "chao co"  # type: ignore[union-attr]
