"""Zalo OA webhook payload validation and signature verification.

Kept separate from the HTTP server in :mod:`dora_edu.bot.zalo.adapter` so this
module's parsing and signature logic can be unit tested without opening a
socket.

NOTE: modelled from Zalo Official Account's public webhook documentation
(the ``X-ZEvent-Signature`` HMAC-SHA256 header and the ``user_send_text``
event shape). Zalo has changed these across API versions before -- verify
both against a real OA app before relying on this in production.
"""

from __future__ import annotations

import hashlib
import hmac

from pydantic import BaseModel, ConfigDict, Field

#: The HTTP header Zalo signs every webhook delivery with.
SIGNATURE_HEADER = "X-ZEvent-Signature"

#: The only event this text-only tutor answers; stickers, images, follows,
#: etc. are acknowledged (200 OK) but otherwise ignored.
TEXT_MESSAGE_EVENT = "user_send_text"


class ZaloSender(BaseModel):
    """The Zalo user who sent the event."""

    model_config = ConfigDict(extra="ignore")

    id: str = Field(min_length=1)


class ZaloMessage(BaseModel):
    """The message payload of a ``user_send_text`` event."""

    model_config = ConfigDict(extra="ignore")

    text: str | None = None


class ZaloWebhookEvent(BaseModel):
    """One validated event from Zalo OA's webhook.

    Only the fields DoraEdu needs are modelled; every other field Zalo sends
    (timestamps, app_id, attachments, ...) is ignored rather than rejected,
    so payload additions on Zalo's side don't break ingestion.
    """

    model_config = ConfigDict(extra="ignore")

    event_name: str
    sender: ZaloSender
    message: ZaloMessage | None = None

    @property
    def is_text_message(self) -> bool:
        """Whether this event carries a non-empty text message to answer."""
        return (
            self.event_name == TEXT_MESSAGE_EVENT
            and self.message is not None
            and bool(self.message.text)
        )


def verify_signature(raw_body: bytes, signature: str | None, oa_secret: str) -> bool:
    """Verify a webhook delivery actually came from Zalo.

    Args:
        raw_body: The exact bytes of the HTTP request body (signing is
            computed over the raw body, not the parsed JSON).
        signature: The ``X-ZEvent-Signature`` header value, or ``None`` when
            the header is missing.
        oa_secret: The Official Account's secret key (``ZALO_OA_SECRET``).

    Returns:
        ``True`` only if ``signature`` matches the HMAC-SHA256 of
        ``raw_body`` keyed by ``oa_secret``.
    """
    if not signature:
        return False
    expected = hmac.new(oa_secret.encode("utf-8"), raw_body, hashlib.sha256).hexdigest()
    return hmac.compare_digest(expected, signature)
