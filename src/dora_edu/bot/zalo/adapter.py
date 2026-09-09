"""Zalo Official Account implementation of the channel adapter.

Zalo delivers messages by POSTing webhook events to a public HTTPS endpoint
(no long polling, unlike Telegram), and replies go out through Zalo's OA
"Send API" instead of a client SDK call. Everything Zalo-specific lives here
and in :mod:`dora_edu.bot.zalo.webhook` -- ``bot/core.py`` and every other
layer are untouched by adding this channel, exactly as
:class:`~dora_edu.bot.telegram.adapter.TelegramAdapter` is for Telegram.

Putting a public webhook URL in front of this server (reverse proxy + TLS,
and registering that URL with the OA) is a deployment concern outside this
module.
"""

from __future__ import annotations

import json
import logging
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import httpx
from pydantic import ValidationError

from dora_edu.bot.adapter import ChannelAdapter
from dora_edu.bot.adapter import MessageHandler as TutorMessageHandler
from dora_edu.bot.zalo.webhook import SIGNATURE_HEADER, ZaloWebhookEvent, verify_signature

logger = logging.getLogger(__name__)

#: Zalo OA "Send API" endpoint for consultation (customer service) messages.
_SEND_MESSAGE_URL = "https://openapi.zalo.me/v3.0/oa/message/cs"

#: Timeout for calls to the Zalo Send API.
_SEND_TIMEOUT_SECONDS = 10.0


class ZaloAdapter(ChannelAdapter):
    """Runs the tutor as a Zalo Official Account webhook receiver."""

    channel_name = "zalo"

    def __init__(
        self,
        handler: TutorMessageHandler,
        access_token: str,
        oa_secret: str,
        *,
        host: str = "0.0.0.0",
        port: int = 8080,
        client: httpx.Client | None = None,
    ) -> None:
        """Build the webhook server, without starting it yet.

        Args:
            handler: The channel-independent tutoring brain.
            access_token: Zalo OA access token used to call the Send API.
            oa_secret: Zalo OA secret key used to verify webhook signatures.
            host: Interface the webhook HTTP server binds to.
            port: Port the webhook HTTP server listens on.
            client: An ``httpx.Client`` to send replies through; a new one is
                created when omitted (tests inject a fake here instead).

        Raises:
            ValueError: If ``access_token`` or ``oa_secret`` is empty.
        """
        super().__init__(handler)
        if not access_token:
            raise ValueError("ZALO_ACCESS_TOKEN is not set; cannot start the Zalo adapter")
        if not oa_secret:
            raise ValueError("ZALO_OA_SECRET is not set; cannot verify Zalo webhook requests")
        self._access_token = access_token
        self._oa_secret = oa_secret
        self._host = host
        self._port = port
        self._client = client or httpx.Client(timeout=_SEND_TIMEOUT_SECONDS)

    def handle_delivery(self, raw_body: bytes, signature: str | None) -> None:
        """Validate, parse and answer one webhook delivery.

        Every failure mode (bad signature, malformed JSON, an event type this
        text-only tutor does not handle) is logged and swallowed rather than
        raised, because Zalo expects a ``200 OK`` for a delivery it accepted
        even when DoraEdu chose not to act on it.

        Args:
            raw_body: The exact, unparsed request body bytes.
            signature: The ``X-ZEvent-Signature`` header value, if present.
        """
        if not verify_signature(raw_body, signature, self._oa_secret):
            logger.warning("Rejected a Zalo webhook delivery with an invalid signature")
            return

        try:
            payload = json.loads(raw_body)
            event = ZaloWebhookEvent.model_validate(payload)
        except (json.JSONDecodeError, ValidationError) as exc:
            logger.warning("Rejected a malformed Zalo webhook payload: %s", exc)
            return

        if not event.is_text_message:
            return

        try:
            # The tutoring pipeline is blocking (vector search + LLM call);
            # this handler already runs on the HTTP server's own worker
            # thread, so no extra offloading is needed here.
            reply = self.dispatch(event.sender.id, event.message.text)  # type: ignore[union-attr]
        except ValueError as exc:
            logger.error("Rejected a malformed Zalo message: %s", exc)
            return

        self._send_message(event.sender.id, reply.text)

    def _send_message(self, user_id: str, text: str) -> None:
        """Send one reply through Zalo OA's Send API."""
        try:
            response = self._client.post(
                _SEND_MESSAGE_URL,
                headers={"access_token": self._access_token},
                json={"recipient": {"user_id": user_id}, "message": {"text": text}},
            )
            response.raise_for_status()
        except httpx.HTTPError as exc:
            logger.error("Zalo rejected the reply to %s: %s", user_id, exc)

    def run(self) -> None:
        """Start the webhook HTTP server and block until the process is stopped."""
        server = ThreadingHTTPServer((self._host, self._port), _make_request_handler(self))
        logger.info("DoraEdu Zalo webhook listening on %s:%d", self._host, self._port)
        try:
            server.serve_forever()
        finally:
            server.server_close()


def _make_request_handler(adapter: ZaloAdapter) -> type[BaseHTTPRequestHandler]:
    """Build a request handler class bound to ``adapter`` (stdlib's API needs a class, not an instance)."""

    class _WebhookHandler(BaseHTTPRequestHandler):
        def do_POST(self) -> None:  # noqa: N802 - stdlib naming convention
            length = int(self.headers.get("Content-Length", 0))
            raw_body = self.rfile.read(length) if length else b""
            adapter.handle_delivery(raw_body, self.headers.get(SIGNATURE_HEADER))
            self.send_response(200)
            self.end_headers()

        def log_message(self, format: str, *args: object) -> None:  # noqa: A002 - stdlib signature
            logger.debug("Zalo webhook: " + format, *args)

    return _WebhookHandler
