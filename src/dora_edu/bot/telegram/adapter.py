"""Telegram implementation of the channel adapter.

Everything Telegram-specific lives here. The adapter converts Telegram updates
into :class:`~dora_edu.bot.adapter.IncomingMessage` and sends the reply back —
it holds no tutoring logic, so the Zalo adapter will be a sibling of this file
rather than a rewrite of the bot.
"""

from __future__ import annotations

import asyncio
import logging

from telegram import Update
from telegram.error import NetworkError, TelegramError
from telegram.ext import Application, ContextTypes, MessageHandler, filters

from dora_edu.bot.adapter import ChannelAdapter, MessageHandler as TutorMessageHandler

logger = logging.getLogger(__name__)

#: Telegram rejects messages longer than this many characters.
_MAX_TELEGRAM_MESSAGE = 4096


def split_for_telegram(text: str, limit: int = _MAX_TELEGRAM_MESSAGE) -> list[str]:
    """Split a reply into Telegram-sized pieces, preferring line boundaries.

    Args:
        text: The full reply text.
        limit: Maximum characters per Telegram message.

    Returns:
        One or more pieces, each within ``limit``.
    """
    if len(text) <= limit:
        return [text]

    pieces: list[str] = []
    remaining = text
    while len(remaining) > limit:
        window = remaining[:limit]
        split_at = window.rfind("\n")
        if split_at <= 0:
            split_at = window.rfind(" ")
        if split_at <= 0:
            split_at = limit
        pieces.append(remaining[:split_at].rstrip())
        remaining = remaining[split_at:].lstrip()
    if remaining:
        pieces.append(remaining)
    return pieces


class TelegramAdapter(ChannelAdapter):
    """Runs the tutor as a Telegram bot using long polling."""

    channel_name = "telegram"

    def __init__(self, handler: TutorMessageHandler, token: str) -> None:
        """Build the Telegram application and register the message handler.

        Args:
            handler: The channel-independent tutoring brain.
            token: Telegram bot token issued by BotFather.

        Raises:
            ValueError: If ``token`` is empty.
        """
        super().__init__(handler)
        if not token:
            raise ValueError("TELEGRAM_BOT_TOKEN is not set; cannot start the Telegram adapter")
        self._application = Application.builder().token(token).build()
        # Commands are plain text to the shared handler, so a single handler
        # covers both questions and commands on every channel identically.
        self._application.add_handler(MessageHandler(filters.TEXT, self._on_message))
        self._application.add_error_handler(self._on_error)

    async def _on_message(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Answer one incoming Telegram message."""
        message = update.effective_message
        user = update.effective_user
        if message is None or user is None or not message.text:
            return

        try:
            # The tutoring pipeline is blocking (vector search + LLM call), so it
            # runs off the event loop to keep the bot responsive.
            reply = await asyncio.to_thread(
                self.dispatch,
                str(user.id),
                message.text,
                user.full_name,
            )
        except ValueError as exc:
            logger.error("Rejected malformed Telegram message: %s", exc)
            return

        try:
            for piece in split_for_telegram(reply.text):
                await message.reply_text(piece)
        except NetworkError as exc:
            logger.error("Network error while replying to %s: %s", user.id, exc)
        except TelegramError as exc:
            logger.error("Telegram rejected the reply to %s: %s", user.id, exc)

    async def _on_error(self, update: object, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Log any exception the Telegram framework surfaces."""
        logger.error("Unhandled Telegram error", exc_info=context.error)

    def run(self) -> None:
        """Start long polling and block until the process is stopped."""
        logger.info("DoraEdu Telegram bot is polling for messages")
        self._application.run_polling(allowed_updates=Update.ALL_TYPES)
