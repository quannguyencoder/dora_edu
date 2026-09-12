"""Discord implementation of the channel adapter.

Everything Discord-specific lives here. The adapter converts Discord message
events into :class:`~dora_edu.bot.adapter.IncomingMessage` and sends the reply
back -- it holds no tutoring logic, exactly like the Telegram and Zalo
adapters. Discord's gateway (a persistent WebSocket connection) needs no
public HTTPS endpoint, so this is the simplest of the three to run.
"""

from __future__ import annotations

import asyncio
import logging

import discord

from dora_edu.bot.adapter import ChannelAdapter
from dora_edu.bot.adapter import MessageHandler as TutorMessageHandler

logger = logging.getLogger(__name__)

#: Discord rejects messages longer than this many characters.
_MAX_DISCORD_MESSAGE = 2000


def split_for_discord(text: str, limit: int = _MAX_DISCORD_MESSAGE) -> list[str]:
    """Split a reply into Discord-sized pieces, preferring line boundaries.

    Args:
        text: The full reply text.
        limit: Maximum characters per Discord message.

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


class _TutorDiscordClient(discord.Client):
    """The discord.py client, delegating every message to the shared adapter.

    Subclassing (rather than ``@client.event``) is discord.py's supported way
    to override ``on_ready``/``on_message`` on an object that also needs its
    own constructor arguments.
    """

    def __init__(self, adapter: "DiscordAdapter", *, intents: discord.Intents) -> None:
        super().__init__(intents=intents)
        self._adapter = adapter

    async def on_ready(self) -> None:
        logger.info("DoraEdu Discord bot logged in as %s", self.user)

    async def on_message(self, message: discord.Message) -> None:
        """Answer one incoming Discord message."""
        if message.author.bot or not message.content:
            return

        try:
            # The tutoring pipeline is blocking (vector search + LLM call), so it
            # runs off the event loop to keep the gateway connection responsive.
            reply = await asyncio.to_thread(
                self._adapter.dispatch,
                str(message.author.id),
                message.content,
                message.author.display_name,
            )
        except ValueError as exc:
            logger.error("Rejected malformed Discord message: %s", exc)
            return

        try:
            for piece in split_for_discord(reply.text):
                await message.channel.send(piece)
        except discord.DiscordException as exc:
            logger.error("Discord rejected the reply to %s: %s", message.author.id, exc)


class DiscordAdapter(ChannelAdapter):
    """Runs the tutor as a Discord bot over the gateway (WebSocket) connection."""

    channel_name = "discord"

    def __init__(self, handler: TutorMessageHandler, token: str) -> None:
        """Build the Discord client and register the message handler.

        Args:
            handler: The channel-independent tutoring brain.
            token: Discord bot token from the Discord Developer Portal.

        Raises:
            ValueError: If ``token`` is empty.
        """
        super().__init__(handler)
        if not token:
            raise ValueError("DISCORD_BOT_TOKEN is not set; cannot start the Discord adapter")
        self._token = token
        intents = discord.Intents.default()
        # Reading message text requires the privileged "Message Content"
        # intent -- enable it for this bot in the Discord Developer Portal too.
        intents.message_content = True
        self._client = _TutorDiscordClient(self, intents=intents)

    def run(self) -> None:
        """Connect to the Discord gateway and block until the process is stopped."""
        logger.info("DoraEdu Discord bot is connecting to the gateway")
        self._client.run(self._token, log_handler=None)
