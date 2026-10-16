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

#: Prefix of the "Báo sai" button's ``custom_id``; the feedback id follows it.
#: Discord sends this string straight back on a click, which is what lets the
#: button keep working after a restart -- the click is handled in
#: ``on_interaction`` from the id alone, with no in-memory view to look up.
_REPORT_CUSTOM_ID_PREFIX = "dora:report:"

#: What the student presses when an answer looks wrong.
_REPORT_BUTTON_LABEL = "🚩 Báo sai"

#: Prefix of the "Chấm điểm" button's ``custom_id``, same scheme as above.
_RATE_CUSTOM_ID_PREFIX = "dora:rate:"

#: What the student presses to score an answer 1-5.
_RATE_BUTTON_LABEL = "⭐ Chấm điểm"


def build_feedback_view(feedback_id: int) -> discord.ui.View:
    """Build the row of feedback controls that sits under an answer.

    Args:
        feedback_id: Id of the answer in the feedback store.

    Returns:
        A view holding the "Báo sai" and "Chấm điểm" buttons, never timing
        out so they stay pressable for as long as the message is on screen.
    """
    view = discord.ui.View(timeout=None)
    view.add_item(
        discord.ui.Button(
            label=_REPORT_BUTTON_LABEL,
            style=discord.ButtonStyle.secondary,
            custom_id=f"{_REPORT_CUSTOM_ID_PREFIX}{feedback_id}",
        )
    )
    view.add_item(
        discord.ui.Button(
            label=_RATE_BUTTON_LABEL,
            style=discord.ButtonStyle.secondary,
            custom_id=f"{_RATE_CUSTOM_ID_PREFIX}{feedback_id}",
        )
    )
    return view


class _RatingModal(discord.ui.Modal):
    """The 1-5 form a student fills in after pressing "Chấm điểm".

    The two questions are the ones the classroom trial asks: was the answer
    easy to understand, and did the hinting help the student work it out
    themselves. Both arrive as text, because Discord modals have no numeric
    input -- :func:`~dora_edu.bot.core._parse_rating` is what decides
    whether what was typed is a usable score.
    """

    clarity: discord.ui.TextInput = discord.ui.TextInput(
        label="Câu trả lời có dễ hiểu không? (1-5)",
        placeholder="5",
        max_length=1,
    )
    guidance: discord.ui.TextInput = discord.ui.TextInput(
        label="Gợi mở có giúp bạn tự nghĩ ra? (1-5)",
        placeholder="4",
        max_length=1,
    )
    comment: discord.ui.TextInput = discord.ui.TextInput(
        label="Góp ý thêm (không bắt buộc)",
        style=discord.TextStyle.paragraph,
        required=False,
        max_length=300,
    )

    def __init__(self, feedback_id: str, client: "_TutorDiscordClient") -> None:
        super().__init__(title="Chấm điểm câu trả lời")
        self._feedback_id = feedback_id
        self._client = client

    async def on_submit(self, interaction: discord.Interaction) -> None:
        """Forward the filled-in form as the ordinary ``/chamdiem`` command."""
        argument = f"#{self._feedback_id} {self.clarity.value} {self.guidance.value}"
        if self.comment.value:
            argument = f"{argument} {self.comment.value}"
        await self._client.handle_feedback_interaction(interaction, f"/chamdiem {argument}")


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

        pieces = split_for_discord(reply.text)
        # The buttons belong under the end of the answer, so a long reply
        # split across several messages only carries them on the last one.
        view = build_feedback_view(reply.feedback_id) if reply.feedback_id else None
        try:
            for index, piece in enumerate(pieces):
                is_last = index == len(pieces) - 1
                await message.channel.send(piece, view=view if is_last else None)
        except discord.DiscordException as exc:
            logger.error("Discord rejected the reply to %s: %s", message.author.id, exc)

    async def on_interaction(self, interaction: discord.Interaction) -> None:
        """Route a feedback button press to the shared tutoring handler.

        Each press is turned into the equivalent text command -- ``/baosai``
        or ``/chamdiem`` -- so the buttons are pure Discord rendering: a
        Telegram or Zalo student typing the same command reaches exactly the
        same code. Dispatching from ``custom_id`` here (rather than from a
        per-message view callback) is also what keeps the buttons working
        after a restart, since nothing has to be looked up in memory.
        """
        custom_id = str((interaction.data or {}).get("custom_id", ""))
        if custom_id.startswith(_REPORT_CUSTOM_ID_PREFIX):
            feedback_id = custom_id[len(_REPORT_CUSTOM_ID_PREFIX) :]
            await self.handle_feedback_interaction(interaction, f"/baosai {feedback_id}")
            return
        if custom_id.startswith(_RATE_CUSTOM_ID_PREFIX):
            feedback_id = custom_id[len(_RATE_CUSTOM_ID_PREFIX) :]
            try:
                # A rating needs two numbers, so it opens a form instead of
                # being recorded straight from the press; the submitted form
                # comes back through this same method as /chamdiem.
                await interaction.response.send_modal(_RatingModal(feedback_id, self))
            except discord.DiscordException as exc:
                logger.error("Discord rejected the rating form for %s: %s", interaction.user.id, exc)

    async def handle_feedback_interaction(
        self, interaction: discord.Interaction, command: str
    ) -> None:
        """Run ``command`` on behalf of the interacting student and reply privately.

        Args:
            interaction: The button press or submitted form.
            command: The equivalent text command to run, e.g. ``"/baosai 12"``.
        """
        try:
            reply = await asyncio.to_thread(
                self._adapter.dispatch,
                str(interaction.user.id),
                command,
                interaction.user.display_name,
            )
        except ValueError as exc:
            logger.error("Rejected malformed Discord interaction: %s", exc)
            return

        try:
            # Ephemeral: the acknowledgement is for the student who pressed,
            # and should not add noise to a shared study channel.
            await interaction.response.send_message(reply.text, ephemeral=True)
        except discord.DiscordException as exc:
            logger.error("Discord rejected the feedback ack for %s: %s", interaction.user.id, exc)


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
