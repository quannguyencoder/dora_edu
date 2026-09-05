"""Channel-agnostic messaging abstraction.

Every messaging platform (Telegram today, Zalo next) is plugged in by
implementing :class:`ChannelAdapter`. Adapters translate platform payloads into
:class:`IncomingMessage` and render :class:`OutgoingMessage` back — they contain
no retrieval, prompting or session logic, so a new channel never requires
touching the RAG or LLM layers.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Protocol

from pydantic import BaseModel, ConfigDict, Field


class IncomingMessage(BaseModel):
    """A student message, normalised away from any platform-specific type."""

    model_config = ConfigDict(frozen=True)

    channel: str = Field(min_length=1, description="Channel id, e.g. 'telegram' or 'zalo'.")
    user_id: str = Field(min_length=1, description="Stable per-channel user identifier.")
    text: str = Field(description="Raw message text as typed by the student.")
    display_name: str | None = Field(default=None, description="Student name, when the channel exposes one.")

    @property
    def session_key(self) -> tuple[str, str]:
        """Return the key isolating this student's session within their channel."""
        return (self.channel, self.user_id)


class OutgoingMessage(BaseModel):
    """A reply to send back to the student."""

    model_config = ConfigDict(frozen=True)

    text: str = Field(min_length=1)


class MessageHandler(Protocol):
    """The channel-independent brain an adapter delegates every message to."""

    def __call__(self, message: IncomingMessage) -> OutgoingMessage:
        """Turn one student message into one reply."""
        ...


class ChannelAdapter(ABC):
    """Base class for a messaging-platform adapter."""

    #: Channel identifier stored on every :class:`IncomingMessage`.
    channel_name: str = "unknown"

    def __init__(self, handler: MessageHandler) -> None:
        """Bind the adapter to the channel-independent message handler."""
        self._handler = handler

    @abstractmethod
    def run(self) -> None:
        """Start receiving messages and block until the bot is stopped."""

    def dispatch(self, user_id: str, text: str, display_name: str | None = None) -> OutgoingMessage:
        """Normalise a platform message and hand it to the shared handler.

        Args:
            user_id: The platform's user identifier, as a string.
            text: The raw message text.
            display_name: The student's name, when available.

        Returns:
            The reply to render back on the platform.
        """
        message = IncomingMessage(
            channel=self.channel_name,
            user_id=user_id,
            text=text,
            display_name=display_name,
        )
        return self._handler(message)
