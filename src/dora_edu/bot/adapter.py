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
    """A reply to send back to the student.

    Beyond the text, a reply may carry structured extras that each channel
    renders in its own way: Discord turns ``feedback_id`` into a "Báo sai"
    button, while Telegram and Zalo leave it to the student to type
    ``/baosai``. Keeping them as data here -- rather than baking a button
    into the text -- is what lets the core stay channel-agnostic.
    """

    model_config = ConfigDict(frozen=True)

    text: str = Field(min_length=1)
    citations: list[str] = Field(
        default_factory=list,
        description="Textbook pages the answer came from, straight from ChromaDB metadata.",
    )
    feedback_id: int | None = Field(
        default=None,
        description="Id of this answer in the feedback store, when it can be flagged as wrong.",
    )


class MessageHandler(Protocol):
    """The channel-independent brain an adapter delegates every message to."""

    def __call__(self, message: IncomingMessage) -> OutgoingMessage:
        """Turn one student message into one reply."""
        ...


class FeedbackHandler(Protocol):
    """The side of the brain that receives a student's verdict on an answer.

    A channel with interactive controls (Discord buttons, a Telegram inline
    keyboard) calls these directly; a channel without them reaches the same
    code through the ``/baosai`` command, which is plain text and so needs
    nothing beyond :class:`MessageHandler`.
    """

    def report_wrong_answer(self, feedback_id: int, reporter_id: str) -> OutgoingMessage:
        """Record that ``feedback_id``'s answer looks wrong to the student."""
        ...

    def rate_answer(
        self,
        feedback_id: int,
        user_id: str,
        *,
        clarity: int | None = None,
        guidance: int | None = None,
        comment: str | None = None,
    ) -> OutgoingMessage:
        """Record one student's 1-5 scores for ``feedback_id``'s answer."""
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
