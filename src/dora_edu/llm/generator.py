"""LLM API communication.

The generator is the last line of the zero-hallucination defence: when the
retriever returns nothing, no request is sent to the LLM at all and the fixed
"not in your textbook" answer is returned instead.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod

from openai import (
    APIConnectionError,
    APIStatusError,
    APITimeoutError,
    OpenAI,
    RateLimitError,
)
from pydantic import BaseModel, ConfigDict, Field

from dora_edu.config import Settings, get_settings
from dora_edu.llm import prompts
from dora_edu.models import RetrievedChunk, StudentProfile

logger = logging.getLogger(__name__)

#: Shown when the LLM provider is unreachable, so students never see a stack trace.
_SERVICE_ERROR_ANSWER = (
    "Xin lỗi em, cô đang gặp chút trục trặc kỹ thuật. 😔 "
    "Em thử hỏi lại sau ít phút nhé!"
)


class GeneratedAnswer(BaseModel):
    """A validated answer ready to be sent back to the student."""

    model_config = ConfigDict(frozen=True)

    answer: str = Field(min_length=1)
    grounded: bool = Field(
        description="True when the answer was produced from retrieved textbook context."
    )
    citations: list[str] = Field(default_factory=list)


class AnswerGenerator(ABC):
    """Interface every LLM backend implements.

    Keeping generation behind this interface lets the project swap OpenAI for
    Gemini or a local model without touching the bot or retrieval layers.
    """

    @abstractmethod
    def generate(
        self,
        question: str,
        chunks: list[RetrievedChunk],
        profile: StudentProfile,
        history: list[dict[str, str]] | None = None,
    ) -> GeneratedAnswer:
        """Produce a tutoring answer grounded in ``chunks``."""


class OpenAIAnswerGenerator(AnswerGenerator):
    """Generates answers through an OpenAI-compatible chat completions API.

    Setting ``OPENAI_BASE_URL`` points this at any compatible endpoint, including
    a locally hosted model, which keeps the offline-first option open.
    """

    def __init__(self, settings: Settings | None = None) -> None:
        """Build the API client.

        Raises:
            ValueError: If no API key is configured.
        """
        self._settings = settings or get_settings()
        if not self._settings.openai_api_key:
            raise ValueError("OPENAI_API_KEY is not set; cannot start the answer generator")
        self._client = OpenAI(
            api_key=self._settings.openai_api_key,
            base_url=self._settings.openai_base_url or None,
            timeout=self._settings.llm_timeout_seconds,
        )

    def generate(
        self,
        question: str,
        chunks: list[RetrievedChunk],
        profile: StudentProfile,
        history: list[dict[str, str]] | None = None,
    ) -> GeneratedAnswer:
        """Answer ``question`` using only the retrieved textbook passages.

        Args:
            question: The student's question, in Vietnamese.
            chunks: Passages returned by the retriever for this question.
            profile: The student's grade and subject.
            history: Prior conversation turns, oldest first.

        Returns:
            The tutoring answer. When ``chunks`` is empty the fixed
            "not in your textbook" message is returned without any API call.
        """
        # Zero-hallucination short circuit: no context means no generation.
        if not chunks:
            logger.info(
                "No context retrieved (grade=%d, subject=%s); returning the fixed refusal",
                profile.grade,
                profile.subject,
            )
            return GeneratedAnswer(answer=prompts.NO_CONTEXT_ANSWER, grounded=False)

        messages = prompts.build_messages(question, chunks, profile, history)

        try:
            response = self._client.chat.completions.create(
                model=self._settings.llm_model,
                messages=messages,
                temperature=self._settings.llm_temperature,
                max_tokens=self._settings.llm_max_tokens,
            )
        except (APITimeoutError, APIConnectionError) as exc:
            logger.error("LLM provider unreachable: %s", exc)
            return GeneratedAnswer(answer=_SERVICE_ERROR_ANSWER, grounded=False)
        except RateLimitError as exc:
            logger.error("LLM rate limit hit: %s", exc)
            return GeneratedAnswer(
                answer="Hiện có nhiều bạn đang hỏi cô cùng lúc. Em chờ cô một chút rồi hỏi lại nhé! ⏳",
                grounded=False,
            )
        except APIStatusError as exc:
            logger.error("LLM provider returned %s: %s", exc.status_code, exc)
            return GeneratedAnswer(answer=_SERVICE_ERROR_ANSWER, grounded=False)

        content = (response.choices[0].message.content or "").strip() if response.choices else ""
        if not content:
            logger.warning("LLM returned an empty completion; falling back to the refusal")
            return GeneratedAnswer(answer=prompts.NO_CONTEXT_ANSWER, grounded=False)

        return GeneratedAnswer(
            answer=content,
            grounded=True,
            citations=[chunk.citation for chunk in chunks],
        )
