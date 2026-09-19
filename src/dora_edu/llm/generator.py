"""LLM API communication.

The generator is the last line of the zero-hallucination defence: when the
retriever returns nothing, no request is sent to the LLM at all and the fixed
"not in your textbook" answer is returned instead.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod

import httpx
from google import genai
from google.genai import types as genai_types
from google.genai.errors import ClientError, ServerError
from openai import (
    APIConnectionError,
    APIStatusError,
    APITimeoutError,
    OpenAI,
    RateLimitError,
)
from pydantic import BaseModel, ConfigDict, Field

from dora_edu.config import Settings, get_settings, parse_model_list
from dora_edu.llm import prompts
from dora_edu.models import RetrievedChunk, StudentProfile, strip_diacritics

logger = logging.getLogger(__name__)

#: Shown when the LLM provider is unreachable, so students never see a stack trace.
_SERVICE_ERROR_ANSWER = (
    "Xin lỗi bạn, mình đang gặp chút trục trặc kỹ thuật. 😔 "
    "Bạn thử hỏi lại sau ít phút nhé!"
)

#: Shown when the LLM provider rejects a request for being rate-limited.
_RATE_LIMIT_ANSWER = (
    "Hiện có nhiều bạn đang hỏi mình cùng lúc. Bạn chờ một chút rồi hỏi lại nhé! ⏳"
)

#: The classification prompt's literal answer for a question that is not
#: about any subject's content at all (e.g. "what subjects can you help
#: with?", small talk, asking about the bot itself). Kept case another
#: constant rather than a plain string so callers can tell it apart from a
#: genuine subject name at a glance.
_NOT_SUBJECT_SPECIFIC_REPLY = "KHAC"

#: Sentinel :meth:`AnswerGenerator.classify_subject` returns for a question
#: that is not about any subject's content -- distinct from ``None``, which
#: means the classifier was inconclusive or unreachable and the caller should
#: fall back to nearest-distance matching instead.
NOT_SUBJECT_SPECIFIC = "__not_subject_specific__"

_CLASSIFY_SUBJECT_PROMPT_TEMPLATE = (
    'Học sinh nhắn: "{question}"\n\n'
    "Nếu tin nhắn này KHÔNG phải là câu hỏi về nội dung kiến thức của một môn học cụ "
    "thể -- ví dụ: hỏi bot hỗ trợ những môn nào, bot làm được gì, bot là ai, chào hỏi, "
    f"hỏi han ngoài lề -- hãy trả lời đúng 1 từ: {_NOT_SUBJECT_SPECIFIC_REPLY}\n\n"
    "Nếu đây đúng là một câu hỏi kiến thức, nó thuộc môn học nào trong danh sách sau: "
    "{subjects}?\n"
    "Chỉ trả lời đúng tên 1 môn có trong danh sách, hoặc đúng 1 từ "
    f"{_NOT_SUBJECT_SPECIFIC_REPLY} như trên, không giải thích gì thêm."
)


def _parse_classification(raw: str, subjects: list[str]) -> str | None:
    """Turn a classification reply into a subject, the meta sentinel, or ``None``.

    Args:
        raw: The LLM's raw reply to the classification prompt.
        subjects: Canonical subject names it was asked to choose from.

    Returns:
        One of ``subjects``, :data:`NOT_SUBJECT_SPECIFIC`, or ``None`` when
        the reply matched neither.
    """
    if raw.strip().upper() == _NOT_SUBJECT_SPECIFIC_REPLY:
        return NOT_SUBJECT_SPECIFIC
    return _match_subject(raw, subjects)


class GeneratedAnswer(BaseModel):
    """A validated answer ready to be sent back to the student."""

    model_config = ConfigDict(frozen=True)

    answer: str = Field(min_length=1)
    grounded: bool = Field(
        description="True when the answer was produced from retrieved textbook context."
    )
    citations: list[str] = Field(default_factory=list)


def _match_subject(raw: str, subjects: list[str]) -> str | None:
    """Match a free-form LLM reply back to one of the offered canonical subjects.

    Diacritic- and case-insensitive substring match, so "Môn Toán." or "toan"
    both resolve to "Toán". If the reply's normalised text contains more than
    one candidate subject (an ambiguous or rambling reply), this refuses to
    guess and returns ``None`` rather than picking one arbitrarily.

    Args:
        raw: The LLM's raw reply to the classification prompt.
        subjects: Canonical subject names it was asked to choose from.

    Returns:
        The one matching subject, or ``None`` if zero or several matched.
    """
    normalised = strip_diacritics(raw).lower()
    matches = [
        subject for subject in subjects if strip_diacritics(subject).lower() in normalised
    ]
    return matches[0] if len(matches) == 1 else None


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

    @abstractmethod
    def classify_subject(self, question: str, subjects: list[str]) -> str | None:
        """Pick which of ``subjects`` ``question`` is most likely about.

        Used when a student has not pinned a subject with ``/mon``: semantic
        vector distance alone is not reliable enough to tell subjects apart
        (a history question can embed closer to a math passage than to the
        right history one), so this asks the LLM directly instead.

        Args:
            question: The student's question, in Vietnamese.
            subjects: Candidate subjects to choose from (that grade's
                indexed subjects); never empty when called by the bot layer.

        Returns:
            One of ``subjects`` verbatim; :data:`NOT_SUBJECT_SPECIFIC` when
            the message is not a subject-content question at all (e.g. it
            asks about the bot itself); or ``None`` when the classifier is
            inconclusive or the provider is unreachable.
        """


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
                answer=_RATE_LIMIT_ANSWER,
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

    def classify_subject(self, question: str, subjects: list[str]) -> str | None:
        """Ask the model which of ``subjects`` best fits ``question``.

        A minimal, single-turn, zero-temperature call -- this only ever needs
        to output one subject name, not a tutoring answer.
        """
        if not subjects:
            return None

        prompt = _CLASSIFY_SUBJECT_PROMPT_TEMPLATE.format(
            question=question, subjects=", ".join(subjects)
        )
        try:
            response = self._client.chat.completions.create(
                model=self._settings.llm_model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0,
                max_tokens=20,
            )
        except (APITimeoutError, APIConnectionError, RateLimitError, APIStatusError) as exc:
            logger.warning("Subject classification unreachable: %s", exc)
            return None

        content = (response.choices[0].message.content or "").strip() if response.choices else ""
        return _parse_classification(content, subjects)


#: OpenAI's chat-message role for an LLM reply; Gemini calls the same turn "model".
_GEMINI_ROLE_MAP = {"user": "user", "assistant": "model"}


class GeminiAnswerGenerator(AnswerGenerator):
    """Generates answers through the Gemini API (``google-genai``).

    Swapping this in for :class:`OpenAIAnswerGenerator` needs no change to the
    bot or retrieval layers -- both sit behind the same :class:`AnswerGenerator`
    interface, and :func:`build_generator` picks between them from ``LLM_PROVIDER``.
    """

    def __init__(self, settings: Settings | None = None) -> None:
        """Build the API client.

        Raises:
            ValueError: If no API key is configured.
        """
        self._settings = settings or get_settings()
        if not self._settings.gemini_api_key:
            raise ValueError("GEMINI_API_KEY is not set; cannot start the answer generator")
        self._client = genai.Client(api_key=self._settings.gemini_api_key)

    def _models_in_order(self) -> list[str]:
        """The configured model first, then each distinct fallback model.

        A fallback that duplicates the primary model (or an earlier
        fallback) is dropped -- retrying the very model that was just
        rate-limited would not help, and would burn an extra call for no
        benefit.
        """
        seen = {self._settings.llm_model}
        ordered = [self._settings.llm_model]
        for model in parse_model_list(self._settings.gemini_fallback_models):
            if model not in seen:
                ordered.append(model)
                seen.add(model)
        return ordered

    def _generate_with_one_model(
        self,
        model: str,
        contents: list[genai_types.Content],
        config: genai_types.GenerateContentConfig,
    ) -> tuple[str | None, bool]:
        """Call a single Gemini model, retrying once on an empty completion.

        Args:
            model: The Gemini model id to call.
            contents: The conversation turns already built for this request.
            config: The generation config shared across every model tried.

        Returns:
            A ``(content, rate_limited)`` pair. ``content`` is ``None`` for a
            non-rate-limit provider error (the caller should stop trying
            models altogether), an empty string when every attempt on this
            model came back empty, or the generated text. ``rate_limited``
            is ``True`` only for an HTTP 429 from this specific model.
        """
        content = ""
        for attempt in range(2):
            try:
                response = self._client.models.generate_content(
                    model=model, contents=contents, config=config
                )
            except ClientError as exc:
                if exc.code == 429:
                    return "", True
                logger.error("LLM provider returned %s: %s", exc.code, exc)
                return None, False
            except ServerError as exc:
                logger.error("LLM provider unreachable: %s", exc)
                return None, False
            except httpx.HTTPError as exc:
                # A connection-level failure (timeout, reset, DNS...) rather
                # than an HTTP error response -- google-genai's own retry
                # wrapper re-raises these as plain httpx exceptions instead
                # of ClientError/ServerError once its retry budget is spent.
                # Uncaught, this crashes the whole message handler and
                # leaves the student with no reply at all, not just a wrong
                # one -- treat it the same as ServerError.
                logger.error("LLM provider connection failed: %s", exc)
                return None, False

            content = (getattr(response, "text", None) or "").strip()
            if content:
                break
            # Sampling occasionally comes back with an empty candidate for no
            # discernible reason (confirmed by hand: the identical request
            # often succeeds on a plain retry) -- one retry on the same model
            # is far cheaper than telling a student who asked a perfectly
            # answerable question "not found".
            logger.warning("%s returned an empty completion (attempt %d/2)", model, attempt + 1)
        return content, False

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
        system_instruction = next(
            (m["content"] for m in messages if m["role"] == "system"), None
        )
        contents = [
            genai_types.Content(
                role=_GEMINI_ROLE_MAP[m["role"]], parts=[genai_types.Part(text=m["content"])]
            )
            for m in messages
            if m["role"] != "system"
        ]

        config = genai_types.GenerateContentConfig(
            system_instruction=system_instruction,
            temperature=self._settings.llm_temperature,
            max_output_tokens=self._settings.llm_max_tokens,
            # Keep reasoning light: a tutoring reply doesn't need deep
            # chain-of-thought, and a heavier budget was eating most
            # of max_output_tokens before any visible answer appeared.
            thinking_config=genai_types.ThinkingConfig(thinking_level="low"),
        )

        content = ""
        any_rate_limited = False
        for model in self._models_in_order():
            content, rate_limited = self._generate_with_one_model(model, contents, config)
            if rate_limited:
                # This model's own daily/per-minute quota is exhausted --
                # the free tier quotas each model separately, so the next
                # model in the list still has its own budget.
                any_rate_limited = True
                logger.warning("%s is rate-limited; trying the next fallback model", model)
                continue
            if content is None:
                # A non-rate-limit provider error: retrying with a different
                # model would not fix a request/connection problem.
                return GeneratedAnswer(answer=_SERVICE_ERROR_ANSWER, grounded=False)
            # Real content, or a definitive empty completion (already retried
            # once on this same model) -- either way, stop here rather than
            # cascading through every fallback model too.
            break

        if not content:
            if any_rate_limited:
                return GeneratedAnswer(answer=_RATE_LIMIT_ANSWER, grounded=False)
            return GeneratedAnswer(answer=prompts.NO_CONTEXT_ANSWER, grounded=False)

        return GeneratedAnswer(
            answer=content,
            grounded=True,
            citations=[chunk.citation for chunk in chunks],
        )

    def classify_subject(self, question: str, subjects: list[str]) -> str | None:
        """Ask the model which of ``subjects`` best fits ``question``.

        A minimal, single-turn, zero-temperature call -- this only ever needs
        to output one subject name, not a tutoring answer.
        """
        if not subjects:
            return None

        prompt = _CLASSIFY_SUBJECT_PROMPT_TEMPLATE.format(
            question=question, subjects=", ".join(subjects)
        )
        contents = [genai_types.Content(role="user", parts=[genai_types.Part(text=prompt)])]
        config = genai_types.GenerateContentConfig(
            temperature=0,
            max_output_tokens=200,
            thinking_config=genai_types.ThinkingConfig(thinking_level="low"),
        )

        # Also falls back across models on a 429: an auto-detect question
        # costs a classification call on top of the answer call, so this can
        # exhaust the primary model's quota faster than generate() alone.
        for model in self._models_in_order():
            content, rate_limited = self._generate_with_one_model(model, contents, config)
            if rate_limited:
                logger.warning("%s is rate-limited; trying the next fallback model", model)
                continue
            if content:
                return _parse_classification(content, subjects)
            break
        return None


def build_generator(settings: Settings | None = None) -> AnswerGenerator:
    """Construct the LLM backend selected by ``LLM_PROVIDER``.

    Args:
        settings: Application settings; the process singleton when omitted.

    Returns:
        An :class:`OpenAIAnswerGenerator` or :class:`GeminiAnswerGenerator`.

    Raises:
        ValueError: If ``llm_provider`` is not ``"openai"`` or ``"gemini"``,
            or the matching API key is not configured.
    """
    settings = settings or get_settings()
    provider = settings.llm_provider.strip().lower()
    if provider == "openai":
        return OpenAIAnswerGenerator(settings)
    if provider == "gemini":
        return GeminiAnswerGenerator(settings)
    raise ValueError(
        f"Unsupported LLM_PROVIDER: {settings.llm_provider!r} (expected 'openai' or 'gemini')"
    )
