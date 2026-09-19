"""Tests for the generator's zero-hallucination short circuit."""

from __future__ import annotations

from typing import Any

import httpx
import pytest
from openai import APIConnectionError, APITimeoutError, RateLimitError

from dora_edu.llm import prompts
from dora_edu.llm.generator import NOT_SUBJECT_SPECIFIC, OpenAIAnswerGenerator, _match_subject
from dora_edu.models import RetrievedChunk, StudentProfile

PROFILE = StudentProfile(grade=6, subject="Toán")


def _chunk() -> RetrievedChunk:
    return RetrievedChunk(
        text="Phan so gom tu so va mau so.",
        distance=0.1,
        metadata={"book_title": "SGK Toán 6", "page_start": 12, "page_end": 12},
    )


class _StubCompletions:
    """Minimal stand-in for ``client.chat.completions``."""

    def __init__(self, content: str | None = "Goi y cua co", error: Exception | None = None) -> None:
        self.content = content
        self.error = error
        self.calls: list[dict[str, Any]] = []

    def create(self, **kwargs: Any) -> Any:
        self.calls.append(kwargs)
        if self.error is not None:
            raise self.error
        message = type("Message", (), {"content": self.content})()
        choice = type("Choice", (), {"message": message})()
        return type("Response", (), {"choices": [choice]})()


@pytest.fixture
def generator(settings, monkeypatch) -> OpenAIAnswerGenerator:
    settings = settings.model_copy(update={"openai_api_key": "test-key"})
    return OpenAIAnswerGenerator(settings)


def _install(generator: OpenAIAnswerGenerator, completions: _StubCompletions) -> None:
    """Swap the OpenAI client for a stub that records its calls."""
    chat = type("Chat", (), {"completions": completions})()
    generator._client = type("Client", (), {"chat": chat})()


def test_missing_api_key_is_rejected_at_construction(settings) -> None:
    with pytest.raises(ValueError):
        OpenAIAnswerGenerator(settings.model_copy(update={"openai_api_key": None}))


def test_no_context_returns_the_refusal_without_calling_the_llm(generator) -> None:
    completions = _StubCompletions()
    _install(generator, completions)

    result = generator.generate("Cau hoi ngoai sach", [], PROFILE)

    assert result.answer == prompts.NO_CONTEXT_ANSWER
    assert result.grounded is False
    assert completions.calls == []


def test_retrieved_context_produces_a_grounded_answer_with_citations(generator) -> None:
    _install(generator, _StubCompletions("Em thu nghi xem..."))

    result = generator.generate("Phan so la gi?", [_chunk()], PROFILE)

    assert result.answer == "Em thu nghi xem..."
    assert result.grounded is True
    assert result.citations == ["SGK Toán 6, trang 12"]


def test_an_empty_completion_falls_back_to_the_refusal(generator) -> None:
    _install(generator, _StubCompletions(""))

    result = generator.generate("Phan so la gi?", [_chunk()], PROFILE)

    assert result.answer == prompts.NO_CONTEXT_ANSWER
    assert result.grounded is False


def _openai_request() -> httpx.Request:
    """Build the minimal request object the OpenAI exception types require."""
    return httpx.Request("POST", "https://api.openai.com/v1/chat/completions")


@pytest.mark.parametrize(
    "make_error",
    [
        pytest.param(lambda: APITimeoutError(request=_openai_request()), id="timeout"),
        pytest.param(
            lambda: APIConnectionError(request=_openai_request()), id="connection"
        ),
        pytest.param(
            lambda: RateLimitError(
                "slow down",
                response=httpx.Response(429, request=_openai_request()),
                body=None,
            ),
            id="rate_limit",
        ),
    ],
)
def test_provider_failures_become_friendly_vietnamese_messages(generator, make_error) -> None:
    _install(generator, _StubCompletions(error=make_error()))

    result = generator.generate("Phan so la gi?", [_chunk()], PROFILE)

    assert result.grounded is False
    assert result.answer
    assert "Traceback" not in result.answer


# --- Subject classification -------------------------------------------------


@pytest.mark.parametrize(
    ("raw", "subjects", "expected"),
    [
        ("Toán", ["Toán", "Lịch sử"], "Toán"),
        ("toan", ["Toán", "Lịch sử"], "Toán"),
        ("Môn học là: Lịch sử.", ["Toán", "Lịch sử"], "Lịch sử"),
        ("Ngữ văn", ["Toán", "Lịch sử"], None),
        ("Toán hoặc Lịch sử", ["Toán", "Lịch sử"], None),
        ("", ["Toán", "Lịch sử"], None),
    ],
)
def test_match_subject(raw, subjects, expected) -> None:
    assert _match_subject(raw, subjects) == expected


def test_classify_subject_returns_none_for_an_empty_subject_list(generator) -> None:
    assert generator.classify_subject("Phan so la gi?", []) is None


def test_classify_subject_matches_the_models_reply(generator) -> None:
    _install(generator, _StubCompletions("Toán"))

    assert generator.classify_subject("Phan so la gi?", ["Toán", "Lịch sử"]) == "Toán"


def test_classify_subject_returns_none_when_the_reply_is_ambiguous(generator) -> None:
    _install(generator, _StubCompletions("Có thể là Toán hoặc Lịch sử"))

    assert generator.classify_subject("abc", ["Toán", "Lịch sử"]) is None


def test_classify_subject_returns_none_when_the_provider_is_unreachable(generator) -> None:
    _install(generator, _StubCompletions(error=APIConnectionError(request=_openai_request())))

    assert generator.classify_subject("abc", ["Toán", "Lịch sử"]) is None


def test_classify_subject_detects_a_non_curriculum_question(generator) -> None:
    _install(generator, _StubCompletions("KHAC"))

    result = generator.classify_subject("Ban co the ho tro nhung mon nao?", ["Toán", "Lịch sử"])

    assert result == NOT_SUBJECT_SPECIFIC


# --- Diacritic restoration ---------------------------------------------------


def test_restore_diacritics_returns_the_models_correction(generator) -> None:
    _install(generator, _StubCompletions("đa thức là gì"))

    assert generator.restore_diacritics("da thuc la gi") == "đa thức là gì"


def test_restore_diacritics_returns_the_original_on_an_empty_completion(generator) -> None:
    _install(generator, _StubCompletions(""))

    assert generator.restore_diacritics("da thuc la gi") == "da thuc la gi"


def test_restore_diacritics_returns_the_original_when_the_provider_is_unreachable(
    generator,
) -> None:
    _install(generator, _StubCompletions(error=APIConnectionError(request=_openai_request())))

    assert generator.restore_diacritics("da thuc la gi") == "da thuc la gi"


def test_restore_diacritics_of_an_empty_string_skips_the_api_call(generator) -> None:
    completions = _StubCompletions()
    _install(generator, completions)

    assert generator.restore_diacritics("   ") == "   "
    assert completions.calls == []
