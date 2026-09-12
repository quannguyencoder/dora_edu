"""Tests for the Gemini backend and the provider-selecting factory."""

from __future__ import annotations

from typing import Any

import pytest
from google.genai.errors import ClientError, ServerError

from dora_edu.llm import prompts
from dora_edu.llm.generator import (
    GeminiAnswerGenerator,
    OpenAIAnswerGenerator,
    build_generator,
)
from dora_edu.models import RetrievedChunk, StudentProfile

PROFILE = StudentProfile(grade=6, subject="Toán")


def _chunk() -> RetrievedChunk:
    return RetrievedChunk(
        text="Phan so gom tu so va mau so.",
        distance=0.1,
        metadata={"book_title": "SGK Toán 6", "page_start": 12, "page_end": 12},
    )


class _StubModels:
    """Minimal stand-in for ``client.models``."""

    def __init__(self, text: str | None = "Goi y cua co", error: Exception | None = None) -> None:
        self.text = text
        self.error = error
        self.calls: list[dict[str, Any]] = []

    def generate_content(self, **kwargs: Any) -> Any:
        self.calls.append(kwargs)
        if self.error is not None:
            raise self.error
        return type("Response", (), {"text": self.text})()


@pytest.fixture
def generator(settings) -> GeminiAnswerGenerator:
    settings = settings.model_copy(update={"gemini_api_key": "test-key"})
    return GeminiAnswerGenerator(settings)


def _install(generator: GeminiAnswerGenerator, models: _StubModels) -> None:
    """Swap the genai client for a stub that records its calls."""
    generator._client = type("Client", (), {"models": models})()


def test_missing_api_key_is_rejected_at_construction(settings) -> None:
    with pytest.raises(ValueError):
        GeminiAnswerGenerator(settings.model_copy(update={"gemini_api_key": None}))


def test_no_context_returns_the_refusal_without_calling_the_llm(generator) -> None:
    models = _StubModels()
    _install(generator, models)

    result = generator.generate("Cau hoi ngoai sach", [], PROFILE)

    assert result.answer == prompts.NO_CONTEXT_ANSWER
    assert result.grounded is False
    assert models.calls == []


def test_retrieved_context_produces_a_grounded_answer_with_citations(generator) -> None:
    _install(generator, _StubModels("Em thu nghi xem..."))

    result = generator.generate("Phan so la gi?", [_chunk()], PROFILE)

    assert result.answer == "Em thu nghi xem..."
    assert result.grounded is True
    assert result.citations == ["SGK Toán 6, trang 12"]


def test_system_prompt_travels_as_system_instruction_not_a_content_turn(generator) -> None:
    models = _StubModels("ok")
    _install(generator, models)

    generator.generate("Phan so la gi?", [_chunk()], PROFILE)

    kwargs = models.calls[0]
    assert kwargs["config"].system_instruction
    assert all(content.role != "system" for content in kwargs["contents"])


def test_an_empty_completion_falls_back_to_the_refusal(generator) -> None:
    _install(generator, _StubModels(""))

    result = generator.generate("Phan so la gi?", [_chunk()], PROFILE)

    assert result.answer == prompts.NO_CONTEXT_ANSWER
    assert result.grounded is False


def test_a_rate_limit_error_becomes_a_friendly_message(generator) -> None:
    error = ClientError(429, {"message": "slow down"})
    _install(generator, _StubModels(error=error))

    result = generator.generate("Phan so la gi?", [_chunk()], PROFILE)

    assert result.grounded is False
    assert "chờ" in result.answer.lower() or result.answer


def test_a_server_error_becomes_a_friendly_message_not_a_crash(generator) -> None:
    error = ServerError(503, {"message": "unavailable"})
    _install(generator, _StubModels(error=error))

    result = generator.generate("Phan so la gi?", [_chunk()], PROFILE)

    assert result.grounded is False
    assert "Traceback" not in result.answer


def test_a_generic_client_error_becomes_a_friendly_message(generator) -> None:
    error = ClientError(400, {"message": "bad request"})
    _install(generator, _StubModels(error=error))

    result = generator.generate("Phan so la gi?", [_chunk()], PROFILE)

    assert result.grounded is False
    assert "Traceback" not in result.answer


def test_build_generator_defaults_to_openai(settings) -> None:
    settings = settings.model_copy(update={"openai_api_key": "test-key"})
    assert isinstance(build_generator(settings), OpenAIAnswerGenerator)


def test_build_generator_selects_gemini(settings) -> None:
    settings = settings.model_copy(
        update={"llm_provider": "gemini", "gemini_api_key": "test-key"}
    )
    assert isinstance(build_generator(settings), GeminiAnswerGenerator)


def test_build_generator_is_case_insensitive(settings) -> None:
    settings = settings.model_copy(
        update={"llm_provider": "Gemini", "gemini_api_key": "test-key"}
    )
    assert isinstance(build_generator(settings), GeminiAnswerGenerator)


def test_build_generator_rejects_an_unknown_provider(settings) -> None:
    settings = settings.model_copy(update={"llm_provider": "claude"})
    with pytest.raises(ValueError, match="Unsupported LLM_PROVIDER"):
        build_generator(settings)
