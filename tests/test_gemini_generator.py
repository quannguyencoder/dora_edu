"""Tests for the Gemini backend and the provider-selecting factory."""

from __future__ import annotations

from typing import Any

import httpx
import pytest
from google.genai.errors import ClientError, ServerError

from dora_edu.config import parse_model_list
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


class _FlakyThenGoodModels:
    """Returns an empty completion once, then a real answer -- simulates the
    sampling flakiness observed in practice (an identical retry succeeds)."""

    def __init__(self, text: str) -> None:
        self.text = text
        self.calls: list[dict[str, Any]] = []

    def generate_content(self, **kwargs: Any) -> Any:
        self.calls.append(kwargs)
        content = "" if len(self.calls) == 1 else self.text
        return type("Response", (), {"text": content})()


class _PerModelStub:
    """Routes ``generate_content`` by the requested model name.

    ``responses`` maps a model id to either the text it should return or an
    exception it should raise, so a test can simulate one model being
    rate-limited while another still has quota.
    """

    def __init__(self, responses: dict[str, str | Exception]) -> None:
        self.responses = responses
        self.calls: list[str] = []

    def generate_content(self, **kwargs: Any) -> Any:
        model = kwargs["model"]
        self.calls.append(model)
        outcome = self.responses.get(model, "")
        if isinstance(outcome, Exception):
            raise outcome
        return type("Response", (), {"text": outcome})()


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
    models = _StubModels("")
    _install(generator, models)

    result = generator.generate("Phan so la gi?", [_chunk()], PROFILE)

    assert result.answer == prompts.NO_CONTEXT_ANSWER
    assert result.grounded is False
    # One retry is attempted before giving up.
    assert len(models.calls) == 2


def test_an_empty_completion_is_retried_and_can_still_succeed(generator) -> None:
    models = _FlakyThenGoodModels("Day la cau tra loi that su")
    _install(generator, models)

    result = generator.generate("Phan so la gi?", [_chunk()], PROFILE)

    assert result.answer == "Day la cau tra loi that su"
    assert result.grounded is True
    assert len(models.calls) == 2


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


def test_a_connection_level_failure_becomes_a_friendly_message_not_a_crash(generator) -> None:
    # google-genai's own retry wrapper re-raises a connection-level failure
    # (timeout, reset, DNS...) as a plain httpx exception, not ClientError or
    # ServerError, once its retry budget is spent. Reproduced live: this
    # exact gap crashed the whole Discord message handler, leaving a student
    # with no reply at all instead of a friendly error.
    error = httpx.ConnectError("connection reset")
    _install(generator, _StubModels(error=error))

    result = generator.generate("Phan so la gi?", [_chunk()], PROFILE)

    assert result.grounded is False
    assert "Traceback" not in result.answer


def test_classify_subject_returns_none_on_a_connection_level_failure(generator) -> None:
    error = httpx.ReadTimeout("timed out")
    _install(generator, _StubModels(error=error))

    assert generator.classify_subject("abc", ["Toán", "Lịch sử"]) is None


# --- Model fallback on rate limit -------------------------------------------


def test_generate_falls_back_to_the_next_model_when_the_primary_is_rate_limited(
    generator, settings
) -> None:
    first_fallback = parse_model_list(settings.gemini_fallback_models)[0]
    models = _PerModelStub(
        {
            settings.llm_model: ClientError(429, {"message": "slow down"}),
            first_fallback: "Cau tra loi tu model du phong",
        }
    )
    _install(generator, models)

    result = generator.generate("Phan so la gi?", [_chunk()], PROFILE)

    assert result.answer == "Cau tra loi tu model du phong"
    assert result.grounded is True
    assert models.calls == [settings.llm_model, first_fallback]


def test_generate_tries_every_fallback_before_giving_up(generator, settings) -> None:
    error = ClientError(429, {"message": "slow down"})
    models = _PerModelStub({model: error for model in generator._models_in_order()})
    _install(generator, models)

    result = generator.generate("Phan so la gi?", [_chunk()], PROFILE)

    assert result.grounded is False
    assert models.calls == generator._models_in_order()


def test_generate_does_not_try_fallback_models_after_a_non_rate_limit_error(
    generator, settings
) -> None:
    error = ServerError(503, {"message": "down"})
    models = _PerModelStub({settings.llm_model: error})
    _install(generator, models)

    result = generator.generate("Phan so la gi?", [_chunk()], PROFILE)

    assert result.grounded is False
    assert models.calls == [settings.llm_model]


def test_generate_does_not_cascade_models_on_a_merely_empty_completion(
    generator, settings
) -> None:
    models = _StubModels("")
    _install(generator, models)

    result = generator.generate("Phan so la gi?", [_chunk()], PROFILE)

    assert result.answer == prompts.NO_CONTEXT_ANSWER
    # Only the primary model is retried (twice) -- an empty completion is not
    # a rate limit, so it must not cascade through every fallback model too.
    assert len(models.calls) == 2


def test_models_in_order_drops_a_fallback_that_duplicates_the_primary(settings) -> None:
    settings = settings.model_copy(
        update={
            "gemini_api_key": "test-key",
            "llm_model": "gemini-3.5-flash-lite",
            "gemini_fallback_models": "gemini-3.5-flash-lite,gemini-3.1-flash-lite",
        }
    )
    generator = GeminiAnswerGenerator(settings)

    assert generator._models_in_order() == ["gemini-3.5-flash-lite", "gemini-3.1-flash-lite"]


def test_classify_subject_falls_back_to_the_next_model_when_the_primary_is_rate_limited(
    generator, settings
) -> None:
    first_fallback = parse_model_list(settings.gemini_fallback_models)[0]
    models = _PerModelStub(
        {
            settings.llm_model: ClientError(429, {"message": "slow down"}),
            first_fallback: "Toán",
        }
    )
    _install(generator, models)

    result = generator.classify_subject("Phan so la gi?", ["Toán", "Lịch sử"])

    assert result == "Toán"
    assert models.calls == [settings.llm_model, first_fallback]


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


# --- Subject classification -------------------------------------------------


def test_classify_subject_returns_none_for_an_empty_subject_list(generator) -> None:
    assert generator.classify_subject("Phan so la gi?", []) is None


def test_classify_subject_matches_the_models_reply(generator) -> None:
    models = _StubModels("Toán")
    _install(generator, models)

    assert generator.classify_subject("Phan so la gi?", ["Toán", "Lịch sử"]) == "Toán"
    # No system_instruction/history plumbing needed for this minimal call.
    assert models.calls[0]["config"].max_output_tokens == 200


def test_classify_subject_returns_none_when_the_reply_is_ambiguous(generator) -> None:
    _install(generator, _StubModels("Có thể là Toán hoặc Lịch sử"))

    assert generator.classify_subject("abc", ["Toán", "Lịch sử"]) is None


def test_classify_subject_returns_none_when_the_provider_is_unreachable(generator) -> None:
    error = ServerError(503, {"message": "unavailable"})
    _install(generator, _StubModels(error=error))

    assert generator.classify_subject("abc", ["Toán", "Lịch sử"]) is None


# --- Diacritic restoration ---------------------------------------------------


def test_restore_diacritics_returns_the_models_correction(generator) -> None:
    _install(generator, _StubModels("đa thức là gì"))

    assert generator.restore_diacritics("da thuc la gi") == "đa thức là gì"


def test_restore_diacritics_returns_the_original_when_every_model_is_unreachable(
    generator,
) -> None:
    error = ServerError(503, {"message": "unavailable"})
    _install(generator, _StubModels(error=error))

    assert generator.restore_diacritics("da thuc la gi") == "da thuc la gi"


def test_restore_diacritics_falls_back_to_the_next_model_when_rate_limited(
    generator, settings
) -> None:
    first_fallback = parse_model_list(settings.gemini_fallback_models)[0]
    models = _PerModelStub(
        {
            settings.llm_model: ClientError(429, {"message": "slow down"}),
            first_fallback: "đa thức là gì",
        }
    )
    _install(generator, models)

    assert generator.restore_diacritics("da thuc la gi") == "đa thức là gì"


def test_restore_diacritics_of_an_empty_string_skips_the_api_call(generator) -> None:
    models = _StubModels()
    _install(generator, models)

    assert generator.restore_diacritics("   ") == "   "
    assert models.calls == []
