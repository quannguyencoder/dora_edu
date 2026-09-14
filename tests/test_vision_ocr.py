"""Tests for the Gemini Vision OCR helper and its LaTeX-stripping safety net."""

from __future__ import annotations

from typing import Any

import pytest
from google.genai.errors import ServerError
from PIL import Image

from dora_edu.data_pipeline import vision_ocr as vision_ocr_module
from dora_edu.data_pipeline.vision_ocr import _strip_stray_latex, ocr_page_with_gemini


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("AB/AC = 2", "AB/AC = 2"),
        ("$AB/AC$", "AB/AC"),
        (r"\frac{AB}{AC}", "AB/AC"),
        (r"tam giác \Delta ABC", "tam giác ABC"),
        ("**Định lí Thalès**", "Định lí Thalès"),
    ],
)
def test_strip_stray_latex_removes_markup_the_model_should_not_use(raw: str, expected: str) -> None:
    assert _strip_stray_latex(raw) == expected


def test_missing_api_key_is_rejected(settings) -> None:
    with pytest.raises(ValueError, match="GEMINI_API_KEY"):
        ocr_page_with_gemini(
            Image.new("RGB", (10, 10)), settings=settings.model_copy(update={"gemini_api_key": None})
        )


class _StubModels:
    def __init__(self, text: str | None = "Noi dung sach", error: Exception | None = None) -> None:
        self.text = text
        self.error = error
        self.calls: list[dict[str, Any]] = []

    def generate_content(self, **kwargs: Any) -> Any:
        self.calls.append(kwargs)
        if self.error is not None:
            raise self.error
        return type("Response", (), {"text": self.text})()


def _fake_client(models: _StubModels) -> Any:
    return type("Client", (), {"models": models})()


def test_ocr_page_with_gemini_returns_the_cleaned_text(monkeypatch, settings) -> None:
    settings = settings.model_copy(update={"gemini_api_key": "test-key"})
    models = _StubModels(r"AB/AC = \frac{2}{3}")
    monkeypatch.setattr(vision_ocr_module.genai, "Client", lambda api_key: _fake_client(models))

    result = ocr_page_with_gemini(Image.new("RGB", (10, 10)), settings=settings)

    assert result == "AB/AC = 2/3"
    assert len(models.calls) == 1


def test_a_provider_failure_returns_empty_string_not_a_crash(monkeypatch, settings) -> None:
    settings = settings.model_copy(update={"gemini_api_key": "test-key"})
    models = _StubModels(error=ServerError(503, {"message": "unavailable"}))
    monkeypatch.setattr(vision_ocr_module.genai, "Client", lambda api_key: _fake_client(models))

    assert ocr_page_with_gemini(Image.new("RGB", (10, 10)), settings=settings) == ""
