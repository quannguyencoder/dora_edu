"""Gemini Vision OCR: a targeted, higher-quality alternative to local Tesseract.

Tesseract handles plain body text reasonably well but does badly on the
things MOET textbooks are full of: fraction/formula notation and diagrams
with embedded point labels and figures (a labelled triangle, a number line...).
Gemini reads the page image directly and transcribes both far more reliably.

This is deliberately kept separate from :mod:`pdf_parser`'s local OCR path:
sending a page image to Gemini leaves the machine, which is only acceptable
with the student's/admin's explicit go-ahead (see the data-privacy rule in
CLAUDE.md) -- so this module is invoked deliberately, page by page, to patch
a specific page whose local OCR came out unusable, not wired into the default
bulk-ingestion pipeline.
"""

from __future__ import annotations

import io
import logging
import re

from google import genai
from google.genai import types as genai_types
from google.genai.errors import ClientError, ServerError
from PIL import Image

from dora_edu.config import Settings, get_settings

logger = logging.getLogger(__name__)

#: Deliberately not ``settings.llm_model``: this runs rarely (one admin
#: patching one bad page), so it's worth spending the flagship model's
#: tighter free-tier quota here rather than the lite model used for every
#: student chat turn -- image transcription accuracy matters more than quota
#: headroom for this one-off, low-volume use.
_VISION_MODEL = "gemini-3.6-flash"

_PROMPT = """\
Hãy chép lại TOÀN BỘ chữ, công thức và nội dung trong hình vẽ ở trang sách \
giáo khoa này thành văn bản thuần, đầy đủ và chính xác nhất có thể.

Quy tắc:
- TUYỆT ĐỐI không dùng cú pháp LaTeX hay ký hiệu $...$, \\frac{}{}, \\Delta, \
^, _ dưới bất kỳ hình thức nào. Viết công thức/phân số bằng chữ và ký tự \
thường, ví dụ: "AB/AC" thay vì "\\frac{AB}{AC}", "tam giác ABC" thay vì \
"\\Delta ABC".
- Với hình vẽ (tam giác, hình học, biểu đồ...), mô tả ngắn gọn các nhãn điểm \
và số liệu xuất hiện trong hình ngay sau đoạn văn bản liên quan, ví dụ: \
"Hình vẽ: tam giác DEF, M thuộc DE, N thuộc DF, DM=2, ME=4, DN=x, NF=5".
- Giữ đúng thứ tự đọc từ trên xuống, trái sang phải, đúng như bố cục trang.
- Không thêm giải thích ngoài lề, không bình luận, không thêm markdown \
(không dùng **, >, #).
"""

#: Catches any LaTeX the model writes despite the prompt, so it never leaks
#: into the chunk store even when the model doesn't fully comply.
_STRAY_DOLLAR = re.compile(r"\$([^$]*)\$")
_STRAY_FRAC = re.compile(r"\\frac\s*\{([^{}]*)\}\s*\{([^{}]*)\}")
_STRAY_LATEX_CMD = re.compile(r"\\(Delta|left|right|cdot|times)\b\s*")


def _strip_stray_latex(text: str) -> str:
    """Remove any LaTeX markup the model used despite being told not to."""
    text = _STRAY_DOLLAR.sub(r"\1", text)
    text = _STRAY_FRAC.sub(r"\1/\2", text)
    text = _STRAY_LATEX_CMD.sub("", text)
    return text.replace("{", "").replace("}", "").replace("**", "")


def ocr_page_with_gemini(image: Image.Image, *, settings: Settings | None = None) -> str:
    """Transcribe one page image with Gemini Vision.

    Args:
        image: The page's embedded raster image (same image Tesseract would OCR).
        settings: Application settings; the process singleton when omitted.

    Returns:
        The transcribed text, or ``""`` on any provider failure -- callers
        should fall back to the existing (Tesseract) text rather than lose
        the page entirely.

    Raises:
        ValueError: If ``GEMINI_API_KEY`` is not configured.
    """
    settings = settings or get_settings()
    if not settings.gemini_api_key:
        raise ValueError("GEMINI_API_KEY is not set; cannot use Gemini Vision OCR")

    client = genai.Client(api_key=settings.gemini_api_key)
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")

    try:
        response = client.models.generate_content(
            model=_VISION_MODEL,
            contents=[
                genai_types.Part.from_bytes(data=buffer.getvalue(), mime_type="image/png"),
                _PROMPT,
            ],
            config=genai_types.GenerateContentConfig(
                thinking_config=genai_types.ThinkingConfig(thinking_level="low"),
                max_output_tokens=4096,
            ),
        )
    except (ClientError, ServerError) as exc:
        logger.warning("Gemini Vision OCR failed: %s", exc)
        return ""

    text = getattr(response, "text", None) or ""
    return _strip_stray_latex(text).strip()
