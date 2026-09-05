"""PDF extraction and cleaning logic.

Raw MOET textbook PDFs are treated as sensitive local data: this module only
ever reads them from the local filesystem and never uploads them anywhere.
"""

from __future__ import annotations

import logging
import re
import unicodedata
from collections import Counter
from pathlib import Path

from pypdf import PdfReader
from pypdf.errors import PdfReadError

from dora_edu.models import ParsedPage

logger = logging.getLogger(__name__)

#: A line that is nothing but a page number (optionally decorated with dashes).
_PAGE_NUMBER_LINE = re.compile(r"^\s*[-–—|]*\s*\d{1,4}\s*[-–—|]*\s*$")

#: Words split across a line break by a hyphen, e.g. "chuy-\nển".
_HYPHEN_LINE_BREAK = re.compile(r"(\w)-\s*\n\s*(\w)")

#: Three or more blank lines collapse to a paragraph break.
_EXCESS_BLANK_LINES = re.compile(r"\n{3,}")

#: Runs of spaces/tabs that survive PDF layout extraction.
_EXCESS_SPACES = re.compile(r"[ \t ]{2,}")

#: Control characters that carry no meaning once extracted.
_CONTROL_CHARS = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")

#: A header/footer must repeat on at least this share of pages to be dropped.
_REPEATED_LINE_RATIO = 0.5

#: Only short lines are ever considered running headers/footers.
_MAX_HEADER_LENGTH = 80


def clean_text(raw: str) -> str:
    """Normalise one page of raw PDF text into clean Vietnamese prose.

    Applies Unicode NFC normalisation (so that pre-composed and decomposed
    Vietnamese vowels compare equal), rejoins hyphenated line breaks, drops
    control characters and standalone page numbers, and collapses redundant
    whitespace.

    Args:
        raw: Text as returned by the PDF extractor.

    Returns:
        The cleaned text, possibly an empty string for image-only pages.
    """
    if not raw:
        return ""

    text = unicodedata.normalize("NFC", raw)
    text = _CONTROL_CHARS.sub("", text)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = _HYPHEN_LINE_BREAK.sub(r"\1\2", text)

    kept_lines: list[str] = []
    for line in text.split("\n"):
        stripped = _EXCESS_SPACES.sub(" ", line).strip()
        if _PAGE_NUMBER_LINE.match(stripped):
            continue
        kept_lines.append(stripped)

    text = "\n".join(kept_lines)
    text = _EXCESS_BLANK_LINES.sub("\n\n", text)
    return text.strip()


def find_repeated_lines(pages: list[str], ratio: float = _REPEATED_LINE_RATIO) -> set[str]:
    """Detect running headers and footers shared by most pages of a book.

    Textbooks repeat the chapter or book title on every page; leaving those in
    pollutes the embeddings, so ingestion strips lines that appear on at least
    ``ratio`` of the pages.

    Args:
        pages: Cleaned page texts.
        ratio: Minimum share of pages a line must appear on to be considered
            boilerplate.

    Returns:
        The set of lines to drop.
    """
    if len(pages) < 3:
        return set()

    counter: Counter[str] = Counter()
    for page in pages:
        # Count each distinct line once per page so a repeated in-page phrase
        # is not mistaken for a running header.
        unique_lines = {
            line.strip()
            for line in page.split("\n")
            if line.strip() and len(line.strip()) <= _MAX_HEADER_LENGTH
        }
        counter.update(unique_lines)

    threshold = max(2, int(len(pages) * ratio))
    return {line for line, count in counter.items() if count >= threshold}


def _drop_lines(page: str, blocked: set[str]) -> str:
    """Remove ``blocked`` boilerplate lines from a single page."""
    if not blocked:
        return page
    kept = [line for line in page.split("\n") if line.strip() not in blocked]
    return _EXCESS_BLANK_LINES.sub("\n\n", "\n".join(kept)).strip()


def parse_pdf(pdf_path: Path, *, strip_boilerplate: bool = True) -> list[ParsedPage]:
    """Extract and clean every text-bearing page of one textbook PDF.

    Args:
        pdf_path: Path to a local PDF file.
        strip_boilerplate: Whether to remove running headers and footers.

    Returns:
        Cleaned pages in reading order; pages that yield no text (e.g. pure
        scans) are omitted.

    Raises:
        FileNotFoundError: If ``pdf_path`` does not exist.
        ValueError: If the file cannot be read as a PDF.
    """
    if not pdf_path.is_file():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    try:
        reader = PdfReader(str(pdf_path))
        raw_pages = [page.extract_text() or "" for page in reader.pages]
    except (PdfReadError, OSError, ValueError) as exc:
        raise ValueError(f"Cannot read PDF {pdf_path.name}: {exc}") from exc

    cleaned = [clean_text(raw) for raw in raw_pages]
    blocked = find_repeated_lines(cleaned) if strip_boilerplate else set()

    parsed: list[ParsedPage] = []
    for page_number, page_text in enumerate(cleaned, start=1):
        final_text = _drop_lines(page_text, blocked)
        if final_text:
            parsed.append(ParsedPage(page_number=page_number, text=final_text))

    empty_count = len(raw_pages) - len(parsed)
    if empty_count:
        logger.info(
            "%s: %d of %d pages produced no text (likely scanned images)",
            pdf_path.name,
            empty_count,
            len(raw_pages),
        )
    return parsed


def discover_pdfs(path: Path) -> list[Path]:
    """Return the PDF files at ``path``, which may be a file or a directory.

    Raises:
        FileNotFoundError: If ``path`` does not exist.
    """
    if path.is_file():
        return [path] if path.suffix.lower() == ".pdf" else []
    if path.is_dir():
        return sorted(p for p in path.rglob("*.pdf") if p.is_file())
    raise FileNotFoundError(f"Path not found: {path}")
