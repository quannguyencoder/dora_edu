"""Translate a PDF page index into the page number printed on the paper book.

A scanned textbook's PDF page 72 is not page 72 of the book: the scan usually
starts with the cover, so the printed number lags the PDF index by a fixed
offset (1 in every MOET book measured so far). That gap does not matter to
retrieval, but it matters a lot to the student: the bot's whole safety story
rests on *"bạn mở sách trang ... để kiểm tra lại nhé"*, and a reminder that
sends a 12-year-old to the wrong page is worse than no reminder at all --
they look, do not find it, and conclude the bot made the whole thing up.

Offsets are measured per book by ``dora-scan-page-offsets`` (which OCRs the
page-number strip at the foot of a sample of pages) and stored in
``data/page_offsets.json``. A book that could not be measured confidently is
simply absent from that file and keeps its PDF numbering -- citing the PDF
page is the status quo, whereas a guessed offset would make a correct
citation wrong.
"""

from __future__ import annotations

import json
import logging
from functools import lru_cache
from pathlib import Path

logger = logging.getLogger(__name__)

#: Where ``dora-scan-page-offsets`` writes what it measured.
OFFSETS_PATH = Path("./data/page_offsets.json")


@lru_cache(maxsize=4)
def load_offsets(path: Path | None = None) -> dict[str, int]:
    """Read the measured per-book offsets.

    Cached: this is consulted for every citation of every answer, and the
    file only changes when the corpus is re-scanned.

    Args:
        path: The JSON file to read; :data:`OFFSETS_PATH` when omitted.
            Resolved on each call rather than bound as a default, so a test
            (or an admin tool) can point it elsewhere.

    Returns:
        A ``{source_file: offset}`` mapping; empty when the file is missing
        or unreadable, which makes every citation fall back to PDF
        numbering.
    """
    path = path or OFFSETS_PATH
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return {}
    except (OSError, json.JSONDecodeError) as exc:
        logger.warning("Cannot read page offsets from %s: %s", path, exc)
        return {}
    return {
        str(book): int(offset)
        for book, offset in raw.get("offsets", {}).items()
        if isinstance(offset, int) and offset >= 0
    }


def printed_page(source_file: str, pdf_page: int) -> int:
    """Return the page number printed in the book for a PDF page index.

    Args:
        source_file: The book's PDF file name, as stored in chunk metadata.
        pdf_page: 1-based page index within that PDF.

    Returns:
        The printed page number, or ``pdf_page`` unchanged when this book's
        offset was never measured (or when subtracting it would produce a
        nonsensical page number, which means the measurement was wrong).
    """
    offset = load_offsets(OFFSETS_PATH).get(source_file, 0)
    printed = pdf_page - offset
    return printed if printed >= 1 else pdf_page
