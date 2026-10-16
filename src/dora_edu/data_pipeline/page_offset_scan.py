"""Measure, per book, how far the PDF page index runs ahead of the printed page.

The page number printed at the foot of a textbook page is the one a student
can act on, and it is not the PDF's page index (the scan starts at the
cover). This scans a sample of pages per book, OCRs just the strip where the
page number sits, and keeps an offset only when the evidence agrees with
itself -- see :data:`_MIN_AGREEING_SAMPLES` and :data:`_MIN_AGREEMENT_SHARE`.

A book whose offset cannot be measured confidently is left out of the
output. That is deliberate: without an entry, citations keep PDF numbering
(today's behaviour, sometimes off by one), whereas a wrong guess would turn
citations that happen to be right into wrong ones.

Run it with ``dora-scan-page-offsets``; results land in
``data/page_offsets.json``, which :mod:`dora_edu.page_numbering` reads.
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
from collections import Counter
from dataclasses import dataclass
from datetime import date
from pathlib import Path

import pytesseract
from pypdf import PdfReader

from dora_edu.data_pipeline.pdf_parser import discover_pdfs
from dora_edu.page_numbering import OFFSETS_PATH, load_offsets

logger = logging.getLogger(__name__)

#: Fraction of the page height, measured from the bottom, that holds the
#: printed page number. Kept narrow so body text does not get OCR'd too.
_FOOTER_BAND = 0.07

#: Some books print the number at the top of the page instead.
_HEADER_BAND = 0.07

#: Tesseract in single-line mode, digits only: the strip holds a page number
#: and (sometimes) a running header, and only the number is wanted.
_OCR_CONFIG = "--psm 7 -c tessedit_char_whitelist=0123456789"

#: A plausible printed page number.
_PAGE_NUMBER = re.compile(r"\d{1,3}")

#: Largest offset worth considering: front matter runs to a handful of pages
#: (cover, title, copyright, contents), never dozens.
_MAX_OFFSET = 8

#: How many sampled pages must agree on the same offset before it is trusted.
_MIN_AGREEING_SAMPLES = 4

#: And what share of all usable samples that agreement must represent, so a
#: book where the numbers disagree is reported as unmeasured rather than
#: decided by a plurality of three against two.
_MIN_AGREEMENT_SHARE = 0.6


@dataclass(frozen=True)
class BookOffset:
    """What the scan concluded about one book."""

    source_file: str
    offset: int | None
    agreeing_samples: int
    usable_samples: int
    #: Every offset seen, for the report -- a book that disagrees with
    #: itself is worth looking at by hand.
    observed: tuple[tuple[int, int], ...]

    @property
    def confident(self) -> bool:
        """Whether the offset is backed by enough agreeing samples."""
        return self.offset is not None


def _page_number_candidates(image, *, band: float, from_top: bool) -> set[int]:
    """OCR one edge strip of a page image and return the numbers found."""
    width, height = image.size
    box = (
        (0, 0, width, int(height * band))
        if from_top
        else (0, int(height * (1 - band)), width, height)
    )
    try:
        text = pytesseract.image_to_string(image.crop(box), config=_OCR_CONFIG)
    except pytesseract.TesseractNotFoundError as exc:
        raise RuntimeError(
            "Tesseract OCR is not installed. Install it (e.g. `brew install "
            "tesseract tesseract-lang` on macOS) to scan page numbers."
        ) from exc
    except pytesseract.TesseractError as exc:
        logger.debug("Page-number OCR failed on one strip: %s", exc)
        return set()
    return {int(match.group()) for match in _PAGE_NUMBER.finditer(text)}


def measure_book(pdf_path: Path, *, samples: int = 14) -> BookOffset:
    """Measure one book's PDF-to-printed page offset.

    Pages are sampled from the middle of the book, skipping front and back
    matter: the cover, contents and back cover carry no printed number (or
    carry a Roman one), which would only add noise.

    Args:
        pdf_path: The book's PDF.
        samples: How many pages to look at.

    Returns:
        What could be concluded, confident or not.
    """
    try:
        reader = PdfReader(str(pdf_path))
        total = len(reader.pages)
    except Exception as exc:  # pypdf raises a variety of parse errors
        logger.warning("Cannot open %s: %s", pdf_path.name, exc)
        return BookOffset(pdf_path.name, None, 0, 0, ())

    if total < 20:
        return BookOffset(pdf_path.name, None, 0, 0, ())

    first, last = int(total * 0.15), int(total * 0.85)
    step = max(1, (last - first) // samples)
    offsets: Counter[int] = Counter()
    usable = 0

    for index in range(first, last, step):
        page_number = index + 1  # 1-based, matching chunk metadata
        try:
            images = list(reader.pages[index].images)
        except Exception as exc:  # a damaged page should not stop the book
            logger.debug("%s p.%d: cannot read image: %s", pdf_path.name, page_number, exc)
            continue
        if not images:
            continue

        found: set[int] = set()
        for from_top, band in ((False, _FOOTER_BAND), (True, _HEADER_BAND)):
            found |= _page_number_candidates(images[0].image, band=band, from_top=from_top)

        plausible = [
            page_number - printed
            for printed in found
            if 0 <= page_number - printed <= _MAX_OFFSET
        ]
        if plausible:
            usable += 1
            # A strip can yield several numbers (a running header, an
            # exercise number); the smallest offset is the one produced by
            # the number closest to this page's own index.
            offsets[min(plausible)] += 1

    if not offsets:
        return BookOffset(pdf_path.name, None, 0, usable, ())

    best, count = offsets.most_common(1)[0]
    confident = count >= _MIN_AGREEING_SAMPLES and count / usable >= _MIN_AGREEMENT_SHARE
    return BookOffset(
        source_file=pdf_path.name,
        offset=best if confident else None,
        agreeing_samples=count,
        usable_samples=usable,
        observed=tuple(sorted(offsets.items())),
    )


def scan_all(root: Path, *, samples: int = 14) -> list[BookOffset]:
    """Measure every PDF under ``root``.

    Args:
        root: Directory holding the textbook PDFs, searched recursively.
        samples: Pages sampled per book.

    Returns:
        One result per book, in file order.
    """
    results: list[BookOffset] = []
    pdfs = discover_pdfs(root)
    for index, pdf_path in enumerate(pdfs, start=1):
        result = measure_book(pdf_path, samples=samples)
        logger.info(
            "[%d/%d] %s: %s",
            index,
            len(pdfs),
            pdf_path.name,
            f"lệch {result.offset} trang ({result.agreeing_samples}/{result.usable_samples} mẫu)"
            if result.confident
            else f"chưa đo được ({result.usable_samples} mẫu đọc được: {result.observed})",
        )
        results.append(result)
    return results


def write_offsets(results: list[BookOffset], path: Path = OFFSETS_PATH) -> None:
    """Merge the confident offsets into the file :mod:`dora_edu.page_numbering` reads.

    Merged rather than overwritten: scanning one grade with more samples
    (``--path data/raw_pdfs/8 --samples 80``) is the normal way to pin down
    the books a broad scan could not measure, and that must not discard the
    hundred-odd books already measured.

    Args:
        results: Output of :func:`scan_all`.
        path: Destination JSON file.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    offsets = dict(load_offsets(path))
    for result in results:
        if result.confident and result.offset is not None:
            offsets[result.source_file] = result.offset

    payload = {
        "measured_on": date.today().isoformat(),
        "note": (
            "Số trang in trên sách = số trang PDF - offset. Sách không đo được "
            "thì không có trong danh sách này và vẫn dùng số trang PDF."
        ),
        "offsets": dict(sorted(offsets.items())),
        "unmeasured": sorted(
            r.source_file
            for r in results
            if not r.confident and r.source_file not in offsets
        ),
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    """Entry point for ``dora-scan-page-offsets``.

    Returns:
        ``0`` when the scan ran, ``1`` when there was nothing to scan.
    """
    parser = argparse.ArgumentParser(
        description="Đo độ lệch giữa số trang PDF và số trang in trên sách."
    )
    parser.add_argument(
        "--path",
        type=Path,
        default=Path("./data/raw_pdfs"),
        help="Thư mục chứa PDF sách (mặc định: ./data/raw_pdfs)",
    )
    parser.add_argument("--samples", type=int, default=14, help="Số trang lấy mẫu mỗi sách")
    parser.add_argument(
        "--out", type=Path, default=OFFSETS_PATH, help="File JSON để ghi kết quả"
    )
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)-8s %(message)s")

    try:
        results = scan_all(args.path, samples=args.samples)
    except RuntimeError as exc:
        logger.error("%s", exc)
        return 1
    if not results:
        logger.error("Không tìm thấy PDF nào trong %s", args.path)
        return 1

    write_offsets(results, args.out)
    confident = [r for r in results if r.confident]
    logger.info(
        "Đo được %d/%d sách. Đã ghi: %s", len(confident), len(results), args.out
    )
    by_offset = Counter(r.offset for r in confident)
    for offset, count in sorted(by_offset.items()):
        logger.info("  lệch %s trang: %d sách", offset, count)
    return 0


if __name__ == "__main__":
    sys.exit(main())
