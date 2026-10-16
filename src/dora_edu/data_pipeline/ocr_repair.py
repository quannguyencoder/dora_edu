"""Re-read pages the OCR audit flagged, and index a clean copy of each.

``dora-ocr-audit`` finds pages where Tesseract mangled the thing that
matters most -- a formula's exponents, a fraction bar, a multiplication dot.
Fixing those one at a time by hand (as
:data:`~dora_edu.data_pipeline.manual_patches.TEXT_FIXES` does) does not
scale past a handful: there are 54 such pages in the lower-secondary maths
books alone.

So this re-reads a flagged page from its page image with Gemini Vision,
which handles formula notation far better than local OCR, and indexes the
result as an extra, clean chunk beside the damaged one. The damaged chunk is
left in place: it usually spans more than one page, so replacing it would
risk losing text from a page nobody checked. A clean chunk next to it is
enough for retrieval, which is what a student's question actually hits.

Every re-read page is saved to ``data/ocr_repairs.json`` and re-applied by
``dora-bulk-ingest``, so a re-ingest cannot silently restore the mangled
version -- the same guarantee ``manual_patches`` gives, without a code change
per page.

Scope is deliberately lower secondary first (grades 6-9): that is where the
project is being used, where the two of us can check a page against our own
printed books, and -- measured by the audit -- where the OCR is in good
enough shape for the remaining damage to be a short list rather than a
rewrite.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import asdict, dataclass
from datetime import date
from pathlib import Path

from pypdf import PdfReader

from dora_edu.config import Settings, get_settings
from dora_edu.data_pipeline.catalog import infer_book_info
from dora_edu.data_pipeline.vision_ocr import ocr_page_with_gemini
from dora_edu.evaluation.ocr_audit import PageAudit, audit_sample
from dora_edu.models import TextbookMetadata, TextChunk
from dora_edu.rag_engine.indexer import TextbookIndexer

logger = logging.getLogger(__name__)

#: Where re-read pages are kept, so they survive a re-ingest.
REPAIRS_PATH = Path("./data/ocr_repairs.json")

#: Chunk indices for repaired pages start here -- clear of both the
#: chunker's own indices and the 900000 block used by hand-written patches,
#: so the three never collide and the source of a chunk is readable from its
#: index alone.
REPAIR_CHUNK_OFFSET = 800_000

#: Issue kinds worth spending a Vision call on: these damage a formula, a
#: fraction or a definition, which is what makes a passage useless (or, for
#: the multiplication dot, actively wrong). "so-lan-vao-chu" is left out --
#: it mostly flags scanned artwork and running headers, where a cleaner
#: transcription buys nothing.
DEFAULT_KINDS: frozenset[str] = frozenset(
    {
        "mat-so-mu",
        "mat-hop-dinh-nghia",
        "phan-so-mat-gach-ngang",
        "dau-nhan-thanh-dau-tru",
        "trang-quet-loi-nang",
    }
)

#: Below this, the transcription came back too short to be a textbook page
#: (a refusal, or a near-blank response); the original is left alone rather
#: than shadowed by a clean-looking but empty chunk.
_MIN_USEFUL_LENGTH = 200


@dataclass(frozen=True)
class RepairedPage:
    """One page re-read from its image, ready to index."""

    grade: int
    subject: str
    book_title: str
    source_file: str
    page: int
    text: str
    #: Which audit flags prompted the re-read, for the record.
    kinds: tuple[str, ...]

    @property
    def chunk_index(self) -> int:
        """A stable chunk index derived from the page number."""
        return REPAIR_CHUNK_OFFSET + self.page


def load_repairs(path: Path | None = None) -> list[RepairedPage]:
    """Read previously repaired pages.

    Args:
        path: The JSON file to read; :data:`REPAIRS_PATH` when omitted.
            Resolved on each call rather than bound as a default, so a test
            can point it elsewhere.

    Returns:
        Every saved page; empty when the file is missing or unreadable.
    """
    path = path or REPAIRS_PATH
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return []
    except (OSError, json.JSONDecodeError) as exc:
        logger.warning("Cannot read OCR repairs from %s: %s", path, exc)
        return []
    return [
        RepairedPage(
            grade=int(page["grade"]),
            subject=str(page["subject"]),
            book_title=str(page["book_title"]),
            source_file=str(page["source_file"]),
            page=int(page["page"]),
            text=str(page["text"]),
            kinds=tuple(page.get("kinds", ())),
        )
        for page in raw.get("pages", [])
    ]


def save_repairs(pages: list[RepairedPage], path: Path | None = None) -> None:
    """Write repaired pages, newest state wins per (book, page).

    Args:
        pages: Pages to save.
        path: Destination JSON file; :data:`REPAIRS_PATH` when omitted.
    """
    path = path or REPAIRS_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    unique: dict[tuple[str, int], RepairedPage] = {}
    for page in pages:
        unique[(page.source_file, page.page)] = page
    payload = {
        "updated_on": date.today().isoformat(),
        "note": (
            "Các trang được đọc lại bằng Gemini Vision vì máy đọc chữ (Tesseract) "
            "làm hỏng công thức/phân số. Mỗi trang được thêm vào kho dưới dạng một "
            "đoạn sạch, và được áp lại sau mỗi lần nạp dữ liệu."
        ),
        "pages": [asdict(page) for page in sorted(unique.values(), key=_sort_key)],
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _sort_key(page: RepairedPage) -> tuple[int, str, int]:
    """Order pages by grade, book and page, so the file reads sensibly."""
    return (page.grade, page.source_file, page.page)


def apply_repairs(indexer: TextbookIndexer, path: Path | None = None) -> int:
    """Index every saved repaired page.

    Safe to re-run: each page upserts under the same stable id, so this
    never duplicates a chunk.

    Args:
        indexer: Destination index.
        path: The JSON file holding the repairs; :data:`REPAIRS_PATH` when
            omitted.

    Returns:
        The number of chunks written.
    """
    written = 0
    for page in load_repairs(path):
        metadata = TextbookMetadata(
            grade=page.grade,
            subject=page.subject,
            book_title=page.book_title,
            source_file=page.source_file,
        )
        chunk = TextChunk(
            text=page.text,
            chunk_index=page.chunk_index,
            page_start=page.page,
            page_end=page.page,
        )
        written += indexer.index_chunks([chunk], metadata)
    return written


def _pdf_path_for(audit: PageAudit, pdf_root: Path) -> Path:
    """Return the PDF a flagged page came from."""
    return pdf_root / str(audit.grade) / f"{audit.book}.pdf"


def reread_page(audit: PageAudit, pdf_root: Path, *, settings: Settings) -> RepairedPage | None:
    """Re-read one flagged page from its image with Gemini Vision.

    Args:
        audit: The flagged page.
        pdf_root: Directory holding the textbook PDFs.
        settings: Application settings.

    Returns:
        The repaired page, or ``None`` when the PDF or page image cannot be
        read, or the transcription came back too short to trust.
    """
    pdf_path = _pdf_path_for(audit, pdf_root)
    try:
        info = infer_book_info(pdf_path)
        reader = PdfReader(str(pdf_path))
        images = list(reader.pages[audit.page - 1].images)
    except (OSError, ValueError, IndexError, KeyError) as exc:
        logger.warning("Bỏ qua %s trang %d: %s", audit.book, audit.page, exc)
        return None
    if not images:
        logger.warning("Trang %d của %s không có ảnh", audit.page, audit.book)
        return None

    text = ocr_page_with_gemini(images[0].image, settings=settings)
    if len(text.strip()) < _MIN_USEFUL_LENGTH:
        logger.warning(
            "Đọc lại trang %d của %s chỉ ra %d ký tự — giữ nguyên bản cũ",
            audit.page,
            audit.book,
            len(text.strip()),
        )
        return None

    return RepairedPage(
        grade=audit.grade,
        subject=info.subject,
        book_title=info.book_title,
        source_file=pdf_path.name,
        page=audit.page,
        text=text,
        kinds=tuple(sorted({issue.kind for issue in audit.issues})),
    )


def flagged_pages(
    *, grades: set[int] | None, kinds: frozenset[str], seed: int = 2026
) -> list[PageAudit]:
    """Audit the corpus and return the pages worth re-reading, worst first.

    Args:
        grades: Grades to audit; every grade when ``None``.
        kinds: Issue kinds that justify a re-read.
        seed: Passed through to the audit so a run is reproducible.

    Returns:
        Flagged pages, those with the most issues first -- a page with
        several flags is the most damaged and the most worth a call.
    """
    audits = audit_sample(sample_size=0, seed=seed, grades=grades)
    wanted = [
        audit
        for audit in audits
        if any(issue.kind in kinds for issue in audit.issues)
    ]
    return sorted(wanted, key=lambda a: len(a.issues), reverse=True)


def parse_page_specs(raw: str) -> list[PageAudit]:
    """Turn ``"6/SGKKhoahoctunhien6:57"`` specs into pages to re-read.

    The audit only flags damage it has a pattern for, and some of the worst
    damage leaves no trace: a definition printed in a coloured box is simply
    absent from the OCR text, with nothing odd-looking left behind. When the
    benchmark turns up a wrong refusal like that, this is how the page gets
    re-read without waiting for a new heuristic.

    Args:
        raw: Comma-separated ``<grade>/<book stem>:<page>`` specs.

    Returns:
        One synthetic :class:`PageAudit` per spec.

    Raises:
        ValueError: If a spec is not in that form.
    """
    pages: list[PageAudit] = []
    for spec in (s.strip() for s in raw.split(",") if s.strip()):
        try:
            location, page = spec.rsplit(":", 1)
            grade, book = location.split("/", 1)
            audit = PageAudit(
                grade=int(grade),
                book=book,
                page=int(page),
                issues=(),
                has_formula=False,
            )
        except ValueError as exc:
            raise ValueError(
                f"{spec!r} không đúng dạng <lớp>/<tên sách>:<trang>, "
                "ví dụ: 6/SGKKhoahoctunhien6:57"
            ) from exc
        pages.append(audit)
    return pages


def main(argv: list[str] | None = None) -> int:
    """Entry point for ``dora-ocr-repair``.

    Returns:
        ``0`` when the run finished, ``1`` when it could not start.
    """
    parser = argparse.ArgumentParser(
        description="Đọc lại bằng Gemini Vision những trang bị công cụ soát OCR gắn cờ."
    )
    parser.add_argument(
        "--grades",
        default="6,7,8,9",
        help="Lớp cần sửa, mặc định 6,7,8,9 (THCS trước, rồi mới mở rộng)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=20,
        help="Số trang đọc lại trong lần chạy này (mỗi trang tốn 1 lượt gọi API)",
    )
    parser.add_argument(
        "--kinds",
        default=",".join(sorted(DEFAULT_KINDS)),
        help="Các dạng lỗi cần sửa, cách nhau bởi dấu phẩy",
    )
    parser.add_argument(
        "--pdf-root", type=Path, default=Path("./data/raw_pdfs"), help="Thư mục chứa PDF"
    )
    parser.add_argument(
        "--pages",
        default="",
        help=(
            "Đọc lại đúng những trang này thay vì chạy soát, dạng "
            "<lớp>/<tên sách>:<trang>, ví dụ: 6/SGKKhoahoctunhien6:57,"
            "6/SGKKhoahoctunhien6:58"
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Chỉ liệt kê các trang sẽ đọc lại, không gọi API",
    )
    args = parser.parse_args(argv)

    settings = get_settings()
    logging.basicConfig(
        level=getattr(logging, settings.log_level.upper(), logging.INFO),
        format="%(levelname)-8s %(message)s",
    )
    logging.getLogger("httpx").setLevel(logging.WARNING)

    grades = {int(g) for g in args.grades.split(",") if g.strip()} or None
    kinds = frozenset(k.strip() for k in args.kinds.split(",") if k.strip())

    try:
        if args.pages:
            candidates = parse_page_specs(args.pages)
        else:
            candidates = flagged_pages(grades=grades, kinds=kinds)
    except (RuntimeError, ValueError) as exc:
        logger.error("%s", exc)
        return 1

    done = load_repairs()
    already = {(page.source_file, page.page) for page in done}
    todo = (
        candidates
        if args.pages
        else [
            audit
            for audit in candidates
            if (f"{audit.book}.pdf", audit.page) not in already
        ]
    )

    logger.info(
        "Có %d trang cần sửa, đã sửa trước đó %d, còn lại %d. Lần này đọc lại %d trang.",
        len(candidates),
        len(candidates) - len(todo),
        len(todo),
        min(args.limit, len(todo)),
    )

    if args.dry_run:
        for audit in todo[: args.limit]:
            kinds_found = ", ".join(sorted({i.kind for i in audit.issues}))
            logger.info("  lớp %d %s trang %d — %s", audit.grade, audit.book, audit.page, kinds_found)
        return 0

    repaired: list[RepairedPage] = list(done)
    added = 0
    for audit in todo[: args.limit]:
        page = reread_page(audit, args.pdf_root, settings=settings)
        if page is None:
            continue
        repaired.append(page)
        added += 1
        logger.info(
            "Đã đọc lại: lớp %d %s trang %d (%d ký tự)",
            page.grade,
            page.source_file,
            page.page,
            len(page.text),
        )
        # Saved after every page: a run interrupted by a rate limit keeps
        # everything it already paid for.
        save_repairs(repaired)

    if not added:
        logger.info("Không đọc lại được trang nào trong lần chạy này.")
        return 0

    indexer = TextbookIndexer(settings)
    written = apply_repairs(indexer)
    logger.info("Đã đọc lại %d trang, ghi %d đoạn sạch vào kho.", added, written)
    logger.info("Kho hiện có %d đoạn.", indexer.count())
    return 0


if __name__ == "__main__":
    sys.exit(main())
