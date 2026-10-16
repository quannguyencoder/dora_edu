"""Spot-check the OCR text behind the index, formulas and fractions first.

Every textbook in this corpus is a scan, so every word in the index came out
of Tesseract -- and Tesseract is at its worst exactly where being wrong
matters most. Three real failures found in this project, each of which the
bot then quoted to a student with a correct-looking page citation:

* ``a . b = b . a`` read as ``a - b = b - a`` -- a true commutative law
  turned into a false statement.
* ``a/b`` read as a bare ``=``, so the one sentence defining "phân số"
  ("ta gọi a/b là một phân số") no longer read as Vietnamese at all.
* Every exponent lost from the seven-identities box (``A² - B²`` came out as
  ``A? ~B?``), leaving the passage with no usable formula.

Rather than re-reading 16,778 pages by hand, this module samples pages and
flags the specific shapes those failures leave behind. A flag is a *suspicion*
to go and check against the printed page, not a verdict: the output is a
shortlist for a person, which is how the three cases above were confirmed and
then corrected in :data:`~dora_edu.data_pipeline.manual_patches.TEXT_FIXES`.

Run it with ``dora-ocr-audit``.
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import re
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

from dora_edu.config import get_settings

logger = logging.getLogger(__name__)

#: Where the PDF parser caches its OCR output: one JSON file per book,
#: under a per-grade folder.
_CACHE_ROOT = Path("./data/processed")

#: How much text to show around a flagged spot.
_SNIPPET_PADDING = 45


@dataclass(frozen=True)
class OcrIssue:
    """One suspicious spot in a page's OCR text."""

    kind: str
    #: Vietnamese explanation of what looks wrong, for the report.
    explanation: str
    snippet: str


@dataclass(frozen=True)
class PageAudit:
    """Everything flagged on one page."""

    grade: int
    book: str
    page: int
    issues: tuple[OcrIssue, ...]
    #: Whether the page carries a formula or fraction at all, which is where
    #: OCR damage is both most likely and most harmful.
    has_formula: bool


#: A commutative/associative law whose operator came out as a minus sign:
#: "a - b = b - a" is false for every a != b, so this shape is almost always
#: a mis-read "a . b = b . a". Matches with or without spaces.
_SWAPPED_MINUS = re.compile(r"\b([a-zA-Z])\s*-\s*([a-zA-Z])\s*=\s*\2\s*-\s*\1\b")

#: A letter immediately followed by "?" or "*" inside a formula: what a lost
#: exponent usually leaves behind ("A? ~B?" for "A² - B²").
_LOST_EXPONENT = re.compile(r"\b[A-Za-z]\s*[?*](?=\s*[-+=~)\]]|\s*$)")

#: "phân số" defined without any fraction visible near it -- the exact shape
#: of the "ta gợi = là một phân s6" failure.
_FRACTION_WITHOUT_SLASH = re.compile(
    r"(?:gọi|gợi|goi)\s*[=~]\s*l[àa]\s*(?:một|mot)\s*ph[âa]n\s*s", re.IGNORECASE
)

#: A digit wedged inside a Vietnamese word ("s6" for "số", "m0t" for "một",
#: "l1" for "li"). Tesseract does this when a diacritic confuses it.
_DIGIT_IN_WORD = re.compile(r"\b[a-zA-ZÀ-ỹ]{1,6}[0-9][a-zA-ZÀ-ỹ]{0,6}\b")

#: Words a digit legitimately belongs to, so they are not flagged: chemistry
#: formulas, grade/volume labels, exercise numbering, units.
_DIGIT_IN_WORD_ALLOWED = re.compile(
    r"^(?:[A-Z][a-z]?[0-9]+|[A-Za-z][0-9]|x[0-9]|y[0-9]|p[0-9]|bài[0-9]+|"
    r"COVID[0-9]+|[0-9]+[a-z])$"
)

#: Characters that belong in Vietnamese textbook text. Anything else in
#: quantity means the page scanned badly.
_EXPECTED_CHARACTERS = re.compile(
    r"[0-9A-Za-zÀ-ỹĐđ\s.,;:!?()\[\]{}\-–—'\"“”‘’/%+=<>*°·×÷…²³√•≈≤≥|&@#$~^_\\]"
)

#: Above this share of unexpected characters, the page is garbled enough that
#: nothing on it should be trusted.
_GARBLED_THRESHOLD = 0.03

#: A page with fewer characters than this is a picture page or a divider;
#: flagging it says nothing useful.
_MIN_PAGE_LENGTH = 120

#: Signs that a page carries mathematical content worth checking closely.
_FORMULA_MARKERS = re.compile(r"[=²³√×÷]|\bphân s|\bcông thức|\bđịnh lí|\btỉ số")


def _snippet(text: str, start: int, end: int) -> str:
    """Return ``text[start:end]`` with a little context, on one line."""
    left = max(0, start - _SNIPPET_PADDING)
    right = min(len(text), end + _SNIPPET_PADDING)
    return " ".join(text[left:right].split())


def scan_page(text: str) -> list[OcrIssue]:
    """Flag the OCR failure shapes known to produce wrong answers.

    Args:
        text: One page of cached OCR text.

    Returns:
        Every suspicion found, in reading order; empty when the page looks
        clean (which most pages are).
    """
    if len(text.strip()) < _MIN_PAGE_LENGTH:
        return []

    issues: list[OcrIssue] = []

    for match in _SWAPPED_MINUS.finditer(text):
        issues.append(
            OcrIssue(
                kind="dau-nhan-thanh-dau-tru",
                explanation=(
                    "Có dạng 'a - b = b - a' — phép trừ KHÔNG giao hoán, nên gần "
                    "chắc chắn sách in 'a . b = b . a' và máy đọc nhầm dấu nhân."
                ),
                snippet=_snippet(text, match.start(), match.end()),
            )
        )

    for match in _FRACTION_WITHOUT_SLASH.finditer(text):
        issues.append(
            OcrIssue(
                kind="phan-so-mat-gach-ngang",
                explanation=(
                    "Định nghĩa phân số nhưng chỗ đáng lẽ là a/b lại thành dấu '=' "
                    "— máy đọc mất gạch ngang phân số."
                ),
                snippet=_snippet(text, match.start(), match.end()),
            )
        )

    lost_exponents = list(_LOST_EXPONENT.finditer(text))
    if len(lost_exponents) >= 2:
        # One stray "A?" is usually a real question mark; several in a row is
        # a formula whose exponents all vanished.
        issues.append(
            OcrIssue(
                kind="mat-so-mu",
                explanation=(
                    f"Có {len(lost_exponents)} chỗ dạng 'A?' hoặc 'A*' — thường là "
                    "số mũ (A², B³) bị máy đọc mất."
                ),
                snippet=_snippet(text, lost_exponents[0].start(), lost_exponents[-1].end()),
            )
        )

    suspicious_words = [
        match.group(0)
        for match in _DIGIT_IN_WORD.finditer(text)
        if not _DIGIT_IN_WORD_ALLOWED.match(match.group(0))
    ]
    if len(suspicious_words) >= 3:
        issues.append(
            OcrIssue(
                kind="so-lan-vao-chu",
                explanation=(
                    "Nhiều từ bị lẫn chữ số (ví dụ 's6' thay cho 'số') — dấu hiệu "
                    "trang này quét kém."
                ),
                snippet=", ".join(dict.fromkeys(suspicious_words).keys()),
            )
        )

    unexpected = len(_EXPECTED_CHARACTERS.sub("", text))
    if unexpected / len(text) > _GARBLED_THRESHOLD:
        issues.append(
            OcrIssue(
                kind="trang-quet-loi-nang",
                explanation=(
                    f"{unexpected / len(text):.1%} ký tự trên trang không phải chữ "
                    "tiếng Việt, số hay dấu câu — trang gần như không đọc được."
                ),
                snippet=_snippet(text, 0, 80),
            )
        )

    return issues


def _cached_books(cache_root: Path) -> list[tuple[int, Path]]:
    """Find every cached book, as ``(grade, json path)`` pairs."""
    books: list[tuple[int, Path]] = []
    for grade_dir in sorted(cache_root.iterdir()) if cache_root.is_dir() else []:
        if not grade_dir.is_dir() or not grade_dir.name.isdigit():
            continue
        for path in sorted(grade_dir.glob("*.json")):
            books.append((int(grade_dir.name), path))
    return books


def audit_sample(
    *,
    sample_size: int,
    seed: int,
    formulas_only: bool = False,
    grades: set[int] | None = None,
    cache_root: Path = _CACHE_ROOT,
) -> list[PageAudit]:
    """Audit a random sample of cached pages.

    Args:
        sample_size: How many pages to check; ``0`` checks every page, which
            turns the sample into a full census of the corpus.
        seed: Random seed, so a run can be repeated exactly -- a benchmark
            nobody can reproduce is not evidence of anything.
        formulas_only: Restrict the sample to pages carrying a formula,
            fraction or theorem, where OCR damage does the most harm.
        grades: Only audit these grades; every grade when omitted.
        cache_root: Where the OCR cache lives.

    Returns:
        One :class:`PageAudit` per sampled page, in the order sampled.

    Raises:
        RuntimeError: If the cache holds no pages at all.
    """
    population: list[tuple[int, str, int, str]] = []
    for grade, path in _cached_books(cache_root):
        if grades is not None and grade not in grades:
            continue
        try:
            pages = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            logger.warning("Bỏ qua %s: %s", path.name, exc)
            continue
        for page in pages:
            text = page.get("text") or ""
            if len(text.strip()) < _MIN_PAGE_LENGTH:
                continue
            if formulas_only and not _FORMULA_MARKERS.search(text):
                continue
            population.append((grade, path.stem, int(page["page_number"]), text))

    if not population:
        raise RuntimeError(
            f"Không tìm thấy trang nào trong {cache_root} — chạy 'dora-bulk-ingest' trước."
        )

    if sample_size:
        rng = random.Random(seed)
        chosen = rng.sample(population, min(sample_size, len(population)))
    else:
        chosen = population
    return [
        PageAudit(
            grade=grade,
            book=book,
            page=page,
            issues=tuple(scan_page(text)),
            has_formula=bool(_FORMULA_MARKERS.search(text)),
        )
        for grade, book, page, text in chosen
    ]


def format_report(audits: list[PageAudit], *, seed: int) -> str:
    """Render an audit run as the Markdown shortlist a person checks.

    Args:
        audits: Result of :func:`audit_sample`.
        seed: The seed used, printed so the run can be repeated.

    Returns:
        The report, in Vietnamese.
    """
    flagged = [audit for audit in audits if audit.issues]
    by_kind = Counter(issue.kind for audit in audits for issue in audit.issues)

    lines = [
        "# Soát lỗi đọc chữ từ ảnh sách (OCR)",
        "",
        f"Số trang đã soát: {len(audits)} (lấy mẫu ngẫu nhiên, seed = {seed} "
        "— chạy lại cùng seed sẽ ra đúng các trang này)",
        f"Số trang có công thức/phân số: {sum(1 for a in audits if a.has_formula)}",
        f"Số trang bị gắn cờ nghi đọc sai: {len(flagged)}",
        "",
        "## Thống kê theo loại lỗi",
        "",
        "| Loại | Số lần |",
        "|---|---|",
    ]
    for kind, count in by_kind.most_common():
        lines.append(f"| {kind} | {count} |")
    if not by_kind:
        lines.append("| (không phát hiện) | 0 |")

    lines += [
        "",
        "## Các trang cần mở sách đối chiếu",
        "",
    ]
    if not flagged:
        lines.append("Không có trang nào bị gắn cờ trong mẫu này.")
    for audit in flagged:
        lines.append(f"### Lớp {audit.grade} — {audit.book} — trang {audit.page}")
        lines.append("")
        for issue in audit.issues:
            lines += [
                f"- **{issue.kind}**: {issue.explanation}",
                f"  - Máy đọc ra: `{issue.snippet}`",
            ]
        lines.append("")

    lines += [
        "## Cách xử lý",
        "",
        "Mỗi dòng ở trên chỉ là NGHI VẤN. Cách làm: mở đúng trang sách giấy, "
        "đọc câu đó, nếu máy đọc sai thật thì thêm một mục vào "
        "`TEXT_FIXES` trong `src/dora_edu/data_pipeline/manual_patches.py` "
        "(sửa đúng chỗ sai), hoặc vào `PATCHES` (thêm một đoạn sạch), rồi chạy "
        "lại `dora-bulk-ingest` và khởi động lại bot.",
    ]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    """Entry point for ``dora-ocr-audit``.

    Returns:
        ``0`` when the audit ran, ``1`` when there was nothing to audit.
    """
    parser = argparse.ArgumentParser(
        description="Soát ngẫu nhiên chất lượng OCR của kho sách đã xử lí."
    )
    parser.add_argument(
        "--sample", type=int, default=60, help="Số trang lấy mẫu (0 = soát toàn bộ)"
    )
    parser.add_argument("--seed", type=int, default=2026, help="Seed để chạy lại được")
    parser.add_argument(
        "--formulas-only",
        action="store_true",
        help="Chỉ lấy trang có công thức, phân số hoặc định lí",
    )
    parser.add_argument(
        "--grades",
        default="",
        help="Chỉ soát các lớp này, ví dụ: 6,7,8,9 (mặc định: tất cả)",
    )
    parser.add_argument(
        "--out", type=Path, default=None, help="Lưu báo cáo Markdown vào đường dẫn này"
    )
    args = parser.parse_args(argv)

    settings = get_settings()
    logging.basicConfig(
        level=getattr(logging, settings.log_level.upper(), logging.INFO),
        format="%(levelname)-8s %(message)s",
    )

    grades = {int(g) for g in args.grades.split(",") if g.strip()} or None
    try:
        audits = audit_sample(
            sample_size=args.sample,
            seed=args.seed,
            formulas_only=args.formulas_only,
            grades=grades,
        )
    except RuntimeError as exc:
        logger.error("%s", exc)
        return 1

    report = format_report(audits, seed=args.seed)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(report, encoding="utf-8")
        logger.info("Đã lưu báo cáo: %s", args.out)
    else:
        print(report)
    return 0


if __name__ == "__main__":
    sys.exit(main())
