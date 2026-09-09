"""Infer (grade, subject, book title) from real MOET filenames.

The textbook corpus in ``data/raw_pdfs/<grade>/<file>.pdf`` was collected out
of band (not by this project's code) and follows the MOET publisher's own
naming convention, e.g. ``SGKToan9tapmot.pdf`` or
``SGKChuyendehoctapDialiI10.pdf``. This module turns that convention into the
same ``(grade, subject, book_title)`` triple a human admin would type by hand
on the single-book ``dora-ingest`` CLI, so :mod:`bulk_ingest` can drive
:func:`dora_edu.data_pipeline.ingest.ingest_file` for the whole corpus without
duplicating any parsing, chunking or indexing logic.

Only the *subject* and *grade* feed the zero-hallucination isolation rule and
are validated against :mod:`dora_edu.models`; the book title is informational
only (shown to students as a citation) and best-effort readable.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from dora_edu.models import normalize_subject, validate_grade

#: The publisher's constant filename prefix, e.g. ``SGKToan9tapmot``.
_SGK_PREFIX = "SGK"

#: Marks a grade 10-12 elective/supplement book, e.g. ``SGKChuyendehoctapToan10``.
#: These stay under the *same* subject/grade as the core textbook (so a
#: student's retrieval scope picks up both) and are only distinguished by
#: their book title.
_CHUYEN_DE_PREFIX = "Chuyendehoctap"

#: Subject letters, then the grade digits, then an optional free-form suffix
#: (volume marker, elective module name, publisher series name, ...).
_FILENAME_RE = re.compile(r"^([A-Za-z]+?)(\d+)(.*)$")

#: Recognised multi-volume markers, normalised to a Vietnamese label.
_VOLUME_LABELS: dict[str, str] = {
    "tapmot": "tập 1",
    "taphai": "tập 2",
    "tap1": "tập 1",
    "tap2": "tập 2",
}


@dataclass(frozen=True)
class BookInfo:
    """The metadata inferred for one textbook file."""

    grade: int
    subject: str
    book_title: str


def _split_camel_case(text: str) -> str:
    """Insert spaces before inner capitals, e.g. ``ModunChebien`` -> ``Modun Chebien``."""
    return re.sub(r"(?<!^)(?=[A-Z])", " ", text).strip()


def _describe_suffix(suffix: str) -> str:
    """Turn a filename suffix into a short human-readable descriptor.

    Args:
        suffix: Everything after the grade digits, e.g. ``"taphai"`` or
            ``"ModunChebienthucpham"``. May be empty.

    Returns:
        A Vietnamese volume label for a known marker, a best-effort
        space-split phrase otherwise, or ``""`` when there is no suffix.
    """
    if not suffix:
        return ""
    label = _VOLUME_LABELS.get(suffix.lower())
    if label is not None:
        return label
    return _split_camel_case(suffix)


def parse_filename(stem: str, *, folder_grade: int) -> BookInfo:
    """Infer grade/subject/title for one textbook from its filename.

    Args:
        stem: The filename without directory or ``.pdf`` extension, e.g.
            ``"SGKToan9tapmot"``.
        folder_grade: The grade implied by the file's parent directory
            (``data/raw_pdfs/<grade>/``), used both as the source of truth
            for ``grade`` and as a cross-check against the digits encoded in
            the filename itself.

    Returns:
        The inferred :class:`BookInfo`.

    Raises:
        ValueError: If the filename does not follow the expected
            ``SGK<Subject><Grade>[Suffix]`` shape, or the grade encoded in
            the filename disagrees with ``folder_grade`` -- both are treated
            as a data anomaly that needs a human to look at the file rather
            than a silent guess.
    """
    grade = validate_grade(folder_grade)

    body = stem[len(_SGK_PREFIX):] if stem.startswith(_SGK_PREFIX) else stem
    is_chuyen_de = body.startswith(_CHUYEN_DE_PREFIX)
    if is_chuyen_de:
        body = body[len(_CHUYEN_DE_PREFIX):]

    match = _FILENAME_RE.match(body)
    if not match:
        raise ValueError(f"filename does not match the SGK<Subject><Grade> pattern: {stem!r}")

    subject_raw, grade_str, suffix = match.groups()
    filename_grade = int(grade_str)
    if filename_grade != grade:
        raise ValueError(
            f"{stem!r}: grade in filename ({filename_grade}) does not match "
            f"its folder (data/raw_pdfs/{grade}/)"
        )

    subject = normalize_subject(subject_raw)
    descriptor = _describe_suffix(suffix)

    title = f"Chuyên đề học tập {subject} {grade}" if is_chuyen_de else f"{subject} {grade}"
    if descriptor:
        title = f"{title} - {descriptor}"

    return BookInfo(grade=grade, subject=subject, book_title=title)


def infer_book_info(pdf_path: Path) -> BookInfo:
    """Infer :class:`BookInfo` for a file under ``data/raw_pdfs/<grade>/<name>.pdf``.

    Args:
        pdf_path: Path to the PDF; its parent directory name must be the
            MOET grade (1-12).

    Returns:
        The inferred :class:`BookInfo`.

    Raises:
        ValueError: If the parent directory is not a valid grade, or the
            filename cannot be parsed (see :func:`parse_filename`).
    """
    folder_grade = validate_grade(pdf_path.parent.name)
    return parse_filename(pdf_path.stem, folder_grade=folder_grade)
