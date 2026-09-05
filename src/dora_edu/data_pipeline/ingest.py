"""Textbook ingestion CLI (``dora-ingest``).

Orchestrates the admin-side pipeline: parse PDFs, chunk them, and index the
chunks into ChromaDB with the grade/subject metadata that later isolates each
student's retrieval scope.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from dora_edu.config import get_settings
from dora_edu.data_pipeline.pdf_parser import discover_pdfs, parse_pdf
from dora_edu.data_pipeline.text_chunker import chunk_pages
from dora_edu.models import TextbookMetadata, normalize_subject, validate_grade
from dora_edu.rag_engine.indexer import TextbookIndexer

logger = logging.getLogger(__name__)


def build_parser() -> argparse.ArgumentParser:
    """Define the ``dora-ingest`` command line interface."""
    parser = argparse.ArgumentParser(
        prog="dora-ingest",
        description="Index MOET textbook PDFs into the DoraEdu vector database.",
    )
    parser.add_argument(
        "--path",
        required=True,
        type=Path,
        help="A textbook PDF, or a directory searched recursively for PDFs.",
    )
    parser.add_argument(
        "--grade",
        required=True,
        help="Grade level the textbooks belong to (1-12).",
    )
    parser.add_argument(
        "--subject",
        required=True,
        help="Subject the textbooks belong to, e.g. 'Toan' or 'Lich su'.",
    )
    parser.add_argument(
        "--book-title",
        default=None,
        help="Book title stored with every chunk; defaults to each PDF's file name.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Parse and chunk without writing anything to ChromaDB.",
    )
    return parser


def ingest_file(
    pdf_path: Path,
    grade: int,
    subject: str,
    book_title: str | None,
    indexer: TextbookIndexer | None,
    chunk_size: int,
    chunk_overlap: int,
) -> int:
    """Parse, chunk and index one textbook PDF.

    Args:
        pdf_path: The PDF to ingest.
        grade: Validated grade level.
        subject: Canonical subject name.
        book_title: Title stored with each chunk; the file stem when ``None``.
        indexer: Destination index, or ``None`` for a dry run.
        chunk_size: Target chunk length in characters.
        chunk_overlap: Overlap between consecutive chunks.

    Returns:
        The number of chunks produced (and indexed, unless this is a dry run).
    """
    pages = parse_pdf(pdf_path)
    if not pages:
        logger.warning("%s: no extractable text, skipping", pdf_path.name)
        return 0

    chunks = chunk_pages(pages, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    metadata = TextbookMetadata(
        grade=grade,
        subject=subject,
        book_title=book_title or pdf_path.stem,
        source_file=pdf_path.name,
    )

    if indexer is None:
        logger.info("[dry-run] %s -> %d pages, %d chunks", pdf_path.name, len(pages), len(chunks))
        return len(chunks)

    indexer.index_chunks(chunks, metadata)
    return len(chunks)


def main(argv: list[str] | None = None) -> int:
    """Run the ingestion CLI.

    Args:
        argv: Command line arguments; ``sys.argv[1:]`` when omitted.

    Returns:
        ``0`` on success, ``1`` on a fatal error.
    """
    args = build_parser().parse_args(argv)
    settings = get_settings()
    logging.basicConfig(
        level=getattr(logging, settings.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)-8s %(name)s: %(message)s",
    )

    try:
        grade = validate_grade(args.grade)
        subject = normalize_subject(args.subject)
        pdf_paths = discover_pdfs(args.path)
    except (ValueError, FileNotFoundError) as exc:
        logger.error("%s", exc)
        return 1

    if not pdf_paths:
        logger.error("No PDF found at %s", args.path)
        return 1

    logger.info(
        "Ingesting %d PDF(s) as grade=%d, subject=%s%s",
        len(pdf_paths),
        grade,
        subject,
        " [dry-run]" if args.dry_run else "",
    )

    indexer: TextbookIndexer | None = None
    if not args.dry_run:
        try:
            indexer = TextbookIndexer(settings)
        except RuntimeError as exc:
            logger.error("%s", exc)
            return 1

    total_chunks = 0
    failed: list[str] = []
    for pdf_path in pdf_paths:
        try:
            total_chunks += ingest_file(
                pdf_path,
                grade=grade,
                subject=subject,
                book_title=args.book_title,
                indexer=indexer,
                chunk_size=settings.chunk_size,
                chunk_overlap=settings.chunk_overlap,
            )
        except (ValueError, FileNotFoundError, RuntimeError) as exc:
            logger.error("Failed to ingest %s: %s", pdf_path.name, exc)
            failed.append(pdf_path.name)

    logger.info(
        "Done: %d chunks from %d/%d PDF(s)%s",
        total_chunks,
        len(pdf_paths) - len(failed),
        len(pdf_paths),
        f"; failed: {', '.join(failed)}" if failed else "",
    )
    if indexer is not None:
        logger.info("Collection now holds %d chunks", indexer.count())
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
