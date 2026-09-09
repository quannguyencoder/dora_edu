"""Bulk ingestion CLI (``dora-bulk-ingest``) for the whole local textbook corpus.

Walks ``data/raw_pdfs/<grade>/*.pdf``, infers each book's grade/subject/title
via :mod:`dora_edu.data_pipeline.catalog`, and drives the same
:func:`dora_edu.data_pipeline.ingest.ingest_file` used by the single-book
``dora-ingest`` CLI -- so the batch path can never disagree with the
single-book path on chunking or indexing behaviour.

A file whose name cannot be parsed, or whose filename-encoded grade disagrees
with its folder, is reported and skipped rather than guessed at: grade and
subject are the load-bearing fields for the zero-hallucination isolation
rule, so a wrong guess here is worse than an admin having to look at one file.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from dora_edu.config import get_settings
from dora_edu.data_pipeline.catalog import infer_book_info
from dora_edu.data_pipeline.ingest import ingest_file
from dora_edu.rag_engine.indexer import TextbookIndexer

logger = logging.getLogger(__name__)


def build_parser() -> argparse.ArgumentParser:
    """Define the ``dora-bulk-ingest`` command line interface."""
    parser = argparse.ArgumentParser(
        prog="dora-bulk-ingest",
        description=(
            "Index every PDF under data/raw_pdfs/<grade>/ into the DoraEdu "
            "vector database, inferring grade/subject from each filename."
        ),
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("data/raw_pdfs"),
        help="Root directory holding one subdirectory per grade (default: data/raw_pdfs).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only parse filenames and chunk PDFs; write nothing to ChromaDB.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help=(
            "Re-run OCR/chunking/embedding even for books already indexed. "
            "By default, a book already present under its (grade, subject, "
            "file name) is skipped, so an interrupted run can resume cheaply."
        ),
    )
    return parser


def discover_catalog(data_root: Path) -> tuple[list[Path], list[tuple[Path, str]]]:
    """Pair every PDF under ``data_root`` with its inferred metadata, or a parse error.

    Args:
        data_root: Directory holding one subdirectory per grade (1-12).

    Returns:
        A ``(parsed, failed)`` pair: ``parsed`` is the list of PDF paths whose
        metadata was inferred successfully (grade/subject already validated);
        ``failed`` pairs each unparseable path with the reason it was skipped.
    """
    parsed: list[Path] = []
    failed: list[tuple[Path, str]] = []
    for pdf_path in sorted(data_root.glob("*/*.pdf")):
        try:
            infer_book_info(pdf_path)
        except ValueError as exc:
            failed.append((pdf_path, str(exc)))
            continue
        parsed.append(pdf_path)
    return parsed, failed


def main(argv: list[str] | None = None) -> int:
    """Run the bulk ingestion CLI.

    Args:
        argv: Command line arguments; ``sys.argv[1:]`` when omitted.

    Returns:
        ``0`` on success (including a dry run with only warnings), ``1`` if
        any book failed to parse or to index.
    """
    args = build_parser().parse_args(argv)
    settings = get_settings()
    logging.basicConfig(
        level=getattr(logging, settings.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)-8s %(name)s: %(message)s",
    )

    if not args.data_root.is_dir():
        logger.error("No such directory: %s", args.data_root)
        return 1

    pdf_paths, failed = discover_catalog(args.data_root)
    for path, reason in failed:
        logger.warning("Skipping %s: %s", path, reason)

    logger.info(
        "Found %d parseable PDF(s) under %s%s (%d skipped)",
        len(pdf_paths),
        args.data_root,
        " [dry-run]" if args.dry_run else "",
        len(failed),
    )

    indexer: TextbookIndexer | None = None
    if not args.dry_run:
        try:
            indexer = TextbookIndexer(settings)
        except RuntimeError as exc:
            logger.error("%s", exc)
            return 1

    total_chunks = 0
    skipped_already_indexed = 0
    ingest_failed: list[str] = []
    for i, pdf_path in enumerate(pdf_paths, start=1):
        info = infer_book_info(pdf_path)

        if indexer is not None and not args.force and indexer.has_source(
            info.grade, info.subject, pdf_path.name
        ):
            skipped_already_indexed += 1
            logger.info(
                "[%d/%d] Already indexed, skipping: %s (grade=%d, subject=%s)",
                i,
                len(pdf_paths),
                pdf_path.name,
                info.grade,
                info.subject,
            )
            continue

        try:
            n_chunks = ingest_file(
                pdf_path,
                grade=info.grade,
                subject=info.subject,
                book_title=info.book_title,
                indexer=indexer,
                chunk_size=settings.chunk_size,
                chunk_overlap=settings.chunk_overlap,
            )
        except (ValueError, FileNotFoundError, RuntimeError) as exc:
            logger.error("[%d/%d] Failed %s: %s", i, len(pdf_paths), pdf_path.name, exc)
            ingest_failed.append(pdf_path.name)
            continue
        total_chunks += n_chunks
        logger.info(
            "[%d/%d] grade=%d subject=%s -> %s (%d chunks): %s",
            i,
            len(pdf_paths),
            info.grade,
            info.subject,
            pdf_path.name,
            n_chunks,
            info.book_title,
        )

    logger.info(
        "Done: %d new chunks; %d/%d book(s) newly indexed, %d already indexed, "
        "%d unparseable filename(s), %d failed",
        total_chunks,
        len(pdf_paths) - len(ingest_failed) - skipped_already_indexed,
        len(pdf_paths),
        skipped_already_indexed,
        len(failed),
        len(ingest_failed),
    )
    if indexer is not None:
        logger.info("Collection now holds %d chunks", indexer.count())
    return 1 if (failed or ingest_failed) else 0


if __name__ == "__main__":
    sys.exit(main())
