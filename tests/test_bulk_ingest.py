"""Tests for the bulk-ingest CLI's resume behaviour.

Exercises ``main()`` with the indexer and the per-book ingestion step faked
out, so the test is about the *orchestration* (which books get skipped, which
get ingested) rather than real PDF parsing or embedding.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from dora_edu.data_pipeline import bulk_ingest as bulk_ingest_module


class FakeIndexer:
    """Records ``has_source``/``index_chunks`` calls; reports fixed already-indexed books."""

    def __init__(self, settings: Any, already_indexed: set[str]) -> None:
        del settings
        self.already_indexed = already_indexed
        self.lookups: list[tuple[int, str, str]] = []
        self.index_chunks_calls: list[Any] = []

    def has_source(self, grade: int, subject: str, source_file: str) -> bool:
        self.lookups.append((grade, subject, source_file))
        return source_file in self.already_indexed

    def index_chunks(self, chunks: list[Any], metadata: Any) -> int:
        self.index_chunks_calls.append((chunks, metadata))
        return len(chunks)

    def count(self) -> int:
        return 999


@pytest.fixture
def data_root(tmp_path: Path) -> Path:
    grade_dir = tmp_path / "6"
    grade_dir.mkdir()
    (grade_dir / "SGKToan6tapmot.pdf").write_bytes(b"")
    (grade_dir / "SGKToan6taphai.pdf").write_bytes(b"")
    return tmp_path


def _patch_common(monkeypatch, settings, already_indexed: set[str]) -> list[Any]:
    ingested: list[Any] = []

    def fake_ingest_file(
        pdf_path, *, grade, subject, book_title, indexer, chunk_size, chunk_overlap, cache_dir=None
    ):
        del chunk_size, chunk_overlap, cache_dir
        ingested.append((pdf_path.name, grade, subject, book_title, indexer))
        return 7

    monkeypatch.setattr(bulk_ingest_module, "get_settings", lambda: settings)
    monkeypatch.setattr(
        bulk_ingest_module,
        "TextbookIndexer",
        lambda s: FakeIndexer(s, already_indexed),
    )
    monkeypatch.setattr(bulk_ingest_module, "ingest_file", fake_ingest_file)
    return ingested


def test_an_already_indexed_book_is_skipped_by_default(monkeypatch, settings, data_root) -> None:
    ingested = _patch_common(monkeypatch, settings, already_indexed={"SGKToan6tapmot.pdf"})

    exit_code = bulk_ingest_module.main(["--data-root", str(data_root)])

    assert exit_code == 0
    ingested_files = {name for name, *_ in ingested}
    assert ingested_files == {"SGKToan6taphai.pdf"}


def test_force_reingests_even_already_indexed_books(monkeypatch, settings, data_root) -> None:
    ingested = _patch_common(monkeypatch, settings, already_indexed={"SGKToan6tapmot.pdf"})

    exit_code = bulk_ingest_module.main(["--data-root", str(data_root), "--force"])

    assert exit_code == 0
    ingested_files = {name for name, *_ in ingested}
    assert ingested_files == {"SGKToan6tapmot.pdf", "SGKToan6taphai.pdf"}


def test_manual_patches_are_reapplied_after_a_normal_run(monkeypatch, settings, data_root) -> None:
    from dora_edu.data_pipeline.manual_patches import PATCHES

    ingested = _patch_common(monkeypatch, settings, already_indexed=set())

    exit_code = bulk_ingest_module.main(["--data-root", str(data_root)])

    assert exit_code == 0
    indexer = ingested[-1][-1]
    assert len(indexer.index_chunks_calls) == len(PATCHES)


def test_manual_patches_are_not_applied_during_a_dry_run(monkeypatch, settings, data_root) -> None:
    # There is no indexer at all in a dry run -- nothing to patch.
    ingested = _patch_common(monkeypatch, settings, already_indexed=set())

    exit_code = bulk_ingest_module.main(["--data-root", str(data_root), "--dry-run"])

    assert exit_code == 0
    assert all(indexer is None for *_, indexer in ingested)


def test_dry_run_never_checks_or_writes_the_index(monkeypatch, settings, data_root) -> None:
    ingested = _patch_common(monkeypatch, settings, already_indexed={"SGKToan6tapmot.pdf"})

    exit_code = bulk_ingest_module.main(["--data-root", str(data_root), "--dry-run"])

    assert exit_code == 0
    # Both books run through ingest_file (indexer=None inside it short-circuits
    # the actual write), because there is no index to check against in dry-run.
    ingested_files = {name for name, *_ in ingested}
    assert ingested_files == {"SGKToan6tapmot.pdf", "SGKToan6taphai.pdf"}
    assert all(indexer is None for *_, indexer in ingested)
