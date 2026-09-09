"""Tests for embedding/upsert logic and the bulk-ingest resume check."""

from __future__ import annotations

from dora_edu.models import TextbookMetadata, TextChunk
from dora_edu.rag_engine import indexer as indexer_module
from dora_edu.rag_engine.indexer import TextbookIndexer, build_chunk_id
from tests.conftest import FakeCollection


def _metadata(**overrides: object) -> TextbookMetadata:
    fields = {
        "grade": 6,
        "subject": "Toán",
        "book_title": "Toán 6 - tập 1",
        "source_file": "SGKToan6tapmot.pdf",
    }
    fields.update(overrides)
    return TextbookMetadata(**fields)  # type: ignore[arg-type]


def _chunk(index: int = 0, text: str = "Noi dung chunk") -> TextChunk:
    return TextChunk(text=text, chunk_index=index, page_start=1, page_end=1)


def _indexer(monkeypatch, settings) -> tuple[TextbookIndexer, FakeCollection]:
    collection = FakeCollection()
    monkeypatch.setattr(indexer_module, "get_collection", lambda *a, **k: collection)
    return TextbookIndexer(settings), collection


def test_index_chunks_writes_grade_and_subject_on_every_row(monkeypatch, settings) -> None:
    indexer, collection = _indexer(monkeypatch, settings)
    metadata = _metadata()

    written = indexer.index_chunks([_chunk(0), _chunk(1)], metadata)

    assert written == 2
    assert collection.count() == 2
    assert all(row["metadata"]["grade"] == 6 for row in collection.rows)
    assert all(row["metadata"]["subject"] == "Toán" for row in collection.rows)


def test_index_chunks_on_no_chunks_writes_nothing(monkeypatch, settings) -> None:
    indexer, collection = _indexer(monkeypatch, settings)

    assert indexer.index_chunks([], _metadata()) == 0
    assert collection.count() == 0


def test_reingesting_the_same_book_upserts_instead_of_duplicating(monkeypatch, settings) -> None:
    indexer, collection = _indexer(monkeypatch, settings)
    metadata = _metadata()
    chunks = [_chunk(0), _chunk(1)]

    indexer.index_chunks(chunks, metadata)
    indexer.index_chunks(chunks, metadata)

    assert collection.count() == 2


def test_build_chunk_id_ignores_book_title(monkeypatch, settings) -> None:
    # Re-running bulk ingestion with a nicer book_title (from catalog.py)
    # must overwrite the same row, not create a duplicate under a new id.
    chunk = _chunk(0)
    id_a = build_chunk_id(_metadata(book_title="SGKToan6tapmot"), chunk)
    id_b = build_chunk_id(_metadata(book_title="Toán 6 - tập 1"), chunk)
    assert id_a == id_b


def test_has_source_is_false_before_indexing(monkeypatch, settings) -> None:
    indexer, _ = _indexer(monkeypatch, settings)
    assert indexer.has_source(6, "Toán", "SGKToan6tapmot.pdf") is False


def test_has_source_is_true_after_indexing(monkeypatch, settings) -> None:
    indexer, _ = _indexer(monkeypatch, settings)
    indexer.index_chunks([_chunk(0)], _metadata())

    assert indexer.has_source(6, "Toán", "SGKToan6tapmot.pdf") is True


def test_has_source_is_scoped_to_the_exact_grade_subject_and_file(monkeypatch, settings) -> None:
    indexer, _ = _indexer(monkeypatch, settings)
    indexer.index_chunks([_chunk(0)], _metadata())

    assert indexer.has_source(7, "Toán", "SGKToan6tapmot.pdf") is False
    assert indexer.has_source(6, "Ngữ văn", "SGKToan6tapmot.pdf") is False
    assert indexer.has_source(6, "Toán", "SGKToan6taphai.pdf") is False
