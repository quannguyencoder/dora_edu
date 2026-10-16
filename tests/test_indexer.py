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


# --- correcting a mis-OCR'd chunk in place ---------------------------------


def _indexer_with(collection: FakeCollection, monkeypatch) -> TextbookIndexer:
    monkeypatch.setattr(indexer_module, "get_collection", lambda *a, **k: collection)
    return TextbookIndexer()


def test_repairing_a_chunk_rewrites_its_text_and_drops_the_stale_copy(monkeypatch) -> None:
    collection = FakeCollection()
    indexer = _indexer_with(collection, monkeypatch)
    metadata = _metadata(grade=8, book_title="Toán 8 - tập 1", source_file="SGKToan8tapmot.pdf")
    indexer.index_chunks([_chunk(55, "thuong gap a-b=b-a la hang dang thuc")], metadata)

    repaired = indexer.repair_chunk_text(
        metadata, chunk_index=55, wrong="a-b=b-a", right="a . b = b . a"
    )

    assert repaired is True
    assert collection.count() == 1
    assert collection.rows[0]["text"] == "thuong gap a . b = b . a la hang dang thuc"
    # The old row is gone, so the false version can never be retrieved again.
    assert len(collection.deleted_ids) == 1


def test_repairing_a_chunk_keeps_its_page_numbers(monkeypatch) -> None:
    collection = FakeCollection()
    indexer = _indexer_with(collection, monkeypatch)
    metadata = _metadata()
    indexer.index_chunks(
        [TextChunk(text="sai roi", chunk_index=7, page_start=30, page_end=31)], metadata
    )

    indexer.repair_chunk_text(metadata, chunk_index=7, wrong="sai roi", right="dung roi")

    row = collection.rows[0]["metadata"]
    assert (row["page_start"], row["page_end"]) == (30, 31)
    assert row["chunk_index"] == 7


def test_repairing_an_already_fixed_chunk_changes_nothing(monkeypatch) -> None:
    # apply_text_fixes re-runs after every ingest, so this must be a no-op.
    collection = FakeCollection()
    indexer = _indexer_with(collection, monkeypatch)
    metadata = _metadata()
    indexer.index_chunks([_chunk(3, "da dung roi")], metadata)

    assert indexer.repair_chunk_text(
        metadata, chunk_index=3, wrong="sai", right="dung"
    ) is False
    assert collection.count() == 1
    assert collection.deleted_ids == []


def test_repairing_a_chunk_that_is_not_indexed_is_reported_not_raised(monkeypatch) -> None:
    collection = FakeCollection()
    indexer = _indexer_with(collection, monkeypatch)

    assert indexer.repair_chunk_text(
        _metadata(), chunk_index=404, wrong="x", right="y"
    ) is False
