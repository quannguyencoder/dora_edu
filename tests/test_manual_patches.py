"""Tests for the re-appliable manual retrieval-quality patches."""

from __future__ import annotations

from dora_edu.data_pipeline import manual_patches
from dora_edu.data_pipeline.manual_patches import PATCHES, apply_patches
from dora_edu.models import TextbookMetadata, TextChunk
from dora_edu.rag_engine import indexer as indexer_module
from dora_edu.rag_engine.indexer import TextbookIndexer
from tests.conftest import FakeCollection


def test_apply_patches_writes_every_known_patch(monkeypatch, settings) -> None:
    collection = FakeCollection()
    monkeypatch.setattr(indexer_module, "get_collection", lambda *a, **k: collection)
    indexer = TextbookIndexer(settings)

    written = apply_patches(indexer)

    assert written == len(PATCHES)
    assert collection.count() == len(PATCHES)


def test_applying_patches_twice_does_not_duplicate_them(monkeypatch, settings) -> None:
    collection = FakeCollection()
    monkeypatch.setattr(indexer_module, "get_collection", lambda *a, **k: collection)
    indexer = TextbookIndexer(settings)

    apply_patches(indexer)
    apply_patches(indexer)

    assert collection.count() == len(PATCHES)


def test_every_patch_is_scoped_to_its_own_grade_and_subject() -> None:
    for patch in PATCHES:
        assert 1 <= patch["grade"] <= 12
        assert patch["subject"]
        assert patch["text"]


def test_every_known_ocr_correction_is_applied(monkeypatch) -> None:
    """Each TEXT_FIXES entry must actually land on a chunk, once applied."""
    collection = FakeCollection()
    monkeypatch.setattr(indexer_module, "get_collection", lambda *a, **k: collection)
    indexer = TextbookIndexer()
    for fix in manual_patches.TEXT_FIXES:
        metadata = TextbookMetadata(
            grade=fix["grade"],
            subject=fix["subject"],
            book_title=fix["book_title"],
            source_file=fix["source_file"],
        )
        indexer.index_chunks(
            [
                TextChunk(
                    text=f"truoc {fix['wrong']} sau",
                    chunk_index=fix["chunk_index"],
                    page_start=1,
                    page_end=1,
                )
            ],
            metadata,
        )

    assert manual_patches.apply_text_fixes(indexer) == len(manual_patches.TEXT_FIXES)
    # Re-running after an ingest must not touch anything a second time.
    assert manual_patches.apply_text_fixes(indexer) == 0
    for fix in manual_patches.TEXT_FIXES:
        assert any(fix["right"] in row["text"] for row in collection.rows)
        assert not any(fix["wrong"] in row["text"] for row in collection.rows)


def test_the_commutative_law_is_never_stored_as_a_minus_sign() -> None:
    """Regression: a grade-8 student was told "a - b = b - a", which is false."""
    fix = next(
        f for f in manual_patches.TEXT_FIXES if f["source_file"] == "SGKToan8tapmot.pdf"
    )
    assert "a-b=b-a" in fix["wrong"]
    assert "a . b = b . a" in fix["right"]
    assert "b - a" not in fix["right"]
