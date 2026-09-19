"""Tests for the re-appliable manual retrieval-quality patches."""

from __future__ import annotations

from dora_edu.data_pipeline.manual_patches import PATCHES, apply_patches
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
