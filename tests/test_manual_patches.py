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


# --- pages re-read from their images ---------------------------------------


def test_a_saved_repair_round_trips_through_the_json_file(tmp_path) -> None:
    from dora_edu.data_pipeline.ocr_repair import RepairedPage, load_repairs, save_repairs

    path = tmp_path / "ocr_repairs.json"
    page = RepairedPage(
        grade=8,
        subject="Toán",
        book_title="Toán 8 - tập 2",
        source_file="SGKToan8taphai.pdf",
        page=6,
        text="Phân thức đại số là biểu thức có dạng A/B, trong đó A, B là các đa thức.",
        kinds=("mat-hop-dinh-nghia",),
    )

    save_repairs([page], path)

    assert load_repairs(path) == [page]


def test_re_reading_the_same_page_twice_keeps_only_the_newer_text(tmp_path) -> None:
    from dora_edu.data_pipeline.ocr_repair import RepairedPage, load_repairs, save_repairs

    path = tmp_path / "ocr_repairs.json"

    def _page(text: str) -> RepairedPage:
        return RepairedPage(
            grade=6,
            subject="Toán",
            book_title="Toán 6 - tập 1",
            source_file="SGKToan6tapmot.pdf",
            page=72,
            text=text,
            kinds=("dau-nhan-thanh-dau-tru",),
        )

    save_repairs([_page("bản đọc lần đầu"), _page("bản đọc lần sau")], path)

    saved = load_repairs(path)
    assert len(saved) == 1
    assert saved[0].text == "bản đọc lần sau"


def test_a_missing_repairs_file_is_not_an_error(tmp_path) -> None:
    from dora_edu.data_pipeline.ocr_repair import load_repairs

    assert load_repairs(tmp_path / "khong-co.json") == []


def test_repaired_pages_are_indexed_with_their_own_index_block(tmp_path, settings, monkeypatch) -> None:
    # 800000 for re-read pages, 900000 for hand-written patches, below that
    # the chunker's own: the source of a chunk is readable from its index.
    from dora_edu.data_pipeline.ocr_repair import (
        REPAIR_CHUNK_OFFSET,
        RepairedPage,
        apply_repairs,
        save_repairs,
    )
    from dora_edu.rag_engine import indexer as indexer_module
    from dora_edu.rag_engine.indexer import TextbookIndexer
    from tests.conftest import FakeCollection

    collection = FakeCollection()
    monkeypatch.setattr(indexer_module, "get_collection", lambda *a, **k: collection)
    path = tmp_path / "ocr_repairs.json"
    save_repairs(
        [
            RepairedPage(
                grade=8,
                subject="Toán",
                book_title="Toán 8 - tập 2",
                source_file="SGKToan8taphai.pdf",
                page=6,
                text="Phân thức đại số là biểu thức có dạng A/B.",
                kinds=("mat-hop-dinh-nghia",),
            )
        ],
        path,
    )

    written = apply_repairs(TextbookIndexer(settings), path)

    assert written == 1
    assert collection.rows[0]["metadata"]["chunk_index"] == REPAIR_CHUNK_OFFSET + 6
    assert collection.rows[0]["metadata"]["page_start"] == 6


def test_only_formula_damaging_flags_justify_a_vision_call() -> None:
    # Re-reading a page costs an API call, so "so-lan-vao-chu" -- which
    # mostly flags scanned artwork and running headers -- is left out.
    from dora_edu.data_pipeline.ocr_repair import DEFAULT_KINDS

    assert "mat-so-mu" in DEFAULT_KINDS
    assert "phan-so-mat-gach-ngang" in DEFAULT_KINDS
    assert "so-lan-vao-chu" not in DEFAULT_KINDS
