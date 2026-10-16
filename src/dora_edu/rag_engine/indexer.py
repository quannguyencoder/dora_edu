"""Embedding and ChromaDB insertion."""

from __future__ import annotations

import hashlib
import logging

import chromadb

from dora_edu.config import Settings, get_settings
from dora_edu.models import TextbookMetadata, TextChunk
from dora_edu.rag_engine.store import get_collection

logger = logging.getLogger(__name__)

#: ChromaDB rejects very large single writes; ingest in batches of this size.
_BATCH_SIZE = 128


def build_chunk_id(metadata: TextbookMetadata, chunk: TextChunk) -> str:
    """Derive a stable, collision-resistant id for one chunk.

    The id is a hash of the book identity plus the chunk position and text, so
    re-ingesting the same PDF upserts existing rows instead of duplicating them.

    Args:
        metadata: Provenance of the book the chunk came from.
        chunk: The chunk itself.

    Returns:
        A hexadecimal id, unique per (book, position, content).
    """
    fingerprint = "|".join(
        [
            str(metadata.grade),
            metadata.subject,
            metadata.source_file,
            str(chunk.chunk_index),
            chunk.text,
        ]
    )
    return hashlib.sha256(fingerprint.encode("utf-8")).hexdigest()


class TextbookIndexer:
    """Writes textbook chunks into ChromaDB with their grade/subject metadata."""

    def __init__(self, settings: Settings | None = None) -> None:
        """Open the persistent collection used for ingestion."""
        self._settings = settings or get_settings()
        self._collection = get_collection(self._settings, create=True)

    def repair_chunk_text(
        self,
        metadata: TextbookMetadata,
        *,
        chunk_index: int,
        wrong: str,
        right: str,
    ) -> bool:
        """Correct a mis-OCR'd passage inside one already-indexed chunk.

        Adding a clean chunk next to a corrupted one (what
        :data:`~dora_edu.data_pipeline.manual_patches.PATCHES` does) is
        enough when the original merely ranks badly. It is *not* enough when
        the original says something false: the bad chunk stays retrievable,
        and the LLM may quote it. This rewrites it in place instead.

        :func:`build_chunk_id` hashes the chunk's text, so the corrected
        chunk gets a new id -- it is written first and the stale row deleted
        afterwards, so a failure in between leaves the chunk duplicated
        rather than missing.

        Args:
            metadata: Provenance of the book the chunk came from.
            chunk_index: Position of the chunk within that book.
            wrong: The exact mis-read substring to replace.
            right: What the page actually says.

        Returns:
            ``True`` when a chunk was corrected, ``False`` when none matched
            or it no longer contains ``wrong`` (already fixed) -- so this is
            safe to re-run after every ingest.
        """
        try:
            found = self._collection.get(
                where={
                    "$and": [
                        {"source_file": {"$eq": metadata.source_file}},
                        {"chunk_index": {"$eq": chunk_index}},
                    ]
                }
            )
        except (chromadb.errors.ChromaError, ValueError) as exc:
            raise RuntimeError(
                f"Failed to look up chunk {chunk_index} of {metadata.source_file}: {exc}"
            ) from exc

        ids = found.get("ids") or []
        documents = found.get("documents") or []
        metadatas = found.get("metadatas") or []
        if not ids or not documents:
            logger.warning(
                "No chunk %d in %s to repair", chunk_index, metadata.source_file
            )
            return False

        stale_id, old_text, row = ids[0], documents[0], metadatas[0]
        if wrong not in old_text:
            logger.debug(
                "Chunk %d of %s needs no repair", chunk_index, metadata.source_file
            )
            return False

        corrected = TextChunk(
            text=old_text.replace(wrong, right),
            chunk_index=chunk_index,
            page_start=row["page_start"],
            page_end=row["page_end"],
        )
        self.index_chunks([corrected], metadata)
        try:
            self._collection.delete(ids=[stale_id])
        except (chromadb.errors.ChromaError, ValueError) as exc:
            raise RuntimeError(
                f"Corrected chunk {chunk_index} of {metadata.source_file} was written "
                f"but the stale copy could not be deleted: {exc}"
            ) from exc
        logger.info(
            "Repaired chunk %d of %s (%r -> %r)",
            chunk_index,
            metadata.source_file,
            wrong,
            right,
        )
        return True

    def index_chunks(self, chunks: list[TextChunk], metadata: TextbookMetadata) -> int:
        """Embed and upsert ``chunks`` under the given book metadata.

        Every stored row carries ``grade`` and ``subject``, which is what the
        retriever later filters on to isolate grade levels and subjects.

        Args:
            chunks: Chunks produced by the text chunker.
            metadata: Provenance shared by all the chunks.

        Returns:
            The number of chunks written.

        Raises:
            RuntimeError: If ChromaDB rejects the write.
        """
        if not chunks:
            logger.warning("No chunks to index for %s", metadata.source_file)
            return 0

        written = 0
        for start in range(0, len(chunks), _BATCH_SIZE):
            batch = chunks[start : start + _BATCH_SIZE]
            try:
                self._collection.upsert(
                    ids=[build_chunk_id(metadata, chunk) for chunk in batch],
                    documents=[chunk.text for chunk in batch],
                    metadatas=[
                        metadata.as_chroma_metadata(
                            page_start=chunk.page_start,
                            page_end=chunk.page_end,
                            chunk_index=chunk.chunk_index,
                        )
                        for chunk in batch
                    ],
                )
            except (chromadb.errors.ChromaError, ValueError, RuntimeError) as exc:
                raise RuntimeError(
                    f"Failed to index chunks {start}-{start + len(batch)} "
                    f"of {metadata.source_file}: {exc}"
                ) from exc
            written += len(batch)
            logger.debug("Indexed %d/%d chunks of %s", written, len(chunks), metadata.source_file)

        logger.info(
            "Indexed %d chunks from %s (grade=%d, subject=%s)",
            written,
            metadata.source_file,
            metadata.grade,
            metadata.subject,
        )
        return written

    def count(self) -> int:
        """Return the total number of chunks currently stored."""
        return self._collection.count()

    def has_source(self, grade: int, subject: str, source_file: str) -> bool:
        """Check whether any chunk from this exact book is already indexed.

        Lets a bulk ingestion run be safely resumed after an interruption
        (e.g. a scheduled stop) without re-running OCR and embedding on
        books that were already indexed.

        Args:
            grade: Grade the book was indexed under.
            subject: Canonical subject the book was indexed under.
            source_file: The PDF's file name, as stored on every chunk's metadata.

        Returns:
            ``True`` if at least one chunk from this exact (grade, subject,
            source_file) is already present.

        Raises:
            RuntimeError: If ChromaDB rejects the lookup.
        """
        where = {
            "$and": [
                {"grade": {"$eq": grade}},
                {"subject": {"$eq": subject}},
                {"source_file": {"$eq": source_file}},
            ]
        }
        try:
            result = self._collection.get(where=where, limit=1)
        except (chromadb.errors.ChromaError, ValueError, RuntimeError) as exc:
            raise RuntimeError(f"Failed to check existing chunks for {source_file}: {exc}") from exc
        return bool(result.get("ids"))
