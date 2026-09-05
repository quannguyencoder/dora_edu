"""Query logic with metadata filtering (Subject/Grade).

Core safety rule: no query ever reaches ChromaDB without both a ``grade`` and a
``subject`` filter. This is what guarantees a 6th-grade student can never be
shown 9th-grade content. The rule is enforced structurally — :meth:`Retriever.retrieve`
only accepts a :class:`~dora_edu.models.StudentProfile`, whose two fields are
both mandatory and validated.
"""

from __future__ import annotations

import logging
from typing import Any

import chromadb

from dora_edu.config import Settings, get_settings
from dora_edu.models import RetrievedChunk, StudentProfile
from dora_edu.rag_engine.store import get_collection

logger = logging.getLogger(__name__)


def build_metadata_filter(profile: StudentProfile) -> dict[str, Any]:
    """Build the mandatory ChromaDB ``where`` clause for one student.

    Args:
        profile: The student's validated grade and subject.

    Returns:
        A ChromaDB filter matching only chunks of that exact grade and subject.
    """
    return {
        "$and": [
            {"grade": {"$eq": profile.grade}},
            {"subject": {"$eq": profile.subject}},
        ]
    }


class Retriever:
    """Retrieves textbook passages scoped to a single grade and subject."""

    def __init__(self, settings: Settings | None = None) -> None:
        """Open the textbook collection for reading.

        Raises:
            RuntimeError: If the collection does not exist yet (nothing ingested).
        """
        self._settings = settings or get_settings()
        self._collection = get_collection(self._settings, create=False)

    def retrieve(
        self,
        question: str,
        profile: StudentProfile,
        *,
        top_k: int | None = None,
    ) -> list[RetrievedChunk]:
        """Find the passages most relevant to ``question`` within the student's scope.

        Args:
            question: The student's question, in Vietnamese.
            profile: Grade and subject; both are mandatory and always applied as
                a metadata filter.
            top_k: Number of passages to fetch; falls back to the configured
                default.

        Returns:
            Passages closer than ``max_retrieval_distance``, nearest first. An
            empty list means the textbook has no relevant content, which the
            generator turns into the "not in your textbook" answer.

        Raises:
            ValueError: If ``question`` is empty.
            RuntimeError: If the ChromaDB query fails.
        """
        cleaned_question = question.strip()
        if not cleaned_question:
            raise ValueError("question must not be empty")

        limit = top_k or self._settings.retrieval_top_k
        where = build_metadata_filter(profile)

        try:
            response = self._collection.query(
                query_texts=[cleaned_question],
                n_results=limit,
                where=where,
                include=["documents", "metadatas", "distances"],
            )
        except (chromadb.errors.ChromaError, ValueError, RuntimeError) as exc:
            raise RuntimeError(f"ChromaDB query failed: {exc}") from exc

        chunks = self._parse_response(response)
        kept = [c for c in chunks if c.distance <= self._settings.max_retrieval_distance]

        logger.info(
            "Retrieved %d/%d passages (grade=%d, subject=%s)",
            len(kept),
            len(chunks),
            profile.grade,
            profile.subject,
        )
        return kept

    @staticmethod
    def _parse_response(response: dict[str, Any]) -> list[RetrievedChunk]:
        """Flatten a single-query ChromaDB response into chunk models."""
        documents = (response.get("documents") or [[]])[0]
        metadatas = (response.get("metadatas") or [[]])[0]
        distances = (response.get("distances") or [[]])[0]

        chunks: list[RetrievedChunk] = []
        for document, metadata, distance in zip(documents, metadatas, distances):
            if not document:
                continue
            chunks.append(
                RetrievedChunk(
                    text=document,
                    distance=float(distance),
                    metadata=dict(metadata or {}),
                )
            )
        return chunks
