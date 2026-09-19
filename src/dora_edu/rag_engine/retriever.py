"""Query logic with metadata filtering (Subject/Grade).

Core safety rule: no query ever reaches ChromaDB without both a ``grade`` and a
``subject`` filter. This is what guarantees a 6th-grade student can never be
shown 9th-grade content. The rule is enforced structurally — :meth:`Retriever.retrieve`
only accepts a :class:`~dora_edu.models.StudentProfile`, whose two fields are
both mandatory and validated.

The grade filter is a ceiling, not an exact match: a grade-9 student can still
be shown grade-1..8 content (useful for reviewing earlier material) but never
grade-10+ content. Only the upper bound is safety-critical, so it is the one
enforced everywhere; the lower bound (grade 1) needs no explicit filter.

:meth:`Retriever.retrieve_best_subject` lets a student skip ``/mon`` by trying
every subject indexed for their grade and keeping whichever matched best --
each attempt is still a normal, fully-filtered :meth:`retrieve` call, so the
rule above holds exactly as strictly as when the subject is chosen by hand.

:meth:`Retriever.retrieve_best_subject` lets a student skip ``/mon`` by trying
every subject indexed for their grade and keeping whichever matched best --
each attempt is still a normal, fully-filtered :meth:`retrieve` call, so the
rule above holds exactly as strictly as when the subject is chosen by hand.
"""

from __future__ import annotations

import logging
from typing import Any

import chromadb

from dora_edu.config import Settings, get_settings
from dora_edu.models import MIN_GRADE, RetrievedChunk, StudentProfile
from dora_edu.rag_engine.store import get_collection

logger = logging.getLogger(__name__)


def build_metadata_filter(profile: StudentProfile) -> dict[str, Any]:
    """Build the mandatory ChromaDB ``where`` clause for one student.

    Grade is a ceiling (``<=``), not an exact match: a student may review
    material from any grade at or below their own, but never from a higher
    one -- that upper bound is the safety-critical part of grade isolation.

    Args:
        profile: The student's validated grade and subject.

    Returns:
        A ChromaDB filter matching chunks of that subject, from ``profile``'s
        grade or any grade below it.
    """
    return {
        "$and": [
            {"grade": {"$lte": profile.grade}},
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
        self._subjects_by_grade: dict[int, list[str]] = {}

    def list_subjects(self, grade: int) -> list[str]:
        """Return every distinct subject indexed at or below ``grade``.

        Cached for the lifetime of this ``Retriever``, since the corpus does
        not change while the bot is running.

        Args:
            grade: The student's grade; subjects from this grade and any
                grade below it are included (mirrors :func:`build_metadata_filter`).

        Returns:
            Canonical subject names with at least one indexed chunk at or
            below ``grade``, sorted alphabetically.

        Raises:
            RuntimeError: If the ChromaDB lookup fails.
        """
        if grade not in self._subjects_by_grade:
            subjects: set[str] = set()
            # One query per grade (each `$eq`, not one big `$lte`) rather than
            # a single query spanning every grade at or below `grade`: at
            # grade 12 a $lte query has to enumerate metadata for nearly the
            # whole corpus (~40k chunks), which blows past SQLite's bound
            # parameter limit ("too many SQL variables") -- confirmed live.
            # Every individual grade's chunk count stays well within that
            # limit, and the result is cached per grade anyway.
            for g in range(MIN_GRADE, grade + 1):
                try:
                    result = self._collection.get(
                        where={"grade": {"$eq": g}}, include=["metadatas"]
                    )
                except (chromadb.errors.ChromaError, ValueError, RuntimeError) as exc:
                    raise RuntimeError(f"ChromaDB metadata lookup failed: {exc}") from exc
                subjects.update(
                    m["subject"] for m in result.get("metadatas") or [] if m.get("subject")
                )
            self._subjects_by_grade[grade] = sorted(subjects)
        return self._subjects_by_grade[grade]

    def retrieve_best_subject(
        self, question: str, grade: int, *, top_k: int | None = None
    ) -> tuple[list[RetrievedChunk], str | None]:
        """Find which subject of ``grade`` best matches ``question``, and its passages.

        Tries every subject indexed for ``grade`` in turn -- each attempt goes
        through :meth:`retrieve`, so every single ChromaDB query stays filtered
        by both grade and subject; nothing is ever searched unscoped.

        Args:
            question: The student's question, in Vietnamese.
            grade: The student's grade; the subject is not yet known.
            top_k: Passages to fetch per subject tried; falls back to the
                configured default.

        Returns:
            The best-matching subject's passages (nearest first) and its
            name; an empty list and ``None`` when no subject of this grade
            has anything close enough to ``question``.

        Raises:
            ValueError: If ``question`` is empty.
            RuntimeError: If a ChromaDB query fails.
        """
        best_chunks: list[RetrievedChunk] = []
        best_subject: str | None = None
        best_distance = float("inf")
        for subject in self.list_subjects(grade):
            profile = StudentProfile(grade=grade, subject=subject)
            chunks = self.retrieve(question, profile, top_k=top_k)
            if chunks and chunks[0].distance < best_distance:
                best_distance = chunks[0].distance
                best_subject = subject
                best_chunks = chunks

        logger.info(
            "Auto-detected subject=%s for grade=%d (%d candidate subjects tried)",
            best_subject,
            grade,
            len(self._subjects_by_grade.get(grade, [])),
        )
        return best_chunks, best_subject

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
