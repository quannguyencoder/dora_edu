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
import re
from typing import Any

import chromadb

from dora_edu.config import Settings, get_settings
from dora_edu.models import MIN_GRADE, RetrievedChunk, StudentProfile
from dora_edu.rag_engine.store import get_collection

logger = logging.getLogger(__name__)

#: Ways a Vietnamese student asks for a definition, stripped to leave the
#: term itself: "Thế nào là tam giác cân?" -> "tam giác cân".
_QUESTION_PREFIXES = (
    "thế nào là",
    "thế nào gọi là",
    "em hiểu thế nào về",
    "cho biết",
    "hãy cho biết",
    "hãy nêu",
    "nêu khái niệm",
    "khái niệm",
    "định nghĩa",
    "giải thích",
    "trình bày",
    "bạn hãy nói về",
    "nói về",
)

_QUESTION_SUFFIXES = (
    "là gì",
    "nghĩa là gì",
    "có nghĩa là gì",
    "được hiểu là gì",
    "phát biểu thế nào",
    "phát biểu như thế nào",
    "là như thế nào",
    "như thế nào",
    "cho biết điều gì",
    "gồm những gì",
    "gồm những hạt nào",
    "có tính chất gì",
    "có đặc điểm gì",
    "có vai trò gì",
    "dùng để làm gì",
    "khác nhau thế nào",
    "khác nhau như thế nào",
)

#: Leading/trailing punctuation and filler a student types around the term.
_EDGE_NOISE = re.compile(r"^[\s,.:;?!\"'-]+|[\s,.:;?!\"'-]+$")

#: A phrase shorter than this matches far too much text to be a useful
#: filter ("ta", "số"), and one much longer is a whole sentence rather than
#: a term -- a substring filter would then match nothing at all.
_MIN_KEYWORD_LENGTH = 4
_MAX_KEYWORD_LENGTH = 40


def keyword_phrase(question: str) -> str | None:
    """Extract the term a definition question is about, for full-text search.

    Vector similarity alone is unreliable on short Vietnamese terms whose
    words are common elsewhere: "tam giác **cân**" embeds close to grade-2
    pages about weighing things in kilograms ("cân nặng"), which in a real
    run pushed the actual tam-giác-cân pages out of the top 10 entirely and
    made the bot refuse a question its textbook answers on two pages. A
    literal substring match on the term does not have that failure mode, so
    :meth:`Retriever.retrieve` runs one as well and puts its hits first.

    Args:
        question: The student's question, in Vietnamese.

    Returns:
        The lower-cased term to search for literally, or ``None`` when the
        question carries no usable term (too short, too long, or not a
        definition-shaped question).
    """
    phrase = _EDGE_NOISE.sub("", question.lower())
    for prefix in _QUESTION_PREFIXES:
        if phrase.startswith(prefix):
            phrase = phrase[len(prefix) :]
            break
    for suffix in _QUESTION_SUFFIXES:
        if phrase.endswith(suffix):
            phrase = phrase[: -len(suffix)]
            break
    phrase = _EDGE_NOISE.sub("", " ".join(phrase.split()))
    if not _MIN_KEYWORD_LENGTH <= len(phrase) <= _MAX_KEYWORD_LENGTH:
        return None
    # Nothing was stripped: the question is not definition-shaped, so a
    # literal search for the whole sentence would only ever match nothing.
    if phrase == _EDGE_NOISE.sub("", question.lower()):
        return None
    return phrase


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


#: Added to a passage's distance for each grade it sits below the student's,
#: when ranking. The grade filter is a ceiling so that a student can revise
#: earlier material, but that also lets a primary-school book answer a
#: lower-secondary question: asked about Điện Biên Phủ, a grade-9 student
#: was answered from the grade-5 book, which does mention it, while the
#: grade-9 chapter on it never made the top ten. Small on purpose -- a
#: lower grade's passage that is a much better match still wins, which is
#: what makes revision work.
_LOWER_GRADE_PENALTY = 0.02

#: How many extra passages to fetch before re-ranking. Re-ranking can only
#: promote what was fetched, so the right passage has to be in the pool
#: first.
_OVERFETCH = 3


def _passage_key(chunk: RetrievedChunk) -> tuple[Any, Any]:
    """Identify a passage across two queries that may both return it."""
    return (chunk.metadata.get("source_file"), chunk.metadata.get("chunk_index"))


def _prefer_the_students_own_grade(
    chunks: list[RetrievedChunk], grade: int, limit: int
) -> list[RetrievedChunk]:
    """Re-rank so a passage from the student's own grade comes first.

    Only the ordering changes; each chunk keeps its real distance, so what
    the logs and the benchmark report show is still what ChromaDB measured.

    Args:
        chunks: Passages from one query, nearest first.
        grade: The student's grade.
        limit: Maximum passages to keep.

    Returns:
        At most ``limit`` passages, best first after the grade adjustment.
    """

    def adjusted(chunk: RetrievedChunk) -> float:
        chunk_grade = chunk.metadata.get("grade")
        if not isinstance(chunk_grade, int):
            return chunk.distance
        return chunk.distance + _LOWER_GRADE_PENALTY * max(0, grade - chunk_grade)

    return sorted(chunks, key=adjusted)[:limit]


def _merge_by_passage(
    preferred: list[RetrievedChunk], rest: list[RetrievedChunk], limit: int
) -> list[RetrievedChunk]:
    """Put ``preferred`` passages first, then fill up from ``rest``.

    Order matters downstream: the prompt numbers the passages it is given,
    and the generator leans on the earlier ones, so a literal-match hit
    belongs ahead of a merely-nearby one even when its vector distance is
    larger.

    Args:
        preferred: Passages to place first (the literal-match pass).
        rest: Passages to fill the remaining slots with.
        limit: Maximum passages to return in total.

    Returns:
        At most ``limit`` passages, no passage repeated.
    """
    merged: list[RetrievedChunk] = []
    seen: set[tuple[Any, Any]] = set()
    for chunk in [*preferred, *rest]:
        key = _passage_key(chunk)
        if key in seen:
            continue
        seen.add(key)
        merged.append(chunk)
        if len(merged) >= limit:
            break
    return merged


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
            Passages closer than ``max_retrieval_distance``, best first --
            ranked by distance, adjusted to favour the student's own grade
            over a lower one (see :func:`_prefer_the_students_own_grade`).
            An empty list means the textbook has no relevant content, which
            the generator turns into the "not in your textbook" answer.

        Raises:
            ValueError: If ``question`` is empty.
            RuntimeError: If the ChromaDB query fails.
        """
        cleaned_question = question.strip()
        if not cleaned_question:
            raise ValueError("question must not be empty")

        limit = top_k or self._settings.retrieval_top_k
        where = build_metadata_filter(profile)

        # Fetched wide, then re-ranked: the grade adjustment can only promote
        # a passage that was fetched in the first place.
        chunks = _prefer_the_students_own_grade(
            self._query(cleaned_question, where, limit * _OVERFETCH), profile.grade, limit
        )

        # Further passes restricted to passages that literally contain the
        # term being asked about. Vector distance alone misses these when
        # the term's individual words are common elsewhere in the corpus
        # (see :func:`keyword_phrase`), and a missed definition means the
        # bot refuses a question its own textbook answers.
        phrase = keyword_phrase(cleaned_question)
        if phrase is not None:
            chunks = self._with_literal_matches(
                cleaned_question, where, limit, phrase, chunks, profile.grade
            )

        kept = [c for c in chunks if c.distance <= self._settings.max_retrieval_distance]

        logger.info(
            "Retrieved %d/%d passages (grade=%d, subject=%s)",
            len(kept),
            len(chunks),
            profile.grade,
            profile.subject,
        )
        return kept

    def _with_literal_matches(
        self,
        question: str,
        where: dict[str, Any],
        limit: int,
        phrase: str,
        vector_hits: list[RetrievedChunk],
        grade: int,
    ) -> list[RetrievedChunk]:
        """Put literal matches for ``phrase`` ahead of the vector results.

        Three passes, most specific first:

        1. ``"<phrase> là"`` -- the shape a Vietnamese textbook definition
           actually takes ("Quang hợp là quá trình lá cây sử dụng..."). This
           is the one sentence a "X là gì?" question needs, and it is often
           ranked well below the lesson's objectives box by vector distance
           alone.
        2. ``"<phrase>"`` -- anywhere the term appears, for questions that
           are not quite definitions.
        3. Only if both came back empty: the last two words of the phrase,
           for a term the student prefixed with a category word ("biện pháp
           chơi chữ" is printed as "chơi chữ").

        Args:
            question: The student's question, embedded for ranking within
                each pass.
            where: The mandatory grade/subject filter.
            limit: Maximum passages to return in total.
            phrase: The term extracted by :func:`keyword_phrase`.
            vector_hits: Results of the plain vector pass, used to fill the
                remaining slots.
            grade: The student's grade, for the same re-ranking the vector
                pass gets.

        Returns:
            At most ``limit`` passages, literal matches first.
        """
        def pass_for(contains: str) -> list[RetrievedChunk]:
            """One literal pass, re-ranked to favour the student's own grade."""
            return _prefer_the_students_own_grade(
                self._query(question, where, limit * _OVERFETCH, contains=contains),
                grade,
                limit,
            )

        definitions = pass_for(f"{phrase} là")
        literal = pass_for(phrase)

        if not definitions and not literal:
            words = phrase.split()
            if len(words) >= 3:
                shorter = " ".join(words[-2:])
                literal = pass_for(shorter)
                if literal:
                    logger.info("Literal match on the shorter term %r", shorter)

        if definitions or literal:
            logger.info(
                "Literal match on %r: %d definition-shaped, %d other passage(s)",
                phrase,
                len(definitions),
                len(literal),
            )
        return _merge_by_passage([*definitions, *literal], vector_hits, limit)

    def _query(
        self,
        question: str,
        where: dict[str, Any],
        limit: int,
        *,
        contains: str | None = None,
    ) -> list[RetrievedChunk]:
        """Run one ChromaDB query, optionally filtered to a literal substring.

        Args:
            question: The text to embed and match against.
            where: The mandatory grade/subject metadata filter.
            limit: Maximum passages to return.
            contains: When given, only passages whose text contains this
                substring are considered.

        Returns:
            Matching passages, nearest first.

        Raises:
            RuntimeError: If the ChromaDB query fails.
        """
        extra: dict[str, Any] = {"where_document": {"$contains": contains}} if contains else {}
        try:
            response = self._collection.query(
                query_texts=[question],
                n_results=limit,
                where=where,
                include=["documents", "metadatas", "distances"],
                **extra,
            )
        except (chromadb.errors.ChromaError, ValueError, RuntimeError) as exc:
            raise RuntimeError(f"ChromaDB query failed: {exc}") from exc
        return self._parse_response(response)

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
