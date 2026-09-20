"""Overlap chunking logic.

Chunks are built from whole sentences rather than fixed character windows so a
retrieved passage never starts or ends mid-sentence, and each chunk records the
page range it came from so the bot can cite the textbook page to the student.
"""

from __future__ import annotations

import re

from dora_edu.models import ParsedPage, TextChunk

#: Sentence boundary: terminal punctuation followed by whitespace and a letter
#: that starts a new sentence. Vietnamese uppercase letters are covered by the
#: Unicode-aware ``str.isupper`` check applied afterwards.
_SENTENCE_BOUNDARY = re.compile(r"(?<=[.!?…:;])\s+")

#: Paragraph boundary produced by the PDF cleaner.
_PARAGRAPH_BOUNDARY = re.compile(r"\n\s*\n")

#: Heading-like lines that mark the start of a new labelled subsection in a
#: MOET textbook page (e.g. "1. TRƯỚC KHI VIẾT", "a. Lựa chọn đề tài",
#: "BÀI 1"). Matched against a single stripped line, not mid-sentence, so
#: ordinary prose that happens to start with a number or "bài" is not
#: mistaken for one.
_HEADING_LINE = re.compile(r"^(?:\d{1,2}\.\s+\S|[a-zđ]\.\s+\S|BÀI\s+\d)")

#: Heading lines are short by construction; a long line matching the pattern
#: by coincidence (e.g. a numbered sentence) is content, not a heading.
_MAX_HEADING_LINE_LENGTH = 80

#: A section shorter than this is a heading with no real body -- most often a
#: run of numbered list items (a book-set back-cover listing, a festival-day
#: programme) rather than an actual labelled subsection. Embedding models
#: rank such short, low-information text deceptively close to many unrelated
#: queries, so instead of becoming its own chunk it merges into the next
#: section, the same way a page with no headings at all behaves.
_MIN_SECTION_LENGTH = 120


def _is_heading_line(line: str) -> bool:
    """Return whether ``line`` looks like a subsection heading on its own line."""
    stripped = line.strip()
    if not stripped or len(stripped) > _MAX_HEADING_LINE_LENGTH:
        return False
    return bool(_HEADING_LINE.match(stripped))


def split_into_sections(text: str) -> list[str]:
    """Split page text into structural sections at heading-like lines.

    Text with no heading lines at all comes back as a single section (the
    whole input), so callers that never see a heading behave exactly as if
    sections did not exist. A heading that starts a section shorter than
    :data:`_MIN_SECTION_LENGTH` is treated as if it were not a heading at
    all -- it merges into the section that follows instead of standing alone.

    Args:
        text: Raw page text, with its original line breaks intact.

    Returns:
        Non-empty sections in reading order, each starting either at the
        top of ``text`` or at a detected heading line.
    """
    sections: list[list[str]] = [[]]
    for line in text.splitlines():
        current = "\n".join(sections[-1]).strip()
        if _is_heading_line(line) and sections[-1] and len(current) >= _MIN_SECTION_LENGTH:
            sections.append([])
        sections[-1].append(line)
    return ["\n".join(section) for section in sections if "".join(section).strip()]


def split_sentences(text: str) -> list[str]:
    """Split one block of Vietnamese text into sentence-like units.

    Falls back to line splitting for textbook content that carries little
    punctuation (exercise lists, tables of contents).

    Args:
        text: A paragraph or page of cleaned text.

    Returns:
        Non-empty, whitespace-trimmed sentences.
    """
    units: list[str] = []
    for paragraph in _PARAGRAPH_BOUNDARY.split(text):
        paragraph = paragraph.strip()
        if not paragraph:
            continue
        for sentence in _SENTENCE_BOUNDARY.split(paragraph):
            sentence = " ".join(sentence.split())
            if sentence:
                units.append(sentence)
    return units


def _tail_for_overlap(sentences: list[str], overlap: int) -> tuple[list[str], int]:
    """Pick the trailing sentences that seed the next chunk's overlap.

    Args:
        sentences: Sentences making up the chunk just emitted.
        overlap: Target overlap size in characters.

    Returns:
        A ``(sentences, length)`` pair; empty when ``overlap`` is zero.
    """
    if overlap <= 0:
        return [], 0

    tail: list[str] = []
    length = 0
    for sentence in reversed(sentences):
        # Always keep at least one sentence of context, then stop once the
        # target overlap is reached.
        if tail and length + len(sentence) + 1 > overlap:
            break
        tail.insert(0, sentence)
        length += len(sentence) + 1
    # Never carry the whole chunk forward, or chunking would not advance.
    if len(tail) == len(sentences) and len(sentences) > 1:
        tail = tail[1:]
        length = sum(len(s) + 1 for s in tail)
    return tail, length


#: Every MOET textbook's table-of-contents page carries this literal marker.
#: A mục lục page is a list of chapter/lesson titles -- e.g. "CHƯƠNG IX. ĐẠO
#: HÀM" -- so a query like "đạo hàm là gì" can match the title line on this
#: page as closely as the real definition dozens of pages later, and being
#: short and near-identical in shape across most of the corpus, it tends to
#: win. There is never a legitimate reason to answer a student from a table
#: of contents, so these pages are dropped before chunking rather than
#: merely down-ranked.
_TABLE_OF_CONTENTS_MARKER = "MỤC LỤC"


def _is_table_of_contents_page(text: str) -> bool:
    """Return whether ``text`` is a table-of-contents page."""
    return _TABLE_OF_CONTENTS_MARKER in text


def chunk_pages(
    pages: list[ParsedPage],
    *,
    chunk_size: int,
    chunk_overlap: int,
) -> list[TextChunk]:
    """Group cleaned pages into overlapping, sentence-aligned chunks.

    Table-of-contents pages (see :func:`_is_table_of_contents_page`) are
    dropped before chunking: they are never useful to answer a student from,
    and their short, repetitive chapter/lesson-title listing otherwise
    tends to out-rank real content that happens to share a keyword with a
    title (e.g. a chapter literally named after the term being asked about).

    Args:
        pages: Cleaned pages in reading order.
        chunk_size: Target chunk length in characters.
        chunk_overlap: Characters of trailing context repeated in the next chunk.

    Returns:
        Chunks in reading order, each tagged with its source page range.

    Raises:
        ValueError: If ``chunk_overlap`` is not smaller than ``chunk_size``.
    """
    if chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be smaller than chunk_size")

    pages = [page for page in pages if not _is_table_of_contents_page(page.text)]

    # Flatten to (sentence, page_number, starts_section) so a chunk can still
    # span a page break when nothing marks one -- only a real heading line
    # forces a new chunk, not merely reaching the top of the next page (a
    # page's first section is a continuation, not a heading, so it never
    # forces a boundary; later sections on the same page always do).
    flat: list[tuple[str, int, bool]] = []
    for page in pages:
        for section_index, section in enumerate(split_into_sections(page.text)):
            for sentence_index, sentence in enumerate(split_sentences(section)):
                starts_section = section_index > 0 and sentence_index == 0
                flat.append((sentence, page.page_number, starts_section))

    chunks: list[TextChunk] = []
    buffer: list[str] = []
    buffer_pages: list[int] = []
    buffer_length = 0

    def flush(*, carry_overlap: bool = True) -> None:
        """Emit the buffered sentences as a chunk.

        Args:
            carry_overlap: Seed the next chunk with trailing context from
                this one. Only meaningful when the flush was triggered by
                the size budget -- a flush at a real section boundary never
                carries overlap, so a labelled subsection's chunk never
                opens with a trailing fragment of the previous one.
        """
        nonlocal buffer, buffer_pages, buffer_length
        if not buffer:
            return
        chunks.append(
            TextChunk(
                text=" ".join(buffer),
                chunk_index=len(chunks),
                page_start=min(buffer_pages),
                page_end=max(buffer_pages),
            )
        )
        tail, tail_length = _tail_for_overlap(buffer, chunk_overlap) if carry_overlap else ([], 0)
        # Carry the page numbers belonging to the retained tail sentences.
        buffer_pages = buffer_pages[len(buffer) - len(tail):] if tail else []
        buffer = tail
        buffer_length = tail_length

    for sentence, page_number, starts_section in flat:
        if buffer and starts_section:
            flush(carry_overlap=False)
        # A single oversized sentence becomes its own chunk rather than being cut.
        if buffer and buffer_length + len(sentence) + 1 > chunk_size:
            flush()
        buffer.append(sentence)
        buffer_pages.append(page_number)
        buffer_length += len(sentence) + 1

    if buffer:
        chunks.append(
            TextChunk(
                text=" ".join(buffer),
                chunk_index=len(chunks),
                page_start=min(buffer_pages),
                page_end=max(buffer_pages),
            )
        )
    return chunks
