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


def chunk_pages(
    pages: list[ParsedPage],
    *,
    chunk_size: int,
    chunk_overlap: int,
) -> list[TextChunk]:
    """Group cleaned pages into overlapping, sentence-aligned chunks.

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

    # Flatten to (sentence, page_number) so a chunk can span a page break while
    # still reporting the exact pages it covers.
    flat: list[tuple[str, int]] = []
    for page in pages:
        for sentence in split_sentences(page.text):
            flat.append((sentence, page.page_number))

    chunks: list[TextChunk] = []
    buffer: list[str] = []
    buffer_pages: list[int] = []
    buffer_length = 0

    def flush() -> None:
        """Emit the buffered sentences as a chunk."""
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
        tail, tail_length = _tail_for_overlap(buffer, chunk_overlap)
        # Carry the page numbers belonging to the retained tail sentences.
        buffer_pages = buffer_pages[len(buffer) - len(tail):] if tail else []
        buffer = tail
        buffer_length = tail_length

    for sentence, page_number in flat:
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
