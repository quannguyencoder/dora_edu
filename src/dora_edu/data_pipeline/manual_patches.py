"""Manual retrieval-quality patches for known problem chunks.

Some passages -- often short, exact definitions or lists -- rank poorly in
the vector index because they compete with something more dominant nearby:
a lesson's generic "Khái niệm, thuật ngữ / Kiến thức, kĩ năng" objectives
box (near-identical across hundreds of lessons, so it clusters tightly in
embedding space and crowds out the real content), or a chunk boundary that
happened to split a short answer away from the heading that made it findable.

Rather than fixing these with an ad-hoc, one-off write straight to ChromaDB
(easy to do, easy to forget, and silently wiped by the next full re-ingest --
this has already happened twice in this project), each known case is listed
here and re-applied automatically by :func:`apply_patches`, which
``dora-bulk-ingest`` calls after every run. Losing one of these to a
re-ingest now just means the next ``dora-bulk-ingest`` restores it.
"""

from __future__ import annotations

from typing import TypedDict

from dora_edu.models import TextbookMetadata, TextChunk
from dora_edu.rag_engine.indexer import TextbookIndexer


class _Patch(TypedDict):
    grade: int
    subject: str
    book_title: str
    source_file: str
    page: int
    #: Offset by 900000 so it can never collide with the natural chunker's
    #: indices for the same book.
    chunk_index: int
    text: str


#: One dedicated, clean chunk per case that needed a manual boost.
PATCHES: list[_Patch] = [
    {
        "grade": 8,
        "subject": "Toán",
        "book_title": "Toán 8 - tập 1",
        "source_file": "SGKToan8tapmot.pdf",
        "page": 12,
        "chunk_index": 900012,
        "text": (
            "Đa thức là gì? Đa thức là tổng của những đơn thức; mỗi đơn thức "
            "trong tổng gọi là một hạng tử của đa thức đó. Chú ý: mỗi đơn thức "
            "cũng được coi là một đa thức."
        ),
    },
    {
        "grade": 6,
        "subject": "Toán",
        "book_title": "Toán 6 - tập 2",
        "source_file": "SGKToan6taphai.pdf",
        "page": 20,
        "chunk_index": 900020,
        "text": (
            "Phép nhân phân số là gì? Muốn nhân hai phân số, ta nhân các tử "
            "với nhau và nhân các mẫu với nhau. Nhận xét: muốn nhân một số "
            "nguyên với một phân số, ta nhân số nguyên đó với tử của phân số "
            "và giữ nguyên mẫu."
        ),
    },
    {
        "grade": 6,
        "subject": "Ngữ văn",
        "book_title": "Ngữ văn 6 - tập 1",
        "source_file": "SGKNguvan6tapmot.pdf",
        "page": 81,
        "chunk_index": 900081,
        "text": (
            "Lập dàn ý bài văn kể lại một trải nghiệm của em. "
            "Mở bài: Giới thiệu câu chuyện. "
            "Thân bài: Kể lại diễn biến của câu chuyện (giới thiệu thời gian, "
            "không gian, nhân vật liên quan; kể lại các sự việc theo trình tự "
            "hợp lí: sự việc 1, sự việc 2, sự việc 3...). "
            "Kết bài: Nêu cảm xúc của người viết và rút ra ý nghĩa, sự quan "
            "trọng của trải nghiệm đối với bản thân."
        ),
    },
    {
        "grade": 11,
        "subject": "Toán",
        "book_title": "Toán 11 - tập 2",
        "source_file": "SGKToan11taphai.pdf",
        "page": 83,
        "chunk_index": 900083,
        "text": (
            "Định nghĩa: Đạo hàm là gì? Đạo hàm của một hàm số tại một điểm cho "
            "biết tốc độ thay đổi tức thời của hàm số tại điểm đó. Nếu hàm số có "
            "đạo hàm tại mọi điểm trong một khoảng, ta nói hàm số có đạo hàm "
            "trên khoảng đó."
        ),
    },
    {
        # Re-adds a patch that existed before this file did (a plain, ad-hoc
        # DB write from an earlier session) -- lost to today's full
        # re-ingests exactly like đa thức and Lập dàn ý were, which is the
        # whole reason this file exists now: every known case belongs here
        # so a future re-ingest restores it instead of silently dropping it.
        "grade": 8,
        "subject": "Toán",
        "book_title": "Toán 8 - tập 1",
        "source_file": "SGKToan8tapmot.pdf",
        "page": 79,
        "chunk_index": 900079,
        "text": (
            "Định lí Thalès là gì? Nếu một đường thẳng song song với một cạnh "
            "của tam giác và cắt hai cạnh còn lại thì nó định ra trên hai cạnh "
            "đó những đoạn thẳng tương ứng tỉ lệ."
        ),
    },
]


def apply_patches(indexer: TextbookIndexer) -> int:
    """Re-apply every known manual chunk patch.

    Safe to call repeatedly: each patch upserts by the same stable id
    (grade + subject + source_file + chunk_index + text), so re-running
    this never creates a duplicate.

    Args:
        indexer: Destination index.

    Returns:
        The number of patch chunks written.
    """
    written = 0
    for patch in PATCHES:
        metadata = TextbookMetadata(
            grade=patch["grade"],
            subject=patch["subject"],
            book_title=patch["book_title"],
            source_file=patch["source_file"],
        )
        chunk = TextChunk(
            text=patch["text"],
            chunk_index=patch["chunk_index"],
            page_start=patch["page"],
            page_end=patch["page"],
        )
        written += indexer.index_chunks([chunk], metadata)
    return written
