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


class _TextFix(TypedDict):
    """One mis-OCR'd passage to correct inside an existing chunk."""

    grade: int
    subject: str
    book_title: str
    source_file: str
    chunk_index: int
    #: The exact string Tesseract produced, as stored in ChromaDB.
    wrong: str
    #: What the printed page actually says, re-read from the page image.
    right: str


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
    {
        # OCR misread the fraction notation "a/b" in this definition as a
        # bare "=" sign ("ta gợi = là một phân s6" instead of "ta gọi a/b là
        # một phân số"), so the one sentence that actually answers "Phân số
        # là gì?" doesn't read as Vietnamese at all -- found live, testing
        # the exact question planned for the grade-6 team member's demo.
        "grade": 6,
        "subject": "Toán",
        "book_title": "Toán 6 - tập 2",
        "source_file": "SGKToan6taphai.pdf",
        "page": 6,
        "chunk_index": 900006,
        "text": (
            "Phân số là gì? Với a, b là các số nguyên và b khác 0, ta gọi "
            "a/b là một phân số, trong đó a là tử số và b là mẫu số của "
            "phân số đó."
        ),
    },
    {
        # The summary box listing all seven identities is printed white-on-
        # blue, and Tesseract lost every exponent in it ("A? ~B? =(A-B)(A+B)",
        # and the cube-of-a-difference line came out as "CN. =....."), so the
        # one passage that answers "có những hằng đẳng thức đáng nhớ nào"
        # carried no usable formula at all. Re-read from the page image with
        # Gemini Vision. Found live: a student asked exactly this and the bot
        # answered from p.31 instead, which is about what a hằng đẳng thức
        # *is*, not which seven are the đáng nhớ ones.
        "grade": 8,
        "subject": "Toán",
        "book_title": "Toán 8 - tập 1",
        "source_file": "SGKToan8tapmot.pdf",
        "page": 40,
        "chunk_index": 900040,
        "text": (
            "Có những hằng đẳng thức đáng nhớ nào? Bảy hằng đẳng thức đáng "
            "nhớ là: "
            "A² - B² = (A - B)(A + B); "
            "(A + B)² = A² + 2AB + B²; "
            "(A - B)² = A² - 2AB + B²; "
            "(A + B)³ = A³ + 3A²B + 3AB² + B³; "
            "(A - B)³ = A³ - 3A²B + 3AB² - B³; "
            "A³ + B³ = (A + B)(A² - AB + B²); "
            "A³ - B³ = (A - B)(A² + AB + B²). "
            "Chú ý: các hằng đẳng thức này được sử dụng thường xuyên trong "
            "các biến đổi đại số nên ta gọi chúng là các hằng đẳng thức "
            "đáng nhớ."
        ),
    },
    {
        # Tesseract read the multiplication dot in "a . b = b . a" as a minus
        # sign, turning a correct commutative law into "a - b = b - a", which
        # is simply false. The bot then quoted it faithfully to a grade-8
        # student -- the worst failure mode there is, since the answer looks
        # authoritative and cites a real page. Re-read with Gemini Vision.
        "grade": 8,
        "subject": "Toán",
        "book_title": "Toán 8 - tập 1",
        "source_file": "SGKToan8tapmot.pdf",
        "page": 31,
        "chunk_index": 900031,
        "text": (
            "Hằng đẳng thức là gì? Nếu hai biểu thức (đại số) A và B luôn "
            "cùng nhận giá trị bằng nhau với mọi giá trị của biến thì ta nói "
            "A = B là một đồng nhất thức hay là một hằng đẳng thức. "
            "Các đẳng thức thường gặp: a + b = b + a; a . b = b . a; "
            "a(b + c) = ab + ac là những hằng đẳng thức."
        ),
    },
]


#: Chunks whose OCR text is not merely unhelpful but *wrong*. A clean extra
#: chunk in :data:`PATCHES` cannot fix these on its own: the corrupted chunk
#: stays retrievable, and a student asking a slightly different question
#: still gets the false version quoted back at them with a real page number.
TEXT_FIXES: list[_TextFix] = [
    {
        # "a . b = b . a" (phép nhân giao hoán): Tesseract read the
        # multiplication dot as a minus sign, producing "a - b = b - a",
        # which is false for every a != b. Found live, in a grade-8
        # student's Discord question about hằng đẳng thức đáng nhớ.
        "grade": 8,
        "subject": "Toán",
        "book_title": "Toán 8 - tập 1",
        "source_file": "SGKToan8tapmot.pdf",
        "chunk_index": 55,
        "wrong": "a+b=b+a; a-b=b-a, a(b + e) = ab + ac",
        "right": "a + b = b + a; a . b = b . a; a(b + c) = ab + ac",
    },
    {
        # The same mis-read dot, in the passage that states all three
        # properties of integer multiplication ("3. TÍNH CHẤT CỦA PHÉP
        # NHÂN", Toán 6 tập 1). Tesseract turned every multiplication dot
        # into a minus sign, so the commutative and associative laws both
        # came out false and the distributive law came out as nonsense.
        # Found by `dora-ocr-audit`, which flags exactly this shape, then
        # confirmed against the page image with Gemini Vision.
        "grade": 6,
        "subject": "Toán",
        "book_title": "Toán 6 - tập 1",
        "source_file": "SGKToan6tapmot.pdf",
        "chunk_index": 179,
        "wrong": (
            "Giao hoán: a - b=b-a Kết hợp: a - (Ð - c) = (a - b) -c "
            "Phân phối của phép nhân đối với phép cộng: a: (b + c) = a - b + a: c."
        ),
        "right": (
            "Giao hoán: a . b = b . a. Kết hợp: a . (b . c) = (a . b) . c. "
            "Phân phối của phép nhân đối với phép cộng: a . (b + c) = a . b + a . c."
        ),
    },
    {
        # And again on the facing page, in the sentence that introduces
        # multiplication as repeated addition: "1 . a = a . 1 = a và
        # a . b = b . a = a + a + ... + a" became "1-a=a-1=a vàa-b=b-a=...".
        "grade": 6,
        "subject": "Toán",
        "book_title": "Toán 6 - tập 1",
        "source_file": "SGKToan6tapmot.pdf",
        "chunk_index": 176,
        "wrong": "N_ 1-a=a-1=a vàa-b=b-a=a+a+...+a",
        "right": "1 . a = a . 1 = a và a . b = b . a = a + a + ... + a",
    },
]


def apply_text_fixes(indexer: TextbookIndexer) -> int:
    """Re-apply every known OCR correction to the indexed chunks.

    Safe to call repeatedly and after every ingest: a chunk that no longer
    contains the mis-read text is left untouched.

    Args:
        indexer: The index holding the chunks to correct.

    Returns:
        The number of chunks actually corrected by this call.
    """
    fixed = 0
    for fix in TEXT_FIXES:
        metadata = TextbookMetadata(
            grade=fix["grade"],
            subject=fix["subject"],
            book_title=fix["book_title"],
            source_file=fix["source_file"],
        )
        if indexer.repair_chunk_text(
            metadata,
            chunk_index=fix["chunk_index"],
            wrong=fix["wrong"],
            right=fix["right"],
        ):
            fixed += 1
    return fixed


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
