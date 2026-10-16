"""The 50-question benchmark set.

Four rules shaped this list, and they are what make the resulting numbers
worth anything:

1. **Nothing hand-patched.** No question here is answered by a chunk in
   :data:`~dora_edu.data_pipeline.manual_patches.PATCHES` or
   :data:`~dora_edu.data_pipeline.manual_patches.TEXT_FIXES`. Those passages
   were written (or corrected) by hand *because* they failed, so including
   them would measure the patches, not the pipeline. The excluded topics are
   listed in :data:`EXCLUDED_TOPICS`.
2. **Spread across subjects and grades.** Grades 6-9 (the lower-secondary
   focus) across Toán, Ngữ văn, Khoa học tự nhiên, Lịch sử và Địa lí, Giáo
   dục công dân, Tin học and Công nghệ.
3. **Ten questions are deliberately outside the textbook** -- either from a
   later grade or from no textbook at all. For these the *correct* behaviour
   is to refuse, so they measure the thing a demo never shows: whether the
   zero-hallucination promise actually holds under pressure.
4. **Every expected answer is a definition or fact printed in the book**,
   not an opinion, so a teacher checking the run has something objective to
   compare against.

``keywords`` are a coarse automatic signal only -- a word the real answer
should contain. A question whose keywords all appear can still be wrong, so
:mod:`dora_edu.evaluation.runner` writes the full answer out for a person to
read; the keyword column just says which rows to look at first.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class EvalQuestion:
    """One benchmark question and what counts as handling it correctly."""

    question: str
    grade: int
    subject: str
    #: Whether the answer is in a textbook this student may be shown (their
    #: own grade or any below it). ``False`` means refusing is the *correct*
    #: outcome.
    in_textbook: bool
    #: Words the real answer should contain; a coarse automatic signal.
    keywords: tuple[str, ...] = field(default_factory=tuple)
    #: Why an out-of-textbook question is out of the textbook, so a reader
    #: of the report can check that judgement rather than trust it.
    note: str = ""


#: Topics deliberately left out, because a hand-written chunk answers them.
#: Keeping this list next to the question set is what stops a future session
#: from "improving" the benchmark by adding a question it already fixed.
EXCLUDED_TOPICS: tuple[str, ...] = (
    "đa thức (Toán 8)",
    "phép nhân phân số (Toán 6)",
    "phân số là gì (Toán 6)",
    "lập dàn ý bài văn kể lại một trải nghiệm (Ngữ văn 6)",
    "đạo hàm (Toán 11)",
    "định lí Thalès (Toán 8)",
    "hằng đẳng thức đáng nhớ (Toán 8)",
)


#: Questions whose answers are printed in the student's own textbook (or a
#: lower grade's), i.e. the bot should answer and cite a page.
IN_TEXTBOOK_QUESTIONS: tuple[EvalQuestion, ...] = (
    # --- Toán ---
    EvalQuestion("Số nguyên tố là gì?", 6, "Toán", True, ("ước",)),
    EvalQuestion("Thế nào là hai số đối nhau?", 6, "Toán", True, ("đối",)),
    EvalQuestion("Trung điểm của đoạn thẳng là gì?", 6, "Toán", True, ("trung điểm",)),
    EvalQuestion("Số hữu tỉ là gì?", 7, "Toán", True, ("phân số",)),
    EvalQuestion("Hai góc đối đỉnh là gì?", 7, "Toán", True, ("đối đỉnh",)),
    EvalQuestion("Thế nào là tam giác cân?", 7, "Toán", True, ("cạnh",)),
    EvalQuestion("Phân thức đại số là gì?", 8, "Toán", True, ("mẫu",)),
    EvalQuestion("Hình thang cân có tính chất gì?", 8, "Toán", True, ("cạnh bên",)),
    EvalQuestion("Thế nào là hai tam giác đồng dạng?", 8, "Toán", True, ("đồng dạng", "tỉ lệ")),
    EvalQuestion("Căn bậc hai của một số là gì?", 9, "Toán", True, ("bình phương",)),
    EvalQuestion("Góc nội tiếp là gì?", 9, "Toán", True, ("đường tròn",)),
    EvalQuestion("Tần số của một giá trị là gì?", 9, "Toán", True, ("số lần",)),
    # --- Ngữ văn ---
    EvalQuestion("Truyện cổ tích là gì?", 6, "Ngữ văn", True, ("dân gian",)),
    EvalQuestion("Từ đơn và từ phức khác nhau thế nào?", 6, "Ngữ văn", True, ("tiếng",)),
    EvalQuestion("Ẩn dụ là gì?", 6, "Ngữ văn", True, ("gọi",)),
    EvalQuestion("Phó từ là gì?", 7, "Ngữ văn", True, ("bổ sung",)),
    EvalQuestion("Tục ngữ là gì?", 7, "Ngữ văn", True, ("kinh nghiệm",)),
    EvalQuestion("Từ tượng hình và từ tượng thanh là gì?", 8, "Ngữ văn", True, ("âm thanh",)),
    EvalQuestion("Nghĩa tường minh và nghĩa hàm ẩn khác nhau thế nào?", 8, "Ngữ văn", True, ("hàm ẩn",)),
    EvalQuestion("Biện pháp chơi chữ là gì?", 9, "Ngữ văn", True, ("âm",)),
    # --- Khoa học tự nhiên ---
    EvalQuestion("Tế bào là gì?", 6, "Khoa học tự nhiên", True, ("cơ thể",)),
    EvalQuestion("Chất tinh khiết và hỗn hợp khác nhau thế nào?", 6, "Khoa học tự nhiên", True, ("chất",)),
    EvalQuestion("Nguyên tử gồm những hạt nào?", 7, "Khoa học tự nhiên", True, ("electron",)),
    EvalQuestion("Quang hợp là gì?", 7, "Khoa học tự nhiên", True, ("ánh sáng",)),
    EvalQuestion("Tốc độ cho biết điều gì?", 7, "Khoa học tự nhiên", True, ("nhanh",)),
    EvalQuestion("Định luật bảo toàn khối lượng phát biểu thế nào?", 8, "Khoa học tự nhiên", True, ("khối lượng",)),
    EvalQuestion("Acid là gì?", 8, "Khoa học tự nhiên", True, ("hydrogen",)),
    EvalQuestion("Khối lượng riêng là gì?", 8, "Khoa học tự nhiên", True, ("thể tích",)),
    EvalQuestion("Định luật Ôm phát biểu thế nào?", 9, "Khoa học tự nhiên", True, ("điện trở",)),
    EvalQuestion("Hiện tượng khúc xạ ánh sáng là gì?", 9, "Khoa học tự nhiên", True, ("tia",)),
    # --- Lịch sử và Địa lí ---
    EvalQuestion("Kinh tuyến và vĩ tuyến là gì?", 6, "Lịch sử và Địa lí", True, ("đường",)),
    EvalQuestion("Nhà nước Văn Lang ra đời vào khoảng thời gian nào?", 6, "Lịch sử và Địa lí", True, ("VII",)),
    EvalQuestion("Phong trào Văn hoá Phục hưng là gì?", 7, "Lịch sử và Địa lí", True, ("châu Âu",)),
    EvalQuestion("Nhà Lý được thành lập năm nào?", 7, "Lịch sử và Địa lí", True, ("1009",)),
    EvalQuestion("Khởi nghĩa Tây Sơn do ai lãnh đạo?", 8, "Lịch sử và Địa lí", True, ("Nguyễn",)),
    EvalQuestion("Cách mạng công nghiệp bắt đầu ở nước nào?", 8, "Lịch sử và Địa lí", True, ("Anh",)),
    EvalQuestion("Chiến thắng Điện Biên Phủ diễn ra năm nào?", 9, "Lịch sử và Địa lí", True, ("1954",)),
    # --- Giáo dục công dân, Tin học, Công nghệ ---
    EvalQuestion("Thế nào là bạo lực học đường?", 8, "Giáo dục công dân", True, ("học sinh",)),
    EvalQuestion("Thông tin và dữ liệu khác nhau thế nào?", 6, "Tin học", True, ("dữ liệu",)),
    EvalQuestion("Nhà ở có vai trò gì với con người?", 6, "Công nghệ", True, ("sinh hoạt",)),
)


#: Questions deliberately outside what this student may be shown. Refusing
#: is the correct outcome; answering is a hallucination.
OUT_OF_TEXTBOOK_QUESTIONS: tuple[EvalQuestion, ...] = (
    EvalQuestion(
        "Đạo hàm của hàm số y = x bình phương là gì?",
        9,
        "Toán",
        False,
        note="Đạo hàm học ở lớp 11, trên lớp 9 nên không được trả lời.",
    ),
    EvalQuestion(
        "Giải phương trình bậc hai x bình phương trừ 5x cộng 6 bằng 0.",
        6,
        "Toán",
        False,
        note="Phương trình bậc hai học ở lớp 9, trên lớp 6.",
    ),
    EvalQuestion(
        "Tích phân dùng để làm gì?",
        7,
        "Toán",
        False,
        note="Tích phân học ở lớp 12.",
    ),
    EvalQuestion(
        "Lò phản ứng hạt nhân hoạt động như thế nào?",
        8,
        "Khoa học tự nhiên",
        False,
        note="Không thuộc chương trình KHTN cấp THCS.",
    ),
    EvalQuestion(
        "Thuyết tương đối hẹp của Einstein phát biểu thế nào?",
        9,
        "Khoa học tự nhiên",
        False,
        note="Không thuộc chương trình THCS.",
    ),
    EvalQuestion(
        "Chiến tranh giữa Nga và Ukraine bắt đầu năm nào?",
        8,
        "Lịch sử và Địa lí",
        False,
        note="Sự kiện sau thời điểm biên soạn sách, không có trong SGK.",
    ),
    EvalQuestion(
        "Tổ chức Thương mại Thế giới (WTO) có vai trò gì?",
        6,
        "Lịch sử và Địa lí",
        False,
        note=(
            "Không có trong SGK Lịch sử và Địa lí từ lớp 1 đến lớp 6 — đã kiểm "
            "bằng cách tìm chữ 'WTO' và 'Thương mại Thế giới' trong toàn bộ kho: "
            "không đoạn nào chứa. Hai câu trước ở vị trí này đều bị thay vì chính "
            "lí do đó: 'Liên Hợp Quốc thành lập năm nào?' và 'Công ước Luật Biển "
            "1982' đều có trong SGK lớp 5, nên bot trả lời (có dẫn nguồn, có nói "
            "rõ là kiến thức lớp dưới) là ĐÚNG — chỗ sai là kỳ vọng của câu hỏi. "
            "Bài học: một câu 'ngoài sách' phải được kiểm bằng tìm chữ trong kho "
            "trước khi tin, chứ không chỉ dựa vào cảm nhận chương trình học."
        ),
    ),
    EvalQuestion(
        "ChatGPT được huấn luyện bằng thuật toán nào?",
        8,
        "Tin học",
        False,
        note="Không có trong SGK Tin học 8.",
    ),
    EvalQuestion(
        "Phân tích bài thơ Tây Tiến của Quang Dũng.",
        7,
        "Ngữ văn",
        False,
        note="Tây Tiến học ở lớp 12.",
    ),
    EvalQuestion(
        "Truyện Harry Potter do ai viết?",
        9,
        "Ngữ văn",
        False,
        note="Không có trong SGK Ngữ văn 9.",
    ),
)


#: The full benchmark: 40 answerable questions and 10 that must be refused.
QUESTION_SET: tuple[EvalQuestion, ...] = (
    IN_TEXTBOOK_QUESTIONS + OUT_OF_TEXTBOOK_QUESTIONS
)
