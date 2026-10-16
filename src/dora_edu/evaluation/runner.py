"""Run the benchmark question set against the real index and LLM.

What this measures, per question, is exactly the three things that decide
whether a tutoring bot is safe to put in front of a 12-year-old:

* **Did it answer or refuse, and was that the right call?** Answering an
  out-of-textbook question is a hallucination; refusing an in-textbook one
  is a wrong refusal. Both are counted separately, because they need
  opposite fixes.
* **Did it cite a real page?** Every page number in the answer is checked
  against the pages actually retrieved from ChromaDB. A number that matches
  none of them was invented by the LLM -- the most dangerous failure here,
  since a made-up citation looks exactly as authoritative as a real one.
* **Did it stay in scope?** Every retrieved passage must belong to the
  question's subject and to the student's grade or below.

Run it with ``dora-eval``. It costs one LLM call per question, so the
default pacing is deliberate rather than fast.
"""

from __future__ import annotations

import argparse
import csv
import logging
import re
import sys
import time
from dataclasses import dataclass
from datetime import date
from pathlib import Path

from dora_edu.config import Settings, get_settings
from dora_edu.evaluation.question_set import EXCLUDED_TOPICS, QUESTION_SET, EvalQuestion
from dora_edu.llm.generator import (
    AnswerGenerator,
    build_generator,
    is_provider_error_answer,
)
from dora_edu.llm.prompts import NO_CONTEXT_ANSWER, hide_internal_wording
from dora_edu.models import StudentProfile
from dora_edu.rag_engine.retriever import Retriever

logger = logging.getLogger(__name__)

#: Page numbers as rule 8 makes the LLM write them ("trang 79", "trang 12-13").
_PAGE_IN_ANSWER = re.compile(r"trang\s+(\d{1,3})(?:\s*[-–]\s*(\d{1,3}))?", re.IGNORECASE)

#: The opening words of the fixed refusal, which the LLM is told to reproduce
#: when the context does not cover the question. Matching on this phrase
#: (rather than on the whole message) also catches the LLM wording it itself.
_REFUSAL_MARKER = "chưa tìm thấy thông tin này trong sách"

#: Seconds between questions. The Gemini free tier limits requests per
#: minute, and a benchmark that trips the rate limit measures the quota
#: rather than the retrieval quality.
DEFAULT_DELAY_SECONDS = 4.0


@dataclass(frozen=True)
class QuestionOutcome:
    """What happened when one benchmark question was asked for real."""

    question: EvalQuestion
    answer: str
    retrieved: int
    #: Pages retrieved from ChromaDB, as ``"book, trang N"`` labels.
    retrieved_sources: tuple[str, ...]
    refused: bool
    #: Whether the provider was unreachable, so this question never really
    #: got asked and must not count as either a right or a wrong answer.
    provider_error: bool
    #: Page numbers the answer names that were not among those retrieved.
    invented_pages: tuple[int, ...]
    #: Whether the answer names at least one page.
    cites_a_page: bool
    #: Keywords from the question set that appear in the answer.
    keywords_found: tuple[str, ...]
    #: Whether every retrieved passage is the right subject and grade.
    in_scope: bool

    @property
    def handled_correctly(self) -> bool:
        """Whether answering/refusing matched what the question expects."""
        if self.provider_error:
            return False
        return self.refused is not self.question.in_textbook

    @property
    def verdict(self) -> str:
        """A short Vietnamese label for the report's rightmost column."""
        if self.provider_error:
            return "LỖI KỸ THUẬT (chưa hỏi được)"
        if self.question.in_textbook:
            if self.refused:
                return "TỪ CHỐI NHẦM"
            if self.invented_pages:
                return "TRÍCH SAI TRANG"
            if self.question.keywords and not self.keywords_found:
                return "CẦN NGƯỜI CHẤM"
            return "ĐẠT"
        return "ĐẠT (từ chối đúng)" if self.refused else "TRẢ LỜI NGOÀI SÁCH"


@dataclass(frozen=True)
class EvalReport:
    """Aggregate results of one benchmark run."""

    outcomes: tuple[QuestionOutcome, ...]

    @property
    def in_textbook(self) -> tuple[QuestionOutcome, ...]:
        """Outcomes for questions the textbook does cover."""
        return tuple(o for o in self.outcomes if o.question.in_textbook)

    @property
    def out_of_textbook(self) -> tuple[QuestionOutcome, ...]:
        """Outcomes for questions the textbook deliberately does not cover."""
        return tuple(o for o in self.outcomes if not o.question.in_textbook)

    @property
    def provider_errors(self) -> tuple[QuestionOutcome, ...]:
        """Questions that never reached the LLM, so they measure nothing."""
        return tuple(o for o in self.outcomes if o.provider_error)

    @property
    def wrong_refusals(self) -> tuple[QuestionOutcome, ...]:
        """In-textbook questions the bot declined to answer."""
        return tuple(o for o in self.in_textbook if o.refused and not o.provider_error)

    @property
    def hallucinations(self) -> tuple[QuestionOutcome, ...]:
        """Out-of-textbook questions the bot answered anyway."""
        return tuple(
            o for o in self.out_of_textbook if not o.refused and not o.provider_error
        )

    @property
    def invented_citations(self) -> tuple[QuestionOutcome, ...]:
        """Answers naming a page that was never retrieved."""
        return tuple(o for o in self.outcomes if o.invented_pages)

    @property
    def out_of_scope(self) -> tuple[QuestionOutcome, ...]:
        """Answers built from a wrong-subject or above-grade passage."""
        return tuple(o for o in self.outcomes if not o.in_scope)

    @property
    def needs_human_review(self) -> tuple[QuestionOutcome, ...]:
        """Answered questions whose expected keyword never appeared."""
        return tuple(
            o
            for o in self.in_textbook
            if not o.refused
            and not o.provider_error
            and o.question.keywords
            and not o.keywords_found
        )


def _looks_like_a_refusal(answer: str) -> bool:
    """Return whether ``answer`` is the "not in your textbook" reply."""
    return _REFUSAL_MARKER in answer.lower() or answer.strip() == NO_CONTEXT_ANSWER.strip()


def _pages_named_in(answer: str) -> set[int]:
    """Collect every page number the answer claims to cite."""
    pages: set[int] = set()
    for match in _PAGE_IN_ANSWER.finditer(answer):
        start = int(match.group(1))
        end = int(match.group(2)) if match.group(2) else start
        # A "trang 12-13" range cites both pages; a reversed or absurd range
        # is read as the single page it starts at rather than guessed at.
        pages.update(range(start, end + 1) if start <= end <= start + 20 else [start])
    return pages


def ask_one(
    question: EvalQuestion,
    retriever: Retriever,
    generator: AnswerGenerator,
) -> QuestionOutcome:
    """Ask one benchmark question through the real pipeline and score it.

    The subject is pinned from the question set rather than auto-detected,
    so the run measures retrieval and generation rather than the subject
    classifier, and costs one LLM call instead of two.

    Args:
        question: The benchmark question.
        retriever: A retriever over the real index.
        generator: The real LLM backend.

    Returns:
        What happened, scored.
    """
    profile = StudentProfile(grade=question.grade, subject=question.subject)
    chunks = retriever.retrieve(question.question, profile)
    result = generator.generate(question.question, chunks, profile)
    # The same post-processing the bot applies before a student sees the
    # answer. Without it the benchmark measures a slightly different answer
    # than the one that gets sent -- which is how "xem lại đoạn 1 trong ngữ
    # cảnh" ended up in a published report after the bot had stopped saying
    # it.
    answer = hide_internal_wording(result.answer)

    retrieved_pages: set[int] = set()
    in_scope = True
    for chunk in chunks:
        # Printed page numbers, so these compare like-for-like against the
        # pages the answer names (which come from the citations).
        start = chunk.page_start
        end = chunk.page_end
        if start is not None:
            retrieved_pages.update(range(int(start), int(end or start) + 1))
        if chunk.metadata.get("subject") != question.subject:
            in_scope = False
        if int(chunk.metadata.get("grade", 0)) > question.grade:
            in_scope = False

    named = _pages_named_in(answer)
    lowered = answer.lower()
    return QuestionOutcome(
        question=question,
        answer=answer,
        retrieved=len(chunks),
        retrieved_sources=tuple(dict.fromkeys(c.citation for c in chunks)),
        refused=_looks_like_a_refusal(answer),
        provider_error=is_provider_error_answer(answer),
        invented_pages=tuple(sorted(named - retrieved_pages)),
        cites_a_page=bool(named),
        keywords_found=tuple(k for k in question.keywords if k.lower() in lowered),
        in_scope=in_scope,
    )


def run_benchmark(
    questions: tuple[EvalQuestion, ...] = QUESTION_SET,
    *,
    settings: Settings | None = None,
    delay_seconds: float = DEFAULT_DELAY_SECONDS,
    retry_delay_seconds: float | None = None,
    retriever: Retriever | None = None,
    generator: AnswerGenerator | None = None,
) -> EvalReport:
    """Ask every question in ``questions`` and collect the results.

    Args:
        questions: The benchmark set to run.
        settings: Application settings; the process singleton when omitted.
        delay_seconds: Pause between questions, to stay inside the free
            tier's per-minute request limit.
        retry_delay_seconds: How long to wait before re-asking a question
            the provider refused; three times ``delay_seconds`` (at least
            15s) when omitted.
        retriever: Override the retriever (used by the tests).
        generator: Override the LLM backend (used by the tests).

    Returns:
        The full report.
    """
    settings = settings or get_settings()
    retriever = retriever or Retriever(settings)
    generator = generator or build_generator(settings)

    outcomes: list[QuestionOutcome] = []
    for index, question in enumerate(questions, start=1):
        logger.info(
            "[%d/%d] lớp %d %s: %s",
            index,
            len(questions),
            question.grade,
            question.subject,
            question.question,
        )
        outcome = ask_one(question, retriever, generator)
        if outcome.provider_error and delay_seconds:
            # The free tier rate-limits by the minute, so one 429 in a
            # 50-question run is routine -- and a question that never
            # reached the LLM measures nothing. One patient retry keeps the
            # run from quietly shrinking its own sample.
            retry_delay = (
                retry_delay_seconds
                if retry_delay_seconds is not None
                else max(delay_seconds * 3, 15.0)
            )
            logger.warning("    -> lỗi kỹ thuật, thử lại sau %.0fs", retry_delay)
            time.sleep(retry_delay)
            outcome = ask_one(question, retriever, generator)
        logger.info("    -> %s (%d đoạn)", outcome.verdict, outcome.retrieved)
        outcomes.append(outcome)
        if delay_seconds and index < len(questions):
            time.sleep(delay_seconds)
    return EvalReport(outcomes=tuple(outcomes))


def _percentage(part: int, whole: int) -> str:
    """Format ``part`` of ``whole`` as a percentage, or "-" when undefined."""
    return f"{part / whole:.1%}" if whole else "-"


def format_markdown(report: EvalReport) -> str:
    """Render ``report`` as the Markdown document a teacher reads.

    Args:
        report: A finished benchmark run.

    Returns:
        The full report, in Vietnamese.
    """
    # Every rate counts only the questions that actually reached the LLM: a
    # provider outage is not evidence either way, and counting it as a
    # correct refusal would flatter the result.
    in_book = tuple(o for o in report.in_textbook if not o.provider_error)
    out_book = tuple(o for o in report.out_of_textbook if not o.provider_error)
    answered = tuple(o for o in in_book if not o.refused)

    lines = [
        "# Kết quả kiểm thử DoraEdu",
        "",
        f"Ngày chạy: {date.today().isoformat()}",
        f"Số câu: {len(report.outcomes)} "
        f"({len(report.in_textbook)} câu có trong sách, "
        f"{len(report.out_of_textbook)} câu cố tình hỏi ngoài sách)",
        "",
        "## 1. Tổng hợp",
        "",
        "| Chỉ số | Kết quả |",
        "|---|---|",
        f"| Câu trong sách được trả lời | {len(answered)}/{len(in_book)} "
        f"({_percentage(len(answered), len(in_book))}) |",
        f"| Từ chối nhầm (sách có mà không trả lời) | {len(report.wrong_refusals)}/{len(in_book)} |",
        f"| Câu ngoài sách bị từ chối đúng | "
        f"{len(out_book) - len(report.hallucinations)}/{len(out_book)} |",
        f"| Trả lời câu ngoài sách (bịa đặt) | {len(report.hallucinations)}/{len(out_book)} |",
        f"| Câu trả lời có nêu số trang | "
        f"{sum(1 for o in answered if o.cites_a_page)}/{len(answered)} |",
        f"| Nêu trang không hề được tra ra | {len(report.invented_citations)} |",
        f"| Dùng đoạn sai môn hoặc vượt lớp | {len(report.out_of_scope)} |",
        f"| Cần người đọc lại để chấm đúng/sai | {len(report.needs_human_review)} |",
        "",
        "Nhóm \"cần người đọc lại\" không phải là lỗi: phần lớn là những câu bot "
        "cố tình KHÔNG nói ngay đáp số mà dẫn dắt để học sinh tự tìm (ví dụ hỏi "
        "năm diễn ra một sự kiện, bot chỉ đúng trang rồi hỏi lại) — đúng cách dạy "
        "gợi mở của dự án. Cột tự động chỉ kiểm được từ khoá, nên những câu này "
        "cần người đọc để kết luận.",
        f"| Lỗi kỹ thuật, chưa hỏi được | {len(report.provider_errors)} |",
        "",
        "Các chỉ số trên là tự động. Riêng việc câu trả lời có ĐÚNG nội dung hay "
        "không thì vẫn cần người (hoặc thầy cô) đọc lại — bảng ở mục 3 có đủ câu "
        "trả lời và trang sách để đối chiếu.",
        "",
    ]
    if report.provider_errors:
        lines += [
            f"Lưu ý: {len(report.provider_errors)} câu không gọi được API (hết lượt "
            "miễn phí trong phút đó) nên KHÔNG được tính vào các tỉ lệ ở trên — "
            "những câu đó chưa thật sự được hỏi, nên không thể coi là đúng hay sai.",
            "",
        ]
    lines += [
        "## 2. Những câu cần xem lại trước",
        "",
    ]

    problems = (
        ("Từ chối nhầm", report.wrong_refusals),
        ("Trả lời câu ngoài sách", report.hallucinations),
        ("Nêu trang không có thật", report.invented_citations),
        ("Sai môn / vượt lớp", report.out_of_scope),
        (
            "Chưa khớp từ khoá, cần người đọc lại (phần lớn là bot cố tình "
            "KHÔNG nói đáp số, để học sinh tự tìm — đúng cách dạy gợi mở, "
            "không phải lỗi)",
            report.needs_human_review,
        ),
    )
    if not any(group for _label, group in problems):
        lines.append("Không có câu nào thuộc 5 nhóm cần xem lại. 🎉")
    for label, group in problems:
        if not group:
            continue
        lines.append(f"**{label}** ({len(group)}):")
        lines.append("")
        for outcome in group:
            lines.append(
                f"- Lớp {outcome.question.grade} {outcome.question.subject}: "
                f"*{outcome.question.question}*"
                + (f" — trang bịa: {outcome.invented_pages}" if outcome.invented_pages else "")
            )
        lines.append("")

    lines += [
        "## 3. Chi tiết từng câu",
        "",
        "| # | Lớp | Môn | Câu hỏi | Đoạn tra được | Kết quả | Trang đã trích |",
        "|---|---|---|---|---|---|---|",
    ]
    for index, outcome in enumerate(report.outcomes, start=1):
        sources = "; ".join(outcome.retrieved_sources[:2]) if not outcome.refused else "-"
        lines.append(
            f"| {index} | {outcome.question.grade} | {outcome.question.subject} | "
            f"{outcome.question.question} | {outcome.retrieved} | {outcome.verdict} | "
            f"{sources} |"
        )

    lines += [
        "",
        "## 4. Ghi chú về cách chọn câu hỏi",
        "",
        "Bộ câu hỏi cố tình KHÔNG dùng những nội dung đã từng được sửa tay "
        "trong quá trình phát triển, để kết quả phản ánh đúng chất lượng chung "
        "của hệ thống chứ không phải các chỗ đã vá:",
        "",
    ]
    lines += [f"- {topic}" for topic in EXCLUDED_TOPICS]
    lines += [
        "",
        "## 5. Toàn văn câu trả lời",
        "",
    ]
    for index, outcome in enumerate(report.outcomes, start=1):
        lines += [
            f"### {index}. Lớp {outcome.question.grade} — {outcome.question.subject} — "
            f"{outcome.question.question}",
            "",
            f"*Kết quả tự động: {outcome.verdict}*"
            + (f" — {outcome.question.note}" if outcome.question.note else ""),
            "",
            outcome.answer.strip() or "(không có nội dung)",
            "",
        ]
    return "\n".join(lines)


def write_csv(report: EvalReport, path: Path) -> None:
    """Write one row per question, for a teacher to mark up in a spreadsheet.

    The ``dung_sai`` column is left empty on purpose: it is the column a
    person fills in after reading the answer.

    Args:
        report: A finished benchmark run.
        path: Destination CSV file.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "stt",
                "lop",
                "mon",
                "cau_hoi",
                "co_trong_sach",
                "da_tra_loi",
                "ket_qua_tu_dong",
                "so_doan_tra_duoc",
                "trang_da_trich",
                "trang_bia_dat",
                "dung_sai_nguoi_cham",
                "cau_tra_loi",
            ]
        )
        for index, outcome in enumerate(report.outcomes, start=1):
            writer.writerow(
                [
                    index,
                    outcome.question.grade,
                    outcome.question.subject,
                    outcome.question.question,
                    "có" if outcome.question.in_textbook else "không",
                    "không" if outcome.refused else "có",
                    outcome.verdict,
                    outcome.retrieved,
                    "; ".join(outcome.retrieved_sources[:3]),
                    "; ".join(str(p) for p in outcome.invented_pages),
                    "",
                    outcome.answer.replace("\n", " ").strip(),
                ]
            )


def main(argv: list[str] | None = None) -> int:
    """Entry point for ``dora-eval``.

    Returns:
        ``0`` when the run finished, ``1`` when it could not start.
    """
    parser = argparse.ArgumentParser(
        description="Chạy bộ câu hỏi kiểm thử DoraEdu trên kho sách thật."
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("./data/eval"),
        help="Thư mục lưu báo cáo (mặc định: ./data/eval)",
    )
    parser.add_argument(
        "--limit", type=int, default=0, help="Chỉ chạy N câu đầu (0 = chạy hết)"
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=DEFAULT_DELAY_SECONDS,
        help=f"Giây nghỉ giữa 2 câu (mặc định: {DEFAULT_DELAY_SECONDS})",
    )
    args = parser.parse_args(argv)

    settings = get_settings()
    logging.basicConfig(
        level=getattr(logging, settings.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)-8s %(message)s",
    )
    # The pipeline's own per-question logging would bury the benchmark's.
    logging.getLogger("dora_edu.rag_engine.retriever").setLevel(logging.WARNING)

    questions = QUESTION_SET[: args.limit] if args.limit else QUESTION_SET
    try:
        report = run_benchmark(questions, settings=settings, delay_seconds=args.delay)
    except (ValueError, RuntimeError) as exc:
        logger.error("Không chạy được bộ kiểm thử: %s", exc)
        logger.error("Kiểm tra lại .env và chạy 'dora-bulk-ingest' trước.")
        return 1

    stamp = date.today().isoformat()
    markdown_path = args.out / f"ket-qua-kiem-thu-{stamp}.md"
    csv_path = args.out / f"ket-qua-kiem-thu-{stamp}.csv"
    markdown_path.parent.mkdir(parents=True, exist_ok=True)
    markdown_path.write_text(format_markdown(report), encoding="utf-8")
    write_csv(report, csv_path)

    logger.info("Đã lưu báo cáo: %s", markdown_path)
    logger.info("Đã lưu bảng chấm: %s", csv_path)
    logger.info(
        "Tóm tắt: %d/%d câu trong sách trả lời được, %d từ chối nhầm, "
        "%d câu ngoài sách bị trả lời, %d trang bịa đặt",
        len(report.in_textbook) - len(report.wrong_refusals),
        len(report.in_textbook),
        len(report.wrong_refusals),
        len(report.hallucinations),
        len(report.invented_citations),
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
