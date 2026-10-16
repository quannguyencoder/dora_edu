"""Turn collected student feedback into something a teacher can review.

The classroom trial produces two kinds of record: answers a student flagged
with "Báo sai", and 1-5 ratings of how clear and how genuinely guiding the
answers were. Neither is useful sitting in a SQLite file, so this renders
both as a Markdown review sheet (what went wrong, with the textbook page to
check) and a CSV the teacher can mark up.

Run it with ``dora-feedback``.
"""

from __future__ import annotations

import argparse
import csv
import logging
import sys
from datetime import date
from pathlib import Path

from dora_edu.bot.feedback import FeedbackStore, ReportRecord
from dora_edu.config import get_settings

logger = logging.getLogger(__name__)


def format_markdown(store: FeedbackStore, *, limit: int = 50) -> str:
    """Render the review sheet.

    Args:
        store: The feedback database to read.
        limit: Maximum flagged answers to list in full.

    Returns:
        The review sheet, in Vietnamese.
    """
    summary = store.summary()
    reports = store.recent_reports(limit=limit)

    def _score(value: float | None) -> str:
        return f"{value:.2f}/5" if value is not None else "chưa có"

    lines = [
        "# Phiếu rà soát câu trả lời DoraEdu",
        "",
        f"Ngày xuất: {date.today().isoformat()}",
        "",
        "## 1. Tổng hợp",
        "",
        "| Chỉ số | Kết quả |",
        "|---|---|",
        f"| Số câu bot đã trả lời | {summary.answers} |",
        f"| Trong đó trả lời được từ sách | {summary.grounded_answers} |",
        f"| Trong đó trả lời \"chưa tìm thấy trong sách\" | {summary.refusals} |",
        f"| Số lần học sinh bấm \"Báo sai\" | {summary.reports} |",
        f"| Số học sinh đã báo sai | {summary.reporters} |",
        f"| Tỉ lệ câu bị báo sai | {summary.report_rate:.1%} |",
        f"| Số lượt chấm điểm | {summary.ratings} (từ {summary.raters} bạn) |",
        f"| Điểm trung bình \"dễ hiểu\" | {_score(summary.average_clarity)} |",
        f"| Điểm trung bình \"gợi mở giúp tự nghĩ ra\" | {_score(summary.average_guidance)} |",
        "",
        "## 2. Các câu học sinh báo sai",
        "",
    ]

    if not reports:
        lines.append("Chưa có câu nào bị báo sai.")
    for report in reports:
        lines.extend(_format_report(report))

    lines += [
        "## 3. Việc cần làm sau khi rà soát",
        "",
        "Với mỗi câu ở mục 2, thầy cô mở đúng trang sách đã ghi rồi kết luận:",
        "",
        "1. **Bot trả lời sai nội dung** — ghi lại câu sai và câu đúng theo sách, "
        "rồi thêm vào `TEXT_FIXES`/`PATCHES` trong "
        "`src/dora_edu/data_pipeline/manual_patches.py`.",
        "2. **Bot trích sai trang** — kiểm tra lại độ lệch số trang của sách đó "
        "bằng `dora-scan-page-offsets`.",
        "3. **Sách có nội dung nhưng bot nói không tìm thấy** — ghi lại đúng từ "
        "ngữ trong sách, rồi bổ sung vào bộ câu hỏi kiểm thử "
        "(`src/dora_edu/evaluation/question_set.py`) để lần sau đo được.",
        "4. **Bot trả lời đúng, học sinh hiểu nhầm** — không sửa dữ liệu; cân nhắc "
        "diễn đạt lại cho dễ hiểu hơn.",
        "",
        "Mỗi lần rút ra một quy tắc chung (ví dụ: \"luôn nêu rõ đây là kiến thức "
        "lớp dưới\"), hãy thêm quy tắc đó vào `src/dora_edu/llm/prompts.py` để lỗi "
        "không lặp lại ở câu khác.",
    ]
    return "\n".join(lines)


def _format_report(report: ReportRecord) -> list[str]:
    """Render one flagged answer as a review entry."""
    answer = report.answer
    pages = "; ".join(answer.citations) if answer.citations else "(không trích trang nào)"
    return [
        f"### Báo sai #{report.report_id} — lớp {answer.grade}, môn {answer.subject}",
        "",
        f"- **Thời điểm:** {report.created_at}",
        f"- **Câu hỏi:** {answer.question}",
        f"- **Trang sách cần mở đối chiếu:** {pages}",
        f"- **Bot đã trả lời được từ sách:** {'có' if answer.grounded else 'không'}",
        "",
        "Câu trả lời của bot:",
        "",
        "> " + answer.answer.replace("\n", "\n> "),
        "",
    ]


def write_csv(store: FeedbackStore, path: Path, *, limit: int = 500) -> None:
    """Write one row per flagged answer, with an empty column for the verdict.

    Args:
        store: The feedback database to read.
        path: Destination CSV file.
        limit: Maximum reports to export.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "bao_sai_id",
                "thoi_diem",
                "lop",
                "mon",
                "cau_hoi",
                "cau_tra_loi",
                "trang_sach",
                "tra_loi_duoc_tu_sach",
                "ket_luan_cua_thay_co",
                "noi_dung_dung_theo_sach",
            ]
        )
        for report in store.recent_reports(limit=limit):
            answer = report.answer
            writer.writerow(
                [
                    report.report_id,
                    report.created_at,
                    answer.grade,
                    answer.subject,
                    answer.question,
                    answer.answer.replace("\n", " ").strip(),
                    "; ".join(answer.citations),
                    "có" if answer.grounded else "không",
                    "",
                    "",
                ]
            )


def main(argv: list[str] | None = None) -> int:
    """Entry point for ``dora-feedback``.

    Returns:
        ``0`` when the report was produced, ``1`` when the database is missing.
    """
    parser = argparse.ArgumentParser(
        description="Xuất phiếu rà soát từ phản hồi của học sinh (Báo sai, chấm điểm)."
    )
    parser.add_argument(
        "--db", type=Path, default=None, help="File feedback.db (mặc định: theo .env)"
    )
    parser.add_argument(
        "--out", type=Path, default=Path("./data/feedback"), help="Thư mục lưu báo cáo"
    )
    parser.add_argument("--limit", type=int, default=50, help="Số câu báo sai liệt kê đầy đủ")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)-8s %(message)s")

    db_path = args.db or get_settings().feedback_db_path
    if not db_path.exists():
        logger.error(
            "Chưa có file phản hồi %s — chạy bot và để học sinh dùng thử trước đã nhé.",
            db_path,
        )
        return 1

    store = FeedbackStore(db_path=db_path)
    try:
        stamp = date.today().isoformat()
        markdown_path = args.out / f"phieu-ra-soat-{stamp}.md"
        csv_path = args.out / f"phieu-ra-soat-{stamp}.csv"
        markdown_path.parent.mkdir(parents=True, exist_ok=True)
        markdown_path.write_text(format_markdown(store, limit=args.limit), encoding="utf-8")
        write_csv(store, csv_path)
        summary = store.summary()
    finally:
        store.close()

    logger.info("Đã lưu phiếu rà soát: %s", markdown_path)
    logger.info("Đã lưu bảng chấm: %s", csv_path)
    logger.info(
        "Tổng: %d câu trả lời, %d lần báo sai, %d lượt chấm điểm",
        summary.answers,
        summary.reports,
        summary.ratings,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
