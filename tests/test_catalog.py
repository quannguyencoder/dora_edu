"""Tests for filename -> (grade, subject, book_title) inference."""

from __future__ import annotations

from pathlib import Path

import pytest

from dora_edu.data_pipeline.catalog import BookInfo, infer_book_info, parse_filename


@pytest.mark.parametrize(
    ("stem", "grade", "expected_subject", "expected_title"),
    [
        ("SGKToan9tapmot", 9, "Toán", "Toán 9 - tập 1"),
        ("SGKToan9taphai", 9, "Toán", "Toán 9 - tập 2"),
        ("SGKNguVan9taphai", 9, "Ngữ văn", "Ngữ văn 9 - tập 2"),
        ("SGKLichsuvaDiali5", 5, "Lịch sử và Địa lí", "Lịch sử và Địa lí 5"),
        ("SGKTiengAnh10GlobalSucess", 10, "Tiếng Anh", "Tiếng Anh 10 - Global Sucess"),
        ("SGKKhoahoc4", 4, "Khoa học", "Khoa học 4"),
        ("SGKKhoahoctunhien6", 6, "Khoa học tự nhiên", "Khoa học tự nhiên 6"),
        ("SGKDaoduc1", 1, "Đạo đức", "Đạo đức 1"),
        ("SGKTunhienvaXahoi2", 2, "Tự nhiên và Xã hội", "Tự nhiên và Xã hội 2"),
        ("SGKGiaoducthechat3", 3, "Giáo dục thể chất", "Giáo dục thể chất 3"),
        ("SGKHoatdongtrainghiem5", 5, "Hoạt động trải nghiệm", "Hoạt động trải nghiệm 5"),
        (
            "SGKHoatdongtrainghiemhuongnghiep7",
            7,
            "Hoạt động trải nghiệm, hướng nghiệp",
            "Hoạt động trải nghiệm, hướng nghiệp 7",
        ),
        ("SGKAmnhac1", 1, "Âm nhạc", "Âm nhạc 1"),
        ("SGKMithuat6", 6, "Mĩ thuật", "Mĩ thuật 6"),
        (
            "SGKGiaoducKinhtevaPhapluat11",
            11,
            "Giáo dục Kinh tế và Pháp luật",
            "Giáo dục Kinh tế và Pháp luật 11",
        ),
        # Grade 10-12 electives stay under the same subject as the core book,
        # only the book title marks them as a "chuyên đề học tập".
        ("SGKChuyendehoctapToan10", 10, "Toán", "Chuyên đề học tập Toán 10"),
        (
            "SGKChuyendehoctapGiaoducKinhtevaPhapluat12",
            12,
            "Giáo dục Kinh tế và Pháp luật",
            "Chuyên đề học tập Giáo dục Kinh tế và Pháp luật 12",
        ),
        # Grade-9 Công nghệ electives: same subject/grade, distinct titles.
        (
            "SGKCongnghe9TrainghiemnghenghiepModunChebienthucpham",
            9,
            "Công nghệ",
            "Công nghệ 9 - Trainghiemnghenghiep Modun Chebienthucpham",
        ),
        ("SGKCongnghe9Dinhhuongnghenghiep", 9, "Công nghệ", "Công nghệ 9 - Dinhhuongnghenghiep"),
    ],
)
def test_parse_filename_matches_real_moet_naming(
    stem: str, grade: int, expected_subject: str, expected_title: str
) -> None:
    info = parse_filename(stem, folder_grade=grade)
    assert info == BookInfo(grade=grade, subject=expected_subject, book_title=expected_title)


def test_parse_filename_rejects_grade_mismatch() -> None:
    with pytest.raises(ValueError, match="does not match"):
        parse_filename("SGKToan9tapmot", folder_grade=5)


def test_parse_filename_rejects_unrecognised_shape() -> None:
    with pytest.raises(ValueError, match="does not match the SGK"):
        parse_filename("NotAnSgkFile", folder_grade=1)


def test_infer_book_info_reads_grade_from_parent_directory() -> None:
    info = infer_book_info(Path("data/raw_pdfs/9/SGKToan9tapmot.pdf"))
    assert info == BookInfo(grade=9, subject="Toán", book_title="Toán 9 - tập 1")


def test_infer_book_info_rejects_invalid_grade_directory() -> None:
    with pytest.raises(ValueError):
        infer_book_info(Path("data/raw_pdfs/not-a-grade/SGKToan9tapmot.pdf"))


@pytest.mark.parametrize("grade_dir", [str(g) for g in range(1, 13)])
def test_all_real_files_for_each_grade_parse_cleanly(grade_dir: str) -> None:
    """Every real PDF collected under data/raw_pdfs/<grade>/ must parse.

    Skipped entirely when the (gitignored, locally-collected) corpus is not
    present, e.g. in CI or a fresh clone.
    """
    root = Path("data/raw_pdfs") / grade_dir
    if not root.is_dir():
        pytest.skip("data/raw_pdfs is not present in this environment")
    pdfs = list(root.glob("*.pdf"))
    if not pdfs:
        pytest.skip(f"no PDFs found under {root}")
    for pdf_path in pdfs:
        info = infer_book_info(pdf_path)
        assert info.grade == int(grade_dir)
        assert info.subject
        assert info.book_title
