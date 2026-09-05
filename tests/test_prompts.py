"""Tests for the anti-hallucination and Socratic prompt guardrails."""

from __future__ import annotations

from dora_edu.llm import prompts
from dora_edu.models import RetrievedChunk, StudentProfile


def _chunk(text: str = "Phan so gom tu so va mau so.") -> RetrievedChunk:
    return RetrievedChunk(
        text=text,
        distance=0.1,
        metadata={"book_title": "SGK Toán 6", "page_start": 12, "page_end": 12},
    )


def test_system_prompt_is_scoped_to_the_student_grade_and_subject() -> None:
    prompt = prompts.build_system_prompt(StudentProfile(grade=6, subject="Toán"))

    assert "LỚP 6" in prompt
    assert "Toán" in prompt


def test_system_prompt_forbids_outside_knowledge() -> None:
    prompt = prompts.build_system_prompt(StudentProfile(grade=6, subject="Toán"))

    assert "CHỈ được dùng thông tin" in prompt
    assert "KHÔNG dùng kiến thức có sẵn" in prompt
    assert "không được bịa" in prompt


def test_system_prompt_mandates_the_textbook_refusal_sentence() -> None:
    prompt = prompts.build_system_prompt(StudentProfile(grade=8, subject="Vật lí"))

    assert "chưa tìm thấy thông tin này trong sách giáo khoa" in prompt


def test_system_prompt_requires_socratic_guidance_over_final_answers() -> None:
    prompt = prompts.build_system_prompt(StudentProfile(grade=6, subject="Toán"))

    assert "gợi mở" in prompt
    assert "KHÔNG đưa ngay lời giải hoàn chỉnh" in prompt
    assert "chép bài" in prompt


def test_refusal_message_is_vietnamese_and_mentions_the_textbook() -> None:
    assert "sách giáo khoa" in prompts.NO_CONTEXT_ANSWER


def test_format_context_numbers_passages_and_keeps_their_citations() -> None:
    context = prompts.format_context([_chunk(), _chunk("Doan hai.")])

    assert "[Đoạn 1]" in context
    assert "[Đoạn 2]" in context
    assert "SGK Toán 6, trang 12" in context


def test_format_context_is_empty_when_nothing_was_retrieved() -> None:
    assert prompts.format_context([]) == ""


def test_user_prompt_embeds_both_the_context_and_the_question() -> None:
    prompt = prompts.build_user_prompt(
        "Phan so la gi?", [_chunk()], StudentProfile(grade=6, subject="Toán")
    )

    assert "Phan so gom tu so va mau so." in prompt
    assert "Phan so la gi?" in prompt


def test_build_messages_orders_system_history_then_question() -> None:
    history = [
        {"role": "user", "content": "Cau hoi cu"},
        {"role": "assistant", "content": "Tra loi cu"},
    ]
    messages = prompts.build_messages(
        "Cau hoi moi", [_chunk()], StudentProfile(grade=6, subject="Toán"), history
    )

    assert [m["role"] for m in messages] == ["system", "user", "assistant", "user"]
    assert "Cau hoi moi" in messages[-1]["content"]
