"""Tests for the channel-independent tutoring logic."""

from __future__ import annotations

import pytest

from dora_edu.bot.adapter import IncomingMessage
from dora_edu.bot.core import TutorService, parse_command
from dora_edu.bot.session import SessionStore
from dora_edu.llm import prompts
from dora_edu.rag_engine import retriever as retriever_module
from dora_edu.rag_engine.retriever import Retriever
from tests.conftest import FakeCollection, FakeGenerator


@pytest.fixture
def collection(textbook_rows) -> FakeCollection:
    return FakeCollection(textbook_rows)


@pytest.fixture
def generator() -> FakeGenerator:
    return FakeGenerator()


@pytest.fixture
def tutor(monkeypatch, collection, generator, settings) -> TutorService:
    monkeypatch.setattr(retriever_module, "get_collection", lambda *a, **k: collection)
    return TutorService(
        retriever=Retriever(settings),
        generator=generator,
        sessions=SessionStore(max_turns=2),
        settings=settings,
    )


def _say(text: str, user_id: str = "1", channel: str = "telegram") -> IncomingMessage:
    return IncomingMessage(channel=channel, user_id=user_id, text=text)


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("/lop 6", ("lop", "6")),
        ("/LOP 6", ("lop", "6")),
        ("/lop@DoraEduBot 6", ("lop", "6")),
        ("/trogiup", ("trogiup", "")),
        # A stray space right after the slash is a common typo on mobile
        # keyboards and must still be recognised as a command.
        ("/ lop 8", ("lop", "8")),
        ("/  mon toan", ("mon", "toan")),
        ("/ trogiup", ("trogiup", "")),
        ("Phan so la gi?", None),
        ("", None),
    ],
)
def test_parse_command_handles_platform_decorations(text, expected) -> None:
    assert parse_command(text) == expected


def test_question_without_a_profile_never_reaches_the_retriever(tutor, collection) -> None:
    reply = tutor.handle(_say("Phan so la gi?"))

    assert reply.text == prompts.PROFILE_REQUIRED_MESSAGE
    assert collection.queries == []


def test_setting_grade_then_subject_completes_the_profile(tutor, collection) -> None:
    tutor.handle(_say("/lop 6"))
    tutor.handle(_say("/mon toan"))
    tutor.handle(_say("Phan so la gi?"))

    where = collection.queries[0]["where"]
    assert {"grade": {"$eq": 6}} in where["$and"]
    assert {"subject": {"$eq": "Toán"}} in where["$and"]


def test_setting_subject_then_grade_works_in_either_order(tutor, collection) -> None:
    tutor.handle(_say("/mon Lich su"))
    tutor.handle(_say("/lop 6"))
    tutor.handle(_say("Bach Dang o dau?"))

    assert {"subject": {"$eq": "Lịch sử"}} in collection.queries[0]["where"]["$and"]


def test_answer_is_generated_only_from_the_students_own_grade(tutor, generator) -> None:
    tutor.handle(_say("/lop 6"))
    tutor.handle(_say("/mon toan"))
    tutor.handle(_say("Toan hoc"))

    chunks = generator.calls[0]["chunks"]
    assert chunks
    assert all(chunk.metadata["grade"] == 6 for chunk in chunks)


def test_changing_grade_clears_the_previous_conversation(tutor, generator) -> None:
    tutor.handle(_say("/lop 6"))
    tutor.handle(_say("/mon toan"))
    tutor.handle(_say("Cau hoi lop 6"))
    tutor.handle(_say("/lop 9"))
    tutor.handle(_say("Cau hoi lop 9"))

    assert generator.calls[-1]["history"] == []
    assert generator.calls[-1]["profile"].grade == 9


def test_history_is_passed_to_the_generator_across_turns(tutor, generator) -> None:
    tutor.handle(_say("/lop 6"))
    tutor.handle(_say("/mon toan"))
    tutor.handle(_say("Cau hoi mot"))
    tutor.handle(_say("Cau hoi hai"))

    assert generator.calls[1]["history"][0]["content"] == "Cau hoi mot"


def test_students_on_different_channels_do_not_share_state(tutor, collection) -> None:
    tutor.handle(_say("/lop 6", user_id="7", channel="telegram"))
    tutor.handle(_say("/mon toan", user_id="7", channel="telegram"))
    reply = tutor.handle(_say("Phan so la gi?", user_id="7", channel="zalo"))

    assert reply.text == prompts.PROFILE_REQUIRED_MESSAGE


@pytest.mark.parametrize("text", ["/lop 0", "/lop 13", "/lop abc"])
def test_invalid_grade_is_refused_in_vietnamese(tutor, text) -> None:
    reply = tutor.handle(_say(text))

    assert "1 đến 12" in reply.text


def test_start_and_help_do_not_require_a_profile(tutor) -> None:
    assert tutor.handle(_say("/start")).text == prompts.WELCOME_MESSAGE
    assert tutor.handle(_say("/trogiup")).text == prompts.HELP_MESSAGE


def test_unknown_command_points_the_student_at_help(tutor) -> None:
    assert "/trogiup" in tutor.handle(_say("/khongton tai")).text


def test_profile_command_reports_the_current_scope(tutor) -> None:
    tutor.handle(_say("/lop 8"))
    tutor.handle(_say("/mon vat li"))

    reply = tutor.handle(_say("/toi"))
    assert "lớp 8" in reply.text
    assert "Vật lí" in reply.text


def test_retrieval_failure_is_reported_without_leaking_internals(tutor, collection) -> None:
    def explode(**kwargs):
        raise RuntimeError("chroma exploded")

    collection.query = explode
    tutor.handle(_say("/lop 6"))
    tutor.handle(_say("/mon toan"))
    reply = tutor.handle(_say("Phan so la gi?"))

    assert "chroma" not in reply.text.lower()
    assert "trục trặc" in reply.text


def test_tutor_service_is_callable_as_a_message_handler(tutor) -> None:
    assert tutor(_say("/start")).text == prompts.WELCOME_MESSAGE


# --- Subject auto-detection (no /mon set) ----------------------------------


def test_a_question_with_only_grade_set_still_gets_answered(tutor, generator) -> None:
    tutor.handle(_say("/lop 6"))
    reply = tutor.handle(_say("Phan so la gi?"))

    assert reply.text == "Cau tra loi mau"
    assert generator.calls[0]["profile"].grade == 6
    assert generator.calls[0]["profile"].subject == "Toán"


def test_auto_detected_queries_still_carry_both_filters(tutor, collection) -> None:
    tutor.handle(_say("/lop 6"))
    tutor.handle(_say("Phan so la gi?"))

    for call in collection.queries:
        fields = {list(clause)[0] for clause in call["where"]["$and"]}
        assert fields == {"grade", "subject"}


def test_an_explicit_mon_still_overrides_auto_detection(tutor, generator) -> None:
    tutor.handle(_say("/lop 6"))
    tutor.handle(_say("/mon Lich su"))
    tutor.handle(_say("Bach Dang o dau?"))

    assert generator.calls[0]["profile"].subject == "Lịch sử"


def test_setting_only_grade_mentions_auto_detection(tutor) -> None:
    reply = tutor.handle(_say("/lop 6"))

    assert "tự nhận diện" in reply.text or "/mon" in reply.text


def test_profile_command_without_a_subject_mentions_auto_detection(tutor) -> None:
    tutor.handle(_say("/lop 6"))
    reply = tutor.handle(_say("/toi"))

    assert "lớp 6" in reply.text
    assert "tự nhận diện" in reply.text


def test_no_subject_detected_gets_a_dedicated_message(monkeypatch, settings, generator) -> None:
    from dora_edu.bot.session import SessionStore
    from dora_edu.rag_engine import retriever as retriever_module
    from dora_edu.rag_engine.retriever import Retriever

    far = [
        {
            "text": "Khong lien quan.",
            "distance": 1.9,
            "metadata": {"grade": 6, "subject": "Toán", "book_title": "SGK", "page_start": 1},
        }
    ]
    monkeypatch.setattr(
        retriever_module, "get_collection", lambda *a, **k: FakeCollection(far)
    )
    tutor = TutorService(
        retriever=Retriever(settings),
        generator=generator,
        sessions=SessionStore(max_turns=2),
        settings=settings,
    )

    tutor.handle(_say("/lop 6"))
    reply = tutor.handle(_say("Cau hoi khong lien quan gi ca"))

    assert reply.text == prompts.SUBJECT_NOT_DETECTED_MESSAGE
    assert generator.calls == []


def test_auto_detect_retrieval_failure_is_reported_without_leaking_internals(
    tutor, collection
) -> None:
    def explode(**kwargs):
        raise RuntimeError("chroma exploded")

    collection.query = explode
    tutor.handle(_say("/lop 6"))
    reply = tutor.handle(_say("Phan so la gi?"))

    assert "chroma" not in reply.text.lower()
    assert "trục trặc" in reply.text


def test_llm_classification_is_tried_before_the_distance_fallback(monkeypatch, settings) -> None:
    from dora_edu.bot.session import SessionStore
    from dora_edu.rag_engine import retriever as retriever_module
    from dora_edu.rag_engine.retriever import Retriever

    # Distances alone would pick "Toán" (see textbook_rows), but the fake
    # classifier is told to prefer "Lịch sử" -- its answer must win.
    rows = [
        {
            "text": "Phan so lop 6: tu so va mau so.",
            "distance": 0.10,
            "metadata": {"grade": 6, "subject": "Toán", "book_title": "SGK Toán 6", "page_start": 12},
        },
        {
            "text": "Chien thang Bach Dang nam 938.",
            "distance": 0.12,
            "metadata": {
                "grade": 6, "subject": "Lịch sử", "book_title": "SGK Lịch sử 6", "page_start": 44,
            },
        },
    ]
    monkeypatch.setattr(retriever_module, "get_collection", lambda *a, **k: FakeCollection(rows))
    generator = FakeGenerator(classify_subject_response="Lịch sử")
    tutor = TutorService(
        retriever=Retriever(settings),
        generator=generator,
        sessions=SessionStore(max_turns=2),
        settings=settings,
    )

    tutor.handle(_say("/lop 6"))
    tutor.handle(_say("Bach Dang o dau?"))

    assert generator.calls[0]["profile"].subject == "Lịch sử"
    assert generator.classify_calls[0]["subjects"] == ["Lịch sử", "Toán"]


def test_falls_back_to_distance_when_the_llm_classification_is_inconclusive(
    tutor, generator
) -> None:
    # FakeGenerator.classify_subject returns None by default (as if the LLM
    # call failed or gave an ambiguous reply); the bot must still answer.
    tutor.handle(_say("/lop 6"))
    tutor.handle(_say("Phan so la gi?"))

    assert generator.calls[0]["profile"].subject == "Toán"


def test_a_meta_question_about_the_bot_lists_subjects_without_touching_retrieval(
    monkeypatch, settings, collection
) -> None:
    from dora_edu.llm.generator import NOT_SUBJECT_SPECIFIC

    monkeypatch.setattr(retriever_module, "get_collection", lambda *a, **k: collection)
    generator = FakeGenerator(classify_subject_response=NOT_SUBJECT_SPECIFIC)
    tutor = TutorService(
        retriever=Retriever(settings),
        generator=generator,
        sessions=SessionStore(max_turns=2),
        settings=settings,
    )

    tutor.handle(_say("/lop 6"))
    reply = tutor.handle(_say("Ban co the ho tro nhung mon nao"))

    assert "Toán" in reply.text
    assert "Lịch sử" in reply.text
    assert collection.queries == []
    assert generator.calls == []
