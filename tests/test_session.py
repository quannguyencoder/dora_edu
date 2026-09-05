"""Tests for the sliding-window session store."""

from __future__ import annotations

from dora_edu.bot.session import Session, SessionStore


def test_profile_is_none_until_both_grade_and_subject_are_set() -> None:
    session = Session(max_turns=3)
    assert session.profile is None

    session.grade = 6
    assert session.profile is None

    session.subject = "Toán"
    assert session.has_profile
    assert session.profile.grade == 6
    assert session.profile.subject == "Toán"


def test_profile_canonicalises_subject_aliases() -> None:
    session = Session(max_turns=3, grade=6, subject="toan")
    assert session.profile.subject == "Toán"


def test_history_keeps_only_the_most_recent_turns() -> None:
    session = Session(max_turns=2)
    for i in range(5):
        session.record_turn(f"hoi {i}", f"dap {i}")

    history = session.history()
    assert len(history) == 4  # 2 turns * (question + answer)
    assert history[0]["content"] == "hoi 3"
    assert history[-1]["content"] == "dap 4"


def test_history_alternates_user_and_assistant_roles() -> None:
    session = Session(max_turns=3)
    session.record_turn("hoi", "dap")

    assert [m["role"] for m in session.history()] == ["user", "assistant"]


def test_clear_history_keeps_the_profile() -> None:
    session = Session(max_turns=3, grade=6, subject="Toán")
    session.record_turn("hoi", "dap")
    session.clear_history()

    assert session.history() == []
    assert session.has_profile


def test_store_isolates_the_same_user_id_across_channels() -> None:
    store = SessionStore(max_turns=3)
    telegram = store.get(("telegram", "123"))
    zalo = store.get(("zalo", "123"))

    telegram.grade = 6
    assert zalo.grade is None
    assert len(store) == 2


def test_store_returns_the_same_session_for_the_same_key() -> None:
    store = SessionStore(max_turns=3)
    assert store.get(("telegram", "1")) is store.get(("telegram", "1"))


def test_reset_forgets_the_profile_too() -> None:
    store = SessionStore(max_turns=3)
    store.get(("telegram", "1")).grade = 6
    store.reset(("telegram", "1"))

    assert store.get(("telegram", "1")).grade is None
