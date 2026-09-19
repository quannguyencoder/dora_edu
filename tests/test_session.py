"""Tests for the sliding-window session store."""

from __future__ import annotations

from pathlib import Path

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


# --- Persistence across a restart (db_path configured) ----------------------


def test_without_a_db_path_nothing_survives_a_new_store(tmp_path: Path) -> None:
    # The default, in-memory-only behaviour every other test above relies on.
    store = SessionStore(max_turns=3)
    store.get(("telegram", "1")).grade = 8
    store.save_profile(("telegram", "1"))

    fresh_store = SessionStore(max_turns=3)
    assert fresh_store.get(("telegram", "1")).grade is None


def test_save_profile_survives_a_new_store_with_the_same_db_path(tmp_path: Path) -> None:
    db_path = tmp_path / "sessions.db"
    store = SessionStore(max_turns=3, db_path=db_path)
    session = store.get(("discord", "1"))
    session.grade = 8
    session.subject = "Toán"
    store.save_profile(("discord", "1"))

    restarted_store = SessionStore(max_turns=3, db_path=db_path)
    restored = restarted_store.get(("discord", "1"))

    assert restored.grade == 8
    assert restored.subject == "Toán"


def test_save_profile_before_the_key_has_a_session_is_a_no_op(tmp_path: Path) -> None:
    db_path = tmp_path / "sessions.db"
    store = SessionStore(max_turns=3, db_path=db_path)

    store.save_profile(("discord", "never-seen"))  # must not raise

    restarted = SessionStore(max_turns=3, db_path=db_path)
    assert restarted.get(("discord", "never-seen")).grade is None


def test_reset_also_forgets_the_persisted_profile(tmp_path: Path) -> None:
    db_path = tmp_path / "sessions.db"
    store = SessionStore(max_turns=3, db_path=db_path)
    store.get(("discord", "1")).grade = 8
    store.save_profile(("discord", "1"))

    store.reset(("discord", "1"))

    restarted = SessionStore(max_turns=3, db_path=db_path)
    assert restarted.get(("discord", "1")).grade is None


def test_saving_again_updates_the_persisted_profile(tmp_path: Path) -> None:
    db_path = tmp_path / "sessions.db"
    store = SessionStore(max_turns=3, db_path=db_path)
    session = store.get(("discord", "1"))
    session.grade = 6
    store.save_profile(("discord", "1"))

    session.grade = 9
    store.save_profile(("discord", "1"))

    restarted = SessionStore(max_turns=3, db_path=db_path)
    assert restarted.get(("discord", "1")).grade == 9
