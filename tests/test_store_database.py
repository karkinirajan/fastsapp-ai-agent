"""The conversation store against a real database engine (SQLite in memory), not a MagicMock."""

from datetime import UTC, datetime, timedelta

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import Session
from sqlalchemy.pool import StaticPool

from app.models import Base, Conversation
from app.services import conversation_store, sentiment_analyzer

A, B = "whatsapp:+85291230001", "whatsapp:+85291230002"


@pytest.fixture
def db():
    engine = create_engine(
        "sqlite://", poolclass=StaticPool, connect_args={"check_same_thread": False}
    )
    Base.metadata.create_all(engine)
    with Session(engine) as session:
        yield session
    engine.dispose()  # StaticPool keeps its one connection open until told otherwise


def add(db, phone, n, at=None):
    db.add(
        Conversation(
            phone_number=phone,
            user_message=f"q{n}",
            bot_response=f"a{n}",
            created_at=at or datetime.now(UTC),
        )
    )
    db.commit()


def test_history_is_oldest_first_as_alternating_user_and_assistant_turns(db):
    base = datetime(2026, 1, 1, tzinfo=UTC)
    for n in (1, 2, 3):
        add(db, A, n, base + timedelta(minutes=n))
    assert conversation_store.get_history(db, A, limit=10) == [
        {"role": "user", "content": "q1"}, {"role": "assistant", "content": "a1"},
        {"role": "user", "content": "q2"}, {"role": "assistant", "content": "a2"},
        {"role": "user", "content": "q3"}, {"role": "assistant", "content": "a3"},
    ]  # fmt: skip


def test_the_limit_keeps_the_most_recent_exchanges(db):
    base = datetime(2026, 1, 1, tzinfo=UTC)
    for n in range(1, 6):
        add(db, A, n, base + timedelta(minutes=n))
    history = conversation_store.get_history(db, A, limit=2)
    assert [m["content"] for m in history] == ["q4", "a4", "q5", "a5"]


def test_exchanges_with_the_same_timestamp_keep_their_order(db):
    same = datetime(2026, 1, 1, tzinfo=UTC)
    for n in (1, 2, 3):
        add(db, A, n, same)
    assert [m["content"] for m in conversation_store.get_history(db, A, limit=10)][::2] == [
        "q1",
        "q2",
        "q3",
    ]
    assert [m["content"] for m in conversation_store.get_history(db, A, limit=1)] == [
        "q3",
        "a3",
    ]  # limit counts exchanges


def test_one_senders_history_never_contains_anothers(db):
    add(db, A, 1)
    add(db, B, 2)
    assert [m["content"] for m in conversation_store.get_history(db, A)] == ["q1", "a1"]
    assert conversation_store.get_history(db, "whatsapp:+85200000000") == []
    assert conversation_store.count_exchanges(db, A) == 1


def test_an_exchange_is_saved_with_its_analysis(db):
    record = conversation_store.save_exchange(
        db, A, "我想預約", "請提供資料", intent="appointment", sentiment="neutral", sentiment_score=0.2,
        entities={"name": "陳大文", "date": "2026-12-01"}, escalated=False,
    )  # fmt: skip
    assert record.id is not None and record.created_at is not None
    saved = db.get(Conversation, record.id)
    assert (saved.intent, saved.sentiment_score, saved.escalated) == ("appointment", 0.2, False)
    assert saved.entities == {"name": "陳大文", "date": "2026-12-01"}  # the JSON column round-trips
    assert saved.user_message == "我想預約"


def test_an_exchange_without_entities_stores_null(db):
    record = conversation_store.save_exchange(db, A, "hi", "hello")
    assert db.get(Conversation, record.id).entities is None


def test_escalation_flags_the_stored_conversation_once(db):
    record = conversation_store.save_exchange(db, A, "this is terrible", "sorry", escalated=False)
    sentiment_analyzer.handle_escalation(db, record.id, A, -0.9)
    db.expire_all()
    assert db.get(Conversation, record.id).escalated is True
    sentiment_analyzer.handle_escalation(db, record.id, A, -0.9)  # again: harmless
    assert db.get(Conversation, record.id).escalated is True


def test_escalating_a_missing_conversation_does_not_raise(db):
    sentiment_analyzer.handle_escalation(db, 999, A, -0.9)
