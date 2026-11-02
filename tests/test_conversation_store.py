"""Tests for app/services/conversation_store.py (Feature 1 — per-user memory)"""

from datetime import UTC, datetime
from unittest.mock import MagicMock


def _make_record(
    id_: int,
    phone: str,
    user_msg: str,
    bot_resp: str,
    intent: str = "faq",
    created_at: datetime | None = None,
):
    r = MagicMock()
    r.id = id_
    r.phone_number = phone
    r.user_message = user_msg
    r.bot_response = bot_resp
    r.intent = intent
    r.created_at = created_at or datetime.now(UTC)
    return r


def test_get_history_returns_openai_format(mock_db):
    """get_history returns interleaved user/assistant dicts, oldest first."""
    from app.services.conversation_store import get_history

    records = [
        _make_record(2, "+852", "second", "ans2"),
        _make_record(1, "+852", "first", "ans1"),
    ]
    # DB query returns newest-first (limit desc), so simulate that
    mock_db.query.return_value.filter.return_value.order_by.return_value.limit.return_value.all.return_value = records

    history = get_history(mock_db, "+852", limit=10)

    # reversed() in get_history makes oldest first → first should be record[1]
    assert len(history) == 4
    assert history[0] == {"role": "user", "content": "first"}
    assert history[1] == {"role": "assistant", "content": "ans1"}
    assert history[2] == {"role": "user", "content": "second"}
    assert history[3] == {"role": "assistant", "content": "ans2"}


def test_get_history_empty_returns_empty_list(mock_db):
    mock_db.query.return_value.filter.return_value.order_by.return_value.limit.return_value.all.return_value = []

    from app.services.conversation_store import get_history

    assert get_history(mock_db, "+852") == []


def test_save_exchange_commits_and_returns_record(mock_db):
    """save_exchange adds the record, commits, and returns it."""
    from app.services.conversation_store import save_exchange

    # db.refresh just sets id on the record — simulate that
    def fake_refresh(obj):
        obj.id = 42

    mock_db.refresh.side_effect = fake_refresh

    result = save_exchange(
        db=mock_db,
        phone_number="whatsapp:+852",
        user_message="hello",
        bot_response="hi",
        intent="greeting",
        sentiment="positive",
        sentiment_score=0.9,
        entities={"name": "Alice"},
        escalated=False,
    )

    mock_db.add.assert_called_once()
    mock_db.commit.assert_called_once()
    assert result.id == 42


def test_count_exchanges(mock_db):
    mock_db.query.return_value.filter.return_value.count.return_value = 7
    from app.services.conversation_store import count_exchanges

    assert count_exchanges(mock_db, "+852") == 7
