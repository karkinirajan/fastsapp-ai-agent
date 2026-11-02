"""Tests for app/services/sentiment_analyzer.py (Feature 4 — escalation)"""

from unittest.mock import MagicMock


def test_escalation_footer_added_when_required():
    from app.services.sentiment_analyzer import maybe_add_escalation_footer

    result = maybe_add_escalation_footer("Regular response.", requires_escalation=True)
    assert "2782 2202" in result  # contact number from footer


def test_escalation_footer_not_added_when_not_required():
    from app.services.sentiment_analyzer import maybe_add_escalation_footer

    result = maybe_add_escalation_footer("Regular response.", requires_escalation=False)
    assert result == "Regular response."


def test_escalation_footer_not_duplicated():
    """Calling twice with the same response should not double-append the footer."""
    from app.services.sentiment_analyzer import _ESCALATION_FOOTER, maybe_add_escalation_footer

    once = maybe_add_escalation_footer("Response.", requires_escalation=True)
    twice = maybe_add_escalation_footer(once, requires_escalation=True)

    assert twice.count(_ESCALATION_FOOTER) == 1


def test_handle_escalation_marks_record(mock_db):
    """handle_escalation sets escalated=True on the DB record and commits."""
    from app.services.sentiment_analyzer import handle_escalation

    fake_record = MagicMock()
    fake_record.escalated = False
    mock_db.query.return_value.filter.return_value.first.return_value = fake_record

    handle_escalation(mock_db, conversation_id=5, phone_number="+852", sentiment_score=-0.8)

    assert fake_record.escalated is True
    mock_db.commit.assert_called_once()


def test_handle_escalation_missing_record(mock_db):
    """handle_escalation should not crash if the record doesn't exist."""
    from app.services.sentiment_analyzer import handle_escalation

    mock_db.query.return_value.filter.return_value.first.return_value = None

    # Should not raise
    handle_escalation(mock_db, conversation_id=99, phone_number="+852", sentiment_score=-0.9)
    mock_db.commit.assert_not_called()


def test_handle_escalation_already_flagged_no_second_commit(mock_db):
    """Already-escalated records are not committed again."""
    from app.services.sentiment_analyzer import handle_escalation

    fake_record = MagicMock()
    fake_record.escalated = True  # already set
    mock_db.query.return_value.filter.return_value.first.return_value = fake_record

    handle_escalation(mock_db, conversation_id=5, phone_number="+852", sentiment_score=-0.9)

    mock_db.commit.assert_not_called()
