"""Logs must not carry phone numbers or what patients wrote."""

import logging
from unittest.mock import MagicMock, patch

SENDER = "whatsapp:+85291234567"
SECRET_TEXT = "I have chest pain and my name is Chan Tai Man"


def run_pipeline(client, escalate=False, entities=None):
    from app.services.intent_classifier import ClassificationResult

    record = MagicMock()
    record.id = 7
    classification = ClassificationResult(
        intent="appointment" if entities else "faq",
        sentiment="negative" if escalate else "neutral",
        sentiment_score=-0.9 if escalate else 0.0,
        entities=entities or {},
        requires_escalation=escalate,
    )
    with (
        patch("app.routes.webhook._send_whatsapp_message"),
        patch("app.services.intent_classifier.classify_message", return_value=classification),
        patch("app.services.conversation_store.get_history", return_value=[]),
        patch("app.services.summarizer.maybe_summarise_history", side_effect=lambda h: h),
        patch("app.services.vectorstore.get_rag_context", return_value=""),
        patch("app.services.ai_service.generate_response", return_value="Bot reply"),
        patch("app.services.entity_extractor.generate_appointment_response", return_value="Appt"),
        patch("app.services.conversation_store.save_exchange", return_value=record),
        patch("app.services.sentiment_analyzer.handle_escalation"),
    ):
        return client.post("/message", data={"Body": SECRET_TEXT, "From": SENDER})


def test_a_message_leaves_no_phone_number_text_or_name_in_the_logs(client, caplog):
    with caplog.at_level(logging.DEBUG):
        run_pipeline(client, entities={"name": "Chan Tai Man", "date": "2026-12-01"})
    logs = caplog.text
    assert "85291234567" not in logs
    assert "chest pain" not in logs and "Chan Tai Man" not in logs and "2026-12-01" not in logs
    assert "4567" in logs  # still enough to tell two senders apart


def test_the_log_says_how_long_the_message_was_but_not_what_it_said(client, caplog):
    with caplog.at_level(logging.INFO):
        run_pipeline(client)
    assert f"{len(SECRET_TEXT)}-character message" in caplog.text


def test_saving_and_escalating_log_only_a_masked_number(mock_db, caplog):
    from app.services import conversation_store, sentiment_analyzer

    mock_db.refresh.side_effect = lambda record: setattr(record, "id", 3)
    mock_db.query.return_value.filter.return_value.first.return_value = MagicMock(escalated=False)
    with caplog.at_level(logging.DEBUG):
        conversation_store.save_exchange(mock_db, SENDER, "text", "reply", intent="faq")
        sentiment_analyzer.handle_escalation(mock_db, 3, SENDER, -0.9)
    assert "85291234567" not in caplog.text
    assert "Saved conversation #3" in caplog.text and "ESCALATION flagged" in caplog.text
