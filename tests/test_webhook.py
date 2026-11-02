"""
Integration-level tests for POST /message (the Twilio webhook).

All external services are patched. Tests verify the full pipeline
orchestration: routing, persistence, escalation, and Twilio dispatch.
"""

from unittest.mock import MagicMock, patch

# ── Shared helpers ────────────────────────────────────────────────────────────


def _classification(
    intent="faq",
    sentiment="neutral",
    score=0.0,
    entities=None,
    escalate=False,
):
    from app.services.intent_classifier import ClassificationResult

    return ClassificationResult(
        intent=intent,
        sentiment=sentiment,
        sentiment_score=score,
        entities=entities or {},
        requires_escalation=escalate,
    )


def _post(client, body="Hello", from_="whatsapp:+85291234567"):
    return client.post("/message", data={"Body": body, "From": from_})


# ── Tests ─────────────────────────────────────────────────────────────────────


@patch("app.routes.webhook._send_whatsapp_message")
@patch("app.services.vectorstore.get_rag_context", return_value="RAG context")
@patch("app.services.ai_service.generate_response", return_value="Bot reply")
@patch("app.services.conversation_store.save_exchange")
@patch("app.services.conversation_store.get_history", return_value=[])
@patch("app.services.summarizer.maybe_summarise_history", side_effect=lambda h: h)
@patch("app.services.intent_classifier.classify_message")
def test_happy_path(
    mock_classify,
    mock_sum,
    mock_history,
    mock_save,
    mock_ai,
    mock_rag,
    mock_send,
    client,
    mock_db,
):
    """Standard FAQ message flows through the full pipeline."""
    mock_classify.return_value = _classification(intent="faq")
    record = MagicMock()
    record.id = 1
    mock_save.return_value = record

    resp = _post(client, "What are your hours?")

    assert resp.status_code == 200
    mock_send.assert_called_once()
    send_args = mock_send.call_args[0]
    assert send_args[1] == "Bot reply"


@patch("app.routes.webhook._send_whatsapp_message")
@patch("app.services.vectorstore.get_rag_context", return_value="")
@patch(
    "app.services.entity_extractor.generate_appointment_response", return_value="Appointment reply"
)
@patch("app.services.conversation_store.save_exchange")
@patch("app.services.conversation_store.get_history", return_value=[])
@patch("app.services.summarizer.maybe_summarise_history", side_effect=lambda h: h)
@patch("app.services.intent_classifier.classify_message")
def test_appointment_intent_uses_entity_extractor(
    mock_classify,
    mock_sum,
    mock_history,
    mock_save,
    mock_entity,
    mock_rag,
    mock_send,
    client,
    mock_db,
):
    """Appointment intent with entities delegates to entity_extractor."""
    mock_classify.return_value = _classification(
        intent="appointment",
        entities={
            "name": "Alice",
            "date": "2024-01-01",
            "time": "09:00",
            "service_type": "checkup",
        },
    )
    record = MagicMock()
    record.id = 2
    mock_save.return_value = record

    resp = _post(client, "I want to book a checkup for Alice on Jan 1 at 9am")

    assert resp.status_code == 200
    mock_entity.assert_called_once()
    send_args = mock_send.call_args[0]
    assert send_args[1] == "Appointment reply"


@patch("app.routes.webhook._send_whatsapp_message")
@patch("app.services.vectorstore.get_rag_context", return_value="")
@patch("app.services.ai_service.generate_response", return_value="Normal reply")
@patch("app.services.sentiment_analyzer.handle_escalation")
@patch("app.services.conversation_store.save_exchange")
@patch("app.services.conversation_store.get_history", return_value=[])
@patch("app.services.summarizer.maybe_summarise_history", side_effect=lambda h: h)
@patch("app.services.intent_classifier.classify_message")
def test_escalation_flag_and_footer(
    mock_classify,
    mock_sum,
    mock_history,
    mock_save,
    mock_escalate,
    mock_ai,
    mock_rag,
    mock_send,
    client,
    mock_db,
):
    """Negative-sentiment message sets escalated flag and adds footer."""
    mock_classify.return_value = _classification(
        intent="complaint", sentiment="negative", score=-0.8, escalate=True
    )
    record = MagicMock()
    record.id = 3
    mock_save.return_value = record

    resp = _post(client, "This is terrible!")

    assert resp.status_code == 200
    # The escalation handler must be called
    mock_escalate.assert_called_once()
    # The footer must appear in the sent message
    sent_body = mock_send.call_args[0][1]
    assert "2782 2202" in sent_body  # escalation contact number


@patch("app.routes.webhook._send_whatsapp_message")
@patch("app.services.vectorstore.get_rag_context", return_value="")
@patch("app.services.ai_service.generate_response", return_value="ok")
@patch("app.services.conversation_store.save_exchange")
@patch("app.services.conversation_store.get_history", return_value=[])
@patch("app.services.summarizer.maybe_summarise_history", side_effect=lambda h: h)
@patch("app.services.intent_classifier.classify_message")
def test_empty_body_is_ignored(
    mock_classify,
    mock_sum,
    mock_history,
    mock_save,
    mock_ai,
    mock_rag,
    mock_send,
    client,
    mock_db,
):
    """Empty message body returns 200 without calling any services."""
    resp = _post(client, body="   ")

    assert resp.status_code == 200
    mock_classify.assert_not_called()
    mock_send.assert_not_called()


@patch("app.routes.webhook._send_whatsapp_message")
@patch("app.services.vectorstore.get_rag_context", return_value="")
@patch("app.services.ai_service.generate_response", side_effect=RuntimeError("AI down"))
@patch("app.services.conversation_store.save_exchange")
@patch("app.services.conversation_store.get_history", return_value=[])
@patch("app.services.summarizer.maybe_summarise_history", side_effect=lambda h: h)
@patch("app.services.intent_classifier.classify_message")
def test_ai_failure_sends_fallback_response(
    mock_classify,
    mock_sum,
    mock_history,
    mock_save,
    mock_ai,
    mock_rag,
    mock_send,
    client,
    mock_db,
):
    """If the AI throws, the fallback message is sent instead of a 500."""
    mock_classify.return_value = _classification()
    record = MagicMock()
    record.id = 4
    mock_save.return_value = record

    resp = _post(client, "hello")

    assert resp.status_code == 200
    sent_body = mock_send.call_args[0][1]
    assert "系統暫時出現問題" in sent_body  # fallback text


@patch("app.routes.webhook._send_whatsapp_message", side_effect=Exception("Twilio down"))
@patch("app.services.vectorstore.get_rag_context", return_value="")
@patch("app.services.ai_service.generate_response", return_value="ok")
@patch("app.services.conversation_store.save_exchange")
@patch("app.services.conversation_store.get_history", return_value=[])
@patch("app.services.summarizer.maybe_summarise_history", side_effect=lambda h: h)
@patch("app.services.intent_classifier.classify_message")
def test_twilio_failure_returns_500(
    mock_classify,
    mock_sum,
    mock_history,
    mock_save,
    mock_ai,
    mock_rag,
    mock_send,
    client,
    mock_db,
):
    """If Twilio send fails, the endpoint returns 500."""
    mock_classify.return_value = _classification()
    record = MagicMock()
    record.id = 5
    mock_save.return_value = record

    resp = _post(client, "hello")

    assert resp.status_code == 500
