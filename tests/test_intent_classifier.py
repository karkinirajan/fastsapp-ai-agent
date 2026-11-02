"""Tests for app/services/intent_classifier.py (Feature 2 — intent classification)"""

import json
from unittest.mock import MagicMock, patch


def _mock_openai_json(data: dict):
    """Return a mock OpenAI client whose chat.completions.create returns *data* as JSON."""
    content = MagicMock()
    content.content = json.dumps(data)
    choice = MagicMock()
    choice.message = content
    completion = MagicMock()
    completion.choices = [choice]

    client = MagicMock()
    client.chat.completions.create.return_value = completion
    return client


@patch("app.services.intent_classifier.get_openai")
def test_classify_faq_message(MockOpenAI, mock_settings):
    from app.services.intent_classifier import classify_message

    MockOpenAI.return_value = _mock_openai_json(
        {
            "intent": "faq",
            "sentiment": "neutral",
            "sentiment_score": 0.0,
            "entities": {
                "name": None,
                "date": None,
                "time": None,
                "service_type": None,
                "location": None,
            },
        }
    )

    result = classify_message("你哋幾點開門？")

    assert result.intent == "faq"
    assert result.sentiment == "neutral"
    assert result.sentiment_score == 0.0
    assert result.requires_escalation is False
    assert result.entities == {}


@patch("app.services.intent_classifier.get_openai")
def test_classify_appointment_with_entities(MockOpenAI, mock_settings):
    from app.services.intent_classifier import classify_message

    MockOpenAI.return_value = _mock_openai_json(
        {
            "intent": "appointment",
            "sentiment": "positive",
            "sentiment_score": 0.6,
            "entities": {
                "name": "陳大文",
                "date": "2024-03-15",
                "time": "10:00",
                "service_type": "全身體檢",
                "location": None,
            },
        }
    )

    result = classify_message("我想預約全身體檢，三月十五號上午十點，我叫陳大文")

    assert result.intent == "appointment"
    assert result.entities.get("name") == "陳大文"
    assert result.entities.get("date") == "2024-03-15"
    assert "location" not in result.entities  # None values stripped


@patch("app.services.intent_classifier.get_openai")
def test_escalation_triggered_for_emergency(MockOpenAI, mock_settings):
    from app.services.intent_classifier import classify_message

    MockOpenAI.return_value = _mock_openai_json(
        {
            "intent": "emergency",
            "sentiment": "urgent",
            "sentiment_score": -0.9,
            "entities": {},
        }
    )

    result = classify_message("我需要緊急幫助！")
    assert result.requires_escalation is True


@patch("app.services.intent_classifier.get_openai")
def test_escalation_triggered_by_low_score(MockOpenAI, mock_settings):
    from app.services.intent_classifier import classify_message

    # threshold is -0.5 in test settings; score -0.6 should trigger
    MockOpenAI.return_value = _mock_openai_json(
        {
            "intent": "complaint",
            "sentiment": "negative",
            "sentiment_score": -0.6,
            "entities": {},
        }
    )

    result = classify_message("服務太差了！")
    assert result.requires_escalation is True


@patch("app.services.intent_classifier.get_openai")
def test_invalid_intent_falls_back_to_other(MockOpenAI, mock_settings):
    from app.services.intent_classifier import classify_message

    MockOpenAI.return_value = _mock_openai_json(
        {
            "intent": "UNKNOWN_GARBAGE",
            "sentiment": "neutral",
            "sentiment_score": 0.0,
            "entities": {},
        }
    )

    result = classify_message("some message")
    assert result.intent == "other"


@patch("app.services.intent_classifier.get_openai")
def test_api_failure_returns_safe_defaults(MockOpenAI, mock_settings):
    from app.services.intent_classifier import classify_message

    client = MagicMock()
    client.chat.completions.create.side_effect = RuntimeError("network error")
    MockOpenAI.return_value = client

    result = classify_message("hello")

    assert result.intent == "other"
    assert result.sentiment == "neutral"
    assert result.requires_escalation is False
