"""Tests for app/services/entity_extractor.py (Feature 5 — entity extraction)"""

from unittest.mock import MagicMock, patch


def _mock_client(reply: str) -> MagicMock:
    content = MagicMock()
    content.content = reply
    choice = MagicMock()
    choice.message = content
    completion = MagicMock()
    completion.choices = [choice]
    client = MagicMock()
    client.chat.completions.create.return_value = completion
    return client


@patch("app.services.entity_extractor.get_openai")
def test_full_entities_generate_confirmation(MockOpenAI, mock_settings):
    """When all required entities are present, a confirmation prompt is used."""
    from app.services.entity_extractor import generate_appointment_response

    MockOpenAI.return_value = _mock_client("已確認預約，請致電確認詳情。")

    result = generate_appointment_response(
        entities={
            "name": "李小明",
            "date": "2024-03-20",
            "time": "14:00",
            "service_type": "全身體檢",
        },
        history=[],
    )

    assert result == "已確認預約，請致電確認詳情。"
    call_kwargs = MockOpenAI.return_value.chat.completions.create.call_args[1]
    # The prompt sent to OpenAI should contain the entity summary
    last_user_msg = call_kwargs["messages"][-1]["content"]
    assert "確認" in last_user_msg  # confirm path
    assert "缺少" not in last_user_msg


@patch("app.services.entity_extractor.get_openai")
def test_missing_entities_generate_follow_up_question(MockOpenAI, mock_settings):
    """When entities are incomplete, a follow-up question prompt is used."""
    from app.services.entity_extractor import generate_appointment_response

    MockOpenAI.return_value = _mock_client("請問您希望預約哪種服務？")

    result = generate_appointment_response(
        entities={"name": "李小明"},  # missing date, time, service_type
        history=[],
    )

    assert result == "請問您希望預約哪種服務？"
    call_kwargs = MockOpenAI.return_value.chat.completions.create.call_args[1]
    last_user_msg = call_kwargs["messages"][-1]["content"]
    assert "缺少" in last_user_msg  # missing-fields path


@patch("app.services.entity_extractor.get_openai")
def test_empty_entities_still_asks_follow_up(MockOpenAI, mock_settings):
    """Empty entity dict should trigger the follow-up path."""
    from app.services.entity_extractor import generate_appointment_response

    MockOpenAI.return_value = _mock_client("請告訴我您的姓名及預約時間。")

    result = generate_appointment_response(entities={}, history=[])
    assert result  # non-empty response


@patch("app.services.entity_extractor.get_openai")
def test_api_failure_returns_empty_string(MockOpenAI, mock_settings):
    """API errors return '' so the caller can fall back to the standard path."""
    from app.services.entity_extractor import generate_appointment_response

    client = MagicMock()
    client.chat.completions.create.side_effect = RuntimeError("API down")
    MockOpenAI.return_value = client

    result = generate_appointment_response(
        entities={"name": "Test", "date": "2024-01-01", "time": "09:00", "service_type": "X"},
        history=[],
    )
    assert result == ""


@patch("app.services.entity_extractor.get_openai")
def test_history_is_included_in_messages(MockOpenAI, mock_settings):
    """Prior conversation history is prepended to the messages list."""
    from app.services.entity_extractor import generate_appointment_response

    MockOpenAI.return_value = _mock_client("ok")

    history = [
        {"role": "user", "content": "我想預約"},
        {"role": "assistant", "content": "請提供詳情"},
    ]
    generate_appointment_response(
        entities={"name": "A", "date": "2024-01-01", "time": "10:00", "service_type": "B"},
        history=history,
    )

    call_kwargs = MockOpenAI.return_value.chat.completions.create.call_args[1]
    messages = call_kwargs["messages"]
    assert messages[0]["role"] == "user"
    assert messages[0]["content"] == "我想預約"
