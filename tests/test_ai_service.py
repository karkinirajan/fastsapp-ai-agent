"""Tests for app/services/ai_service.py"""

from unittest.mock import MagicMock, patch

import pytest


def _make_completion(content: str):
    """Build a minimal mock that looks like an openai.ChatCompletion response."""
    msg = MagicMock()
    msg.content = content
    choice = MagicMock()
    choice.message = msg
    completion = MagicMock()
    completion.choices = [choice]
    return completion


@patch("app.services.ai_service.get_openai")
def test_generate_response_basic(MockOpenAI, mock_settings):
    """Standard call returns the model's reply."""
    from app.services.ai_service import generate_response

    mock_client = MagicMock()
    mock_client.chat.completions.create.return_value = _make_completion("  Hello!  ")
    MockOpenAI.return_value = mock_client

    result = generate_response("Hi", [], "", "greeting")

    assert result == "Hello!"
    mock_client.chat.completions.create.assert_called_once()


@patch("app.services.ai_service.get_openai")
def test_generate_response_includes_history(MockOpenAI, mock_settings):
    """History turns are included in the messages sent to OpenAI."""
    from app.services.ai_service import generate_response

    mock_client = MagicMock()
    mock_client.chat.completions.create.return_value = _make_completion("reply")
    MockOpenAI.return_value = mock_client

    history = [
        {"role": "user", "content": "previous question"},
        {"role": "assistant", "content": "previous answer"},
    ]
    generate_response("new question", history, "", "faq")

    call_kwargs = mock_client.chat.completions.create.call_args[1]
    messages = call_kwargs["messages"]
    roles = [m["role"] for m in messages]
    assert roles == ["system", "user", "assistant", "user"]


@patch("app.services.ai_service.get_openai")
def test_generate_response_injects_rag_context(MockOpenAI, mock_settings):
    """RAG context is appended to the system prompt."""
    from app.services.ai_service import generate_response

    mock_client = MagicMock()
    mock_client.chat.completions.create.return_value = _make_completion("ok")
    MockOpenAI.return_value = mock_client

    generate_response("question", [], "REFERENCE DATA", "faq")

    call_kwargs = mock_client.chat.completions.create.call_args[1]
    system_msg = call_kwargs["messages"][0]
    assert system_msg["role"] == "system"
    assert "REFERENCE DATA" in system_msg["content"]


@patch("app.services.ai_service.get_openai")
def test_generate_response_propagates_api_error(MockOpenAI, mock_settings):
    """API errors bubble up to the caller so the webhook can use a fallback."""
    from app.services.ai_service import generate_response

    mock_client = MagicMock()
    mock_client.chat.completions.create.side_effect = RuntimeError("API down")
    MockOpenAI.return_value = mock_client

    with pytest.raises(RuntimeError, match="API down"):
        generate_response("hello", [], "", "other")
