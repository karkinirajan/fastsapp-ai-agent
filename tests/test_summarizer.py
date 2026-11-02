"""Tests for app/services/summarizer.py (Feature 3 — conversation summarisation)"""

from unittest.mock import MagicMock, patch


def _make_history(n_pairs: int) -> list[dict]:
    """Create a flat list of n_pairs user/assistant pairs."""
    history = []
    for i in range(n_pairs):
        history.append({"role": "user", "content": f"question {i}"})
        history.append({"role": "assistant", "content": f"answer {i}"})
    return history


@patch("app.services.summarizer.get_openai")
def test_short_history_is_unchanged(MockOpenAI, mock_settings):
    """History at or below the threshold should be returned unmodified."""
    from app.services.summarizer import maybe_summarise_history

    # threshold=8 pairs → 16 entries; create 5 pairs (10 entries) — no summarise
    history = _make_history(5)
    result = maybe_summarise_history(history)

    assert result == history
    MockOpenAI.assert_not_called()


@patch("app.services.summarizer.get_openai")
def test_long_history_triggers_summarisation(MockOpenAI, mock_settings):
    """History beyond the threshold triggers a summarisation API call."""
    from app.services.summarizer import maybe_summarise_history

    # threshold=8 → 16 entries; 20 pairs (40 entries) → should summarise
    content = MagicMock()
    content.content = "Summary of earlier conversation."
    choice = MagicMock()
    choice.message = content
    completion = MagicMock()
    completion.choices = [choice]

    client = MagicMock()
    client.chat.completions.create.return_value = completion
    MockOpenAI.return_value = client

    history = _make_history(20)
    result = maybe_summarise_history(history)

    # Should have called OpenAI
    client.chat.completions.create.assert_called_once()

    # First entry should be the summary context marker
    assert result[0]["role"] == "user"
    assert "Summary of earlier conversation" in result[0]["content"]


@patch("app.services.summarizer.get_openai")
def test_summarisation_failure_falls_back_to_truncation(MockOpenAI, mock_settings):
    """If the summarisation API call fails, we truncate to recent entries."""
    from app.services.summarizer import maybe_summarise_history

    client = MagicMock()
    client.chat.completions.create.side_effect = RuntimeError("API error")
    MockOpenAI.return_value = client

    history = _make_history(20)
    result = maybe_summarise_history(history)

    # Should be the recent entries only (max_history_messages * 2 entries)
    from app.config import get_settings

    settings = get_settings()
    assert len(result) == settings.max_history_messages * 2


@patch("app.services.summarizer.get_openai")
def test_summary_is_prepended_not_replacing_recent(MockOpenAI, mock_settings):
    """Recent messages must still be present after summarisation."""
    from app.services.summarizer import maybe_summarise_history

    content = MagicMock()
    content.content = "Prior summary."
    choice = MagicMock()
    choice.message = content
    completion = MagicMock()
    completion.choices = [choice]
    client = MagicMock()
    client.chat.completions.create.return_value = completion
    MockOpenAI.return_value = client

    history = _make_history(20)
    last_user_msg = history[-2]["content"]

    result = maybe_summarise_history(history)

    # The most recent user message should still be present
    contents = [m["content"] for m in result]
    assert last_user_msg in contents
