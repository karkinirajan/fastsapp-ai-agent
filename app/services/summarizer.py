"""
Agentic Feature 3 — Conversation Summarisation
===============================================
When a user's conversation history grows beyond ``summarize_threshold``
message pairs, this module summarises the older portion into a single
compact paragraph and returns a condensed history list.

This solves two problems:
  1. Token budgets — long histories can overflow the context window.
  2. Noise — early small-talk or repeated questions dilute recent context.

The summarised history is injected as a single system-style ``user`` turn
so that OpenAI sees it as background knowledge rather than active dialogue.
"""

import logging

from app.config import get_settings
from app.services.openai_client import get_openai

logger = logging.getLogger(__name__)

_SUMMARISE_PROMPT = (
    "Summarise the following conversation excerpt in 2–3 sentences. "
    "Retain names, dates, service types, and any important requests the "
    "user mentioned. Write only the summary, no preamble.\n\n{transcript}"
)


def maybe_summarise_history(
    history: list[dict[str, str]],
) -> list[dict[str, str]]:
    """
    If *history* (in OpenAI message-dict format, interleaved user/assistant)
    exceeds the configured threshold, summarise the older half and prepend
    the summary as a context marker, keeping the most-recent turns verbatim.

    Returns the (possibly condensed) history list.
    """
    settings = get_settings()
    # Each exchange = 2 entries (user + assistant)
    threshold_entries = settings.summarize_threshold * 2
    if len(history) <= threshold_entries:
        return history

    # Keep the most-recent ``max_history_messages`` pairs verbatim
    keep_entries = settings.max_history_messages * 2
    older = history[:-keep_entries]
    recent = history[-keep_entries:]

    summary_text = _summarise(older)
    if not summary_text:
        # Summarisation failed — just truncate to recent entries
        logger.warning("Summarisation returned empty; truncating history.")
        return recent

    summary_turn: dict[str, str] = {
        "role": "user",
        "content": f"[Context from earlier in this conversation: {summary_text}]",
    }
    return [summary_turn] + recent


def _summarise(history: list[dict[str, str]]) -> str:
    """Call OpenAI to produce a short summary of *history*."""
    settings = get_settings()
    client = get_openai()

    transcript = "\n".join(f"{msg['role'].upper()}: {msg['content']}" for msg in history)
    prompt = _SUMMARISE_PROMPT.format(transcript=transcript)

    try:
        response = client.chat.completions.create(
            model=settings.openai_model,
            messages=[{"role": "user", "content": prompt}],
            max_completion_tokens=150,
            temperature=0.3,
        )
        return response.choices[0].message.content.strip()
    except Exception as exc:
        logger.error("Summarisation API call failed: %s", exc)
        return ""
