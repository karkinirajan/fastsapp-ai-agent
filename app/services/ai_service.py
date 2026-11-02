"""
OpenAI chat-completion wrapper.

Builds the full prompt from system instructions, RAG context, and
conversation history, then calls the OpenAI API using the v1 client.
"""

import logging

from app.config import get_settings
from app.services.openai_client import get_openai

logger = logging.getLogger(__name__)


def generate_response(
    user_message: str,
    history: list[dict[str, str]],
    rag_context: str = "",
    intent: str = "general",
) -> str:
    """
    Generate a bot response.

    Args:
        user_message: The current message from the user.
        history:      Prior conversation turns as OpenAI message dicts
                      (role/content), oldest first.
        rag_context:  Formatted reference material from the vectorstore.
        intent:       Classified intent — used for future prompt steering.

    Returns:
        The assistant's reply as a plain string.
    """
    settings = get_settings()
    client = get_openai()

    system_content = settings.system_prompt
    if rag_context:
        system_content = f"{system_content}\n\n{rag_context}"

    messages: list[dict[str, str]] = [
        {"role": "system", "content": system_content},
        *history,
        {"role": "user", "content": user_message},
    ]

    try:
        completion = client.chat.completions.create(
            model=settings.openai_model,
            messages=messages,
            max_completion_tokens=settings.max_completion_tokens,
            temperature=settings.temperature,
        )
        text = (completion.choices[0].message.content or "").strip()
        if not text:
            raise RuntimeError("OpenAI returned an empty completion")
        return text
    except Exception as exc:
        logger.error("OpenAI API error: %s", exc)
        raise
