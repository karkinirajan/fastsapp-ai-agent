"""
Agentic Feature 1 — Per-user Conversation Memory
=================================================
Stores every message exchange in the database keyed by the user's WhatsApp
phone number and retrieves the N most-recent turns to build multi-turn
context for the language model.

Without this, every reply starts from scratch and the bot cannot remember
anything the user said earlier in the same conversation.
"""

import logging

from sqlalchemy.orm import Session

from app.models import Conversation
from app.security import mask_phone

logger = logging.getLogger(__name__)


def get_history(
    db: Session,
    phone_number: str,
    limit: int = 10,
) -> list[dict[str, str]]:
    """
    Return the most-recent ``limit`` exchanges for *phone_number* as a list
    of OpenAI-format message dicts, interleaved user/assistant, oldest first.

    Example output::

        [
            {"role": "user",      "content": "你哋幾點開門？"},
            {"role": "assistant", "content": "我哋朝早八點開門。"},
            ...
        ]
    """
    records = (
        db.query(Conversation)
        .filter(Conversation.phone_number == phone_number)
        # id breaks ties: two rows can share a timestamp, and the history order must not depend on luck
        .order_by(Conversation.created_at.desc(), Conversation.id.desc())
        .limit(limit)
        .all()
    )
    history: list[dict[str, str]] = []
    for record in reversed(records):  # oldest first
        history.append({"role": "user", "content": record.user_message})
        history.append({"role": "assistant", "content": record.bot_response})
    return history


def save_exchange(
    db: Session,
    phone_number: str,
    user_message: str,
    bot_response: str,
    intent: str | None = None,
    sentiment: str | None = None,
    sentiment_score: float | None = None,
    entities: dict | None = None,
    escalated: bool = False,
) -> Conversation:
    """
    Persist a user message + bot response exchange to the database.

    Returns the newly created ``Conversation`` record (with its ``id``
    set by the database).
    """
    record = Conversation(
        phone_number=phone_number,
        user_message=user_message,
        bot_response=bot_response,
        intent=intent,
        sentiment=sentiment,
        sentiment_score=sentiment_score,
        entities=entities,
        escalated=escalated,
    )
    db.add(record)
    db.commit()
    db.refresh(record)
    logger.info(
        "Saved conversation #%d for %s (intent=%s)", record.id, mask_phone(phone_number), intent
    )
    return record


def count_exchanges(db: Session, phone_number: str) -> int:
    """Return the total number of stored exchanges for *phone_number*."""
    return db.query(Conversation).filter(Conversation.phone_number == phone_number).count()
