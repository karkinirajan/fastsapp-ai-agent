"""
Agentic Feature 4 — Sentiment Analysis & Human Escalation
==========================================================
When the intent classifier flags a conversation as requiring escalation
(very negative sentiment, urgent tone, or emergency intent), this module:

  1. Marks the database record as ``escalated = True``.
  2. Appends a human-contact footer to the bot's reply so the user always
     receives clear escalation options rather than a dead end.
  3. Logs a warning that operators can monitor via log aggregation.

In a production deployment the ``handle_escalation`` function can be
extended to send an alert notification (e.g. email, Slack, PagerDuty)
without changing the caller.
"""

import logging

from sqlalchemy.orm import Session

from app.config import get_settings
from app.models import Conversation
from app.security import mask_phone

logger = logging.getLogger(__name__)

# Appended to bot responses when escalation is triggered.
# Adjust the contact details via the SYSTEM_PROMPT env var or by editing here.
_ESCALATION_FOOTER = (
    "\n\n如您需要緊急協助或希望直接聯繫我們的客服人員，"
    "請致電 (852) 2782 2202 或發送電郵至 info@adventistmedical.hk。"
    "我們的團隊很樂意為您提供進一步協助。"
)


def maybe_add_escalation_footer(response: str, requires_escalation: bool) -> str:
    """
    Append the escalation contact footer to *response* if *requires_escalation*
    is True and the footer is not already present.
    """
    if requires_escalation and _ESCALATION_FOOTER not in response:
        return response + _ESCALATION_FOOTER
    return response


def handle_escalation(
    db: Session,
    conversation_id: int,
    phone_number: str,
    sentiment_score: float,
) -> None:
    """
    Persist the escalation flag on *conversation_id* and emit a log warning.

    Extend this function to add real-time notifications (Slack, email, etc.)
    without modifying the calling code.
    """
    record = db.query(Conversation).filter(Conversation.id == conversation_id).first()
    if record and not record.escalated:
        record.escalated = True
        db.commit()

    logger.warning(
        "ESCALATION flagged — conversation #%d from %s "
        "(sentiment_score=%.2f). Human follow-up required.",
        conversation_id,
        mask_phone(phone_number),
        sentiment_score,
    )

    settings = get_settings()
    if settings.escalation_contact_phone:
        # Placeholder: integrate with Twilio or another alerting service here.
        logger.info(
            "Would notify escalation contact %s for conversation #%d.",
            settings.escalation_contact_phone,
            conversation_id,
        )
