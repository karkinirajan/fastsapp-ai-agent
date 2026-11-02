"""
Twilio WhatsApp webhook — the primary request handler.

POST /message  ← Twilio calls this when a WhatsApp message arrives.

Processing pipeline
───────────────────
The sender's phone number is taken directly from the ``From`` form field.

1. Classify intent + sentiment + entities (Feature 2).
2. Load per-user conversation history (Feature 1).
3. Summarise history if it has grown long (Feature 3).
4. Query vectorstore for RAG context.
5. Generate AI response (appointment path if intent == 'appointment', Feature 5).
6. Append escalation footer if required (Feature 4).
7. Persist exchange to the database.
8. Persist escalation flag if triggered (Feature 4).
9. Send reply via Twilio Messaging API.
"""

import logging

from fastapi import APIRouter, Depends, Form, HTTPException
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session
from twilio.rest import Client as TwilioClient

from app.config import get_settings
from app.database import get_db
from app.security import is_whatsapp_number, mask_phone, verify_twilio_request
from app.services import (
    ai_service,
    conversation_store,
    entity_extractor,
    intent_classifier,
    sentiment_analyzer,
    summarizer,
    vectorstore,
)

logger = logging.getLogger(__name__)
router = APIRouter()

_FALLBACK_RESPONSE = "抱歉，系統暫時出現問題，請稍後再試。如有緊急需要，請致電 (852) 2782 2202。"


# ── Twilio helper ─────────────────────────────────────────────────────────────


def _send_whatsapp_message(to: str, body: str) -> None:
    """Send *body* to *to* via Twilio. Raises on failure."""
    settings = get_settings()
    client = TwilioClient(settings.twilio_account_sid, settings.twilio_auth_token)
    from_number = f"whatsapp:{settings.twilio_number}"
    to_number = to if to.startswith("whatsapp:") else f"whatsapp:{to}"
    try:
        msg = client.messages.create(from_=from_number, body=body, to=to_number)
        logger.info("Sent message to %s (sid=%s)", mask_phone(to_number), msg.sid)
    except Exception as exc:
        logger.error("Twilio send failed to %s: %s", mask_phone(to_number), exc)
        raise


# ── Endpoint ──────────────────────────────────────────────────────────────────


@router.post("/message", tags=["webhook"], dependencies=[Depends(verify_twilio_request)])
def receive_message(
    Body: str = Form(...),
    From: str = Form(...),
    db: Session = Depends(get_db),
) -> str:
    """
    Handle an inbound WhatsApp message forwarded (and signed) by Twilio.

    ``Body`` — the message text.
    ``From`` — the sender's number in ``whatsapp:+XXXXXXXXXXX`` format.

    A plain ``def``: the handler makes blocking OpenAI, Cohere and database calls, which as an
    ``async def`` stalled every other request. FastAPI runs it in a worker thread.
    """
    user_phone = From
    user_message = Body.strip()

    if not is_whatsapp_number(user_phone):
        raise HTTPException(status_code=400, detail="Invalid sender")

    if not user_message:
        logger.warning("Empty message received from %s — ignoring.", mask_phone(user_phone))
        return ""

    settings = get_settings()
    # Health information: log the size of the message, never its text.
    logger.info(
        "Received a %d-character message from %s", len(user_message), mask_phone(user_phone)
    )

    # ── Step 1: Classify intent, sentiment, entities ──────────────────────────
    classification = intent_classifier.classify_message(user_message)
    logger.info(
        "[%s] intent=%s sentiment=%s(%.2f) escalate=%s",
        mask_phone(user_phone),
        classification.intent,
        classification.sentiment,
        classification.sentiment_score,
        classification.requires_escalation,
    )

    # ── Step 2: Load conversation history ────────────────────────────────────
    history = conversation_store.get_history(db, user_phone, limit=settings.max_history_messages)

    # ── Step 3: Summarise if history is long ─────────────────────────────────
    history = summarizer.maybe_summarise_history(history)

    # ── Step 4: RAG context from vectorstore ─────────────────────────────────
    try:
        rag_context = vectorstore.get_rag_context(user_message)
    except Exception as exc:
        logger.error("Vectorstore query failed: %s", exc)
        rag_context = ""

    # ── Step 5: Generate response ─────────────────────────────────────────────
    try:
        if classification.intent == "appointment" and classification.entities:
            # Feature 5: structured appointment handling
            bot_response = entity_extractor.generate_appointment_response(
                entities=classification.entities,
                history=history,
            )
            if not bot_response:
                # Appointment generator returned empty — fall through to standard path
                bot_response = ai_service.generate_response(
                    user_message, history, rag_context, classification.intent
                )
        else:
            bot_response = ai_service.generate_response(
                user_message, history, rag_context, classification.intent
            )
    except Exception as exc:
        logger.error("Response generation error: %s", exc)
        bot_response = _FALLBACK_RESPONSE

    # ── Step 6: Escalation footer ─────────────────────────────────────────────
    bot_response = sentiment_analyzer.maybe_add_escalation_footer(
        bot_response, classification.requires_escalation
    )

    # ── Step 7: Persist to database ───────────────────────────────────────────
    record_id: int | None = None
    try:
        record = conversation_store.save_exchange(
            db=db,
            phone_number=user_phone,
            user_message=user_message,
            bot_response=bot_response,
            intent=classification.intent,
            sentiment=classification.sentiment,
            sentiment_score=classification.sentiment_score,
            entities=classification.entities or None,
            escalated=classification.requires_escalation,
        )
        record_id = record.id
    except SQLAlchemyError as exc:
        logger.error("Failed to persist conversation: %s", exc)
        # Don't abort — still send the response to the user

    # ── Step 8: Escalation DB flag + logging ──────────────────────────────────
    if classification.requires_escalation and record_id is not None:
        try:
            sentiment_analyzer.handle_escalation(
                db=db,
                conversation_id=record_id,
                phone_number=user_phone,
                sentiment_score=classification.sentiment_score,
            )
        except Exception as exc:
            logger.error("Escalation handler error: %s", exc)

    # ── Step 9: Send via Twilio ───────────────────────────────────────────────
    try:
        _send_whatsapp_message(user_phone, bot_response)
    except Exception as exc:
        logger.error("Could not deliver response to %s: %s", mask_phone(user_phone), exc)
        raise HTTPException(status_code=500, detail="Failed to send WhatsApp response") from exc

    return ""
