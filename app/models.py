"""
SQLAlchemy ORM models.

The table schema extends the original 'conversations' table with:
- phone_number  — the sender's WhatsApp number (was missing; 'sender' was wrong)
- created_at    — timestamp for ordering and analytics
- intent        — classified intent from the agentic pipeline
- sentiment     — sentiment label (positive/neutral/negative/urgent)
- sentiment_score — numeric sentiment score [-1.0, 1.0]
- entities      — JSON bag of extracted entities (name, date, service, …)
- escalated     — True when the conversation was flagged for human follow-up
"""

from sqlalchemy import JSON, Boolean, Column, DateTime, Float, Integer, String, Text, func
from sqlalchemy.orm import declarative_base

Base = declarative_base()


class Conversation(Base):
    """One exchange: a single user message and the bot's reply."""

    __tablename__ = "conversations"

    id = Column(Integer, primary_key=True, index=True)
    phone_number = Column(String(50), nullable=False, index=True)
    user_message = Column(Text, nullable=False)
    bot_response = Column(Text, nullable=False)
    intent = Column(String(50), nullable=True)
    sentiment = Column(String(20), nullable=True)
    sentiment_score = Column(Float, nullable=True)
    entities = Column(JSON, nullable=True)
    escalated = Column(Boolean, default=False, nullable=False)
    created_at = Column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
        index=True,
    )
