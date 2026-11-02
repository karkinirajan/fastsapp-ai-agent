"""
Centralised configuration using pydantic-settings.

All values are loaded from environment variables (or a .env file).
Call get_settings() anywhere in the codebase; the result is cached so
the .env file is parsed only once.
"""

from functools import lru_cache

from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # ── OpenAI ────────────────────────────────────────────────────────────────
    openai_api_key: str
    openai_model: str = "gpt-4o-mini"
    openai_embedding_model: str = "text-embedding-3-small"
    # max_completion_tokens replaces the deprecated max_tokens in openai v2
    max_completion_tokens: int = 512
    temperature: float = 0.7
    openai_timeout: float = 30.0

    # ── Cohere ────────────────────────────────────────────────────────────────
    cohere_api_key: str
    cohere_rerank_model: str = "rerank-v4.0-fast"
    cohere_rerank_top_n: int = 3
    vectorstore_query_n: int = 8

    # ── Twilio ────────────────────────────────────────────────────────────────
    twilio_account_sid: str
    twilio_auth_token: str
    twilio_number: str  # Twilio WhatsApp sandbox/production number, e.g. +14155238886
    # Twilio signs every webhook call; requests without a valid signature are refused (403).
    # Set PUBLIC_URL to the address Twilio calls (e.g. https://bot.example.com) when the app runs
    # behind a proxy. Turn validation off only for local experiments.
    validate_twilio_signature: bool = True
    public_url: str = ""

    # ── Database ──────────────────────────────────────────────────────────────
    db_user: str
    db_password: str
    db_host: str = "localhost"
    db_name: str = "fastsapp_ai_agent"
    db_port: int = 5432

    # ── Vectorstore ───────────────────────────────────────────────────────────
    vectorstore_path: str = "./vectorstore"
    content_csv_path: str = "./content/adventistFAQ.csv"

    # ── Conversation ──────────────────────────────────────────────────────────
    max_history_messages: int = 10
    # Summarise when history (in message pairs) exceeds this threshold
    summarize_threshold: int = 8

    # ── Bot persona ───────────────────────────────────────────────────────────
    system_prompt: str = (
        "你是港安醫療中心的客服，請禮貌地在WhatsApp上協助客戶的問題。"
        "你在參考資料中可以找到港安醫療中心的相關資料。"
        "你是在WhatsApp上回答客戶，所以請盡量保持簡短，每次回覆不多於50個字。"
        "若無相關資料，請誠實告知並建議客戶直接聯繫醫療中心。"
    )

    # ── Escalation ────────────────────────────────────────────────────────────
    # Conversations with sentiment_score below this are flagged for escalation
    escalation_sentiment_threshold: float = -0.5
    # Optional: phone number to notify when escalation is triggered (SMS/WhatsApp)
    escalation_contact_phone: str = ""

    @field_validator("twilio_number")
    @classmethod
    def strip_whatsapp_prefix(cls, v: str) -> str:
        """Accept either '+14155238886' or 'whatsapp:+14155238886'."""
        return v.removeprefix("whatsapp:")


@lru_cache
def get_settings() -> Settings:
    return Settings()
