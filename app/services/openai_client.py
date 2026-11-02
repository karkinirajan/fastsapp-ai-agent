"""One shared OpenAI client. Every call used to build a new client (and a new connection pool)."""

from functools import lru_cache

from openai import OpenAI

from app.config import get_settings


@lru_cache
def get_openai() -> OpenAI:
    settings = get_settings()
    return OpenAI(api_key=settings.openai_api_key, timeout=settings.openai_timeout)
