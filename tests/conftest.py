"""
Shared pytest fixtures.

All external dependencies (OpenAI, Cohere, ChromaDB, Twilio, database) are
mocked here so that the test suite runs without any network access or
running services.
"""

from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

# ── Settings fixture ──────────────────────────────────────────────────────────


@pytest.fixture(autouse=True)
def mock_settings(monkeypatch):
    """
    Patch get_settings() everywhere with a test Settings object.
    autouse=True means every test gets this automatically.
    """
    from app.config import Settings

    test_settings = Settings(
        openai_api_key="test-openai-key",
        openai_model="gpt-4o-mini",
        openai_embedding_model="text-embedding-3-small",
        max_completion_tokens=256,
        temperature=0.7,
        cohere_api_key="test-cohere-key",
        cohere_rerank_model="rerank-v4.0-fast",
        cohere_rerank_top_n=3,
        vectorstore_query_n=8,
        twilio_account_sid="ACtest",
        twilio_auth_token="test-token",
        twilio_number="+14155238886",
        validate_twilio_signature=False,  # tests/test_security.py turns it on
        public_url="",
        db_user="testuser",
        db_password="testpass",
        db_host="localhost",
        db_name="testdb",
        db_port=5432,
        vectorstore_path="/tmp/test-vectorstore",
        content_csv_path="./content/adventistFAQ.csv",
        max_history_messages=10,
        summarize_threshold=8,
        system_prompt="You are a helpful test assistant.",
        escalation_sentiment_threshold=-0.5,
        escalation_contact_phone="",
    )

    # Clear lru_cache so the patched version takes effect
    from app import config

    config.get_settings.cache_clear()

    monkeypatch.setattr("app.config.get_settings", lambda: test_settings)
    # Patch in every module that imports get_settings directly
    for module in [
        "app.services.ai_service",
        "app.services.conversation_store",
        "app.services.entity_extractor",
        "app.services.intent_classifier",
        "app.services.sentiment_analyzer",
        "app.services.summarizer",
        "app.services.vectorstore",
        "app.services.openai_client",
        "app.security",
        "app.routes.webhook",
        "app.database",
    ]:
        try:
            monkeypatch.setattr(f"{module}.get_settings", lambda: test_settings)
        except AttributeError:
            pass  # module may not import get_settings

    return test_settings


# ── Database fixture ──────────────────────────────────────────────────────────


@pytest.fixture
def mock_db():
    """Return a MagicMock that quacks like a SQLAlchemy Session."""
    return MagicMock()


# ── FastAPI test client ───────────────────────────────────────────────────────


@pytest.fixture
def client(mock_db):
    """
    TestClient with the real FastAPI app, but with the database session
    and all external services mocked out.
    """
    # Import here to avoid module-level side effects before mocking
    with patch("app.database.init_db"), patch("app.services.vectorstore.get_collection"):
        from app.database import get_db
        from main import app

        app.dependency_overrides[get_db] = lambda: mock_db
        with TestClient(app, raise_server_exceptions=True) as c:
            yield c
        app.dependency_overrides.clear()
