"""The webhook end to end on a real PostgreSQL (psycopg 3 dialect), with a real Twilio signature.

Needs TEST_DATABASE_URL (e.g. postgresql+psycopg://postgres:password@localhost:5432/postgres) and skips
without it. Each test gets its own schema, dropped afterwards. OpenAI, Cohere, Chroma and Twilio's send
API are replaced; everything else is the real application.
"""

import os
import uuid
from unittest.mock import MagicMock, patch

import pytest
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker
from twilio.request_validator import RequestValidator

from app.models import Base, Conversation

URL = os.environ.get("TEST_DATABASE_URL", "")
pytestmark = pytest.mark.postgres
SENDER = "whatsapp:+85291234567"


@pytest.fixture
def pg(monkeypatch):
    if not URL:
        pytest.skip("TEST_DATABASE_URL is not set")
    schema = "t_" + uuid.uuid4().hex[:10]
    admin = create_engine(URL, isolation_level="AUTOCOMMIT")
    with admin.connect() as connection:
        connection.execute(text(f"CREATE SCHEMA {schema}"))
    engine = create_engine(URL).execution_options(schema_translate_map={None: schema})
    Base.metadata.create_all(engine)
    factory = sessionmaker(bind=engine, autoflush=False)
    monkeypatch.setattr("app.database._get_session_factory", lambda: factory)
    monkeypatch.setattr("app.routes.health._get_session_factory", lambda: factory)
    yield factory
    engine.dispose()
    with admin.connect() as connection:
        connection.execute(text(f"DROP SCHEMA {schema} CASCADE"))
    admin.dispose()


@pytest.fixture
def live_client(pg, mock_settings):
    """The real app and the real get_db, signature checking on."""
    from fastapi.testclient import TestClient

    mock_settings.validate_twilio_signature = True
    with patch("app.database.init_db"), patch("app.services.vectorstore.get_collection"):
        from main import app

        with TestClient(app) as client:
            yield client


def openai_returning(*texts):
    """A fake OpenAI client answering each call with the next text."""
    client = MagicMock()
    replies = []
    for content in texts:
        reply = MagicMock()
        reply.choices = [MagicMock()]
        reply.choices[0].message.content = content
        replies.append(reply)
    client.chat.completions.create.side_effect = replies
    return client


def post(client, body, sender=SENDER, sign=True):
    form = {"Body": body, "From": sender}
    headers = (
        {
            "X-Twilio-Signature": RequestValidator("test-token").compute_signature(
                "http://testserver/message", form
            )
        }
        if sign
        else {}
    )
    return client.post("/message", data=form, headers=headers)


CLASSIFICATION = (
    '{"intent": "appointment", "sentiment": "negative", "sentiment_score": -0.8,'
    ' "entities": {"name": "陳大文", "date": "2026-12-01", "time": null}}'
)


def test_a_signed_message_is_answered_and_stored_on_postgres(live_client, pg):
    client = openai_returning(CLASSIFICATION, "請問你想預約什麼時間？")
    with (
        patch("app.services.intent_classifier.get_openai", return_value=client),
        patch("app.services.entity_extractor.get_openai", return_value=client),
        patch("app.services.summarizer.get_openai", return_value=client),
        patch("app.services.vectorstore.get_rag_context", return_value=""),
        patch("app.routes.webhook._send_whatsapp_message") as send,
    ):
        response = post(live_client, "我想預約，你們太慢了")
    assert response.status_code == 200
    reply = send.call_args[0][1]
    assert (
        reply.startswith("請問你想預約什麼時間？") and "(852) 2782 2202" in reply
    )  # escalation footer appended
    with pg() as db:
        (row,) = db.query(Conversation).all()
        assert (row.phone_number, row.user_message, row.intent, row.escalated) == (
            SENDER,
            "我想預約，你們太慢了",
            "appointment",
            True,
        )
        assert row.entities == {
            "name": "陳大文",
            "date": "2026-12-01",
        }  # the JSON column, null dropped
        assert row.created_at is not None and row.sentiment_score == pytest.approx(-0.8)


def test_the_next_message_sees_the_stored_history(live_client, pg):
    first = openai_returning('{"intent": "faq"}', "九點開門")
    with (
        patch("app.services.intent_classifier.get_openai", return_value=first),
        patch("app.services.ai_service.get_openai", return_value=first),
        patch("app.services.vectorstore.get_rag_context", return_value=""),
        patch("app.routes.webhook._send_whatsapp_message"),
    ):
        post(live_client, "你們幾點開門？")
    second = openai_returning('{"intent": "faq"}', "星期一至五")
    with (
        patch("app.services.intent_classifier.get_openai", return_value=second),
        patch("app.services.ai_service.get_openai", return_value=second),
        patch("app.services.vectorstore.get_rag_context", return_value=""),
        patch("app.routes.webhook._send_whatsapp_message"),
    ):
        post(live_client, "星期六呢？")
    sent = second.chat.completions.create.call_args_list[-1].kwargs["messages"]
    assert [m["content"] for m in sent[1:-1]] == ["你們幾點開門？", "九點開門"]


def test_an_unsigned_message_is_refused_and_nothing_is_stored(live_client, pg):
    with patch("app.routes.webhook._send_whatsapp_message") as send:
        assert post(live_client, "hello", sign=False).status_code == 403
    send.assert_not_called()
    with pg() as db:
        assert db.query(Conversation).count() == 0


def test_two_senders_keep_separate_histories_on_postgres(live_client, pg):
    other = "whatsapp:+85298765432"
    for sender, answer in ((SENDER, "A"), (other, "B")):
        client = openai_returning('{"intent": "faq"}', answer)
        with (
            patch("app.services.intent_classifier.get_openai", return_value=client),
            patch("app.services.ai_service.get_openai", return_value=client),
            patch("app.services.vectorstore.get_rag_context", return_value=""),
            patch("app.routes.webhook._send_whatsapp_message"),
        ):
            post(live_client, f"hi from {answer}", sender=sender)
    with pg() as db:
        assert {(r.phone_number, r.bot_response) for r in db.query(Conversation)} == {
            (SENDER, "A"),
            (other, "B"),
        }


def test_readiness_runs_a_real_query(live_client):
    with patch("app.services.vectorstore.get_collection") as collection:
        collection.return_value.count.return_value = 12
        assert live_client.get("/ready").json() == {
            "status": "ready",
            "database": "ok",
            "vectorstore": "ok",
        }
