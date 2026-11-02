"""The Twilio signature check, the sender format and the phone-number helpers."""

from unittest.mock import MagicMock, patch

import pytest
from twilio.request_validator import RequestValidator

from app.security import is_whatsapp_number, mask_phone

TOKEN = "test-token"  # the auth token in conftest
SENDER = "whatsapp:+85291234567"
FORM = {"Body": "Hello", "From": SENDER, "MessageSid": "SM123", "NumMedia": "0"}


def sign(url, params=FORM):
    return RequestValidator(TOKEN).compute_signature(url, params)


@pytest.fixture
def secured(mock_settings):
    mock_settings.validate_twilio_signature = True
    return mock_settings


@pytest.fixture
def pipeline():
    """Everything behind the webhook replaced, so only the front door is being tested."""
    from app.services.intent_classifier import ClassificationResult

    record = MagicMock()
    record.id = 1
    with (
        patch("app.routes.webhook._send_whatsapp_message") as send,
        patch(
            "app.services.intent_classifier.classify_message", return_value=ClassificationResult()
        ),
        patch("app.services.conversation_store.get_history", return_value=[]),
        patch("app.services.summarizer.maybe_summarise_history", side_effect=lambda h: h),
        patch("app.services.vectorstore.get_rag_context", return_value=""),
        patch("app.services.ai_service.generate_response", return_value="Bot reply"),
        patch("app.services.conversation_store.save_exchange", return_value=record),
    ):
        yield send


class TestSignature:
    def test_a_correctly_signed_request_is_accepted(self, client, secured, pipeline):
        response = client.post(
            "/message", data=FORM, headers={"X-Twilio-Signature": sign("http://testserver/message")}
        )
        assert response.status_code == 200
        pipeline.assert_called_once()

    def test_a_request_without_a_signature_is_refused_before_anything_runs(
        self, client, secured, pipeline
    ):
        response = client.post("/message", data=FORM)
        assert response.status_code == 403
        pipeline.assert_not_called()

    @pytest.mark.parametrize(
        "signature", ["", "AAAA", "not-base64!!", sign("http://testserver/message", {"Body": "x"})]
    )
    def test_a_wrong_signature_is_refused(self, client, secured, pipeline, signature):
        response = client.post("/message", data=FORM, headers={"X-Twilio-Signature": signature})
        assert response.status_code == 403
        pipeline.assert_not_called()

    def test_a_signature_made_with_another_token_is_refused(self, client, secured, pipeline):
        forged = RequestValidator("someone-elses-token").compute_signature(
            "http://testserver/message", FORM
        )
        assert (
            client.post("/message", data=FORM, headers={"X-Twilio-Signature": forged}).status_code
            == 403
        )

    def test_changing_a_field_after_signing_is_refused(self, client, secured, pipeline):
        signature = sign("http://testserver/message")
        tampered = {
            **FORM,
            "From": "whatsapp:+85200000000",
        }  # a different recipient for the bot's reply
        assert (
            client.post(
                "/message", data=tampered, headers={"X-Twilio-Signature": signature}
            ).status_code
            == 403
        )
        pipeline.assert_not_called()

    def test_behind_a_proxy_the_public_url_is_what_was_signed(self, client, secured, pipeline):
        secured.public_url = "https://bot.example.com/"
        good = sign("https://bot.example.com/message")
        assert (
            client.post("/message", data=FORM, headers={"X-Twilio-Signature": good}).status_code
            == 200
        )
        internal = sign(
            "http://testserver/message"
        )  # the address the app sees is not the one Twilio called
        assert (
            client.post("/message", data=FORM, headers={"X-Twilio-Signature": internal}).status_code
            == 403
        )

    def test_the_query_string_is_part_of_what_was_signed(self, client, secured, pipeline):
        secured.public_url = "https://bot.example.com"
        good = sign("https://bot.example.com/message?token=abc")
        response = client.post(
            "/message?token=abc", data=FORM, headers={"X-Twilio-Signature": good}
        )
        assert response.status_code == 200

    def test_validation_can_be_switched_off_for_local_work(self, client, mock_settings, pipeline):
        mock_settings.validate_twilio_signature = False
        assert client.post("/message", data=FORM).status_code == 200

    def test_validation_is_on_by_default(self):
        from app.config import Settings

        assert Settings.model_fields["validate_twilio_signature"].default is True

    def test_the_other_endpoints_stay_open(self, client, secured):
        assert client.get("/health").json() == {"status": "ok"}


class TestSender:
    @pytest.mark.parametrize(
        "sender",
        [
            "+852",
            "85291234567",
            "whatsapp:abc",
            "whatsapp:+12",
            "whatsapp:+1234567890123456",
            "tel:+85291234567",
        ],
    )
    def test_an_invalid_sender_is_a_400_and_nothing_is_sent(self, client, pipeline, sender):
        response = client.post("/message", data={"Body": "Hello", "From": sender})
        assert response.status_code == 400
        pipeline.assert_not_called()

    def test_a_missing_sender_is_rejected_by_form_validation(self, client, pipeline):
        assert client.post("/message", data={"Body": "Hello", "From": ""}).status_code == 422
        pipeline.assert_not_called()

    @pytest.mark.parametrize(
        "sender", ["whatsapp:+85291234567", "whatsapp:+14155238886", "whatsapp:+447911123456"]
    )
    def test_a_whatsapp_sender_is_accepted(self, client, pipeline, sender):
        assert client.post("/message", data={"Body": "Hello", "From": sender}).status_code == 200

    def test_the_format_check(self):
        assert is_whatsapp_number("whatsapp:+85291234567")
        assert not is_whatsapp_number("+85291234567")
        assert not is_whatsapp_number(None)


class TestMask:
    def test_only_the_last_four_digits_are_kept(self):
        assert mask_phone("whatsapp:+85291234567") == "whatsapp:+*******4567"

    @pytest.mark.parametrize(
        ("value", "expected"), [("", "***"), (None, "***"), ("garbage", "***"), ("+1234", "+1234")]
    )
    def test_odd_input_is_safe(self, value, expected):
        assert mask_phone(value) == expected
