"""Request authentication and privacy helpers for the WhatsApp webhook."""

import logging
import re

from fastapi import HTTPException, Request
from twilio.request_validator import RequestValidator

from app.config import get_settings

logger = logging.getLogger(__name__)

# Twilio sends the sender as "whatsapp:+<country code and number>".
WHATSAPP_NUMBER = re.compile(r"^whatsapp:\+\d{6,15}$")


def is_whatsapp_number(value: str) -> bool:
    return bool(WHATSAPP_NUMBER.fullmatch(value or ""))


def mask_phone(value: str) -> str:
    """'whatsapp:+85291234567' -> 'whatsapp:+*******4567'. Logs are not the place for phone numbers."""
    prefix, _, number = (value or "").partition("+")
    if not number:
        return "***"
    return f"{prefix}+{'*' * max(len(number) - 4, 0)}{number[-4:]}"


def public_url(request: Request) -> str:
    """The URL Twilio called, which is what it signed. Behind a TLS-terminating proxy the scheme and host
    the app sees are not the public ones, so PUBLIC_URL (e.g. https://bot.example.com) takes precedence."""
    settings = get_settings()
    if settings.public_url:
        url = settings.public_url.rstrip("/") + request.url.path
        return f"{url}?{request.url.query}" if request.url.query else url
    return str(request.url)


async def verify_twilio_request(request: Request) -> None:
    """Reject requests that Twilio did not sign.

    Without this check anyone who can reach /message chooses the ``From`` number the bot then writes to,
    so they could make your Twilio account message arbitrary numbers and spend your OpenAI and Cohere credit.
    """
    settings = get_settings()
    if not settings.validate_twilio_signature:
        return
    signature = request.headers.get("X-Twilio-Signature", "")
    form = await request.form()  # Starlette caches it, so the Form() parameters still work
    validator = RequestValidator(settings.twilio_auth_token)
    if not signature or not validator.validate(public_url(request), form, signature):
        logger.warning("Rejected a webhook request with a missing or invalid Twilio signature.")
        raise HTTPException(status_code=403, detail="Invalid Twilio signature")
