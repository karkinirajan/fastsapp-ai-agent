"""
Agentic Feature 5 — Structured Entity Extraction & Appointment Handling
========================================================================
When the intent classifier detects an ``appointment`` intent, this module
takes over response generation to provide a structured, guided experience:

  * If required entities (name, date, time, service_type) are ALL present:
    generate a confirmation response that summarises the booking details and
    tells the user what to do next.

  * If one or more required entities are MISSING:
    generate a focused follow-up question asking for only the missing items,
    avoiding a generic "tell me more" dead end.

Entities are extracted by the intent classifier in a single upstream API
call (see ``intent_classifier.py``) so this module avoids a redundant call.
It only calls the OpenAI API to *generate the appointment-specific reply*.
"""

import logging

from app.config import get_settings
from app.services.openai_client import get_openai

logger = logging.getLogger(__name__)

_REQUIRED_FIELDS: tuple[str, ...] = ("name", "date", "time", "service_type")

_FIELD_LABELS: dict[str, str] = {
    "name": "姓名",
    "date": "日期",
    "time": "時間",
    "service_type": "服務類型",
    "location": "診所地點",
}

_CONFIRM_PROMPT = (
    "客戶要求預約，已確認以下資料：{entity_summary}。\n"
    "請以禮貌的粵語確認預約，告知客戶下一步（例如：致電或發電郵確認）。\n"
    "回覆不多於40個字。"
)

_MISSING_PROMPT = (
    "客戶要求預約，但缺少以下資料：{missing_labels}。\n"
    "請以禮貌的粵語逐一詢問缺少的資料。\n"
    "回覆不多於40個字。"
)


def generate_appointment_response(
    entities: dict[str, str],
    history: list[dict[str, str]],
) -> str:
    """
    Generate a structured appointment reply based on *entities*.

    Returns an empty string if the API call fails so the caller can fall
    back to the standard response path.
    """
    settings = get_settings()
    client = get_openai()

    missing = [f for f in _REQUIRED_FIELDS if not entities.get(f)]

    if missing:
        missing_labels = "、".join(_FIELD_LABELS.get(f, f) for f in missing)
        prompt = _MISSING_PROMPT.format(missing_labels=missing_labels)
    else:
        entity_summary = "，".join(
            f"{_FIELD_LABELS.get(k, k)}：{v}" for k, v in entities.items() if v
        )
        prompt = _CONFIRM_PROMPT.format(entity_summary=entity_summary)

    try:
        response = client.chat.completions.create(
            model=settings.openai_model,
            messages=[
                *history,
                {"role": "user", "content": prompt},
            ],
            max_completion_tokens=120,
            temperature=0.5,
        )
        return response.choices[0].message.content.strip()
    except Exception as exc:
        logger.error("Appointment response generation failed: %s", exc)
        return ""
