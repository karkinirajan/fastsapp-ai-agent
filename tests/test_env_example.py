from pathlib import Path

from app.config import Settings

EXAMPLE = Path(__file__).resolve().parent.parent / ".env.example"
# documented in a comment instead (a multi-line Chinese default lives in app/config.py)
COMMENTED = {"SYSTEM_PROMPT"}


def names():
    found = set()
    for line in EXAMPLE.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            found.add(line.split("=", 1)[0].strip())
    return found


def test_every_setting_is_documented():
    missing = {n.upper() for n in Settings.model_fields} - names() - COMMENTED
    assert missing == set(), f"add these to .env.example: {sorted(missing)}"


def test_it_names_no_unknown_setting():
    unknown = names() - {n.upper() for n in Settings.model_fields}
    assert unknown == set(), f"not settings (typo?): {sorted(unknown)}"


def test_it_loads_as_valid_settings():
    """`cp .env.example .env` must give a working configuration (inline values, empty values)."""
    s = Settings(_env_file=str(EXAMPLE))
    assert s.validate_twilio_signature is True and s.public_url == ""
    assert s.twilio_number == "+14155238886" and s.db_port == 5432 and s.openai_timeout == 30
