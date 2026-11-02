# Versions

Read from `uv.lock` and the installed packages on 2026-11-02.

| Component | Version | Was (pinned in the old `requirements.txt` and `pyproject.toml`) | Where it is set |
|---|---|---|---|
| Python | 3.13 (3.13.18 here) | "3.11 or higher" | `.python-version`, `pyproject.toml`, `Dockerfile` |
| FastAPI | 0.162.0 | 0.135.3 | `pyproject.toml` (lower bound), `uv.lock` |
| uvicorn | 0.54.0 | 0.44.0 | |
| python-multipart | 0.0.33 | 0.0.24 | |
| openai | 3.42.0 | 2.30.0 | |
| cohere | 7.3.0 | 5.21.1 | |
| chromadb | 1.5.9 | 1.5.6 | |
| SQLAlchemy | 2.1.6 | 2.0.49 | |
| psycopg | 3.3.6 | 3.3.3 | |
| pydantic / pydantic-settings | 2.14.2 / 2.16.2 | 2.12.5 / 2.13.1 | |
| twilio | 9.12.0 | 9.10.4 | |
| pytest / pytest-asyncio / pytest-cov (dev) | 9.2.0 / 1.4.1 / 7.1.0 | 9.0.2 / 1.3.0 / absent | |
| httpx (dev) | 0.28.1 | 0.28.1 | |
| ruff (dev) | 0.17.3 | absent | |
| PostgreSQL | 17 (tested on 17.11) | not stated (the README recommended 17) | `compose.yaml`, CI service container |
| uv (in the image) | 0.13.6 | | `Dockerfile` |

108 packages are locked. The old setup pinned every version exactly in two places (`pyproject.toml` and `requirements.txt`) with a comment about ceilings imposed by chromadb's OpenTelemetry dependencies; `pyproject.toml` now states lower bounds and `uv.lock` holds the exact versions, so there is one source of truth. The ceilings no longer bind: the lock resolves cleanly on current versions.

The default model names (`gpt-4o-mini`, `text-embedding-3-small`, `rerank-v4.0-fast`) are unchanged. Whether they are still the right choice for your accounts was not checked; they are settings.

## Verified here

- 106 tests pass (91% line coverage of `app` and `main.py`); `ruff check` and `ruff format --check` are clean. The old suite (41 tests) passed unchanged on the original pins, and again on the latest versions after one configuration fix.
- The 5 PostgreSQL integration tests ran on PostgreSQL 17.11 (psycopg 3, the JSON column, a really signed request) and skip without `TEST_DATABASE_URL`.
- The real server (`uvicorn`) on PostgreSQL with local stand-ins for OpenAI and Cohere: schema created at startup, `/ready` goes green after the FAQ is ingested, unsigned, forged and tampered calls get 403, a signed call is processed and stored, and the server log holds no phone number, message text or name.
- The Docker image builds (1.6 GB, non-root), `docker compose up` starts the bot and PostgreSQL, the health check turns healthy, and the table is created in the compose database.

## Not verified here

- Any call to the real OpenAI, Cohere or Twilio APIs (stand-ins and mocks were used; the Twilio send was blocked locally), or a real WhatsApp message end to end.
- That the default model names are available to your accounts.
- A GitHub Actions run.
