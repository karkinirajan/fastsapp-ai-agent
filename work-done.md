# Work done: fastsapp-ai-agent

Branch: `chore/modernize` (local, not pushed). Date: 2026-11-02.

## What was there
A FastAPI WhatsApp customer-service bot for a Hong Kong clinic (OpenAI, Cohere rerank, ChromaDB, PostgreSQL, Twilio), already in a good shape: a package layout, a `pyproject.toml`, 41 mocked tests that pass, and a long README. 32 tracked files, no Dockerfile, no CI, no lock file, no linter, dependencies pinned in two places, and no LICENSE file although the README says "MIT, see LICENSE".

So this pass verified it and fixed what the verification found, rather than restructuring it.

## The most important finding: the webhook was open
`POST /message` did not check that the request came from Twilio, and the sender in the `From` field is the number the bot then messages. Anyone who could reach the URL could make **your Twilio account send WhatsApp messages to any number**, and spend your OpenAI and Cohere credit, by posting a form. The webhook now verifies Twilio's `X-Twilio-Signature` (403 otherwise; `PUBLIC_URL` for use behind a proxy), and the sender must look like `whatsapp:+<number>`. Checked on a real running server: unsigned, forged, and signed-then-tampered calls all get 403.

## Other problems found and fixed
| Problem | Fix |
|---|---|
| Phone numbers, the first 60 characters of every message and the extracted names and dates were logged at INFO: health-related personal data in the logs | Numbers are masked to the last four digits, message text and entities are never logged (verified in a real server log) |
| The customer's message was pasted into the classification instructions between quotes (it could rewrite them) | The instructions are a system message, the customer's text is its own user turn |
| The handler was `async def` but made blocking OpenAI, Cohere and database calls, stalling every other request | A plain `def` (FastAPI runs it in a worker thread); a test pins it |
| Startup swallowed a failed database or vectorstore and `/health` still said "ok" | `/ready` checks the database and the vectorstore (503 with the state of each); `/health` stays a cheap liveness check |
| A new OpenAI client (and connection pool) was built on every call, with no timeout | One cached client with `OPENAI_TIMEOUT` |
| An empty completion would have been sent as an empty WhatsApp message | It is an error, and the existing fallback reply is sent |
| A classifier answer with `entities` of the wrong type crashed it | Ignored |
| History order depended on the timestamp alone (two exchanges can share one) | The id breaks ties |
| `pytest-asyncio` 1.4 raised a deprecation notice, which the project's own "warnings are errors" rule turned into a crash | The loop scope is set explicitly |
| Dependencies were pinned twice (`pyproject.toml` and `requirements.txt`) and a month stale | `uv.lock` is the single source; `requirements.txt` removed; latest versions (see VERSIONS.md) |
| README claimed a LICENSE file that does not exist | Added the MIT text (as in your other repositories), since the README already states MIT |

## Added
`app/security.py`, `app/services/openai_client.py`, a `/ready` endpoint, a non-root `Dockerfile` and a `compose.yaml` (bot + PostgreSQL with volumes), CI, ruff, a complete `.env.example` (a test keeps it in sync with the settings), and 65 new tests. The test setup also gained real database coverage: the conversation store runs on a real engine instead of a `MagicMock`, and five integration tests run the webhook on PostgreSQL.

## Changes by file
| Path | Change |
|---|---|
| `app/routes/webhook.py` | signature dependency, sender check, sync handler, masked logs |
| `app/routes/health.py` | `/ready` |
| `app/services/*.py` | shared client, classifier prompt, empty-completion guard, masked logs, history tie-break |
| `app/config.py` | `validate_twilio_signature`, `public_url`, `openai_timeout` |
| `app/security.py`, `app/services/openai_client.py` | new |
| `pyproject.toml`, `uv.lock`, `.python-version`, `requirements.txt` | uv, lower bounds, dev group, ruff and pytest config; requirements removed |
| `Dockerfile`, `.dockerignore`, `compose.yaml`, `.github/workflows/ci.yml` | new |
| `.env.example`, `README.md`, `VERSIONS.md`, `work-done.md`, `LICENSE` | updated or new |
| `tests/` | 41 -> 106 tests (existing ones re-pointed at the shared client; 12 files changed, 5 new) |
| everything under `app/`, `tests/` | reformatted with ruff |

## Verification (actual runs)
- A count-only secret scan of the full history (15 commits) and of the staged changes: no OpenAI-, Twilio- or private-key-shaped strings, and no URL with a password. Nothing needs rotating.
- Baseline: the original 41 tests pass on the original pins (Python 3.13); they pass on the latest versions after the loop-scope fix.
- `uv run pytest`: **106 passed**, 91% coverage; with a database configured, 5 more integration tests run on PostgreSQL 17.11 (without one they skip). `ruff check` and `ruff format --check` clean.
- Mutation checks (the defect put back, the suite must fail), 18, all caught: signature never checked or always skipped, `PUBLIC_URL` ignored, query string dropped from the signed URL, numbers not masked in three places, sender check removed or accepting anything, validation off by default, dependency removed from the route, message text logged again, handler `async` again, customer text back in the instructions, empty completion sent, client rebuilt per call, `/ready` always 200 or leaking error text, history tie-break removed, and a typo in `.env.example`.
- The real server on PostgreSQL with local stand-ins for OpenAI and Cohere: `/ready` green, 403 for unsigned, forged and tampered requests, 400 for a bad sender, a signed request stored (intent, escalation, entities) with the log free of the number, text and name. The Twilio send was blocked locally, so that request answers 500 after storing, as intended for the test.
- Docker: the image builds and runs as a non-root user, compose starts the bot and PostgreSQL, the table is created, the health check turns healthy, and tests and env files are not in the image.
- **Not verified:** real OpenAI, Cohere and Twilio calls or a real WhatsApp conversation; that the default model names are available to your accounts; a GitHub Actions run.

## Needs your attention
- **Set `PUBLIC_URL`** when the bot runs behind a proxy or ngrok. Without it the URL the app sees (`http://...`) differs from the one Twilio signed, and every real message would get 403. This is the one change that can break a working deployment if forgotten.
- Conversations (including names and dates the bot extracted) are stored in full in the `conversations` table. Decide who may read it and how long to keep it. There is no retention or deletion feature.
- `ESCALATION_CONTACT_PHONE` is only logged, never used to notify anyone: an escalated conversation needs a human to read the logs or query the table.
- There are no database migrations (the table is created with `create_all`); changing the model later needs Alembic or manual SQL.
- The image is large (1.6 GB) because of chromadb and its dependencies.
- The ingestion of the FAQ at first start calls the OpenAI embeddings API; if the CSV changes, delete the vectorstore volume to re-ingest.
- Nothing pushed. To publish: `git push -u origin chore/modernize`
