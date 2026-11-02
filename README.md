# FastsApp AI Agent

A production-grade WhatsApp customer-service bot for **Adventist Medical Centre (Hong Kong)**.
Powered by **OpenAI**, **Cohere Rerank**, **ChromaDB**, **PostgreSQL**, and **Twilio WhatsApp**,
with five built-in agentic features.

---

## Architecture

```
Twilio WhatsApp
      │
      ▼ POST /message
┌─────────────────────────────────────────────┐
│  FastAPI webhook handler                    │
│                                             │
│  1. Intent + Sentiment Classification  ◄──── Feature 2
│  2. Load per-user Conversation History ◄──── Feature 1
│  3. Summarise long history (if needed) ◄──── Feature 3
│  4. Query ChromaDB → Cohere Rerank v4       │
│  5. Generate response via OpenAI            │
│     └─ Appointment path if intent=appt ◄──── Feature 5
│  6. Append escalation footer (if needed) ◄── Feature 4
│  7. Persist to PostgreSQL                   │
│  8. Send reply via Twilio                   │
└─────────────────────────────────────────────┘
      │
      ▼
  PostgreSQL (conversations table)
  ChromaDB   (persistent FAQ vectorstore)
```

### Module layout

```
fastsapp-ai-agent/
├── main.py                       # FastAPI app + lifespan startup
├── pyproject.toml  uv.lock       # Dependencies (uv), pytest and ruff config
├── Dockerfile  compose.yaml      # Container image and app + PostgreSQL stack
├── app/
│   ├── config.py                 # pydantic-settings centralised config
│   ├── security.py               # Twilio signature check, sender format, phone masking
│   ├── database.py               # SQLAlchemy engine/session (psycopg3 dialect)
│   ├── models.py                 # Conversation ORM model
│   ├── routes/
│   │   ├── health.py             # GET /health (liveness), GET /ready (database + vectorstore)
│   │   └── webhook.py            # POST /message
│   └── services/
│       ├── openai_client.py      # one shared, cached OpenAI client
│       ├── ai_service.py         # OpenAI chat completion
│       ├── vectorstore.py        # ChromaDB + Cohere rerank
│       ├── conversation_store.py # Feature 1 — per-user history
│       ├── intent_classifier.py  # Feature 2 — intent + sentiment
│       ├── summarizer.py         # Feature 3 — history summarisation
│       ├── sentiment_analyzer.py # Feature 4 — escalation
│       └── entity_extractor.py   # Feature 5 — appointment entity flow
├── content/
│   └── adventistFAQ.csv          # FAQ source data for RAG ingestion
├── tests/                        # 106 tests (mocked services; real SQLite and PostgreSQL for the store)
└── .env.example                  # Environment variable template
```

---

## Five Agentic Features

| # | Feature | Description |
|---|---|---|
| 1 | **Per-user Memory** | Every exchange stored in PostgreSQL; latest N turns injected into each prompt |
| 2 | **Intent Classification** | Single OpenAI call classifies intent (`faq`, `appointment`, `complaint`, `emergency`, `greeting`, `farewell`, `other`) + sentiment score |
| 3 | **Conversation Summarisation** | Older turns condensed into a compact context block when history exceeds threshold |
| 4 | **Sentiment Escalation** | Negative/urgent conversations flagged in DB, given a human-contact footer, and `WARNING`-logged |
| 5 | **Appointment Entity Flow** | Guided booking: confirms when all fields are present, asks for missing ones otherwise |

---

## Security and privacy

- **Every webhook call must be signed by Twilio.** `POST /message` checks the `X-Twilio-Signature` header against your auth token and answers `403` otherwise. This matters because the `From` field decides who the bot writes to: without the check, anyone who could reach the URL could make your Twilio account message arbitrary numbers and spend your OpenAI and Cohere credit. Behind a TLS-terminating proxy set `PUBLIC_URL` (the address Twilio calls, e.g. `https://bot.example.com`) so the URL that was signed can be rebuilt. `VALIDATE_TWILIO_SIGNATURE=false` exists for local experiments only.
- The sender must look like `whatsapp:+<number>`; anything else is a `400`.
- **Logs carry no phone numbers (only the last four digits), no message text and no extracted names or dates.** Conversations are stored in the database in full, as before, so treat it as health-related personal data: restrict access, back it up, and decide how long you keep it.
- The customer's message is sent to the intent classifier as its own turn and the classification instructions call it data, so a message cannot rewrite them (it used to be pasted into the instructions).
- Errors from OpenAI, Cohere and the database are logged, not returned to callers.

---

## Setup

Requires [uv](https://docs.astral.sh/uv/) (Python 3.13) and PostgreSQL 17 (14 or later works).

```bash
git clone https://github.com/kneeraazon404/fastsapp-ai-agent.git
cd fastsapp-ai-agent
uv sync
cp .env.example .env              # fill in the keys and the database settings
psql -U postgres -c "CREATE DATABASE fastsapp_ai_agent;"
uv run uvicorn main:app --reload --port 8000
```

The `conversations` table is created on startup, and the FAQ CSV is ingested into ChromaDB on the first start (it calls the OpenAI embeddings API) and skipped afterwards.

**Local Twilio testing:** run `ngrok http 8000`, set `https://<id>.ngrok.io/message` as the WhatsApp sandbox webhook, and set `PUBLIC_URL=https://<id>.ngrok.io` in `.env` so the signature can be verified.

**Required variables:** `OPENAI_API_KEY`, `COHERE_API_KEY`, `TWILIO_ACCOUNT_SID`, `TWILIO_AUTH_TOKEN`, `TWILIO_NUMBER`, `DB_USER`, `DB_PASSWORD` (the rest have defaults; `.env.example` lists everything).

---

## Configuration reference

| Variable | Default | Description |
|---|---|---|
| `OPENAI_MODEL` | `gpt-4o-mini` | Chat model |
| `OPENAI_EMBEDDING_MODEL` | `text-embedding-3-small` | Embedding model |
| `MAX_COMPLETION_TOKENS` | `512` | Max reply tokens |
| `TEMPERATURE` | `0.7` | Response creativity |
| `OPENAI_TIMEOUT` | `30` | Seconds before an OpenAI call gives up |
| `COHERE_RERANK_MODEL` | `rerank-v4.0-fast` | Rerank model |
| `COHERE_RERANK_TOP_N` | `3` | Documents kept after rerank |
| `VECTORSTORE_QUERY_N` | `8` | Candidate documents from ChromaDB |
| `VALIDATE_TWILIO_SIGNATURE` | `true` | Refuse webhook calls Twilio did not sign |
| `PUBLIC_URL` | empty | The address Twilio calls (needed behind a proxy) |
| `DB_HOST` / `DB_PORT` / `DB_NAME` | `localhost` / `5432` / `fastsapp_ai_agent` | PostgreSQL |
| `VECTORSTORE_PATH` | `./vectorstore` | ChromaDB persistence directory |
| `CONTENT_CSV_PATH` | `./content/adventistFAQ.csv` | FAQ source data |
| `MAX_HISTORY_MESSAGES` | `10` | Prior exchanges included in each prompt |
| `SUMMARIZE_THRESHOLD` | `8` | Exchanges before summarisation triggers |
| `SYSTEM_PROMPT` | (Chinese hospital persona) | Bot persona |
| `ESCALATION_SENTIMENT_THRESHOLD` | `-0.5` | Score below which escalation fires |
| `ESCALATION_CONTACT_PHONE` | empty | Phone to notify on escalation (not wired to anything yet: it only logs) |

---

## API endpoints

| Method | Path | Description |
|---|---|---|
| `GET` | `/health` | Liveness: `{"status": "ok"}`, touches no dependency |
| `GET` | `/ready` | Readiness: `200` when the database answers and the FAQ vectorstore holds documents, else `503` with the state of each |
| `POST` | `/message` | Twilio webhook: `Body` + `From` form fields, signed by Twilio |

---

## Testing

```bash
uv run pytest                      # 106 tests; no external services needed
docker run -d --name fsa-test-db -e POSTGRES_PASSWORD=demo-only -p 15434:5432 postgres:17-alpine
export TEST_DATABASE_URL=postgresql+psycopg://postgres:demo-only@127.0.0.1:15434/postgres
uv run pytest                      # also runs the 5 PostgreSQL integration tests (they skip without it)
uv run ruff check . && uv run ruff format --check .
```

OpenAI, Cohere, ChromaDB and Twilio are mocked. The conversation store is tested on a real database engine (SQLite in memory), and the webhook end to end on PostgreSQL with a really signed request. Deprecation warnings in project code are errors; one targeted suppression remains for chromadb's own `asyncio.iscoroutinefunction` use.

---

## Deployment

```bash
cp .env.example .env               # set the keys and DB_PASSWORD
docker compose up -d --build       # the bot (port 8000) and PostgreSQL 17, each with a volume
```

The image runs as a non-root user and has a health check. Put TLS in front (nginx, Caddy or your platform's ingress) and set `PUBLIC_URL`. Keep `VECTORSTORE_PATH` on persistent storage and secrets in your platform's secret manager. For more than one worker: `uvicorn main:app --workers 2`.

---

## Versions

[VERSIONS.md](./VERSIONS.md). What changed and what needs your attention: [work-done.md](./work-done.md).

---

## License

MIT License. See [LICENSE](LICENSE).
