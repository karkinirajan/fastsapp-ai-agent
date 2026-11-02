"""
Entry point — run with:  uvicorn main:app --reload
"""

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.routes.health import router as health_router
from app.routes.webhook import router as webhook_router

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Startup: initialise the database schema and warm up the vectorstore
    (ingests FAQ data if the collection is empty).
    Shutdown: nothing special needed.
    """
    logger.info("Starting up…")
    try:
        from app.database import init_db

        init_db()
        logger.info("Database schema ready.")
    except Exception as exc:
        logger.error("Database init failed: %s", exc)

    try:
        from app.services.vectorstore import get_collection

        col = get_collection()
        logger.info("Vectorstore ready (%d documents).", col.count())
    except Exception as exc:
        logger.error("Vectorstore init failed: %s", exc)

    yield
    logger.info("Shutting down.")


app = FastAPI(
    title="FastsApp AI Agent",
    description=(
        "An AI-powered WhatsApp customer-service bot backed by RAG, "
        "per-user conversation memory, intent routing, sentiment escalation, "
        "and structured appointment handling."
    ),
    version="2.0.0",
    lifespan=lifespan,
)

app.include_router(health_router)
app.include_router(webhook_router)
