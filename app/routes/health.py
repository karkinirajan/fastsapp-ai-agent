"""Liveness and readiness endpoints for load balancers and uptime monitors."""

import logging

from fastapi import APIRouter
from fastapi.responses import JSONResponse
from sqlalchemy import text

from app.database import _get_session_factory

logger = logging.getLogger(__name__)
router = APIRouter()


@router.get("/health", tags=["ops"])
def health_check() -> JSONResponse:
    """Liveness: the process is up. It does not touch any dependency."""
    return JSONResponse({"status": "ok"})


@router.get("/ready", tags=["ops"])
def readiness() -> JSONResponse:
    """Readiness: the database answers and the FAQ vectorstore holds documents (503 otherwise).
    Startup used to swallow a failed database or vectorstore and /health still said "ok"."""
    checks: dict[str, str] = {}
    try:
        with _get_session_factory()() as db:
            db.execute(text("SELECT 1"))
        checks["database"] = "ok"
    except Exception as exc:  # noqa: BLE001
        logger.error("Readiness: database check failed: %s", exc)
        checks["database"] = "unavailable"
    try:
        from app.services.vectorstore import get_collection

        checks["vectorstore"] = "ok" if get_collection().count() > 0 else "empty"
    except Exception as exc:  # noqa: BLE001
        logger.error("Readiness: vectorstore check failed: %s", exc)
        checks["vectorstore"] = "unavailable"
    ready = all(value == "ok" for value in checks.values())
    return JSONResponse(
        {"status": "ready" if ready else "not ready", **checks}, status_code=200 if ready else 503
    )
