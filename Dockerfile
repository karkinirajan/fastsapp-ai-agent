FROM python:3.13-slim-trixie

COPY --from=ghcr.io/astral-sh/uv:0.13.6 /uv /usr/local/bin/uv

ENV UV_PROJECT_ENVIRONMENT=/opt/venv \
    UV_PYTHON_DOWNLOADS=never \
    UV_COMPILE_BYTECODE=1 \
    PATH="/opt/venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    ANONYMIZED_TELEMETRY=False \
    VECTORSTORE_PATH=/data/vectorstore

WORKDIR /app

# dependencies first so they stay cached when only the source changes
COPY pyproject.toml uv.lock .python-version ./
RUN uv sync --frozen --no-dev --no-install-project

COPY main.py ./
COPY app ./app
COPY content ./content

RUN useradd -m -u 1000 appuser && mkdir -p /data/vectorstore && chown -R appuser:appuser /data
USER appuser

VOLUME /data
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=40s \
    CMD ["python", "-c", "import urllib.request as u; u.urlopen('http://127.0.0.1:8000/health', timeout=3)"]
# Behind a TLS-terminating proxy set PUBLIC_URL (the address Twilio calls) so webhook signatures verify.
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
