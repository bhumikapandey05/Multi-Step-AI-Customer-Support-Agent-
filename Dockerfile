# One image for all three services (API, customer UI, staff UI).
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    # SQLite files live on a named volume mounted at /data.
    SUPPORT_AGENT_DATABASE_URL=sqlite:////data/crm.db \
    SUPPORT_AGENT_CHECKPOINT_DB_PATH=/data/checkpoints.db \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

WORKDIR /app

# Dependencies first, so code changes don't reinstall them.
COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install . \
    && useradd --create-home --uid 10001 app \
    && mkdir -p /data \
    && chown app:app /data

USER app
EXPOSE 8000 8501 8502

HEALTHCHECK --interval=10s --timeout=3s --start-period=20s --retries=5 \
    CMD python -c "import os, urllib.request; urllib.request.urlopen(os.environ.get('HEALTH_URL', 'http://127.0.0.1:8000/health'), timeout=2)"

CMD ["uvicorn", "--factory", "support_agent.api.app:create_app", "--host", "0.0.0.0", "--port", "8000"]
