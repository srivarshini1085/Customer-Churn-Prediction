# syntax=docker/dockerfile:1
# Multi-stage build: deps in a venv, then a slim runtime image.

FROM python:3.12-slim AS builder
ENV PIP_NO_CACHE_DIR=1 PIP_DISABLE_PIP_VERSION_CHECK=1
WORKDIR /app
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"
COPY requirements.txt .
RUN pip install -r requirements.txt

FROM python:3.12-slim AS runtime
ENV PATH="/opt/venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    CHURN_LOG_JSON=true
WORKDIR /app
RUN useradd --create-home --uid 1000 churn
COPY --from=builder /opt/venv /opt/venv
COPY churn/ churn/
COPY api.py app.py pyproject.toml README.md ./
COPY data/ data/
USER churn
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=3s --start-period=10s \
    CMD python -c "import httpx,sys; sys.exit(0 if httpx.get('http://localhost:8000/health').status_code==200 else 1)"
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]
