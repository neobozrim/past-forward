FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PAST_FORWARD_PROJECT_ROOT=/app

WORKDIR /app

COPY pyproject.toml ./
COPY src ./src
RUN python -m pip install --no-cache-dir .

COPY config.yaml ./config.yaml
COPY search_data ./search_data
COPY digitized ./digitized

RUN mkdir -p uploads digitized search_data/index \
    && useradd --create-home --uid 10001 appuser \
    && chown -R appuser:appuser /app

USER appuser

EXPOSE 8000

CMD ["sh", "-c", "uvicorn sofia_harness.agent_api:app --host 0.0.0.0 --port ${PORT:-8000}"]
