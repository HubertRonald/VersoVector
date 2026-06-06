FROM python:3.10-slim AS runtime

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PORT=8001
ENV VERSOVECTOR_MODEL_BUNDLE_DIR=/app/artifacts/model_bundle

RUN apt-get update && \
    apt-get install -y --no-install-recommends curl && \
    rm -rf /var/lib/apt/lists/*

RUN python -m pip install --upgrade pip setuptools wheel

COPY requirements.txt requirements-api.txt ./
RUN pip install --no-cache-dir -r requirements-api.txt

COPY --from=builder /wheelhouse/*.whl /tmp/wheels/
RUN pip install --no-cache-dir --no-deps /tmp/wheels/*.whl && \
    rm -rf /tmp/wheels

RUN python -m spacy download en_core_web_lg

RUN useradd --create-home --shell /bin/bash appuser && \
    mkdir -p /app/artifacts/model_bundle && \
    chown -R appuser:appuser /app

USER appuser

EXPOSE 8001

HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:${PORT}/health || exit 1

CMD ["sh", "-c", "uvicorn versovector.api.main:app --host 0.0.0.0 --port ${PORT}"]