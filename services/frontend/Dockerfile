FROM python:3.10-slim AS runtime

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PORT=7860
ENV VERSOVECTOR_API_BASE_URL=http://api:8001

RUN python -m pip install --upgrade pip setuptools wheel

COPY requirements-frontend.txt ./
RUN pip install --no-cache-dir -r requirements-frontend.txt

COPY apps/frontend ./apps/frontend

RUN useradd --create-home --shell /bin/bash appuser && \
    chown -R appuser:appuser /app

USER appuser

EXPOSE 7860

CMD ["python", "apps/frontend/app.py"]