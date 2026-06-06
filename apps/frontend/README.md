
---

# 14. `apps/frontend/README.md`

```markdown
# VersoVector Gradio Frontend

This is the first user-facing demo app for VersoVector.

The app provides a simple interface to:

- submit a poem or lyric-like text;
- predict emotional or thematic tags;
- retrieve semantically similar poems;
- inspect topic and cluster information;
- display the full API response.

## Run Locally

Start the API first:

```bash
PYTHONPATH=src:. uvicorn versovector.api.main:app \
  --host 0.0.0.0 \
  --port 8001 \
  --reload
```

Then run the frontend:

```bash
python apps/frontend/app.py
```
Open:

```bash
http://localhost:7860
```

## Environment Variables

```text
VERSOVECTOR_API_BASE_URL
    Base URL of the FastAPI backend.
    Default: http://localhost:8001

PORT
    Frontend port.
    Default: 7860
```

Example:

```bash
VERSOVECTOR_API_BASE_URL=http://localhost:8001 \
PORT=7860 \
python apps/frontend/app.py
```

## Docker Compose

From the repository root:

```bash
docker compose -f services/compose.yaml up --build
```

Open:

```bash
http://localhost:7860
```

## Design Notes

This frontend does not use external fonts or externally hosted UI assets.

That makes it safer for private Cloud Run deployments and restricted corporate environments.