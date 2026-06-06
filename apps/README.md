# VersoVector Gradio Frontend

This is the first polished user-facing demo app for VersoVector.

It implements the `v0.7.0-ui-product-experience-foundation` direction: a calm, elegant, public AI reading experience with a hidden technical layer.

## Features

- Poem or reflective text analysis.
- Optional title and poet fields.
- Suggested tag chips.
- Advanced options hidden by default.
- Predicted tags as percentage bars.
- Similar poems with cleaned title and poet display.
- Literary insights in a collapsed section.
- API and developer guide in a secondary section.
- Raw JSON hidden from the main user experience.
- No external fonts or externally hosted UI assets.

## Run Locally

Start the API first:

```bash
PYTHONPATH=src:. uvicorn versovector.api.main:app \
  --host 0.0.0.0 \
  --port 8001 \
  --reload
```

Warm up the model bundle:

```bash
curl http://localhost:8001/ready | jq .
```

Then run the frontend:

```bash
VERSOVECTOR_API_BASE_URL=http://localhost:8001 \
VERSOVECTOR_API_TIMEOUT_SECONDS=300 \
PORT=7860 \
python apps/frontend/app.py
```

Open:

```bash
http://localhost:7860
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

The frontend is intentionally public-first:

1. poem input;
2. predicted tags;
3. similar poems.

Technical details are available but secondary:

- topic summary;
- cluster assignment;
- API health;
- raw JSON;
- developer guide.
