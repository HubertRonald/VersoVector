# VersoVector Frontend Service

This service packages the Gradio frontend demo for VersoVector.

The frontend does not load model artifacts directly. It calls the API service over HTTP.

## Build

From the repository root:

```bash
docker build \
  -f services/frontend/Dockerfile \
  -t versovector-frontend:local \
  .
```

## Run

When running outside Docker Compose, point the frontend to a running API:

```bash
docker run --rm \
  --name versovector-frontend \
  -p 7860:7860 \
  -e PORT=7860 \
  -e VERSOVECTOR_API_BASE_URL=http://host.docker.internal:8001 \
  versovector-frontend:local
```

## Open

```bash
http://localhost:7860
```

## Notes

In Docker Compose, the frontend calls:
```bash
http://api:8001
```

In Cloud Run, this URL should become the deployed API service URL.