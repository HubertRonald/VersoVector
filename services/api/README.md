# VersoVector API Service

This service packages the FastAPI serving layer for VersoVector.

The API exposes the `PoemAnalyzer` inference layer through HTTP endpoints.

## Build

From the repository root:

```bash
docker build \
  -f services/api/Dockerfile \
  -t versovector-api:local \
  .
```

## Run

The API expects a local model bundle at:

```bash
artifacts/model_bundle/
```

Run:

```bash
docker run --rm \
  --name versovector-api \
  -p 8001:8001 \
  -e PORT=8001 \
  -e VERSOVECTOR_MODEL_BUNDLE_DIR=/app/artifacts/model_bundle \
  -v "$(pwd)/artifacts/model_bundle:/app/artifacts/model_bundle:ro" \
  versovector-api:local
```

## Health Check

```bash
curl http://localhost:8001/health | jq .
```

## API Docs

```bash
http://localhost:8001/docs
```

## Notes

The image installs the VersoVector package from a wheel generated during the Docker build.

The model bundle is not baked into the image in this local foundation stage.