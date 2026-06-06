# Services

This directory contains service-level packaging assets for VersoVector.

It is different from the Python package under `src/versovector/`.

```text
services/
├── api/
├── frontend/
└── compose.yaml
```

## Purpose

The services/ layer defines how the project components are packaged and executed as local or deployable services.

Current services:

- `api`: FastAPI serving layer for PoemAnalyzer.
- `frontend`: Gradio demo app that calls the API.

## Local Execution

From the repository root:

```bash
docker compose -f services/compose.yaml up --build
```

The local services will be available at:

```text
API:
    http://localhost:8001

API docs:
    http://localhost:8001/docs

Frontend:
    http://localhost:7860
```

## Model Bundle

The API container expects a local model bundle mounted at:

```bash
artifacts/model_bundle/
```

Generate it first with:
```bash
PYTHONPATH=src:. python -m versovector.training.build_dataset \
  --config configs/model_config.toml && \
PYTHONPATH=src:. python -m versovector.training.train_features \
  --config configs/model_config.toml && \
PYTHONPATH=src:. python -m versovector.training.train_supervised \
  --config configs/model_config.toml && \
PYTHONPATH=src:. python -m versovector.training.train_unsupervised \
  --config configs/model_config.toml && \
PYTHONPATH=src:. python -m versovector.training.register_model \
  --config configs/model_config.toml
```

## Notes

Generated model artifacts are not included in the Docker image by default.

For local development, the model bundle is mounted as a read-only volume.

For future Cloud Run deployment, the model bundle should be retrieved from a controlled artifact location such as Google Cloud Storage, Artifact Registry, or an MLflow artifact store.