# VersoVector Python Package

This directory contains the Python package layer for VersoVector.

VersoVector is an emotional-semantic NLP and MLOps project for poetic and lyrical language analysis. The package under `src/versovector/` provides scriptable training, inference, and API serving components that complement the exploratory notebooks and reusable modules in the repository.

The goal of this package is to move the project from notebook-based experimentation toward reproducible training, model bundle generation, local inference, and FastAPI serving.

## Package Structure

```text
src/versovector/
├── __init__.py
├── README.md
├── training/
│   ├── __init__.py
│   ├── build_dataset.py
│   ├── train_features.py
│   ├── train_supervised.py
│   ├── train_unsupervised.py
│   ├── register_model.py
│   └── mlflow_utils.py
├── inference/
│   ├── __init__.py
│   ├── artifact_loader.py
│   ├── poem_analyzer.py
│   ├── schemas.py
│   ├── similarity_search.py
│   ├── tag_predictor.py
│   └── topic_clusterer.py
└── api/
    ├── __init__.py
    ├── dependencies.py
    ├── main.py
    ├── schemas.py
    └── settings.py
```

## Relationship with the Rest of the Repository

The repository is organized into several complementary layers:

```text
notebook/
    Explains, validates, and visualizes the modeling workflow.

modules/
    Provides reusable analytical logic:
    preprocessing, feature engineering, classification, clustering,
    evaluation, integration, and artifact I/O helpers.

src/versovector/training/
    Runs the same pipeline without Jupyter.
    Generates datasets, features, supervised artifacts,
    unsupervised artifacts, and model bundles.

src/versovector/inference/
    Loads a generated model bundle and exposes reusable inference logic.

src/versovector/api/
    Exposes the inference layer through FastAPI endpoints.

artifacts/
    Stores generated local artifacts.
    These files are intentionally ignored by Git.
```

## Training Pipeline

The training scripts are the production-oriented counterpart of the notebooks.

| Script                  | Notebook equivalent                | Purpose                                                                     |
| ----------------------- | ---------------------------------- | --------------------------------------------------------------------------- |
| `build_dataset.py`      | `01_cleaning_pipeline.ipynb`       | Builds the processed corpus.                                                |
| `train_features.py`     | `02_feature_pipeline.ipynb`        | Fits and serializes the shared feature pipeline.                            |
| `train_supervised.py`   | `03_embeddings_supervised.ipynb`   | Trains the supervised multilabel classifier.                                |
| `train_unsupervised.py` | `04_embeddings_unsupervised.ipynb` | Generates unsupervised artifacts: neighbors, topics, clusters, projections. |
| `register_model.py`     | Model packaging step               | Builds `artifacts/model_bundle/` and optionally logs it with MLflow.        |

## Configuration

The training and packaging scripts use:

```text
configs/model_config.toml
```

This file defines the project metadata, data locations, feature configuration, supervised model settings, unsupervised settings, MLflow behavior, and artifact directories.

A generated model bundle also contains a snapshot copy of the same configuration:

```text
artifacts/model_bundle/model_config.toml
```

The source configuration is versioned in Git. The bundle snapshot is generated locally and should not be committed.

## Training Commands

Run the scripts from the repository root.

### 1. Syntax Check

```bash
python -m py_compile \
  src/versovector/training/mlflow_utils.py \
  src/versovector/training/build_dataset.py \
  src/versovector/training/train_features.py \
  src/versovector/training/train_supervised.py \
  src/versovector/training/train_unsupervised.py \
  src/versovector/training/register_model.py
```

### 2. Execute the Training Pipeline

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

`PYTHONPATH=src:.` allows Python to resolve both:

```text
src/versovector/
modules/
utils/
```

This is useful during the transition phase while reusable analytical logic still lives under `modules/`.

## MLflow

MLflow support is optional.

The scripts use `src/versovector/training/mlflow_utils.py` to enable or disable MLflow based on:

```toml
[mlflow]
enabled = true
```

If MLflow is disabled, the scripts can run without `mlflow` installed.

If MLflow is enabled, install the MLOps dependencies first:

```bash
pip install -r requirements-mlops.txt
```

Then start the local MLflow UI:

```bash
mlflow ui
```

Local MLflow files are ignored by Git:

```text
mlruns/
mlflow.db
```

## Model Bundle

The final training step creates:

```text
artifacts/model_bundle/
```

The model bundle is the contract between training and inference.

Expected bundle contents may include:

```text
artifacts/model_bundle/
├── model_config.toml
├── model_metadata.json
├── feature_pipeline.joblib
├── supervised_classifier.joblib
├── multilabel_binarizer.joblib
├── nearest_neighbors.joblib
├── reference_metadata.csv
├── lda_model.joblib
├── lda_count_vectorizer.joblib
├── dimensionality_reducer.joblib
├── kmeans_model.joblib
├── gmm_model.joblib
├── lda_topics.csv
├── supervised_metrics.csv
├── unsupervised_metadata.json
└── unsupervised_results.csv
```

Generated artifacts are intentionally not versioned in Git.

## Inference Layer

The inference layer loads a generated model bundle and exposes reusable Python components.

Main entrypoint:

```python
from versovector.inference import PoemAnalyzer

analyzer = PoemAnalyzer.from_bundle("artifacts/model_bundle")

result = analyzer.analyze_dict(
    title="test poem",
    poet="anonymous",
    poem="I walk through the rain carrying a memory of light.",
    top_k_tags=5,
    top_n_similar=5,
)

print(result)
```

### Inference Components

| File                   | Purpose                                                             |
| ---------------------- | ------------------------------------------------------------------- |
| `artifact_loader.py`   | Loads the model bundle artifacts.                                   |
| `poem_analyzer.py`     | Orchestrates preprocessing, tags, similarity, topics, and clusters. |
| `tag_predictor.py`     | Predicts emotional or thematic tags.                                |
| `similarity_search.py` | Finds semantically similar poems.                                   |
| `topic_clusterer.py`   | Predicts dominant topics and cluster assignments.                   |
| `schemas.py`           | Defines inference response structures.                              |

### Syntax Check

```bash
python -m py_compile \
  src/versovector/inference/__init__.py \
  src/versovector/inference/artifact_loader.py \
  src/versovector/inference/poem_analyzer.py \
  src/versovector/inference/schemas.py \
  src/versovector/inference/similarity_search.py \
  src/versovector/inference/tag_predictor.py \
  src/versovector/inference/topic_clusterer.py
```

### Local Inference Test

```bash
PYTHONPATH=src:. python - <<'PY'
from versovector.inference import PoemAnalyzer

analyzer = PoemAnalyzer.from_bundle("artifacts/model_bundle")

result = analyzer.analyze_dict(
    title="test poem",
    poet="anonymous",
    poem="I walk through the rain carrying a memory of light.",
    top_k_tags=5,
    top_n_similar=5,
)

print(result)
PY
```

## API Layer

The API layer exposes the inference package through FastAPI.

### API Components

| File              | Purpose                                               |
| ----------------- | ----------------------------------------------------- |
| `main.py`         | FastAPI application and route registration.           |
| `schemas.py`      | Request and response models.                          |
| `settings.py`     | Runtime settings and environment configuration.       |
| `dependencies.py` | Lazy `PoemAnalyzer` loading and dependency injection. |
| `__init__.py`     | Package-level exports.                                |

### Syntax Check

```bash
python -m py_compile \
  src/versovector/api/__init__.py \
  src/versovector/api/dependencies.py \
  src/versovector/api/main.py \
  src/versovector/api/schemas.py \
  src/versovector/api/settings.py
```

### Launch API Locally

```bash
PYTHONPATH=src:. uvicorn versovector.api.main:app \
  --host 0.0.0.0 \
  --port 8001 \
  --reload
```

### Health Check

```bash
curl http://localhost:8001/health | jq .
```

### Model Information

```bash
curl http://localhost:8001/v1/model-info | jq .
```

### Full Poem Analysis

```bash
curl -X POST http://localhost:8001/v1/analyze \
  -H "Content-Type: application/json" \
  -d '{
    "title": "test poem",
    "poet": "anonymous",
    "poem": "I walk through the rain carrying a memory of light.",
    "top_k_tags": 5,
    "top_n_similar": 5
  }' | jq .
```

### Tag Prediction

```bash
curl -X POST http://localhost:8001/v1/predict-tags \
  -H "Content-Type: application/json" \
  -d '{
    "title": "test poem",
    "poet": "anonymous",
    "poem": "I walk through the rain carrying a memory of light.",
    "top_k_tags": 5
  }' | jq .
```

### Similarity Search

```bash
curl -X POST http://localhost:8001/v1/similar \
  -H "Content-Type: application/json" \
  -d '{
    "title": "test poem",
    "poet": "anonymous",
    "poem": "I walk through the rain carrying a memory of light.",
    "top_n_similar": 5
  }' | jq .
```

## API Endpoints

```text
GET  /
GET  /health
GET  /v1/model-info
POST /v1/analyze
POST /v1/predict-tags
POST /v1/similar
```

## Environment Variables

API settings use the `VERSOVECTOR_` prefix.

Examples:

```bash
export VERSOVECTOR_MODEL_BUNDLE_DIR="artifacts/model_bundle"
export VERSOVECTOR_DEFAULT_TOP_K_TAGS=5
export VERSOVECTOR_DEFAULT_TOP_N_SIMILAR=5
```

## Dependency Groups

The project separates dependency groups by purpose:

```text
requirements.txt
    Core runtime dependencies.

requirements-dev.txt
    Notebooks, tests, linting, formatting, packaging.

requirements-mlops.txt
    MLflow and model tracking dependencies.

requirements-api.txt
    FastAPI serving dependencies.

requirements-umap.txt
    Optional UMAP dependencies.
```

Install the API dependencies with:

```bash
pip install -r requirements-api.txt
```

Install MLOps dependencies with:

```bash
pip install -r requirements-mlops.txt
```

## Artifact Policy

Generated artifacts should not be committed to Git.

Ignored paths include:

```text
artifacts/features/*
artifacts/supervised/*
artifacts/unsupervised/*
artifacts/integration/*
artifacts/model_bundle/*
mlruns/
mlflow.db
*.joblib
*.pkl
*.pickle
*.npy
*.npz
```

The source configuration remains versioned:

```text
configs/model_config.toml
```

The generated bundle snapshot is ignored:

```text
artifacts/model_bundle/model_config.toml
```

## Current Development Stage

This package currently supports:

* script-based dataset building;
* feature pipeline training;
* supervised multilabel training;
* unsupervised artifact generation;
* model bundle registration;
* local inference through `PoemAnalyzer`;
* local API serving through FastAPI.

Next steps:

* add tests for training, inference, and API layers;
* add Docker packaging under `services/api/`;
* add Cloud Run deployment support;
* add Terraform infrastructure;
* add GitHub Actions CI/CD;
* evolve toward a production-ready emotional-semantic recommendation service.
