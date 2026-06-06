<p align="left">
    <a href="https://www.python.org/" target="_blank">
        <img src="https://img.shields.io/badge/Python-3.10.11-3670A0?style=flat-square&logo=python&logoColor=ffdd54" />
    </a>
    <a href="https://scikit-learn.org/" target="_blank">
        <img src="https://img.shields.io/badge/scikit--learn-ML-orange?style=flat-square&logo=scikit-learn&logoColor=white" />
    </a>
    <a href="https://spacy.io/" target="_blank">
        <img src="https://img.shields.io/badge/spaCy-NLP-09A3D5?style=flat-square&logo=spacy&logoColor=white" />
    </a>
    <a href="https://numpy.org/" target="_blank">
        <img src="https://img.shields.io/badge/NumPy-Arrays-013243?style=flat-square&logo=numpy&logoColor=white" />
    </a>
    <a href="https://pandas.pydata.org/" target="_blank">
        <img src="https://img.shields.io/badge/Pandas-DataFrames-150458?style=flat-square&logo=pandas&logoColor=white" />
    </a>
    <a href="https://scipy.org/" target="_blank">
        <img src="https://img.shields.io/badge/SciPy-Sparse%20Matrix-8CAAE6?style=flat-square&logo=scipy&logoColor=white" />
    </a>
    <a href="https://matplotlib.org/" target="_blank">
        <img src="https://img.shields.io/badge/Matplotlib-Visualizations-11557C?style=flat-square" />
    </a>
    <a href="https://umap-learn.readthedocs.io/" target="_blank">
        <img src="https://img.shields.io/badge/UMAP-Optional%20Reduction-6A5ACD?style=flat-square" />
    </a>
    <a href="https://jupyter.org/" target="_blank">
        <img src="https://img.shields.io/badge/Jupyter-Notebooks-F37626?style=flat-square&logo=jupyter&logoColor=white" />
    </a>
    <a href="https://mlflow.org/" target="_blank">
        <img src="https://img.shields.io/badge/MLflow-Tracking-0194E2?style=flat-square&logo=mlflow&logoColor=white" />
    </a>
    <a href="https://fastapi.tiangolo.com/" target="_blank">
        <img src="https://img.shields.io/badge/FastAPI-Serving-009688?style=flat-square&logo=fastapi&logoColor=white" />
    </a>
    <a href="https://www.gradio.app/" target="_blank">
        <img src="https://img.shields.io/badge/Gradio-Frontend-F97316?style=flat-square&logo=gradio&logoColor=white" />
    </a>
    <a href="https://www.docker.com/" target="_blank">
        <img src="https://img.shields.io/badge/Docker-Services-2496ED?style=flat-square&logo=docker&logoColor=white" />
    </a>
    <a href="https://cloud.google.com/run" target="_blank">
        <img src="https://img.shields.io/badge/Cloud%20Run-Blueprint-4285F4?style=flat-square&logo=googlecloud&logoColor=white" />
    </a>
    <a href="https://www.terraform.io/" target="_blank">
        <img src="https://img.shields.io/badge/Terraform-IaC-844FBA?style=flat-square&logo=terraform&logoColor=white" />
    </a>
    <a href="https://github.com/features/actions" target="_blank">
        <img src="https://img.shields.io/badge/GitHub%20Actions-CI%2FCD-2088FF?style=flat-square&logo=githubactions&logoColor=white" />
    </a>
    <img src="https://img.shields.io/github/last-commit/HubertRonald/VersoVector?style=flat-square" />
    <img src="https://img.shields.io/github/license/HubertRonald/VersoVector?style=flat-square&color=success" />
</p>

# VersoVector

## Overview

VersoVector is an emotional-semantic NLP and MLOps project for poetry and lyrical language analysis.

The project starts with poetry as a dense form of emotional and symbolic language, using César Vallejo and English poetry corpora as a key reference point. Its long-term goal is to evolve into a mood-aware recommendation engine capable of mapping poems, lyric-like text, and user-provided fragments into interpretable affective spaces.

VersoVector combines:

* supervised multilabel tag prediction;
* semantic similarity;
* topic modeling;
* clustering;
* visual interpretation;
* model packaging;
* inference through a reusable Python package;
* FastAPI serving;
* a Gradio frontend product experience;
* Docker-based local services;
* testing;
* a sanitized Google Cloud deployment blueprint.

This public repository is designed as a portfolio-grade implementation. It demonstrates the technical and product direction of the system while intentionally excluding production-only assets such as real cloud project identifiers, secrets, billing configuration, private datasets, authentication logic, premium access control, analytics pipelines, and production infrastructure state.

## Product Vision

The target product is an emotional-semantic recommendation API and user-facing analysis experience.

Given a poem, lyric-like text, or short user-provided fragment, the system can return:

* predicted emotional or thematic tags;
* semantically similar poems or texts;
* dominant topic information;
* cluster assignment;
* explainable similarity signals;
* optional visualization metadata.

The long-term vision is to support mood-aware discovery for poetic and lyrical content.

For copyrighted lyrics, production usage should rely only on licensed, public-domain, metadata-based, short-excerpt, or user-provided content. Public demos should avoid storing or redistributing full copyrighted lyrics unless properly licensed.

## Long-Term Vision: From Poetry to Lyrics

VersoVector starts with poetry because poetry provides dense emotional and symbolic language.

The long-term vision is to evolve the system into an emotional-semantic recommendation engine for poetic and lyrical content.

The goal is not only to recommend similar texts, but to explain why they feel emotionally close:

* shared moods;
* symbolic intensity;
* semantic neighborhoods;
* dominant topics;
* affective tags;
* interpretability signals.

A future version could support mood-aware discovery for music, lyrics, journaling, education, literary exploration, libraries, editorial platforms, ecommerce catalog enrichment, and creative recommendation systems.

## GCP Deployment Blueprint

The public repository includes a sanitized Google Cloud deployment blueprint.

This blueprint shows how VersoVector could be deployed as a containerized emotional-semantic recommendation system using Cloud Run, Artifact Registry, Terraform, Docker services, and optional MLflow or remote artifact storage.

![VersoVector GCP Architecture](figs/VersoVector-GCP-Architecture.png)

The diagram separates the public portfolio architecture from the private production-only layer.

Public repository scope:

```text
architecture
demo frontend
FastAPI backend
Docker services
tests
documentation
sanitized Terraform blueprint
```

Private production scope:

```text
real GCP projects
real IAM bindings
secrets
billing
authentication
premium features
analytics
production monitoring
private datasets
deployment state
```

The Terraform blueprint lives under:

```text
infra/gcp-cloud-run-blueprint/
```

It is intentionally generic and does not include real production values.

## Current Analytical Pipeline

The current analytical pipeline is organized as six reproducible notebooks:

| Notebook                                       | Purpose                                                                                           |
| ---------------------------------------------- | ------------------------------------------------------------------------------------------------- |
| `01_cleaning_pipeline.ipynb`                   | Cleans raw datasets, normalizes metadata, creates `poem_id`, and generates `poems_processed.csv`. |
| `02_feature_pipeline.ipynb`                    | Fits the shared feature pipeline on the reference corpus and transforms external poems.           |
| `03_embeddings_supervised.ipynb`               | Trains the supervised multilabel tag prediction model.                                            |
| `04_embeddings_unsupervised.ipynb`             | Generates unsupervised artifacts: similarity, topics, clustering, and 2D projections.             |
| `05_supervised_unsupervised_integration.ipynb` | Integrates supervised predictions with unsupervised outputs.                                      |
| `06_visualizations.ipynb`                      | Generates final visualizations from integration artifacts.                                        |

## Additional Documentation

The project documentation is split by layer:

* [`docs/model_topology.md`](docs/model_topology.md): conceptual model topology, modeling rationale, and feature flow.
* [`notebook/README.md`](notebook/README.md): notebook-first analytical guide for the exploratory pipeline.
* [`src/versovector/README.md`](src/versovector/README.md): Python package guide for training, inference, and API serving.
* [`apps/README.md`](apps/README.md): application-layer documentation for user-facing apps.
* [`apps/frontend/README.md`](apps/frontend/README.md): Gradio frontend documentation and local execution guide.
* [`services/README.md`](services/README.md): local services overview and Docker Compose guide.
* [`services/api/README.md`](services/api/README.md): FastAPI service packaging and local Docker execution.
* [`services/frontend/README.md`](services/frontend/README.md): frontend service packaging and Docker execution.
* [`tests/README.md`](tests/README.md): unit and integration testing strategy.
* [`infra/gcp-cloud-run-blueprint/README.md`](infra/gcp-cloud-run-blueprint/README.md): sanitized Cloud Run deployment blueprint.

## Repository Structure

```text
VersoVector/
├── apps/
│   └── frontend/                 # Gradio user-facing demo application
├── artifacts/                    # Generated artifacts, mostly ignored by Git
├── configs/                      # Project and model configuration
├── data/                         # Raw and processed local datasets
├── docs/                         # Technical documentation and model topology
├── figs/                         # Figures used by notebooks and README
├── infra/
│   └── gcp-cloud-run-blueprint/  # Sanitized Terraform blueprint for public portfolio use
├── modules/                      # Reusable analytical project modules
│   ├── classification/
│   ├── clustering/
│   ├── evaluation/
│   ├── features/
│   ├── integration/
│   ├── io/
│   └── preprocessing/
├── notebook/                     # Analytical notebooks 01–06
├── services/
│   ├── api/                      # FastAPI Docker/service packaging
│   ├── frontend/                 # Gradio Docker/service packaging
│   └── compose.yaml              # Local API + frontend orchestration
├── src/
│   └── versovector/
│       ├── training/             # Scripted training pipeline
│       ├── inference/            # Model bundle loading and analysis logic
│       └── api/                  # FastAPI serving layer
├── tests/
│   ├── unit/                     # Unit tests
│   └── integration/              # Integration tests
├── utils/                        # Shared constants and utility helpers
├── requirements.txt
├── requirements-dev.txt
├── requirements-test.txt
├── requirements-mlops.txt
├── requirements-api.txt
├── requirements-frontend.txt
├── requirements-umap.txt
├── pyproject.toml
├── tox.ini
└── README.md
```

## Modeling Approach

VersoVector combines supervised and unsupervised NLP methods.

### Supervised Branch

The supervised branch predicts multilabel poetic tags using a sparse normalized feature representation and a multilabel classifier.

The strongest current candidate is:

```text
OneVsRestClassifier
└── StackingClassifier
    ├── ComplementNB
    ├── MultinomialNB
    └── LogisticRegression final estimator
```

The sparse pipeline avoids converting the full feature matrix to a dense representation, which significantly improves runtime and memory usage.

### Unsupervised Branch

The unsupervised branch generates:

* cosine similarity recommendations;
* nearest-neighbor search;
* LDA topic modeling;
* KMeans and GMM cluster assignments;
* exploratory Agglomerative and DBSCAN clusters;
* 2D UMAP or t-SNE projections.

For clustering, the pipeline avoids densifying the full sparse matrix. Dimensionality reduction is applied first, then clustering is performed on the reduced dense representation.

## Key Design Principles

### Reference vs External Corpus

The project separates the corpus into two roles:

| Role        | Meaning                                                                                 |
| ----------- | --------------------------------------------------------------------------------------- |
| `reference` | Poetry Foundation corpus used to fit feature and model spaces.                          |
| `external`  | External poets projected into the learned space, such as César Vallejo or future poets. |

This avoids fitting the vector space directly on external poems and makes extrapolation clearer.

### Stable Poem Identity

Every poem receives a unique technical identifier:

```text
poem_id
```

This avoids joining outputs by non-unique fields such as title, poet, source, or corpus role.

### Sparse-First Pipeline

The main feature pipeline is sparse and normalized:

```python
build_feature_pipeline(
    input_is_processed=True,
    to_dense=False,
    normalize=True,
)
```

Dense matrices are only created after dimensionality reduction when required by downstream algorithms.

### Artifact-First Serving

The serving layer does not retrain models.

It consumes a generated model bundle:

```text
artifacts/model_bundle/
```

This keeps training, packaging, inference, API serving, and frontend usage clearly separated.

## MLOps Workflow

The notebook pipeline is the analytical foundation. The MLOps workflow turns that foundation into reproducible scripts and deployable artifacts.

### Phase 1 — Scripted Training

Notebook logic is exposed through reproducible scripts:

```text
src/versovector/training/
├── build_dataset.py
├── train_features.py
├── train_supervised.py
├── train_unsupervised.py
├── register_model.py
└── mlflow_utils.py
```

The scripts are mapped to the notebook pipeline:

| Script                  | Notebook equivalent                | Purpose                                          |
| ----------------------- | ---------------------------------- | ------------------------------------------------ |
| `build_dataset.py`      | `01_cleaning_pipeline.ipynb`       | Builds the processed corpus.                     |
| `train_features.py`     | `02_feature_pipeline.ipynb`        | Fits and serializes the shared feature pipeline. |
| `train_supervised.py`   | `03_embeddings_supervised.ipynb`   | Trains the supervised multilabel classifier.     |
| `train_unsupervised.py` | `04_embeddings_unsupervised.ipynb` | Generates unsupervised artifacts.                |
| `register_model.py`     | Model packaging step               | Builds `artifacts/model_bundle/`.                |

### Phase 2 — MLflow Experiment Tracking

MLflow is used to track:

* parameters;
* metrics;
* model artifacts;
* feature configuration;
* dataset versions;
* model versions.

MLflow support is optional and controlled by:

```toml
[mlflow]
enabled = true
```

If MLflow is disabled, the scripts can run without MLflow installed.

### Phase 3 — Model Packaging

The trained system is packaged as a model bundle:

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

The first deployable product is a single `PoemAnalyzer` bundle that can return tags, similar poems, topics, and clusters.

### Phase 4 — Inference Package

The inference layer loads a generated model bundle and exposes reusable Python components:

```text
src/versovector/inference/
├── artifact_loader.py
├── poem_analyzer.py
├── schemas.py
├── similarity_search.py
├── tag_predictor.py
└── topic_clusterer.py
```

Example local usage:

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

### Phase 5 — FastAPI Serving Layer

The API layer exposes the inference package through FastAPI:

```text
src/versovector/api/
├── dependencies.py
├── main.py
├── schemas.py
└── settings.py
```

Current endpoints:

```text
GET  /
GET  /health
GET  /ready
GET  /v1/model-info
POST /v1/analyze
POST /v1/predict-tags
POST /v1/similar
```

Example request:

```json
{
  "title": "Untitled",
  "poet": "anonymous",
  "poem": "I walk through the rain carrying a memory...",
  "user_tags": ["memory", "sadness"],
  "top_k_tags": 5,
  "top_n_similar": 5
}
```

Example response shape:

```json
{
  "predicted_tags": [
    {"tag": "sadness", "score": 0.82},
    {"tag": "memory", "score": 0.71}
  ],
  "similar_poems": [
    {
      "title": "black herald",
      "poet": "cesar vallejo",
      "score": 0.76
    }
  ],
  "topic": {
    "topic_id": 3,
    "terms": "death, night, heart, god, pain"
  },
  "cluster": {
    "kmeans": 2,
    "gmm": 1
  }
}
```

### Phase 6 — Testing Foundation

The project includes a separate testing layer for unit and integration validation.

The test suite is intended to validate:

* preprocessing utilities;
* artifact loading helpers;
* inference schemas;
* tag prediction logic;
* similarity search formatting;
* model bundle loading;
* `PoemAnalyzer` end-to-end execution;
* FastAPI endpoints through `TestClient`.

### Phase 7 — Packaging and Services

VersoVector includes a packaging and local services foundation.

This layer separates Python package code, user-facing applications, and service-level deployment assets:

```text
apps/
└── frontend/

services/
├── api/
├── frontend/
└── compose.yaml
```

The API and frontend can run locally through Docker Compose.

### Phase 8 — UI Product Experience

The first frontend is implemented with Gradio because it is well suited for interactive AI demos.

The frontend allows users to:

* submit a poem or lyric-like fragment;
* select suggested tags;
* request emotional or thematic tag prediction;
* retrieve semantically similar poems;
* inspect topic and cluster information;
* access a secondary API and developer guide.

The frontend does not load model artifacts directly. It communicates with the FastAPI backend.

### Phase 9 — Cloud Deployment Blueprint

The public repository includes a sanitized Terraform blueprint under:

```text
infra/gcp-cloud-run-blueprint/
```

It models the following public portfolio architecture:

```text
Users
  ↓
Gradio Frontend
  Cloud Run
  ↓
FastAPI API
  Cloud Run
  ↓
PoemAnalyzer
  ↓
Model Bundle / Reference Metadata
```

Supporting infrastructure:

```text
Artifact Registry
Cloud Run services
runtime service accounts
public frontend access
private API access from frontend
optional remote model artifact store
```

The blueprint is not a production deployment.

It intentionally avoids:

```text
real project IDs
real service account keys
real backend state
real IAM policies
real secrets
real billing setup
real domains
private datasets
premium access logic
production monitoring configuration
```

A production implementation should live in a private repository, for example:

```text
VersoVector-Platform
```

That private repository would contain real environments such as:

```text
infra/
├── environments/
│   ├── dev/
│   ├── staging/
│   └── prod/
├── backend.tf
├── terraform.tfvars
├── iam/
├── secrets/
├── cloud-run/
├── artifact-registry/
└── monitoring/
```

### Phase 10 — CI/CD

GitHub Actions will be used to automate:

* linting;
* unit tests;
* integration tests;
* package builds;
* Docker image builds;
* model validation;
* optional deployment workflows.

The public repository may include CI/CD examples or non-sensitive workflows. Production deployment automation should remain private when it involves real cloud projects, service accounts, secrets, billing, private environments, or production access rules.

## Packaging and Services Foundation

VersoVector includes an initial packaging and local services foundation.

This layer prepares the project for containerized execution by separating Python package code, user-facing applications, and service-level deployment assets.

```text
apps/
└── frontend/
    ├── README.md
    ├── app.py
    ├── client.py
    ├── tag_catalog.py
    ├── ui_formatters.py
    └── assets/

services/
├── README.md
├── api/
│   ├── Dockerfile
│   ├── README.md
│   └── .dockerignore
├── frontend/
│   ├── Dockerfile
│   ├── README.md
│   └── .dockerignore
└── compose.yaml
```

### Python Package Build

The project uses `pyproject.toml` as the package metadata and build configuration source.

A local wheel can be generated with:

```bash
python -m build
```

The generated files are written to:

```text
dist/
├── versovector-<version>-py3-none-any.whl
└── versovector-<version>.tar.gz
```

The `dist/` directory is a generated build output and should not be committed to Git.

The wheel packages project code, including:

```text
src/versovector/
modules/
utils/
```

It does not include generated model artifacts such as:

```text
artifacts/model_bundle/
*.joblib
mlruns/
mlflow.db
```

### Local Services

The services layer currently defines two local services:

* `api`: FastAPI backend powered by `PoemAnalyzer`.
* `frontend`: Gradio demo app that calls the API over HTTP.

Run both services locally with Docker Compose:

```bash
docker compose -f services/compose.yaml up --build
```

Local URLs:

```text
API:
    http://localhost:8001

API docs:
    http://localhost:8001/docs

Frontend:
    http://localhost:7860
```

### Model Bundle Strategy

In this foundation stage, the API image does not bake model artifacts into the container.

Instead, the local model bundle is mounted as a read-only volume:

```text
artifacts/model_bundle:/app/artifacts/model_bundle:ro
```

This keeps code packaging and model artifact management separated.

For future Cloud Run deployment, the model bundle should be retrieved from a controlled artifact location such as Google Cloud Storage, Artifact Registry, or an MLflow artifact store.

## Cloud Deployment Blueprint

The `infra/gcp-cloud-run-blueprint/` directory contains a sanitized Terraform blueprint for a possible Google Cloud deployment.

The blueprint includes conceptual resources for:

* Artifact Registry;
* Cloud Run API service;
* Cloud Run frontend service;
* runtime service accounts;
* public frontend invoker access;
* private API access from the frontend service account;
* optional remote model artifact store configuration.

This layer is intended to show how the system could be deployed without exposing real production infrastructure.

Basic Terraform workflow:

```bash
cd infra/gcp-cloud-run-blueprint

cp terraform.tfvars.example terraform.tfvars

terraform init
terraform fmt -recursive
terraform validate
terraform plan -var-file=terraform.tfvars
```

Do not commit `terraform.tfvars`, `.terraform/`, or `*.tfstate` files.

## Local Setup

Create a virtual environment:

```bash
python3.10 -m venv .venv
source .venv/bin/activate
```

Install core dependencies:

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

Install development dependencies:

```bash
pip install -r requirements-dev.txt
```

Install test dependencies:

```bash
pip install -r requirements-test.txt
```

Install MLOps dependencies:

```bash
pip install -r requirements-mlops.txt
```

Install API dependencies:

```bash
pip install -r requirements-api.txt
```

Install frontend dependencies:

```bash
pip install -r requirements-frontend.txt
```

Install optional UMAP dependencies:

```bash
pip install -r requirements-umap.txt
```

Download the spaCy model:

```bash
python -m spacy download en_core_web_lg
```

Start Jupyter:

```bash
jupyter notebook
```

## Running the Analytical Notebook Pipeline

Run notebooks in order:

```text
01_cleaning_pipeline.ipynb
02_feature_pipeline.ipynb
03_embeddings_supervised.ipynb
04_embeddings_unsupervised.ipynb
05_supervised_unsupervised_integration.ipynb
06_visualizations.ipynb
```

Generated files are stored under:

```text
artifacts/
figs/
data/
```

Large local artifacts are not intended to be committed to Git.

## Running the Scripted Training Pipeline

Run from the repository root:

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

## Running the API Locally

Start the API:

```bash
PYTHONPATH=src:. uvicorn versovector.api.main:app \
  --host 0.0.0.0 \
  --port 8001 \
  --reload
```

Health check:

```bash
curl http://localhost:8001/health | jq .
```

Readiness check:

```bash
curl http://localhost:8001/ready | jq .
```

Model information:

```bash
curl http://localhost:8001/v1/model-info | jq .
```

Full analysis:

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

Tag prediction:

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

Similarity search:

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

## Running the Frontend Locally

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

Run the frontend:

```bash
VERSOVECTOR_API_BASE_URL=http://localhost:8001 \
VERSOVECTOR_API_TIMEOUT_SECONDS=300 \
PORT=7860 \
python apps/frontend/app.py
```

Open:

```text
http://localhost:7860
```

## Running Local Services with Docker Compose

Run both API and frontend:

```bash
docker compose -f services/compose.yaml up --build
```

Local URLs:

```text
API:
    http://localhost:8001

API docs:
    http://localhost:8001/docs

Frontend:
    http://localhost:7860
```

## Testing Strategy

Testing is a separate quality layer.

Recommended structure:

```text
tests/
├── unit/
│   ├── test_preprocessing.py
│   ├── test_artifact_store.py
│   ├── test_inference_schemas.py
│   ├── test_similarity_search.py
│   └── test_tag_predictor.py
└── integration/
    ├── test_model_bundle_loading.py
    ├── test_poem_analyzer_end_to_end.py
    └── test_api_endpoints.py
```

### Unit Tests

Unit tests validate isolated Python components without requiring a full model bundle.

Examples:

* text cleaning and tag parsing;
* artifact path helpers;
* schema serialization;
* similarity result formatting;
* tag ranking logic.

### Integration Tests

Integration tests validate that multiple layers work together.

Examples:

* loading a minimal test model bundle;
* running `PoemAnalyzer.analyze_dict`;
* calling FastAPI endpoints with `TestClient`;
* validating API response shapes.

### Services vs Integration Tests

The `services/` directory is for deployment packaging and service-level assets.

The `tests/integration/` directory is for validating interactions between components.

They are related, but not the same thing:

```text
services/
    How the API or frontend is packaged and deployed.

tests/integration/
    Whether components work together correctly.
```

### Test Commands

Run unit tests:

```bash
PYTHONPATH=src:. pytest tests/unit
```

Run integration tests:

```bash
PYTHONPATH=src:. pytest tests/integration
```

Run coverage:

```bash
PYTHONPATH=src:. pytest --cov=src --cov=modules tests/
```

Run through tox:

```bash
tox -e unit
tox -e integration
tox -e coverage
tox -e build
```

## Dependency Groups

The project separates dependencies by purpose:

```text
requirements.txt
    Core runtime dependencies.

requirements-dev.txt
    Development dependencies: notebooks, linting, formatting, packaging, and local development tools.

requirements-test.txt
    Test-only dependencies for unit and integration tests.

requirements-mlops.txt
    MLflow and model tracking dependencies.

requirements-api.txt
    FastAPI serving dependencies.

requirements-frontend.txt
    Gradio frontend/demo dependencies.

requirements-umap.txt
    Optional UMAP dependencies.
```

## Artifact Policy

The repository does not version large generated artifacts such as:

```text
*.joblib
*.pkl
*.pickle
*.npy
*.npz
```

Ignored generated paths include:

```text
artifacts/features/*
artifacts/supervised/*
artifacts/unsupervised/*
artifacts/integration/*
artifacts/model_bundle/*
mlruns/
mlflow.db
dist/
build/
*.egg-info/
```

These artifacts are regenerated locally from notebooks or training scripts.

A production MLOps version should store model artifacts in a dedicated artifact store, such as MLflow artifacts, Google Cloud Storage, or another controlled registry.

## Infrastructure Policy

The public repository may include sanitized infrastructure blueprints.

It should not include:

```text
real project IDs
real terraform.tfvars files
Terraform state
service account keys
secrets
private IAM bindings
billing configuration
production domains
premium access rules
customer data
private datasets
```

The production infrastructure should live in a private repository or private environment-specific configuration.

## Figures

The following figures are intentionally kept in the repository:

```text
figs/vallejo_tfidf_vectors.png
figs/poemas_2d_umap_clustering_kmeans.png
figs/VersoVector-GCP-Architecture.png
```

`figs/vallejo_tfidf_vectors.png` is used in the project documentation.

`figs/poemas_2d_umap_clustering_kmeans.png` is kept as a visual reference for the organic UMAP structure observed during exploration.

`figs/VersoVector-GCP-Architecture.png` documents the sanitized public GCP blueprint for the project.

Additional figures generated by `06_visualizations.ipynb` may include:

```text
figs/poemas_2d_integracion.png
figs/vallejo_projection.png
figs/top_tags_by_cluster.png
figs/cluster_source_matrix.png
figs/cluster_topic_matrix.png
```

## Current Status

Completed:

* modular notebook pipeline;
* reusable feature, classification, clustering, evaluation, I/O, preprocessing, and integration modules;
* supervised multilabel classifier;
* unsupervised similarity, topic, and clustering outputs;
* integration artifacts;
* final visualizations;
* script-based training layer;
* optional MLflow helper utilities;
* model bundle packaging;
* inference package through `PoemAnalyzer`;
* FastAPI serving layer;
* unit and integration testing foundation;
* Python package build foundation;
* Docker-based local services foundation;
* Gradio frontend product experience foundation;
* sanitized Cloud Run deployment blueprint.

Next:

* refine GitHub Actions for linting, tests, package build, and image build;
* define a private production infrastructure repository;
* move real Terraform environments and secrets to a private repo;
* prepare a controlled model artifact strategy using GCS, MLflow artifacts, or Artifact Registry;
* add production observability and analytics outside the public portfolio repo;
* evaluate authentication, billing, and premium features in a private product track.

## .gitignore

The base `.gitignore` was generated from [gitignore.io](https://www.toptal.com/developers/gitignore/) with the filters `python`, `macos`, and `windows`.

Additional project-specific ignore rules exclude generated datasets, model artifacts, MLflow local files, Terraform local state, local package build outputs, and large binary files.

## Authors

* **Hubert Ronald** - Initial Work - [HubertRonald](https://github.com/HubertRonald)
* See also the list of [contributors](https://github.com/HubertRonald/VersoVector/contributors) who participated in this project.

## License and Copyright

The source code in this repository is distributed under the MIT License. See the [LICENSE](./LICENSE) file for more details.

The repository may reference public-domain poems, open datasets, translations, or third-party textual materials for experimentation. These materials may be subject to their own licenses, terms of use, or copyright restrictions and are not automatically covered by the MIT License of this codebase.

For production, commercial deployment, hosted inference, or redistribution of trained model artifacts, dataset and text usage rights should be reviewed separately. A production version of VersoVector should rely only on public-domain, properly licensed, or otherwise cleared textual sources.
