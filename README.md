
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
    <a href="https://mermaid.js.org/" target="_blank">
        <img src="https://img.shields.io/badge/Mermaid-Diagrams-FF3670?style=flat-square&logo=mermaid&logoColor=white" />
    </a>
    <a href="https://code.visualstudio.com/download" target="_blank">
        <img src="https://img.shields.io/badge/VS%20Code-Editor-007ACC?style=flat-square&logo=visualstudiocode&logoColor=white" />
    </a>
    <img src="https://img.shields.io/github/last-commit/HubertRonald/VersoVector?style=flat-square" />
    <img src="https://img.shields.io/github/commit-activity/t/HubertRonald/VersoVector?style=flat-square&color=dodgerblue" />
    <img src="https://img.shields.io/github/license/HubertRonald/VersoVector?style=flat-square&color=success" />
</p>

# VersoVector

## Overview

VersoVector is an emotional-semantic NLP and MLOps project for poetry and lyrical language analysis.

The project starts with poetry as a dense form of emotional and symbolic language, using César Vallejo and English poetry corpora as a reference point. Its long-term goal is to evolve into a mood-aware recommendation engine capable of mapping poems, lyrics, and user-provided text into interpretable affective spaces.

VersoVector combines supervised multilabel tag prediction, semantic similarity, topic modeling, clustering, and visual interpretation. The current version is notebook-driven and modularized through reusable Python package sunder `modules/`. The next phase focuses on MLflow tracking, model packaging, FastAPI inference, Docker, Cloud Run deployment, Terraform infrastructure, and CI/CD.

## Product Vision

The target product is an emotional-semantic recommendation API.

Given a poem, lyric-like text, or short user-provided fragment, the system should return:

- predicted emotional or thematic tags;
- semantically similar poems or texts;
- dominant topics;
- cluster assignment;
- explainable similarity signals;
- optional visualization metadata.

The long-term vision is to support mood-aware discovery for poetic and lyrical content. For copyrighted lyrics, production usage should rely only on licensed, public-domain, metadata-based, or user-provided content.

## Long-Term Vision: From Poetry to Lyrics

VersoVector starts with poetry because poetry provides dense emotional and symbolic language. The long-term vision is to evolve the system into an emotional-semantic recommendation engine for poetic and lyrical content.

The goal is not only to recommend similar texts, but to explain why they feel emotionally close: shared moods, symbolic intensity, topics, semantic neighborhoods, and affective tags.

For lyrical content, the project will avoid storing or redistributing full copyrighted lyrics unless properly licensed. Public demos should rely on public-domain texts, licensed datasets, short excerpts, metadata, or user-provided text.

## Documentation

- [Model topology](docs/model_topology.md)
- [Python package guide](src/versovector/README.md)

## Current Analytical Pipeline

The current pipeline is organized as six reproducible notebooks:

| Notebook                                       | Purpose                                                                                           |
| ---------------------------------------------- | ------------------------------------------------------------------------------------------------- |
| `01_cleaning_pipeline.ipynb`                   | Cleans raw datasets, normalizes metadata, creates `poem_id`, and generates `poems_processed.csv`. |
| `02_feature_pipeline.ipynb`                    | Fits the shared feature pipeline on the reference corpus and transforms external poems.           |
| `03_embeddings_supervised.ipynb`               | Trains the supervised multilabel tag prediction model.                                            |
| `04_embeddings_unsupervised.ipynb`             | Generates unsupervised artifacts: similarity, topics, clustering, and 2D projections.             |
| `05_supervised_unsupervised_integration.ipynb` | Integrates supervised predictions with unsupervised outputs.                                      |
| `06_visualizations.ipynb`                      | Generates final visualizations from integration artifacts.                                        |

The detailed model topology, feature engineering flow, and mathematical notes are documented separately in:

```text
docs/model_topology.md
```

or, during the notebook-first phase:

```text
notebook/README.md
```

## Repository Structure

```text
VersoVector/
├── artifacts/                 # Generated artifacts, mostly ignored by Git
├── data/                      # Raw and processed local datasets
├── docs/                      # Technical documentation and model topology
├── figs/                      # Figures used by notebooks and README
├── modules/                   # Reusable project modules
│   ├── classification/
│   ├── clustering/
│   ├── evaluation/
│   ├── features/
│   ├── integration/
│   ├── io/
│   └── preprocesing/
├── notebook/                  # Analytical notebooks 01–06
├── requirements.txt
├── requirements-dev.txt
├── requirements-umap.txt
└── README.md
```

## Modeling Approach

VersoVector combines supervised and unsupervised NLP methods.

### Supervised branch

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

### Unsupervised branch

The unsupervised branch generates:

* cosine similarity recommendations;
* LDA topic modeling;
* KMeans and GMM cluster assignments;
* exploratory Agglomerative and DBSCAN clusters;
* 2D UMAP or t-SNE projections.

For clustering, the pipeline avoids densifying the full sparse matrix. Instead, dimensionality reduction is applied first, then clustering is performed on the reduced dense representation.

## Key Design Principles

### Reference vs external corpus

The project separates the corpus into two roles:

| Role        | Meaning                                                                                 |
| ----------- | --------------------------------------------------------------------------------------- |
| `reference` | Poetry Foundation corpus used to fit feature and model spaces.                          |
| `external`  | External poets projected into the learned space, such as César Vallejo or future poets. |

This avoids fitting the vector space directly on external poems and makes extrapolation clearer.

### Stable poem identity

Every poem receives a unique technical identifier:

```text
poem_id
```

This avoids joining outputs by non-unique fields such as title, poet, source, or corpus role.

### Sparse-first pipeline

The main feature pipeline is sparse and normalized:

```python
build_feature_pipeline(
    input_is_processed=True,
    to_dense=False,
    normalize=True,
)
```

Dense matrices are only created after dimensionality reduction when required by downstream algorithms.

## MLOps Roadmap

The notebook pipeline is the analytical foundation. The next phase is to convert the project into a production-oriented MLOps workflow.

### Phase 1 — Scripted training

Move notebook logic into reproducible scripts:

```text
src/versovector/training/
├── build_dataset.py
├── train_features.py
├── train_supervised.py
├── train_unsupervised.py
└── register_model.py
```

### Phase 2 — MLflow experiment tracking

Use MLflow to track:

* parameters;
* metrics;
* model artifacts;
* feature configuration;
* dataset versions;
* model versions.

MLflow Tracking records runs with metadata, metrics, parameters, start/end times, and artifacts, making it suitable for comparing model experiments over time.

### Phase 3 — Model packaging

Package the trained system as a model bundle:

```text
artifacts/model_bundle/
├── feature_pipeline.joblib
├── supervised_classifier.joblib
├── multilabel_binarizer.joblib
├── nearest_neighbors.joblib
├── reference_metadata.csv
├── topic_model.joblib
├── cluster_models.joblib
└── model_config.toml
```

The first deployable product should be a single `PoemAnalyzer` bundle that can return tags, similar poems, topics, and clusters.

### Phase 4 — FastAPI inference service

Expose the model through a REST API:

```text
GET  /health
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
  "user_tags": ["memory", "sadness"]
}
```

Example response:

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
    "terms": ["death", "night", "heart", "god", "pain"]
  },
  "cluster": {
    "kmeans": 2,
    "gmm": 1
  }
}
```

### Phase 5 — Cloud Run deployment

The target serving architecture is:

```text
User
  ↓
Frontend Cloud Run service
  ↓
Orchestrator API Cloud Run service
  ↓
PoemAnalyzer model bundle
  ↓
Model artifacts / MLflow registry / GCS
```

Cloud Run is a managed platform for running containerized applications invoked through requests or events.

### Phase 6 — Infrastructure as Code

Terraform will be used to define and version cloud infrastructure, including:

* Artifact Registry;
* Cloud Run services;
* Cloud Storage buckets;
* service accounts;
* IAM bindings;
* secrets;
* CI/CD permissions.

Terraform is an infrastructure-as-code tool for safely building, changing, and versioning infrastructure.

### Phase 7 — CI/CD

GitHub Actions will be used to automate:

* linting;
* tests;
* Docker image builds;
* model validation;
* deployment to Cloud Run.

GitHub Actions supports automated software workflows, including CI/CD workflows directly from the repository.

## Local Setup

Create a virtual environment:

```bash
python3.10 -m venv .venv
source .venv/bin/activate
```

Install dependencies:

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

Install development dependencies if needed:

```bash
pip install -r requirements-dev.txt
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

## Running the Analytical Pipeline

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

## Artifact Policy

The repository does not version large generated artifacts such as:

```text
*.joblib
*.pkl
*.pickle
*.npy
*.npz
```

These artifacts are regenerated locally from notebooks or future training scripts.

The long-term MLOps version will store model artifacts in a dedicated artifact store, such as MLflow artifacts or Google Cloud Storage.

## Figures

The following figures are intentionally kept in the repository:

```text
figs/vallejo_tfidf_vectors.png
figs/poemas_2d_umap_clustering_kmeans.png
```

Additional figures generated by `06_visualizations.ipynb` may include:

```text
figs/poemas_2d_integracion.png
figs/vallejo_projection.png
figs/top_tags_by_cluster.png
figs/cluster_source_matrix.png
figs/cluster_topic_matrix.png
```

## Current Status

The project is currently in the notebook-to-MLOps transition stage.

Completed:

* modular notebook pipeline;
* reusable feature, classification, clustering, evaluation, I/O and integration modules;
* supervised multilabel classifier;
* unsupervised similarity, topic and clustering outputs;
* integration artifacts;
* final visualizations.

Next:

* move training logic from notebooks into scripts;
* add MLflow tracking;
* define a `PoemAnalyzer` inference class;
* package a model bundle;
* expose a FastAPI service;
* containerize the service;
* deploy to Cloud Run;
* manage infrastructure with Terraform;
* automate CI/CD with GitHub Actions.

## .gitignore

It was generated in [gitignore.io](https://www.toptal.com/developers/gitignore/) with the filters `python`, `macos`, and `windows`, and consumed through its API as a raw file from the terminal:

```bash
curl -L https://www.toptal.com/developers/gitignore/api/python,macos,windows > .gitignore
```

## Authors

* **Hubert Ronald** - Initial Work - [HubertRonald](https://github.com/HubertRonald)
* See also the list of [contributors](https://github.com/HubertRonald/VersoVector/contributors) who participated in this project.


## License and Copyright

The source code in this repository is distributed under the MIT License. See the [LICENSE](./LICENSE) file for more details.

The repository may reference public-domain poems, open datasets, translations, or third-party textual materials for experimentation. These materials may be subject to their own licenses, terms of use, or copyright restrictions and are not automatically covered by the MIT License of this codebase.

For production, commercial deployment, hosted inference, or redistribution of trained model artifacts, dataset and text usage rights should be reviewed separately. A production version of VersoVector should rely only on public-domain, properly licensed, or otherwise cleared textual sources.
