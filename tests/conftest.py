from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest


class DummyFeaturePipeline:
    """Small fake feature pipeline for tests."""

    def transform(self, texts):
        return np.ones((len(texts), 3), dtype=float)


class DummyClassifier:
    """Small fake multilabel classifier for tests."""

    def predict_proba(self, X):
        return np.tile(np.array([[0.20, 0.80, 0.55]], dtype=float), (X.shape[0], 1))

    def predict(self, X):
        return (self.predict_proba(X) >= 0.50).astype(int)


class DummyMultiLabelBinarizer:
    """Small fake MultiLabelBinarizer for tests."""

    classes_ = np.array(["memory", "sadness", "light"])


class DummyNearestNeighbors:
    """Small fake nearest-neighbor index for tests."""

    def __init__(self, n_reference: int = 2):
        self.n_reference = n_reference

    def kneighbors(self, X, n_neighbors: int = 5):
        k = min(n_neighbors, self.n_reference)
        indices = np.tile(np.arange(k), (X.shape[0], 1))
        distances = np.tile(np.linspace(0.10, 0.30, k), (X.shape[0], 1))
        return distances, indices


class DummyLdaVectorizer:
    """Small fake topic vectorizer for tests."""

    def transform(self, texts):
        return np.ones((len(texts), 3), dtype=float)


class DummyLdaModel:
    """Small fake LDA model for tests."""

    n_components = 2

    def transform(self, X):
        return np.tile(np.array([[0.25, 0.75]], dtype=float), (X.shape[0], 1))


class DummyDimensionalityReducer:
    """Small fake dimensionality reducer for tests."""

    def transform(self, X):
        return np.ones((X.shape[0], 2), dtype=float)


class DummyClusterModel:
    """Small fake cluster model for tests."""

    def predict(self, X):
        return np.zeros(X.shape[0], dtype=int)


@pytest.fixture
def sample_reference_metadata() -> pd.DataFrame:
    """Return a minimal reference metadata DataFrame."""
    return pd.DataFrame(
        {
            "poem_id": ["poetry_foundation::000000::abc", "poetry_foundation::000001::def"],
            "title": ["first poem", "second poem"],
            "title_raw": ["First Poem", "Second Poem"],
            "poet": ["poet one", "poet two"],
            "poet_raw": ["Poet One", "Poet Two"],
            "source": ["poetry_foundation", "poetry_foundation"],
            "corpus_role": ["reference", "reference"],
            "poem_processed": ["memory light", "sadness night"],
            "tags": ['["memory"]', '["sadness"]'],
        }
    )


@pytest.fixture
def minimal_model_bundle_dir(
        tmp_path: Path,
        sample_reference_metadata: pd.DataFrame,
    ) -> Path:
    """Create a minimal model bundle under tmp_path."""
    bundle_dir = tmp_path / "model_bundle"
    bundle_dir.mkdir(parents=True, exist_ok=True)

    (bundle_dir / "model_config.toml").write_text(
        """
[project]
name = "versovector"
task = "emotional_semantic_recommendation"

[supervised]
top_k_tags = 3

[unsupervised]
top_n_neighbors = 2

[mlflow]
registered_model_name = "versovector-test-model"
""".strip(),
        encoding="utf-8",
    )

    (bundle_dir / "model_metadata.json").write_text(
        json.dumps(
            {
                "registered_model_name": "versovector-test-model",
                "bundle_dir": "artifacts/model_bundle",
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    sample_reference_metadata.to_csv(
        bundle_dir / "reference_metadata.csv",
        sep="|",
        index=False,
        encoding="utf-8",
    )

    lda_topics = pd.DataFrame(
        {
            "topic_id": [0, 1],
            "top_words": ["death night heart", "memory light rain"],
        }
    )
    lda_topics.to_csv(
        bundle_dir / "lda_topics.csv",
        sep="|",
        index=False,
        encoding="utf-8",
    )

    joblib.dump(DummyFeaturePipeline(), bundle_dir / "feature_pipeline.joblib")
    joblib.dump(DummyClassifier(), bundle_dir / "supervised_classifier.joblib")
    joblib.dump(DummyMultiLabelBinarizer(), bundle_dir / "multilabel_binarizer.joblib")
    joblib.dump(DummyNearestNeighbors(n_reference=2), bundle_dir / "nearest_neighbors.joblib")
    joblib.dump(DummyLdaModel(), bundle_dir / "lda_model.joblib")
    joblib.dump(DummyLdaVectorizer(), bundle_dir / "lda_count_vectorizer.joblib")
    joblib.dump(DummyDimensionalityReducer(), bundle_dir / "dimensionality_reducer.joblib")
    joblib.dump(DummyClusterModel(), bundle_dir / "kmeans_model.joblib")
    joblib.dump(DummyClusterModel(), bundle_dir / "gmm_model.joblib")

    return bundle_dir
