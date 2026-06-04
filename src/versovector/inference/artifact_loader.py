from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from modules.io import (
    load_csv,
    load_joblib,
    load_json,
    load_toml_config,
    project_path,
)


@dataclass
class ModelBundle:
    """Container for all artifacts required by inference."""

    bundle_dir: Path
    config: dict[str, Any]
    model_metadata: dict[str, Any]
    feature_pipeline: Any
    supervised_classifier: Any
    multilabel_binarizer: Any
    nearest_neighbors: Any
    reference_metadata: pd.DataFrame
    lda_model: Any | None = None
    lda_count_vectorizer: Any | None = None
    dimensionality_reducer: Any | None = None
    kmeans_model: Any | None = None
    gmm_model: Any | None = None
    lda_topics: pd.DataFrame | None = None
    unsupervised_metadata: dict[str, Any] | None = None


def resolve_bundle_path(bundle_dir: str | Path = "artifacts/model_bundle") -> Path:
    """Resolve a bundle path relative to the project root."""
    path = Path(bundle_dir)
    return path if path.is_absolute() else project_path(str(path))


def load_optional_joblib(bundle_dir: Path, filename: str) -> Any | None:
    """Load an optional joblib artifact from the model bundle."""
    path = bundle_dir / filename
    return load_joblib(path) if path.is_file() else None


def load_optional_csv(bundle_dir: Path, filename: str) -> pd.DataFrame | None:
    """Load an optional CSV artifact from the model bundle."""
    path = bundle_dir / filename
    return load_csv(path) if path.is_file() else None


def load_optional_json(bundle_dir: Path, filename: str) -> dict[str, Any] | None:
    """Load an optional JSON artifact from the model bundle."""
    path = bundle_dir / filename
    return load_json(path) if path.is_file() else None


def load_model_bundle(bundle_dir: str | Path = "artifacts/model_bundle") -> ModelBundle:
    """Load a complete VersoVector model bundle for inference."""
    resolved_bundle_dir = resolve_bundle_path(bundle_dir)

    if not resolved_bundle_dir.is_dir():
        raise FileNotFoundError(f"Model bundle directory not found: {resolved_bundle_dir}")

    config_path = resolved_bundle_dir / "model_config.toml"
    metadata_path = resolved_bundle_dir / "model_metadata.json"

    required_files = [
        config_path,
        resolved_bundle_dir / "feature_pipeline.joblib",
        resolved_bundle_dir / "supervised_classifier.joblib",
        resolved_bundle_dir / "multilabel_binarizer.joblib",
        resolved_bundle_dir / "nearest_neighbors.joblib",
        resolved_bundle_dir / "reference_metadata.csv",
    ]

    missing_files = [path.name for path in required_files if not path.is_file()]

    if missing_files:
        raise FileNotFoundError(
            "The model bundle is incomplete. Missing files: "
            f"{', '.join(missing_files)}"
        )

    config = load_toml_config(config_path)
    model_metadata = load_json(metadata_path) if metadata_path.is_file() else {}

    return ModelBundle(
        bundle_dir=resolved_bundle_dir,
        config=config,
        model_metadata=model_metadata,
        feature_pipeline=load_joblib(resolved_bundle_dir / "feature_pipeline.joblib"),
        supervised_classifier=load_joblib(resolved_bundle_dir / "supervised_classifier.joblib"),
        multilabel_binarizer=load_joblib(resolved_bundle_dir / "multilabel_binarizer.joblib"),
        nearest_neighbors=load_joblib(resolved_bundle_dir / "nearest_neighbors.joblib"),
        reference_metadata=load_csv(resolved_bundle_dir / "reference_metadata.csv"),
        lda_model=load_optional_joblib(resolved_bundle_dir, "lda_model.joblib"),
        lda_count_vectorizer=load_optional_joblib(
            resolved_bundle_dir,
            "lda_count_vectorizer.joblib",
        ),
        dimensionality_reducer=load_optional_joblib(
            resolved_bundle_dir,
            "dimensionality_reducer.joblib",
        ),
        kmeans_model=load_optional_joblib(resolved_bundle_dir, "kmeans_model.joblib"),
        gmm_model=load_optional_joblib(resolved_bundle_dir, "gmm_model.joblib"),
        lda_topics=load_optional_csv(resolved_bundle_dir, "lda_topics.csv"),
        unsupervised_metadata=load_optional_json(
            resolved_bundle_dir,
            "unsupervised_metadata.json",
        ),
    )
