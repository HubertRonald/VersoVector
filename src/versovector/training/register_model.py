# MLflow tracking + model bundle registration
from __future__ import annotations

import argparse
import shutil
from pathlib import Path
from typing import Any

import mlflow

from modules.io import (
    copy_file,
    ensure_dir,
    get_nested,
    load_toml_config,
    project_path,
    save_json,
)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Build and register VersoVector model bundle.")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/model_config.toml",
        help="Path to the TOML configuration file.",
    )
    return parser.parse_args()


def copy_optional(source: Path, target: Path) -> bool:
    """Copy an optional artifact if available."""
    return copy_file(source, target, required=False)


def main() -> None:
    """Build model bundle and log it to MLflow."""
    args = parse_args()
    config = load_toml_config(args.config)

    experiment_name = str(
        get_nested(
            config,
            "mlflow",
            "experiment_name",
            default="versovector-model-bundle",
        )
    )

    registered_model_name = str(
        get_nested(
            config,
            "mlflow",
            "registered_model_name",
            default="versovector-poem-analyzer",
        )
    )

    features_dir = project_path(
        str(get_nested(config, "artifacts", "features_dir", default="artifacts/features"))
    )

    supervised_dir = project_path(
        str(get_nested(config, "artifacts", "supervised_dir", default="artifacts/supervised"))
    )

    unsupervised_dir = project_path(
        str(get_nested(config, "artifacts", "unsupervised_dir", default="artifacts/unsupervised"))
    )

    model_bundle_dir = project_path(
        str(get_nested(config, "artifacts", "model_bundle_dir", default="artifacts/model_bundle"))
    )

    if model_bundle_dir.exists():
        shutil.rmtree(model_bundle_dir)

    ensure_dir(model_bundle_dir)

    config_source = project_path(args.config)
    config_target = model_bundle_dir / "model_config.toml"

    copy_file(config_source, config_target, required=True)

    required_files = {
        features_dir / "feature_pipeline.joblib": model_bundle_dir / "feature_pipeline.joblib",
        features_dir / "reference_metadata.csv": model_bundle_dir / "reference_metadata.csv",
        supervised_dir / "supervised_classifier.joblib": model_bundle_dir / "supervised_classifier.joblib",
        supervised_dir / "multilabel_binarizer.joblib": model_bundle_dir / "multilabel_binarizer.joblib",
    }

    for source, target in required_files.items():
        copy_file(source, target, required=True)

    optional_files = {
        features_dir / "feature_pipeline_metadata.json": model_bundle_dir / "feature_pipeline_metadata.json",
        supervised_dir / "supervised_metrics.csv": model_bundle_dir / "supervised_metrics.csv",
        supervised_dir / "supervised_model_metadata.json": model_bundle_dir / "supervised_model_metadata.json",
        supervised_dir / "label_filter_summary.csv": model_bundle_dir / "label_filter_summary.csv",
        supervised_dir / "label_filter_metadata.json": model_bundle_dir / "label_filter_metadata.json",
        unsupervised_dir / "lda_model.joblib": model_bundle_dir / "lda_model.joblib",
        unsupervised_dir / "lda_count_vectorizer.joblib": model_bundle_dir / "lda_count_vectorizer.joblib",
        unsupervised_dir / "kmeans_model.joblib": model_bundle_dir / "kmeans_model.joblib",
        unsupervised_dir / "gmm_model.joblib": model_bundle_dir / "gmm_model.joblib",
        unsupervised_dir / "unsupervised_metadata.json": model_bundle_dir / "unsupervised_metadata.json",
        unsupervised_dir / "unsupervised_results.csv": model_bundle_dir / "unsupervised_results.csv",
    }

    copied_optional = []

    for source, target in optional_files.items():
        if copy_optional(source, target):
            copied_optional.append(target.name)

    bundle_metadata: dict[str, Any] = {
        "registered_model_name": registered_model_name,
        "bundle_dir": "artifacts/model_bundle",
        "required_artifacts": [target.name for target in required_files.values()],
        "optional_artifacts": copied_optional,
        "source_config": str(args.config),
    }

    save_json(bundle_metadata, model_bundle_dir / "model_metadata.json")

    mlflow.set_experiment(experiment_name)

    with mlflow.start_run(run_name="register-model-bundle"):
        mlflow.log_param("registered_model_name", registered_model_name)
        mlflow.log_param("bundle_dir", "artifacts/model_bundle")
        mlflow.log_artifacts(str(model_bundle_dir), artifact_path="model_bundle")

        print("Model bundle built and logged to MLflow.")
        print(f"Bundle path: {model_bundle_dir}")
        print(f"Run ID: {mlflow.active_run().info.run_id}")


if __name__ == "__main__":
    main()
