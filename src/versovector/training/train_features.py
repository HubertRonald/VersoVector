# equivalent to notebook 02
from __future__ import annotations

import argparse
from typing import Any

import pandas as pd
from versovector.training.mlflow_utils import (
    log_mlflow_artifacts,
    start_mlflow_run,
)
from modules.features import build_feature_pipeline
from modules.io import (
    get_nested,
    load_csv,
    load_toml_config,
    project_path,
    save_csv,
    save_joblib,
    save_json,
)
from modules.preprocessing import parse_tags


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Train VersoVector feature pipeline.")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/model_config.toml",
        help="Path to the TOML configuration file.",
    )
    return parser.parse_args()


def maybe_parse_tags(df: pd.DataFrame) -> pd.DataFrame:
    """Parse tags column if available."""
    df = df.copy()

    if "tags" in df.columns:
        df["tags"] = df["tags"].apply(parse_tags)

    return df


def main() -> None:
    """Fit the shared sparse normalized feature pipeline."""
    args = parse_args()
    config = load_toml_config(args.config)

    processed_corpus_path = project_path(
        get_nested(config, "data", "processed_corpus", default="data/poems_processed.csv")
    )

    features_dir = project_path(
        get_nested(config, "artifacts", "features_dir", default="artifacts/features")
    )

    reference_role = get_nested(config, "data", "reference_role", default="reference")
    external_role = get_nested(config, "data", "external_role", default="external")
    text_column = get_nested(config, "data", "text_column", default="poem_processed")

    input_is_processed = bool(
        get_nested(config, "features", "input_is_processed", default=True)
    )
    to_dense = bool(get_nested(config, "features", "to_dense", default=False))
    normalize = bool(get_nested(config, "features", "normalize", default=True))

    poems_df = maybe_parse_tags(load_csv(processed_corpus_path))

    required_cols = {
        "poem_id",
        "title",
        "poet",
        "source",
        "corpus_role",
        text_column,
    }

    missing_cols = required_cols.difference(poems_df.columns)

    if missing_cols:
        raise ValueError(f"Processed corpus is missing columns: {missing_cols}")

    reference_df = (
        poems_df
        .loc[poems_df["corpus_role"].eq(reference_role)]
        .reset_index(drop=True)
    )

    external_df = (
        poems_df
        .loc[poems_df["corpus_role"].eq(external_role)]
        .reset_index(drop=True)
    )

    reference_texts = reference_df[text_column].astype(str).tolist()
    external_texts = external_df[text_column].astype(str).tolist()

    feature_pipeline = build_feature_pipeline(
        input_is_processed=input_is_processed,
        to_dense=to_dense,
        normalize=normalize,
    )

    X_reference = feature_pipeline.fit_transform(reference_texts)

    # Optional cleanup:
    if external_texts:
        external_shape = feature_pipeline.transform(external_texts).shape
    else:
        external_shape = (0, X_reference.shape[1])

    feature_pipeline_path = features_dir / "feature_pipeline.joblib"
    reference_metadata_path = features_dir / "reference_metadata.csv"
    external_metadata_path = features_dir / "external_metadata.csv"
    metadata_path = features_dir / "feature_pipeline_metadata.json"

    save_joblib(feature_pipeline, feature_pipeline_path)
    save_csv(reference_df, reference_metadata_path)
    save_csv(external_df, external_metadata_path)

    metadata: dict[str, Any] = {
        "input_column": text_column,
        "pipeline": (
            "build_feature_pipeline("
            f"input_is_processed={input_is_processed}, "
            f"to_dense={to_dense}, "
            f"normalize={normalize})"
        ),
        "fit_corpus_role": reference_role,
        "transform_corpus_role": external_role,
        "n_reference_documents": int(X_reference.shape[0]),
        "n_external_documents": int(external_df.shape[0]),
        "n_features": int(X_reference.shape[1]),
        "to_dense": to_dense,
        "normalize": normalize,
        "serialized_feature_matrices": False,
        "reason": (
            "The feature matrix is regenerated from feature_pipeline.joblib "
            "to avoid large dense artifacts."
        ),
    }

    save_json(metadata, metadata_path)

    mlflow_client, mlflow_run = start_mlflow_run(
        config=config,
        run_name="train-features",
    )

    with mlflow_run:
        if mlflow_client is not None:
            mlflow_client.log_params(
                {
                    "input_is_processed": input_is_processed,
                    "to_dense": to_dense,
                    "normalize": normalize,
                    "n_features": int(X_reference.shape[1]),
                    "n_reference_documents": int(X_reference.shape[0]),
                    "n_external_documents": int(external_df.shape[0]),
                }
            )

            log_mlflow_artifacts(
                mlflow_client,
                [
                    metadata_path,
                    feature_pipeline_path,
                ],
            )

    print("Feature pipeline trained.")
    print(f"Reference shape: {X_reference.shape}")
    print(f"External shape: {external_shape}")
    print(f"Output: {features_dir}")


if __name__ == "__main__":
    main()
