# equivalent to notebook 03
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import joblib
import mlflow
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer


from modules.classification import (
    build_fast_multilabel_classifier,
    build_stacking_classifier,
    filter_rare_multilabel_tags,
)
from modules.evaluation import compute_multilabel_metrics
from modules.preprocessing import parse_tags
from modules.io import (
    ensure_dir,
    get_nested,
    load_toml_config,
    project_path,
    load_csv,
    save_csv,
    save_json,
)

def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Train supervised VersoVector model.")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/model_config.toml",
        help="Path to the TOML configuration file.",
    )
    return parser.parse_args()


def build_classifier(model_type: str, seed: int):
    """Build the supervised multilabel classifier."""
    model_type = model_type.strip()

    if model_type == "stacking_classifier":
        return build_stacking_classifier(seed=seed)

    if model_type in {"complement_nb", "multinomial_nb", "logreg"}:
        return build_fast_multilabel_classifier(
            model_type=model_type,
            seed=seed,
            n_jobs=-1,
        )

    raise ValueError(f"Unsupported classifier_model_type: {model_type}")


def main() -> None:
    """Train supervised multilabel model and log experiment with MLflow."""
    args = parse_args()
    config = load_toml_config(args.config)

    seed = int(get_nested(config, "training", "seed", default=42))

    classifier_model_type = str(
        get_nested(config, "training", "classifier_model_type", default="stacking_classifier")
    ).strip()

    min_label_count = int(
        get_nested(config, "supervised", "min_label_count", default=10)
    )

    test_size = float(
        get_nested(config, "supervised", "test_size", default=0.20)
    )

    experiment_name = str(
        get_nested(
            config,
            "mlflow",
            "experiment_name",
            default="versovector-supervised",
        )
    )

    features_dir = project_path(
        str(get_nested(config, "artifacts", "features_dir", default="artifacts/features"))
    )

    supervised_dir = project_path(
        str(get_nested(config, "artifacts", "supervised_dir", default="artifacts/supervised"))
    )

    ensure_dir(supervised_dir)

    reference_metadata_path = features_dir / "reference_metadata.csv"
    external_metadata_path = features_dir / "external_metadata.csv"
    feature_pipeline_path = features_dir / "feature_pipeline.joblib"
    feature_metadata_path = features_dir / "feature_pipeline_metadata.json"

    reference_df = load_csv(reference_metadata_path)
    external_df = load_csv(external_metadata_path) if external_metadata_path.is_file() else pd.DataFrame()

    reference_df["tags"] = reference_df["tags"].apply(parse_tags)

    if not external_df.empty and "tags" in external_df.columns:
        external_df["tags"] = external_df["tags"].apply(parse_tags)

    feature_pipeline = joblib.load(feature_pipeline_path)

    reference_texts = reference_df["poem_processed"].astype(str).tolist()
    external_texts = (
        external_df["poem_processed"].astype(str).tolist()
        if not external_df.empty
        else []
    )

    X_reference = feature_pipeline.transform(reference_texts)
    X_external = feature_pipeline.transform(external_texts) if external_texts else None

    reference_df = reference_df.reset_index(drop=True)
    reference_df["reference_row_id"] = np.arange(len(reference_df))

    supervised_source_df = (
        reference_df
        .loc[reference_df["tags"].map(len) > 0]
        .reset_index(drop=True)
    )

    supervised_df, label_summary_df, label_filter_metadata = filter_rare_multilabel_tags(
        df=supervised_source_df,
        tags_col="tags",
        output_col="filtered_tags",
        min_label_count=min_label_count,
    )

    supervised_row_idx = supervised_df["reference_row_id"].to_numpy()
    X_supervised = X_reference[supervised_row_idx]

    mlb = MultiLabelBinarizer(classes=label_filter_metadata["kept_labels"])
    y = mlb.fit_transform(supervised_df["filtered_tags"])

    indices = np.arange(X_supervised.shape[0])

    idx_train, idx_test, y_train, y_test = train_test_split(
        indices,
        y,
        test_size=test_size,
        random_state=seed,
    )

    X_train = X_supervised[idx_train]
    X_test = X_supervised[idx_test]

    classifier = build_classifier(
        model_type=classifier_model_type,
        seed=seed,
    )

    mlflow.set_experiment(experiment_name)

    run_name = f"supervised-{classifier_model_type}-min-label-{min_label_count}"

    with mlflow.start_run(run_name=run_name):
        classifier.fit(X_train, y_train)

        y_pred = classifier.predict(X_test)

        try:
            y_score = classifier.predict_proba(X_test)
        except Exception:
            y_score = None

        metrics_df = compute_multilabel_metrics(
            y_true=y_test,
            y_pred=y_pred,
            y_score=y_score,
        )

        metrics_df["model_type"] = classifier_model_type
        metrics_df["min_label_count"] = min_label_count
        metrics_df["n_train"] = len(idx_train)
        metrics_df["n_test"] = len(idx_test)
        metrics_df["n_labels"] = len(mlb.classes_)
        metrics_df["n_reference_documents"] = int(reference_df.shape[0])
        metrics_df["n_external_documents"] = int(external_df.shape[0])
        metrics_df["feature_source"] = "artifacts/features/feature_pipeline.joblib"

        classifier_path = supervised_dir / "supervised_classifier.joblib"
        mlb_path = supervised_dir / "multilabel_binarizer.joblib"
        metrics_path = supervised_dir / "supervised_metrics.csv"
        predictions_path = supervised_dir / "supervised_predictions.csv"
        label_summary_path = supervised_dir / "label_filter_summary.csv"
        label_metadata_path = supervised_dir / "label_filter_metadata.json"
        model_metadata_path = supervised_dir / "supervised_model_metadata.json"

        joblib.dump(classifier, classifier_path)
        joblib.dump(mlb, mlb_path)

        save_csv(metrics_df, metrics_path)
        save_csv(label_summary_df, label_summary_path)
        save_json(label_filter_metadata, label_metadata_path)

        reference_pred = classifier.predict(X_reference)
        reference_tags = mlb.inverse_transform(reference_pred)

        reference_predictions = reference_df[
            [
                "poem_id",
                "title",
                "title_raw",
                "poet",
                "poet_raw",
                "source",
                "corpus_role",
            ]
        ].copy()

        reference_predictions["predicted_tags"] = list(map(list, reference_tags))
        reference_predictions["predicted_tags_json"] = [
            json.dumps(list(tags), ensure_ascii=False)
            for tags in reference_tags
        ]

        if X_external is not None and not external_df.empty:
            external_pred = classifier.predict(X_external)
            external_tags = mlb.inverse_transform(external_pred)

            external_predictions = external_df[
                [
                    "poem_id",
                    "title",
                    "title_raw",
                    "poet",
                    "poet_raw",
                    "source",
                    "corpus_role",
                ]
            ].copy()

            external_predictions["predicted_tags"] = list(map(list, external_tags))
            external_predictions["predicted_tags_json"] = [
                json.dumps(list(tags), ensure_ascii=False)
                for tags in external_tags
            ]

            supervised_predictions = pd.concat(
                [reference_predictions, external_predictions],
                ignore_index=True,
            )
        else:
            supervised_predictions = reference_predictions

        save_csv(supervised_predictions, predictions_path)

        model_metadata: dict[str, Any] = {
            "model_type": classifier_model_type,
            "classifier": "OneVsRestClassifier",
            "feature_pipeline_artifact": "artifacts/features/feature_pipeline.joblib",
            "classifier_artifact": "artifacts/supervised/supervised_classifier.joblib",
            "mlb_artifact": "artifacts/supervised/multilabel_binarizer.joblib",
            "input_column": "poem_processed",
            "fit_corpus_role": "reference",
            "prediction_corpus_roles": ["reference", "external"],
            "min_label_count": int(min_label_count),
            "test_size": float(test_size),
            "seed": int(seed),
            "n_train": int(len(idx_train)),
            "n_test": int(len(idx_test)),
            "n_labels": int(len(mlb.classes_)),
            "classes": mlb.classes_.tolist(),
        }

        save_json(model_metadata, model_metadata_path)

        params = {
            "classifier_model_type": classifier_model_type,
            "min_label_count": min_label_count,
            "to_dense": bool(get_nested(config, "features", "to_dense", default=False)),
            "normalize": bool(get_nested(config, "features", "normalize", default=True)),
            "n_labels": int(len(mlb.classes_)),
            "n_train": int(len(idx_train)),
            "n_test": int(len(idx_test)),
            "n_reference_documents": int(reference_df.shape[0]),
            "n_external_documents": int(external_df.shape[0]),
        }

        mlflow.log_params(params)

        metric_row = metrics_df.iloc[0].to_dict()
        for key in ["jaccard_micro", "jaccard_macro", "roc_auc_micro"]:
            value = metric_row.get(key)
            if value is not None and not pd.isna(value):
                mlflow.log_metric(key, float(value))

        mlflow.log_artifact(str(metrics_path))
        mlflow.log_artifact(str(label_summary_path))
        mlflow.log_artifact(str(label_metadata_path))
        mlflow.log_artifact(str(model_metadata_path))

        if feature_metadata_path.is_file():
            mlflow.log_artifact(str(feature_metadata_path))

        mlflow.log_artifact(str(classifier_path))
        mlflow.log_artifact(str(mlb_path))

        print("Supervised training completed.")
        print(f"Run ID: {mlflow.active_run().info.run_id}")
        print(metrics_df)


if __name__ == "__main__":
    main()
