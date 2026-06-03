# equivalent to notebook 04
from __future__ import annotations

import argparse
import json
from typing import Any

import numpy as np
import pandas as pd
from scipy import sparse
from versovector.training.mlflow_utils import (
    log_mlflow_artifacts,
    start_mlflow_run,
)

from sklearn.cluster import AgglomerativeClustering, DBSCAN
from sklearn.manifold import TSNE
from sklearn.metrics.pairwise import cosine_distances
from sklearn.neighbors import NearestNeighbors

try:
    import umap

    UMAP_AVAILABLE = True
except ImportError:
    umap = None
    UMAP_AVAILABLE = False

from modules.clustering import (
    extract_top_words,
    fit_gmm,
    fit_lda_topics,
    fit_minibatch_kmeans_range_fast,
    reduce_features_dimensionality,
    topic_terms_map,
    transform_lda_topics,
)
from modules.evaluation import cluster_metrics_from_silhouette
from modules.io import (
    get_nested,
    load_csv,
    load_joblib,
    load_toml_config,
    project_path,
    save_csv,
    save_joblib,
    save_json,
)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Train VersoVector unsupervised artifacts.")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/model_config.toml",
        help="Path to the TOML configuration file.",
    )
    return parser.parse_args()


def neighbors_to_lists(
    indices: np.ndarray,
    distances: np.ndarray,
    metadata: pd.DataFrame,
) -> tuple[list[list[str]], list[list[float]]]:
    """Convert nearest-neighbor indices and distances into title and score lists."""
    titles = metadata["title"].astype(str).tolist()

    nearest_titles: list[list[str]] = []
    nearest_scores: list[list[float]] = []

    for row_idx in range(indices.shape[0]):
        row_titles: list[str] = []
        row_scores: list[float] = []

        for neighbor_idx, distance in zip(indices[row_idx], distances[row_idx]):
            if int(neighbor_idx) == row_idx:
                continue

            row_titles.append(titles[int(neighbor_idx)])
            row_scores.append(round(float(1.0 - distance), 4))

        nearest_titles.append(row_titles)
        nearest_scores.append(row_scores)

    return nearest_titles, nearest_scores


def main() -> None:
    """Train unsupervised artifacts and log them with MLflow."""
    args = parse_args()
    config = load_toml_config(args.config)

    seed = int(get_nested(config, "training", "seed", default=42))
    text_column = get_nested(config, "data", "text_column", default="poem_processed")

    features_dir = project_path(
        get_nested(config, "artifacts", "features_dir", default="artifacts/features")
    )
    unsupervised_dir = project_path(
        get_nested(config, "artifacts", "unsupervised_dir", default="artifacts/unsupervised")
    )

    n_topics = int(get_nested(config, "unsupervised", "n_topics", default=7))
    topic_max_features = int(
        get_nested(config, "unsupervised", "topic_max_features", default=5000)
    )
    reduction_components = int(
        get_nested(config, "unsupervised", "reduction_components", default=100)
    )
    kmin = int(get_nested(config, "unsupervised", "kmin", default=2))
    kmax = int(get_nested(config, "unsupervised", "kmax", default=8))
    top_n_neighbors = int(
        get_nested(config, "unsupervised", "top_n_neighbors", default=5)
    )
    sample_size = int(get_nested(config, "unsupervised", "sample_size", default=1000))
    batch_size = int(get_nested(config, "unsupervised", "batch_size", default=1024))

    compute_exploratory = bool(
        get_nested(config, "unsupervised", "compute_exploratory_clusters", default=True)
    )
    compute_2d_projection = bool(
        get_nested(config, "unsupervised", "compute_2d_projection", default=True)
    )

    mlflow_enabled = bool(get_nested(config, "mlflow", "enabled", default=False))
    experiment_name = get_nested(
        config,
        "mlflow",
        "experiment_name",
        default="versovector-emotional-semantic-recommender",
    )

    feature_pipeline = load_joblib(features_dir / "feature_pipeline.joblib")
    reference_df = load_csv(features_dir / "reference_metadata.csv")
    external_df = load_csv(features_dir / "external_metadata.csv")

    reference_texts = reference_df[text_column].astype(str).tolist()
    external_texts = external_df[text_column].astype(str).tolist()

    X_reference = feature_pipeline.transform(reference_texts)
    X_external = feature_pipeline.transform(external_texts) if external_texts else None

    if X_external is not None and X_external.shape[0] > 0:
        X_all = sparse.vstack([X_reference, X_external])
        all_metadata = pd.concat([reference_df, external_df], ignore_index=True)
    else:
        X_all = X_reference
        all_metadata = reference_df.copy()

    all_metadata = all_metadata.reset_index(drop=True)

    nn_all = NearestNeighbors(metric="cosine", algorithm="brute")
    nn_all.fit(X_all)

    distances, indices = nn_all.kneighbors(
        X_all,
        n_neighbors=min(top_n_neighbors + 1, X_all.shape[0]),
    )

    nearest_titles_cosine, nearest_scores_cosine = neighbors_to_lists(
        indices=indices,
        distances=distances,
        metadata=all_metadata,
    )

    nearest_neighbors_reference = NearestNeighbors(metric="cosine", algorithm="brute")
    nearest_neighbors_reference.fit(X_reference)

    lda_model, topic_vectorizer, _, lda_topics_reference = fit_lda_topics(
        reference_texts,
        n_components=n_topics,
        max_features=topic_max_features,
        random_state=seed,
    )

    if external_texts:
        _, lda_topics_external = transform_lda_topics(
            lda_model,
            topic_vectorizer,
            external_texts,
        )
        lda_topics = np.vstack([lda_topics_reference, lda_topics_external])
    else:
        lda_topics = lda_topics_reference

    top_words_df = extract_top_words(
        lda_model=lda_model,
        topic_vectorizer=topic_vectorizer,
        n_top=10,
    )

    topic_terms = topic_terms_map(top_words_df, n_terms=5)

    X_reference_cluster, dimensionality_reducer = reduce_features_dimensionality(
        X_reference,
        n_components=reduction_components,
        random_state=seed,
        method="auto",
    )

    if X_external is not None and X_external.shape[0] > 0:
        X_external_cluster = dimensionality_reducer.transform(X_external)
        X_cluster_all = np.vstack([X_reference_cluster, X_external_cluster])
    else:
        X_external_cluster = np.empty((0, X_reference_cluster.shape[1]))
        X_cluster_all = X_reference_cluster

    best_km, km_sils = fit_minibatch_kmeans_range_fast(
        X_reference_cluster,
        kmin=kmin,
        kmax=kmax,
        random_state=seed,
        sample_size=sample_size,
        batch_size=batch_size,
    )

    reference_cluster_km = best_km["labels"]

    if X_external_cluster.shape[0] > 0:
        external_cluster_km = best_km["model"].predict(X_external_cluster)
        cluster_km = np.concatenate([reference_cluster_km, external_cluster_km])
    else:
        cluster_km = reference_cluster_km

    cluster_metrics_df = cluster_metrics_from_silhouette(
        km_sils,
        model_name="MiniBatchKMeans",
    )

    gmm_model, reference_cluster_gmm = fit_gmm(
        X_reference_cluster,
        n_components=int(best_km["k"]),
        covariance_type="diag",
        random_state=seed,
    )

    if X_external_cluster.shape[0] > 0:
        external_cluster_gmm = gmm_model.predict(X_external_cluster)
        cluster_gmm = np.concatenate([reference_cluster_gmm, external_cluster_gmm])
    else:
        cluster_gmm = reference_cluster_gmm

    if compute_exploratory:
        distance_matrix = cosine_distances(X_cluster_all)
        agg_model = AgglomerativeClustering(
            n_clusters=int(best_km["k"]),
            metric="precomputed",
            linkage="average",
        )
        cluster_agg = agg_model.fit_predict(distance_matrix)
    else:
        cluster_agg = np.full(X_cluster_all.shape[0], -1)

    if compute_2d_projection:
        if UMAP_AVAILABLE:
            reducer_2d = umap.UMAP(
                n_neighbors=15,
                min_dist=0.1,
                random_state=seed,
            )
            X2 = reducer_2d.fit_transform(X_cluster_all)
            reducer_name = "UMAP_combined"
        else:
            reducer_2d = TSNE(
                n_components=2,
                random_state=seed,
                metric="cosine",
            )
            X2 = reducer_2d.fit_transform(X_cluster_all)
            reducer_name = "t-SNE_combined"

        cluster_dbscan = DBSCAN(
            eps=0.5,
            min_samples=5,
            metric="euclidean",
        ).fit_predict(X2)
    else:
        X2 = np.zeros((X_cluster_all.shape[0], 2))
        reducer_name = "not_computed"
        cluster_dbscan = np.full(X_cluster_all.shape[0], -1)

    lda_topic_id = lda_topics.argmax(axis=1)
    lda_topic_prob = lda_topics.max(axis=1)

    unsupervised_results = pd.DataFrame(
        {
            "poem_id": all_metadata["poem_id"].values,
            "title": all_metadata["title"].values,
            "title_raw": all_metadata["title_raw"].values,
            "poet": all_metadata["poet"].values,
            "poet_raw": all_metadata["poet_raw"].values,
            "source": all_metadata["source"].values,
            "corpus_role": all_metadata["corpus_role"].values,
            "cluster_km": cluster_km,
            "cluster_gmm": cluster_gmm,
            "cluster_agg": cluster_agg,
            "cluster_dbscan": cluster_dbscan,
            "lda_topic_id": lda_topic_id,
            "lda_topic_prob": np.round(lda_topic_prob, 4),
            "lda_topic_terms": [
                topic_terms.get(int(topic_id), "")
                for topic_id in lda_topic_id
            ],
            "nearest_titles_cosine_json": [
                json.dumps(titles, ensure_ascii=False)
                for titles in nearest_titles_cosine
            ],
            "nearest_scores_cosine_json": [
                json.dumps(scores, ensure_ascii=False)
                for scores in nearest_scores_cosine
            ],
            "embedding_x": X2[:, 0],
            "embedding_y": X2[:, 1],
        }
    )

    embeddings_2d = unsupervised_results[
        [
            "poem_id",
            "title",
            "poet",
            "source",
            "corpus_role",
            "embedding_x",
            "embedding_y",
            "cluster_km",
            "cluster_gmm",
            "cluster_agg",
            "cluster_dbscan",
        ]
    ].copy()

    unsupervised_results_path = unsupervised_dir / "unsupervised_results.csv"
    lda_topics_path = unsupervised_dir / "lda_topics.csv"
    cluster_metrics_path = unsupervised_dir / "cluster_metrics.csv"
    embeddings_2d_path = unsupervised_dir / "embeddings_2d.csv"
    metadata_path = unsupervised_dir / "unsupervised_metadata.json"

    save_csv(unsupervised_results, unsupervised_results_path)
    save_csv(top_words_df.drop(columns=["top_words_list"], errors="ignore"), lda_topics_path)
    save_csv(cluster_metrics_df, cluster_metrics_path)
    save_csv(embeddings_2d, embeddings_2d_path)

    save_joblib(nearest_neighbors_reference, unsupervised_dir / "nearest_neighbors.joblib")
    save_joblib(lda_model, unsupervised_dir / "lda_model.joblib")
    save_joblib(topic_vectorizer, unsupervised_dir / "lda_count_vectorizer.joblib")
    save_joblib(dimensionality_reducer, unsupervised_dir / "dimensionality_reducer.joblib")
    save_joblib(best_km["model"], unsupervised_dir / "kmeans_model.joblib")
    save_joblib(gmm_model, unsupervised_dir / "gmm_model.joblib")

    metadata: dict[str, Any] = {
        "feature_pipeline_artifact": "artifacts/features/feature_pipeline.joblib",
        "input_column": text_column,
        "n_reference_documents": int(reference_df.shape[0]),
        "n_external_documents": int(external_df.shape[0]),
        "n_total_documents": int(all_metadata.shape[0]),
        "n_features": int(X_reference.shape[1]),
        "n_cluster_components": int(X_reference_cluster.shape[1]),
        "dimensionality_reducer": dimensionality_reducer.__class__.__name__,
        "best_kmeans_k": int(best_km["k"]),
        "best_kmeans_silhouette": float(best_km["sil"]),
        "lda_n_topics": int(lda_model.n_components),
        "reducer_name": reducer_name,
        "top_n_neighbors": int(top_n_neighbors),
        "compute_exploratory_clusters": compute_exploratory,
        "compute_2d_projection": compute_2d_projection,
    }

    save_json(metadata, metadata_path)

    mlflow_client, mlflow_run = start_mlflow_run(
        config=config,
        run_name="train-unsupervised",
    )

    with mlflow_run:
        if mlflow_client is not None:
            mlflow_client.log_params(
                {
                    "n_topics": n_topics,
                    "topic_max_features": topic_max_features,
                    "reduction_components": reduction_components,
                    "kmin": kmin,
                    "kmax": kmax,
                    "best_kmeans_k": int(best_km["k"]),
                    "top_n_neighbors": top_n_neighbors,
                    "reducer_name": reducer_name,
                }
            )

            mlflow_client.log_metric(
                "best_kmeans_silhouette",
                float(best_km["sil"]),
            )

            log_mlflow_artifacts(
                mlflow_client,
                [
                    unsupervised_results_path,
                    lda_topics_path,
                    cluster_metrics_path,
                    embeddings_2d_path,
                    metadata_path,
                    unsupervised_dir / "nearest_neighbors.joblib",
                    unsupervised_dir / "lda_model.joblib",
                    unsupervised_dir / "lda_count_vectorizer.joblib",
                    unsupervised_dir / "dimensionality_reducer.joblib",
                    unsupervised_dir / "kmeans_model.joblib",
                    unsupervised_dir / "gmm_model.joblib",
                ],
            )

    print("Unsupervised training completed.")
    print(f"X_reference: {X_reference.shape}")
    print(f"X_cluster_all: {X_cluster_all.shape}")
    print(f"Best KMeans k: {best_km['k']}")
    print(f"Best KMeans silhouette: {best_km['sil']:.4f}")


if __name__ == "__main__":
    main()
