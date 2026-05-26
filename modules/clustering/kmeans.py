from __future__ import annotations

__all__ = ["reduce_features_pca", "fit_minibatch_kmeans_range_fast"]

from typing import Any

import numpy as np
from sklearn.cluster import MiniBatchKMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score


def reduce_features_pca(
        X,
        n_components: int = 100,
        random_state: int = 42,
    ) -> tuple[np.ndarray, PCA]:
    """Reduce a dense feature matrix with PCA before clustering."""
    
    max_components = min(n_components, X.shape[0] - 1, X.shape[1])
    reducer = PCA(n_components=max_components, random_state=random_state)
    X_reduced = reducer.fit_transform(X)
    
    return X_reduced, reducer


def fit_minibatch_kmeans_range_fast(
        X,
        kmin: int = 2,
        kmax: int = 8,
        random_state: int = 42,
        sample_size: int = 1000,
        batch_size: int = 1024,
    ) -> tuple[dict[str, Any], list[tuple[int, float]]]:
    """Search a compact KMeans solution using MiniBatchKMeans and sampled silhouette."""
    
    best = {"k": None, "sil": -1.0, "model": None, "labels": None}
    sils: list[tuple[int, float]] = []
    
    n_samples = X.shape[0]
    sample_size = min(sample_size, n_samples)

    for k in range(kmin, kmax + 1):
        model = MiniBatchKMeans(
            n_clusters=k,
            random_state=random_state,
            batch_size=batch_size,
            n_init=3,
        )
        labels = model.fit_predict(X)

        sil = silhouette_score(
            X,
            labels,
            metric="cosine",
            sample_size=sample_size,
            random_state=random_state,
        )
        sils.append((k, float(sil)))

        if sil > best["sil"]:
            best = {"k": k, "sil": float(sil), "model": model, "labels": labels}

    return best, sils
