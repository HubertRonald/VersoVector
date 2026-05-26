from __future__ import annotations

__all__ = [
    "cluster_metrics_from_silhouette"
]

import pandas as pd


def cluster_metrics_from_silhouette(
        sils: list[tuple[int, float]],
        model_name: str = "MiniBatchKMeans",
    ) -> pd.DataFrame:
    """Convert silhouette search results into a metrics DataFrame."""
    return pd.DataFrame([
        {"model": model_name, "k": int(k), "silhouette": float(sil),}
        for k, sil in sils
    ])
