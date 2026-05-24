from __future__ import annotations

__all__ = [
    "compute_multilabel_metrics"
]

import pandas as pd
from sklearn.metrics import jaccard_score, roc_auc_score


def compute_multilabel_metrics(y_true, y_pred, y_score=None) -> pd.DataFrame:
    """Compute basic multilabel metrics and return a one-row DataFrame."""
    metrics = {
        "jaccard_micro": float(jaccard_score(y_true, y_pred, average="micro", zero_division=0)),
        "jaccard_macro": float(jaccard_score(y_true, y_pred, average="macro", zero_division=0)),
    }
    if y_score is not None:
        try:
            metrics["roc_auc_micro"] = float(roc_auc_score(y_true, y_score, average="micro"))
        except Exception:
            metrics["roc_auc_micro"] = None
    return pd.DataFrame([metrics])
