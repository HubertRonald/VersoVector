from __future__ import annotations

import numpy as np
from scipy import sparse

def row_l2_norms(X, n_rows: int = 10) -> np.ndarray:
    """
    Calcula norma L2 por fila para matrices sparse o dense.
    """
    X_sample = X[:n_rows]

    if sparse.issparse(X_sample):
        return np.sqrt(X_sample.multiply(X_sample).sum(axis=1)).A1

    return np.linalg.norm(X_sample, axis=1)
