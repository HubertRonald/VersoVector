from __future__ import annotations

__all__ = ["fit_gmm"]

from sklearn.mixture import GaussianMixture


def fit_gmm(X, n_components: int, covariance_type: str = "diag", random_state: int = 42):
    """Fit a lightweight Gaussian Mixture model and return model plus labels."""
    model = GaussianMixture(
        n_components=n_components,
        covariance_type=covariance_type,
        random_state=random_state,
    )
    labels = model.fit_predict(X)
    return model, labels
