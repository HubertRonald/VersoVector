from __future__ import annotations

from typing import Any

import numpy as np

from .schemas import TagPrediction


class TagPredictor:
    """Predict emotional or thematic tags from vectorized poem features."""

    def __init__(
            self,
            classifier: Any,
            multilabel_binarizer: Any,
        ) -> None:
        self.classifier = classifier
        self.multilabel_binarizer = multilabel_binarizer

    def _predict_scores(self, X) -> np.ndarray | None:
        """Return probability-like scores when the classifier supports them."""
        if not hasattr(self.classifier, "predict_proba"):
            return None

        scores = self.classifier.predict_proba(X)

        if isinstance(scores, list):
            scores = np.vstack([np.asarray(item)[:, 1] for item in scores]).T

        scores = np.asarray(scores)

        if scores.ndim == 1:
            scores = scores.reshape(1, -1)

        return scores

    def predict(
            self,
            X,
            top_k: int = 5,
            threshold: float | None = None,
        ) -> list[TagPrediction]:
        """
        Predict tags for a single vectorized poem.

        If probability scores are available, tags are ranked by score.
        Otherwise, binary predictions are returned with score 1.0.
        """
        classes = list(self.multilabel_binarizer.classes_)
        scores = self._predict_scores(X)

        if scores is not None:
            row_scores = scores[0]
            ranked_indices = np.argsort(row_scores)[::-1]

            predictions: list[TagPrediction] = []

            for idx in ranked_indices:
                score = float(row_scores[idx])

                if threshold is not None and score < threshold:
                    continue

                predictions.append(
                    TagPrediction(
                        tag=str(classes[idx]),
                        score=round(score, 6),
                    )
                )

                if len(predictions) >= top_k:
                    break

            return predictions

        binary_pred = np.asarray(self.classifier.predict(X))[0]
        active_indices = np.where(binary_pred == 1)[0]

        predictions = [
            TagPrediction(
                tag=str(classes[idx]),
                score=1.0,
            )
            for idx in active_indices[:top_k]
        ]

        return predictions
