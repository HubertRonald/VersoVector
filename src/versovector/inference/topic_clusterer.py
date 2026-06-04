from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from .schemas import ClusterPrediction, TopicPrediction


class TopicClusterer:
    """Infer topics and clusters from trained unsupervised artifacts."""

    def __init__(
            self,
            lda_model: Any | None = None,
            lda_count_vectorizer: Any | None = None,
            dimensionality_reducer: Any | None = None,
            kmeans_model: Any | None = None,
            gmm_model: Any | None = None,
            lda_topics: pd.DataFrame | None = None,
        ) -> None:
        self.lda_model = lda_model
        self.lda_count_vectorizer = lda_count_vectorizer
        self.dimensionality_reducer = dimensionality_reducer
        self.kmeans_model = kmeans_model
        self.gmm_model = gmm_model
        self.lda_topics = lda_topics

    def _topic_terms(self, topic_id: int) -> str | None:
        """Return human-readable terms for a topic when available."""
        if self.lda_topics is None or self.lda_topics.empty:
            return None

        if "topic_id" not in self.lda_topics.columns:
            return None

        topic_rows = self.lda_topics.loc[self.lda_topics["topic_id"].astype(int).eq(topic_id)]

        if topic_rows.empty:
            return None

        row = topic_rows.iloc[0]

        if "top_words" in row.index and not pd.isna(row["top_words"]):
            return str(row["top_words"])

        if "lda_topic_terms" in row.index and not pd.isna(row["lda_topic_terms"]):
            return str(row["lda_topic_terms"])

        return None

    def predict_topic(self, processed_text: str) -> TopicPrediction | None:
        """Predict the dominant LDA topic for one processed poem."""
        if self.lda_model is None or self.lda_count_vectorizer is None:
            return None

        X_topic = self.lda_count_vectorizer.transform([processed_text])
        topic_distribution = self.lda_model.transform(X_topic)[0]

        topic_id = int(np.argmax(topic_distribution))
        probability = float(np.max(topic_distribution))

        return TopicPrediction(
            topic_id=topic_id,
            probability=round(probability, 6),
            terms=self._topic_terms(topic_id),
        )

    def _cluster_features(self, X):
        """Project features into the clustering space when a reducer exists."""
        if self.dimensionality_reducer is None:
            return X

        return self.dimensionality_reducer.transform(X)

    def predict_cluster(self, X) -> ClusterPrediction | None:
        """Predict KMeans and GMM clusters for one vectorized poem."""
        if self.kmeans_model is None and self.gmm_model is None:
            return None

        X_cluster = self._cluster_features(X)

        kmeans_cluster = None
        gmm_cluster = None

        if self.kmeans_model is not None:
            kmeans_cluster = int(self.kmeans_model.predict(X_cluster)[0])

        if self.gmm_model is not None:
            gmm_cluster = int(self.gmm_model.predict(X_cluster)[0])

        return ClusterPrediction(
            kmeans=kmeans_cluster,
            gmm=gmm_cluster,
        )