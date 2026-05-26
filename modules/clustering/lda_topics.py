from __future__ import annotations

__all__ = [
    "fit_lda_topics",
    "transform_lda_topics",
    "extract_top_words",
    "topic_terms_map",
]

import pandas as pd
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer

from utils import Constants


def fit_lda_topics(
        texts: list[str],
        n_components: int = 7,
        max_features: int = 5000,
        random_state: int = 42,
    ):
    """Fit LDA over a document-term count matrix."""
    vectorizer = CountVectorizer(stop_words="english", max_features=max_features)
    X_topics = vectorizer.fit_transform(texts)
    model = LatentDirichletAllocation(
        n_components=n_components,
        random_state=random_state,
    )
    topic_distribution = model.fit_transform(X_topics)
    return model, vectorizer, X_topics, topic_distribution


def transform_lda_topics(
        model: LatentDirichletAllocation,
        vectorizer: CountVectorizer,
        texts: list[str],
    ):
    """Transforma nuevos textos usando vectorizer + LDA ya ajustados."""
    X_topics = vectorizer.transform(texts)
    lda_topics = model.transform(X_topics)
    return X_topics, lda_topics


def extract_top_words(
        lda_model: LatentDirichletAllocation,
        topic_vectorizer: CountVectorizer,
        n_top: int = 10,
    ) -> pd.DataFrame:
    """Extrae palabras principales por tópico."""
    feature_names = topic_vectorizer.get_feature_names_out()

    rows = []

    for topic_id, topic in enumerate(lda_model.components_):
        top_idx = topic.argsort()[::-1][:n_top]
        words = [feature_names[i] for i in top_idx]

        rows.append({
            "topic_id": topic_id,
            "top_words": ", ".join(words),
            "top_words_list": words,
        })

    return pd.DataFrame(rows)


def topic_terms_map(
        top_words_df: pd.DataFrame,
        n_terms: int = 5,
    ) -> dict[int, str]:
    """Construye mapa topic_id -> términos principales."""
    topic_terms = {}

    for _, row in top_words_df.iterrows():
        words = row["top_words_list"]

        if isinstance(words, str):
            words = [
                word.strip()
                for word in words.split(",")
                if word.strip()
            ]

        topic_terms[int(row["topic_id"])] = ", ".join(words[:n_terms])

    return topic_terms