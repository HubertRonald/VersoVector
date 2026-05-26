from __future__ import annotations

__all__ = ["fit_lda_topics", "extract_top_words", "topic_terms_map"]

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


def extract_top_words(model, vectorizer, n_top: int = 10) -> pd.DataFrame:
    """Return top words by topic as a long DataFrame."""
    feature_names = vectorizer.get_feature_names_out()
    records = []
    for topic_id, topic in enumerate(model.components_):
        top_idx = topic.argsort()[::-1][:n_top]
        for rank, term_idx in enumerate(top_idx, start=1):
            records.append({
                "topic_id": topic_id,
                "rank": rank,
                "term": feature_names[term_idx],
                "weight": float(topic[term_idx]),
            })
    return pd.DataFrame(records)


def topic_terms_map(top_words_df: pd.DataFrame, n_terms: int = 5) -> dict[int, str]:
    """Build a topic_id -> comma-separated top terms mapping."""
    return (
        top_words_df.sort_values(["topic_id", "rank"])
        .groupby("topic_id")["term"]
        .apply(lambda terms: Constants.COMMA_STR.join(list(terms)[:n_terms]))
        .to_dict()
    )
