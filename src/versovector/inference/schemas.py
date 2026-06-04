from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class PoemInput:
    """Input payload for poem analysis."""

    poem: str
    title: str | None = None
    poet: str | None = None
    user_tags: list[str] = field(default_factory=list)
    already_processed: bool = False


@dataclass
class TagPrediction:
    """Predicted emotional or thematic tag."""

    tag: str
    score: float | None = None


@dataclass
class SimilarPoem:
    """Nearest poem returned by semantic similarity search."""

    poem_id: str | None
    title: str
    title_raw: str | None
    poet: str | None
    poet_raw: str | None
    source: str | None
    corpus_role: str | None
    score: float


@dataclass
class TopicPrediction:
    """Dominant topic prediction."""

    topic_id: int | None
    probability: float | None
    terms: str | None = None


@dataclass
class ClusterPrediction:
    """Cluster assignment returned by unsupervised models."""

    kmeans: int | None = None
    gmm: int | None = None


@dataclass
class AnalysisResponse:
    """Full response returned by PoemAnalyzer."""

    title: str | None
    poet: str | None
    poem_processed: str
    predicted_tags: list[TagPrediction]
    similar_poems: list[SimilarPoem]
    topic: TopicPrediction | None
    cluster: ClusterPrediction | None
    model_info: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert the response into a serializable dictionary."""
        return asdict(self)
