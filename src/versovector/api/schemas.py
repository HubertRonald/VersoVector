from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    """Health check response."""

    status: str
    app_name: str
    app_version: str
    model_bundle_dir: str


class ModelInfoResponse(BaseModel):
    """Model information response."""

    project: str | None = None
    task: str | None = None
    registered_model_name: str | None = None
    bundle_dir: str
    model_metadata: dict[str, Any] = Field(default_factory=dict)


class PoemAnalyzeRequest(BaseModel):
    """Request payload for full poem analysis."""

    poem: str = Field(..., min_length=1)
    title: str | None = None
    poet: str | None = None
    user_tags: list[str] = Field(default_factory=list)
    already_processed: bool = False
    top_k_tags: int | None = Field(default=None, ge=1, le=50)
    top_n_similar: int | None = Field(default=None, ge=1, le=50)
    tag_threshold: float | None = Field(default=None, ge=0.0, le=1.0)


class PredictTagsRequest(BaseModel):
    """Request payload for tag prediction."""

    poem: str = Field(..., min_length=1)
    title: str | None = None
    poet: str | None = None
    already_processed: bool = False
    top_k_tags: int | None = Field(default=None, ge=1, le=50)
    tag_threshold: float | None = Field(default=None, ge=0.0, le=1.0)


class SimilarPoemsRequest(BaseModel):
    """Request payload for semantic similarity search."""

    poem: str = Field(..., min_length=1)
    title: str | None = None
    poet: str | None = None
    already_processed: bool = False
    top_n_similar: int | None = Field(default=None, ge=1, le=50)


class TagPredictionResponse(BaseModel):
    """Predicted tag response item."""

    tag: str
    score: float | None = None


class SimilarPoemResponse(BaseModel):
    """Similar poem response item."""

    poem_id: str | None = None
    title: str
    title_raw: str | None = None
    poet: str | None = None
    poet_raw: str | None = None
    source: str | None = None
    corpus_role: str | None = None
    score: float


class TopicPredictionResponse(BaseModel):
    """Topic prediction response."""

    topic_id: int | None = None
    probability: float | None = None
    terms: str | None = None


class ClusterPredictionResponse(BaseModel):
    """Cluster prediction response."""

    kmeans: int | None = None
    gmm: int | None = None


class AnalysisAPIResponse(BaseModel):
    """Full poem analysis API response."""

    title: str | None = None
    poet: str | None = None
    poem_processed: str
    predicted_tags: list[TagPredictionResponse]
    similar_poems: list[SimilarPoemResponse]
    topic: TopicPredictionResponse | None = None
    cluster: ClusterPredictionResponse | None = None
    model_info: dict[str, Any] = Field(default_factory=dict)


class PredictTagsResponse(BaseModel):
    """Tag prediction endpoint response."""

    title: str | None = None
    poet: str | None = None
    poem_processed: str
    predicted_tags: list[TagPredictionResponse]
    model_info: dict[str, Any] = Field(default_factory=dict)


class SimilarPoemsResponse(BaseModel):
    """Similarity endpoint response."""

    title: str | None = None
    poet: str | None = None
    poem_processed: str
    similar_poems: list[SimilarPoemResponse]
    model_info: dict[str, Any] = Field(default_factory=dict)
