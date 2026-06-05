from __future__ import annotations

from versovector.inference.schemas import (
    AnalysisResponse,
    ClusterPrediction,
    SimilarPoem,
    TagPrediction,
    TopicPrediction,
)


def test_analysis_response_to_dict() -> None:
    response = AnalysisResponse(
        title="test poem",
        poet="anonymous",
        poem_processed="memory light rain",
        predicted_tags=[
            TagPrediction(tag="memory", score=0.8),
        ],
        similar_poems=[
            SimilarPoem(
                poem_id="poem-1",
                title="similar poem",
                title_raw="Similar Poem",
                poet="poet",
                poet_raw="Poet",
                source="poetry_foundation",
                corpus_role="reference",
                score=0.75,
            )
        ],
        topic=TopicPrediction(topic_id=1, probability=0.9, terms="memory light rain"),
        cluster=ClusterPrediction(kmeans=2, gmm=1),
        model_info={"project": "versovector"},
    )

    result = response.to_dict()

    assert result["title"] == "test poem"
    assert result["predicted_tags"][0]["tag"] == "memory"
    assert result["similar_poems"][0]["score"] == 0.75
    assert result["topic"]["topic_id"] == 1
    assert result["cluster"]["kmeans"] == 2
