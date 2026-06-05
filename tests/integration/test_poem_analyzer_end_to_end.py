from __future__ import annotations

from versovector.inference import PoemAnalyzer


def test_poem_analyzer_end_to_end_with_minimal_bundle(minimal_model_bundle_dir) -> None:
    analyzer = PoemAnalyzer.from_bundle(minimal_model_bundle_dir)

    result = analyzer.analyze_dict(
        title="test poem",
        poet="anonymous",
        poem="memory light rain",
        already_processed=True,
        top_k_tags=2,
        top_n_similar=2,
    )

    assert result["title"] == "test poem"
    assert result["poem_processed"] == "memory light rain"
    assert [tag["tag"] for tag in result["predicted_tags"]] == ["sadness", "light"]
    assert len(result["similar_poems"]) == 2
    assert result["topic"]["topic_id"] == 1
    assert result["cluster"]["kmeans"] == 0
    assert result["model_info"]["project"] == "versovector"
