from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from fastapi.testclient import TestClient

from versovector.api.dependencies import get_poem_analyzer
from versovector.api.main import app


class FakePoemAnalyzer:
    """Fake analyzer used by API integration tests."""

    bundle = SimpleNamespace(
        config={
            "project": {
                "name": "versovector",
                "task": "emotional_semantic_recommendation",
            },
            "mlflow": {
                "registered_model_name": "versovector-test-model",
            },
        },
        model_metadata={
            "registered_model_name": "versovector-test-model",
        },
        bundle_dir=Path("artifacts/model_bundle"),
    )

    def analyze_dict(
        self,
        poem: str,
        title: str | None = None,
        poet: str | None = None,
        user_tags: list[str] | None = None,
        already_processed: bool = False,
        top_k_tags: int | None = None,
        top_n_similar: int | None = None,
        tag_threshold: float | None = None,
    ):
        return {
            "title": title,
            "poet": poet,
            "poem_processed": poem if already_processed else "memory light rain",
            "predicted_tags": [
                {"tag": "memory", "score": 0.8},
                {"tag": "light", "score": 0.7},
            ],
            "similar_poems": [
                {
                    "poem_id": "poem-1",
                    "title": "similar poem",
                    "title_raw": "Similar Poem",
                    "poet": "poet",
                    "poet_raw": "Poet",
                    "source": "poetry_foundation",
                    "corpus_role": "reference",
                    "score": 0.9,
                }
            ],
            "topic": {
                "topic_id": 1,
                "probability": 0.75,
                "terms": "memory light rain",
            },
            "cluster": {
                "kmeans": 0,
                "gmm": 0,
            },
            "model_info": {
                "project": "versovector",
                "task": "emotional_semantic_recommendation",
            },
        }


def override_analyzer() -> FakePoemAnalyzer:
    """Return a fake analyzer for API tests."""
    return FakePoemAnalyzer()


def test_health_endpoint() -> None:
    client = TestClient(app)

    response = client.get("/health")

    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_model_info_endpoint() -> None:
    app.dependency_overrides[get_poem_analyzer] = override_analyzer
    client = TestClient(app)

    response = client.get("/v1/model-info")

    app.dependency_overrides.clear()

    assert response.status_code == 200
    assert response.json()["project"] == "versovector"
    assert response.json()["registered_model_name"] == "versovector-test-model"


def test_analyze_endpoint() -> None:
    app.dependency_overrides[get_poem_analyzer] = override_analyzer
    client = TestClient(app)

    response = client.post(
        "/v1/analyze",
        json={
            "title": "test poem",
            "poet": "anonymous",
            "poem": "I walk through the rain carrying a memory of light.",
            "top_k_tags": 2,
            "top_n_similar": 1,
        },
    )

    app.dependency_overrides.clear()

    assert response.status_code == 200
    body = response.json()
    assert body["title"] == "test poem"
    assert body["predicted_tags"][0]["tag"] == "memory"
    assert body["similar_poems"][0]["title"] == "similar poem"


def test_predict_tags_endpoint() -> None:
    app.dependency_overrides[get_poem_analyzer] = override_analyzer
    client = TestClient(app)

    response = client.post(
        "/v1/predict-tags",
        json={
            "poem": "I walk through the rain carrying a memory of light.",
            "top_k_tags": 2,
        },
    )

    app.dependency_overrides.clear()

    assert response.status_code == 200
    body = response.json()
    assert "predicted_tags" in body
    assert "similar_poems" not in body


def test_similar_endpoint() -> None:
    app.dependency_overrides[get_poem_analyzer] = override_analyzer
    client = TestClient(app)

    response = client.post(
        "/v1/similar",
        json={
            "poem": "I walk through the rain carrying a memory of light.",
            "top_n_similar": 1,
        },
    )

    app.dependency_overrides.clear()

    assert response.status_code == 200
    body = response.json()
    assert "similar_poems" in body
    assert "predicted_tags" not in body
