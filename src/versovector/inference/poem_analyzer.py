from __future__ import annotations

from pathlib import Path
from typing import Any

from modules.io import get_nested
from modules.preprocessing import preprocess

from .artifact_loader import ModelBundle, load_model_bundle
from .schemas import AnalysisResponse, PoemInput
from .similarity_search import SimilaritySearcher
from .tag_predictor import TagPredictor
from .topic_clusterer import TopicClusterer


class PoemAnalyzer:
    """
    Analyze poems using a trained VersoVector model bundle.

    Responsibilities:
    - preprocess raw poem text;
    - transform text through the feature pipeline;
    - predict emotional or thematic tags;
    - retrieve semantically similar poems;
    - infer dominant topic;
    - infer cluster assignments;
    - return a structured response.
    """

    def __init__(self, bundle: ModelBundle) -> None:
        self.bundle = bundle
        self.feature_pipeline = bundle.feature_pipeline

        self.tag_predictor = TagPredictor(
            classifier=bundle.supervised_classifier,
            multilabel_binarizer=bundle.multilabel_binarizer,
        )

        self.similarity_searcher = SimilaritySearcher(
            nearest_neighbors=bundle.nearest_neighbors,
            reference_metadata=bundle.reference_metadata,
        )

        self.topic_clusterer = TopicClusterer(
            lda_model=bundle.lda_model,
            lda_count_vectorizer=bundle.lda_count_vectorizer,
            dimensionality_reducer=bundle.dimensionality_reducer,
            kmeans_model=bundle.kmeans_model,
            gmm_model=bundle.gmm_model,
            lda_topics=bundle.lda_topics,
        )

    @classmethod
    def from_bundle(
            cls,
            bundle_dir: str | Path = "artifacts/model_bundle",
        ) -> "PoemAnalyzer":
        """Create a PoemAnalyzer from a serialized model bundle."""
        return cls(load_model_bundle(bundle_dir))

    def _process_text(self, poem_input: PoemInput) -> str:
        """Return processed poem text expected by the feature pipeline."""
        if poem_input.already_processed:
            return poem_input.poem

        return preprocess(str(poem_input.poem))

    def _model_info(self) -> dict[str, Any]:
        """Return compact model information for response metadata."""
        config = self.bundle.config
        metadata = self.bundle.model_metadata or {}

        return {
            "project": get_nested(config, "project", "name", default="versovector"),
            "task": get_nested(
                config,
                "project",
                "task",
                default="emotional_semantic_recommendation",
            ),
            "bundle_dir": str(self.bundle.bundle_dir),
            "registered_model_name": metadata.get(
                "registered_model_name",
                get_nested(
                    config,
                    "mlflow",
                    "registered_model_name",
                    default="versovector-poem-analyzer",
                ),
            ),
        }

    def analyze(
            self,
            poem: str,
            title: str | None = None,
            poet: str | None = None,
            user_tags: list[str] | None = None,
            already_processed: bool = False,
            top_k_tags: int | None = None,
            top_n_similar: int | None = None,
            tag_threshold: float | None = None,
        ) -> AnalysisResponse:
        """Run the full emotional-semantic analysis for one poem."""
        poem_input = PoemInput(
            poem=poem,
            title=title,
            poet=poet,
            user_tags=user_tags or [],
            already_processed=already_processed,
        )

        processed_text = self._process_text(poem_input)
        X = self.feature_pipeline.transform([processed_text])

        config = self.bundle.config

        effective_top_k_tags = int(
            top_k_tags
            if top_k_tags is not None
            else get_nested(config, "supervised", "top_k_tags", default=5)
        )

        effective_top_n_similar = int(
            top_n_similar
            if top_n_similar is not None
            else get_nested(config, "unsupervised", "top_n_neighbors", default=5)
        )

        predicted_tags = self.tag_predictor.predict(
            X,
            top_k=effective_top_k_tags,
            threshold=tag_threshold,
        )

        similar_poems = self.similarity_searcher.find_similar(
            X,
            top_n=effective_top_n_similar,
        )

        topic = self.topic_clusterer.predict_topic(processed_text)
        cluster = self.topic_clusterer.predict_cluster(X)

        return AnalysisResponse(
            title=title,
            poet=poet,
            poem_processed=processed_text,
            predicted_tags=predicted_tags,
            similar_poems=similar_poems,
            topic=topic,
            cluster=cluster,
            model_info=self._model_info(),
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
        ) -> dict[str, Any]:
        """Run analysis and return a plain dictionary."""
        return self.analyze(
            poem=poem,
            title=title,
            poet=poet,
            user_tags=user_tags,
            already_processed=already_processed,
            top_k_tags=top_k_tags,
            top_n_similar=top_n_similar,
            tag_threshold=tag_threshold,
        ).to_dict()
