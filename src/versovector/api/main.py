from __future__ import annotations

from typing import Any

from fastapi import Depends, FastAPI, Response
from fastapi.middleware.cors import CORSMiddleware

from modules.io import get_nested
from versovector.inference import PoemAnalyzer

from .dependencies import get_poem_analyzer
from .schemas import (
    AnalysisAPIResponse,
    HealthResponse,
    ModelInfoResponse,
    PoemAnalyzeRequest,
    PredictTagsRequest,
    PredictTagsResponse,
    SimilarPoemsRequest,
    SimilarPoemsResponse,
)
from .settings import APISettings, get_settings


def create_app() -> FastAPI:
    """Create and configure the VersoVector FastAPI application."""
    settings = get_settings()

    app = FastAPI(
        title=settings.app_name,
        version=settings.app_version,
        description=(
            "Emotional-semantic analysis and recommendation API for "
            "poetic and lyrical language."
        ),
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_allowed_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    register_routes(app)

    return app


def register_routes(app: FastAPI) -> None:
    """Register API routes."""
    
    @app.get("/favicon.ico", include_in_schema=False)
    def favicon() -> Response:
        """Avoid browser favicon 404 noise during local development."""
        return Response(status_code=204)

    @app.get("/", tags=["Root"])
    def root(settings: APISettings = Depends(get_settings)) -> dict[str, str]:
        """Return a compact root message."""
        return {
            "message": "VersoVector API",
            "version": settings.app_version,
            "docs": "/docs",
        }

    @app.get("/health", response_model=HealthResponse, tags=["Health"])
    def health(settings: APISettings = Depends(get_settings)) -> HealthResponse:
        """
        Return API health information.

        This endpoint does not force model loading, so it remains lightweight for
        basic service checks.
        """
        return HealthResponse(
            status="ok",
            app_name=settings.app_name,
            app_version=settings.app_version,
            model_bundle_dir=settings.model_bundle_dir,
        )
    
    @app.get("/ready", tags=["Health"])
    def readiness(
            analyzer: PoemAnalyzer = Depends(get_poem_analyzer),
        ) -> dict[str, str]:
        """Return readiness information after loading the model bundle."""
        return {
            "status": "ready",
            "bundle_dir": str(analyzer.bundle.bundle_dir),
        }

    @app.get(
        "/v1/model-info",
        response_model=ModelInfoResponse,
        tags=["Model"],
    )
    def model_info(
            analyzer: PoemAnalyzer = Depends(get_poem_analyzer),
        ) -> ModelInfoResponse:
        """Return information about the loaded model bundle."""
        config = analyzer.bundle.config
        metadata: dict[str, Any] = analyzer.bundle.model_metadata or {}

        return ModelInfoResponse(
            project=get_nested(config, "project", "name", default="versovector"),
            task=get_nested(
                config,
                "project",
                "task",
                default="emotional_semantic_recommendation",
            ),
            registered_model_name=metadata.get(
                "registered_model_name",
                get_nested(
                    config,
                    "mlflow",
                    "registered_model_name",
                    default="versovector-poem-analyzer",
                ),
            ),
            bundle_dir=str(analyzer.bundle.bundle_dir),
            model_metadata=metadata,
        )

    @app.post(
        "/v1/analyze",
        response_model=AnalysisAPIResponse,
        tags=["Analysis"],
    )
    def analyze_poem(
            request: PoemAnalyzeRequest,
            settings: APISettings = Depends(get_settings),
            analyzer: PoemAnalyzer = Depends(get_poem_analyzer),
        ) -> dict[str, Any]:
        """Run full emotional-semantic analysis for one poem."""
        return analyzer.analyze_dict(
            poem=request.poem,
            title=request.title,
            poet=request.poet,
            user_tags=request.user_tags,
            already_processed=request.already_processed,
            top_k_tags=request.top_k_tags or settings.default_top_k_tags,
            top_n_similar=request.top_n_similar or settings.default_top_n_similar,
            tag_threshold=request.tag_threshold,
        )

    @app.post(
        "/v1/predict-tags",
        response_model=PredictTagsResponse,
        tags=["Prediction"],
    )
    def predict_tags(
            request: PredictTagsRequest,
            settings: APISettings = Depends(get_settings),
            analyzer: PoemAnalyzer = Depends(get_poem_analyzer),
        ) -> dict[str, Any]:
        """Predict emotional or thematic tags for one poem."""
        result = analyzer.analyze_dict(
            poem=request.poem,
            title=request.title,
            poet=request.poet,
            already_processed=request.already_processed,
            top_k_tags=request.top_k_tags or settings.default_top_k_tags,
            top_n_similar=1,
            tag_threshold=request.tag_threshold,
        )

        return {
            "title": result["title"],
            "poet": result["poet"],
            "poem_processed": result["poem_processed"],
            "predicted_tags": result["predicted_tags"],
            "model_info": result["model_info"],
        }

    @app.post(
        "/v1/similar",
        response_model=SimilarPoemsResponse,
        tags=["Similarity"],
    )
    def find_similar(
        request: SimilarPoemsRequest,
        settings: APISettings = Depends(get_settings),
        analyzer: PoemAnalyzer = Depends(get_poem_analyzer),
    ) -> dict[str, Any]:
        """Find semantically similar poems from the reference corpus."""
        result = analyzer.analyze_dict(
            poem=request.poem,
            title=request.title,
            poet=request.poet,
            already_processed=request.already_processed,
            top_k_tags=1,
            top_n_similar=request.top_n_similar or settings.default_top_n_similar,
        )

        return {
            "title": result["title"],
            "poet": result["poet"],
            "poem_processed": result["poem_processed"],
            "similar_poems": result["similar_poems"],
            "model_info": result["model_info"],
        }


app = create_app()
