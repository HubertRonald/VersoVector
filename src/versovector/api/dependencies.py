from __future__ import annotations

from functools import lru_cache

from fastapi import HTTPException, status

from versovector.inference import PoemAnalyzer

from .settings import get_settings


@lru_cache(maxsize=1)
def get_cached_poem_analyzer() -> PoemAnalyzer:
    """Load and cache the PoemAnalyzer instance."""
    settings = get_settings()
    return PoemAnalyzer.from_bundle(settings.model_bundle_dir)


def get_poem_analyzer() -> PoemAnalyzer:
    """
    Return the cached analyzer or raise a service-unavailable error.

    This keeps model loading lazy while providing a clean API error when the
    local model bundle is missing or incomplete.
    """
    try:
        return get_cached_poem_analyzer()
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=(
                "The VersoVector model bundle could not be loaded. "
                f"Reason: {exc}"
            ),
        ) from exc
