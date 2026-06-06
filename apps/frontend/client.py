from __future__ import annotations

import os
from typing import Any

import requests


DEFAULT_API_BASE_URL = "http://localhost:8001"
DEFAULT_TIMEOUT_SECONDS = int(
    os.getenv("VERSOVECTOR_API_TIMEOUT_SECONDS", "300")
)


def get_api_base_url() -> str:
    """Return the configured API base URL."""
    return os.getenv("VERSOVECTOR_API_BASE_URL", DEFAULT_API_BASE_URL).rstrip("/")


def build_url(path: str) -> str:
    """Build a full API URL from a path."""
    return f"{get_api_base_url()}/{path.lstrip('/')}"


def post_json(
        path: str,
        payload: dict[str, Any],
        timeout: int = DEFAULT_TIMEOUT_SECONDS,
    ) -> dict[str, Any]:
    """POST JSON to the API and return the decoded response."""
    response = requests.post(
        build_url(path),
        json=payload,
        timeout=timeout,
    )
    response.raise_for_status()
    return response.json()


def get_json(
        path: str,
        timeout: int = DEFAULT_TIMEOUT_SECONDS,
    ) -> dict[str, Any]:
    """GET JSON from the API and return the decoded response."""
    response = requests.get(
        build_url(path),
        timeout=timeout,
    )
    response.raise_for_status()
    return response.json()


def health_check() -> dict[str, Any]:
    """Call the API health endpoint."""
    return get_json("/health")


def readiness_check() -> dict[str, Any]:
    """Call the API readiness endpoint."""
    return get_json("/ready")


def analyze_poem(
        poem: str,
        title: str | None = None,
        poet: str | None = None,
        user_tags: list[str] | None = None,
        top_k_tags: int = 5,
        top_n_similar: int = 5,
        tag_threshold: float | None = None,
    ) -> dict[str, Any]:
    """Call the full analysis endpoint."""
    payload: dict[str, Any] = {
        "poem": poem,
        "title": title or None,
        "poet": poet or None,
        "user_tags": user_tags or [],
        "top_k_tags": top_k_tags,
        "top_n_similar": top_n_similar,
        "already_processed": False,
    }

    if tag_threshold is not None:
        payload["tag_threshold"] = tag_threshold

    return post_json("/v1/analyze", payload)


def predict_tags(
        poem: str,
        title: str | None = None,
        poet: str | None = None,
        top_k_tags: int = 5,
        tag_threshold: float | None = None,
    ) -> dict[str, Any]:
    """Call the tag prediction endpoint."""
    payload: dict[str, Any] = {
        "poem": poem,
        "title": title or None,
        "poet": poet or None,
        "top_k_tags": top_k_tags,
        "already_processed": False,
    }

    if tag_threshold is not None:
        payload["tag_threshold"] = tag_threshold

    return post_json("/v1/predict-tags", payload)


def find_similar(
        poem: str,
        title: str | None = None,
        poet: str | None = None,
        top_n_similar: int = 5,
    ) -> dict[str, Any]:
    """Call the semantic similarity endpoint."""
    payload: dict[str, Any] = {
        "poem": poem,
        "title": title or None,
        "poet": poet or None,
        "top_n_similar": top_n_similar,
        "already_processed": False,
    }

    return post_json("/v1/similar", payload)