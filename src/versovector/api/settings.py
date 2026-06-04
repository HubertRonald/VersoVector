from __future__ import annotations

from functools import lru_cache
from importlib.metadata import PackageNotFoundError, version

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


def get_package_version(package_name: str = "versovector") -> str:
    """Return the installed package version when available."""
    try:
        return version(package_name)
    except PackageNotFoundError:
        return "0.1.0"


class APISettings(BaseSettings):
    """Runtime settings for the VersoVector API."""

    model_config = SettingsConfigDict(
        env_prefix="VERSOVECTOR_",
        env_file=".env",
        extra="ignore",
    )

    app_name: str = "VersoVector API"
    app_version: str = Field(default_factory=get_package_version)
    api_prefix: str = "/v1"

    model_bundle_dir: str = "artifacts/model_bundle"

    default_top_k_tags: int = 5
    default_top_n_similar: int = 5

    cors_allowed_origins: list[str] = ["*"]


@lru_cache(maxsize=1)
def get_settings() -> APISettings:
    """Return cached API settings."""
    return APISettings()
