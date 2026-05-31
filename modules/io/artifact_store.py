# modules/io/artifact_store.py

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

import joblib
import pandas as pd

try:
    import tomllib
except ModuleNotFoundError:  # Python 3.10 compatibility
    import tomli as tomllib

from utils import Constants

__all__ = [
    "PROJECT_ROOT",
    "ARTIFACTS_DIR",
    "DATA_DIR",
    "FIGS_DIR",
    "project_path",
    "artifact_path",
    "data_path",
    "fig_path",
    "display_path",
    "ensure_dir",
    "copy_file",
    "save_joblib",
    "load_joblib",
    "save_csv",
    "load_csv",
    "save_json",
    "load_json",
    "load_toml_config",
    "get_nested",
]



PROJECT_ROOT = Path(__file__).resolve().parents[2]
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
DATA_DIR = PROJECT_ROOT / "data"
FIGS_DIR = PROJECT_ROOT / "figs"


def project_path(*parts: str) -> Path:
    """Build a path inside the project root."""
    return PROJECT_ROOT.joinpath(*parts)


def artifact_path(*parts: str) -> Path:
    """Build a path inside artifacts/."""
    return ARTIFACTS_DIR.joinpath(*parts)


def data_path(*parts: str) -> Path:
    """Build a path inside data/."""
    return DATA_DIR.joinpath(*parts)


def fig_path(*parts: str) -> Path:
    """Build a path inside figs/."""
    return FIGS_DIR.joinpath(*parts)


def ensure_dir(path: Path) -> Path:
    """Create a directory if it does not exist and return the path."""
    path.mkdir(parents=True, exist_ok=True)
    return path


def copy_file(source: str | Path, target: str | Path, required: bool = True) -> bool:
    """
    Copy a file from source to target.

    Returns True when the file was copied.
    If required is False and the source does not exist, returns False.
    """
    source = Path(source)
    target = Path(target)

    if not source.is_file():
        if required:
            raise FileNotFoundError(f"Required file not found: {source}")
        return False

    ensure_dir(target.parent)
    shutil.copy2(source, target)
    return True


def save_joblib(obj: Any, path: Path) -> Path:
    """Save a serializable object with joblib."""
    ensure_dir(path.parent)
    joblib.dump(obj, path)
    return path


def load_joblib(path: Path) -> Any:
    """Load a joblib artifact."""
    if not path.is_file():
        raise FileNotFoundError(f"Artifact not found: {path}")
    return joblib.load(path)


def save_csv(
    df: pd.DataFrame,
    path: str | Path,
    sep: str = Constants.PIPE_STR,
    encoding: str = Constants.ENCODING,
) -> Path:
    """Save a DataFrame as CSV."""
    path = Path(path)
    ensure_dir(path.parent)
    df.to_csv(path, index=False, sep=sep, encoding=encoding)
    return path


def load_csv(
        path: str | Path,
        sep: str = Constants.PIPE_STR,
        encoding: str = Constants.ENCODING,
    ) -> pd.DataFrame:
    """Load a CSV file as a DataFrame."""
    path = Path(path)

    if not path.is_file():
        raise FileNotFoundError(f"CSV not found: {path}")

    return pd.read_csv(path, sep=sep, encoding=encoding)


def save_json(
        data: dict[str, Any],
        path: str | Path,
        encoding: str = Constants.ENCODING,
    ) -> Path:
    """Save a dictionary as JSON."""
    path = Path(path)
    ensure_dir(path.parent)

    with path.open("w", encoding=encoding) as file:
        json.dump(data, file, ensure_ascii=False, indent=2)

    return path


def load_json(
        path: str | Path,
        encoding: str = Constants.ENCODING
    ) -> dict[str, Any]:
    """Load a JSON file as a dictionary."""
    path = Path(path)

    if not path.is_file():
        raise FileNotFoundError(f"JSON not found: {path}")

    with path.open("r", encoding=encoding) as file:
        return json.load(file)


def load_toml_config(config_path: str | Path) -> dict[str, Any]:
    """Load a TOML configuration file."""
    path = Path(config_path)

    if not path.is_absolute():
        path = project_path(str(path))

    if not path.is_file():
        raise FileNotFoundError(f"Config file not found: {path}")

    with path.open("rb") as file:
        return tomllib.load(file)


def get_nested(config: dict[str, Any], *keys: str, default: Any = None) -> Any:
    """Safely read a nested value from a dictionary."""
    value: Any = config

    for key in keys:
        if not isinstance(value, dict) or key not in value:
            return default

        value = value[key]

    return value


def display_path(path: Path, include_project_name: bool = True) -> str:
    """
    Return a readable path relative to the project root.

    This avoids exposing absolute local paths in notebooks.
    Example:
        /VersoVector/data/CleanedPoetryFoundationData.csv
    """
    path = Path(path).resolve()

    try:
        relative_path = path.relative_to(PROJECT_ROOT)
    except ValueError:
        return path.as_posix()

    if include_project_name:
        return f"/{PROJECT_ROOT.name}/{relative_path.as_posix()}"

    return relative_path.as_posix()
