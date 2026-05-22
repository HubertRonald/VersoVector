# modules/io/artifact_store.py

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import pandas as pd

from utils import Constants

__all__ = [
    "ARTIFACTS_DIR",
    "DATA_DIR",
    "artifact_path",
    "data_path",
    "ensure_dir",
    "save_joblib",
    "load_joblib",
    "save_csv",
    "load_csv",
    "save_json",
    "load_json",
]


PROJECT_ROOT = Path(__file__).resolve().parents[2]
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
DATA_DIR = PROJECT_ROOT / "data"


def ensure_dir(path: Path) -> Path:
    """Crea un directorio si no existe y retorna el path."""
    path.mkdir(parents=True, exist_ok=True)
    return path


def artifact_path(*parts: str) -> Path:
    """Construye una ruta dentro de artifacts/."""
    return ARTIFACTS_DIR.joinpath(*parts)


def data_path(*parts: str) -> Path:
    """Construye una ruta dentro de data/."""
    return DATA_DIR.joinpath(*parts)


def save_joblib(obj: Any, path: Path) -> Path:
    """Guarda un objeto serializable con joblib."""
    ensure_dir(path.parent)
    joblib.dump(obj, path)
    return path


def load_joblib(path: Path) -> Any:
    """Carga un objeto serializado con joblib."""
    if not path.is_file():
        raise FileNotFoundError(f"No existe el artifact: {path}")
    return joblib.load(path)


def save_csv(df: pd.DataFrame, path: Path, sep: str = Constants.PIPE_STR, encoding: str = Constants.ENCODING) -> Path:
    """Guarda un DataFrame como CSV."""
    ensure_dir(path.parent)
    df.to_csv(path, index=False, sep=sep, encoding=encoding)
    return path


def load_csv(path: Path, sep: str = Constants.PIPE_STR, encoding: str = Constants.ENCODING) -> pd.DataFrame:
    """Carga un CSV como DataFrame."""
    if not path.is_file():
        raise FileNotFoundError(f"No existe el CSV: {path}")
    return pd.read_csv(path, sep=sep, encoding=encoding)


def save_json(data: dict, path: Path, encoding: str = Constants.ENCODING) -> Path:
    """Guarda un diccionario como JSON."""
    ensure_dir(path.parent)
    with path.open("w", encoding=encoding) as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    return path


def load_json(path: Path, encoding: str = Constants.ENCODING) -> dict:
    """Carga un JSON como diccionario."""
    if not path.is_file():
        raise FileNotFoundError(f"No existe el JSON: {path}")
    with path.open("r", encoding=encoding) as f:
        return json.load(f)


if __name__ == '__main__':
    print(f"{PROJECT_ROOT=}")
