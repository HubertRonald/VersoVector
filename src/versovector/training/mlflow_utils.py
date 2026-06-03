from __future__ import annotations

from contextlib import nullcontext
from pathlib import Path
from typing import Any, ContextManager

from modules.io import get_nested

try:
    import mlflow as _mlflow
except ModuleNotFoundError:
    _mlflow = None


def is_mlflow_enabled(config: dict[str, Any]) -> bool:
    """Return whether MLflow tracking is enabled in the project config."""
    return bool(get_nested(config, "mlflow", "enabled", default=False))


def get_mlflow_client(config: dict[str, Any]):
    """
    Return the MLflow module when tracking is enabled.

    If tracking is disabled, return None.
    If tracking is enabled but MLflow is not installed, raise a clear error.
    """
    enabled = is_mlflow_enabled(config)

    if not enabled:
        return None

    if _mlflow is None:
        raise ModuleNotFoundError(
            "MLflow is enabled in configs/model_config.toml, but mlflow is not installed. "
            "Install it with: pip install -r requirements-mlops.txt"
        )

    experiment_name = str(
        get_nested(
            config,
            "mlflow",
            "experiment_name",
            default="versovector-emotional-semantic-recommender",
        )
    )

    _mlflow.set_experiment(experiment_name)

    return _mlflow


def start_mlflow_run(
        config: dict[str, Any],
        run_name: str,
    ) -> tuple[Any | None, ContextManager[Any]]:
    """
    Start an MLflow run only when tracking is enabled.

    Returns:
        tuple:
            - mlflow module or None
            - MLflow run context or nullcontext
    """
    mlflow_client = get_mlflow_client(config)

    if mlflow_client is None:
        return None, nullcontext()

    return mlflow_client, mlflow_client.start_run(run_name=run_name)


def log_mlflow_artifacts(
        mlflow_client: Any | None,
        artifact_paths: list[str | Path],
    ) -> None:
    """Log existing artifact files when MLflow is enabled."""
    if mlflow_client is None:
        return

    for artifact_path in artifact_paths:
        path = Path(artifact_path)

        if path.is_file():
            mlflow_client.log_artifact(str(path))
