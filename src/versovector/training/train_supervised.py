# equivalent to notebook 03
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import joblib
import mlfow
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer


from modules.classification import (
    build_fast_multilabel_classifier,
    build_stacking_classifier,
    filter_rare_multilabel_tags,
)
from modules.evaluation import compute_multilabel_metrics
from modules.preprocessing import parse_tags
from modules.io import (
    ensure_dir,
    get_nested,
    load_toml_config,
    project_path,
    load_csv,
    save_csv,
    save_json,
)

def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Train supervised VersoVector model.")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/model_config.toml",
        help="Path to the TOML configuration file.",
    )
    return parser.parse_args()


def build_classifier(model_type: str, seed: int):
    """Build the supervised multilabel classifier."""
    model_type = model_type.strip()

    if model_type == "stacking_classifier":
        return build_stacking_classifier(seed=seed)

    if model_type in {"complement_nb", "multinomial_nb", "logreg"}:
        return build_fast_multilabel_classifier(
            model_type=model_type,
            seed=seed,
            n_jobs=-1,
        )

    raise ValueError(f"Unsupported classifier_model_type: {model_type}")


def main() -> None:
    """Train supervised multilabel model and log experiment with MLflow."""
    args = parse_args()
    config = load_toml_config(args.config)

    seed = int(get_nested(config, "training", "seed", default=42))

    classifier_model_type = str(
        get_nested(config, "training", "classifier_model_type", default="stacking_classifier")
    ).strip()



if __name__ == "__main__":
    main()