from __future__ import annotations

from .classification import (
    build_stacking_classifier,
    build_fast_multilabel_classifier,
    build_supervised_pipeline,
    count_multilabel_tags,
    filter_rare_multilabel_tags,
)

from .evaluation import compute_multilabel_metrics
from .preprocessing import (
    clean,
    remove_stopwords,
    lematize,
    preprocess,
    parse_tags,
    preprocess_tags,
    normalize_poetry_columns
)
from .features import build_feature_pipeline
from .io import (
    PROJECT_ROOT,
    ARTIFACTS_DIR,
    DATA_DIR,
    artifact_path,
    data_path,
    ensure_dir,
    save_joblib,
    load_joblib,
    save_csv,
    load_csv,
    save_json,
    load_json,
)

__all__ = [
    'build_stacking_classifier',
    'build_fast_multilabel_classifier',
    'build_supervised_pipeline',
    'count_multilabel_tags',
    'filter_rare_multilabel_tags',
    'compute_multilabel_metrics'
    'clean',
    'remove_stopwords',
    'lematize',
    'preprocess',
    'parse_tags',
    'preprocess_tags',
    'normalize_poetry_columns'
    'build_feature_pipeline',
    'PROJECT_ROOT',
    'ARTIFACTS_DIR',
    'DATA_DIR',
    'artifact_path',
    'data_path',
    'ensure_dir',
    'save_joblib',
    'load_joblib',
    'save_csv',
    'load_csv',
    'save_json',
    'load_json',
]
