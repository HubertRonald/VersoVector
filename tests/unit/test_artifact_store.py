from __future__ import annotations

import pandas as pd

from modules.io import (
    copy_file,
    load_csv,
    load_json,
    load_joblib,
    save_csv,
    save_json,
    save_joblib
)


def test_save_and_load_json(tmp_path) -> None:
    path = tmp_path / "metadata.json"
    data = {"name": "versovector", "version": 1}

    save_json(data, path)

    assert load_json(path) == data


def test_save_and_load_csv(tmp_path) -> None:
    path = tmp_path / "data.csv"
    df = pd.DataFrame({"title": ["poem"], "score": [0.75]})

    save_csv(df, path)
    result = load_csv(path)

    assert result.to_dict(orient="records") == [{"title": "poem", "score": 0.75}]


def test_save_and_load_joblib(tmp_path) -> None:
    path = tmp_path / "object.joblib"
    obj = {"model": "dummy"}

    save_joblib(obj, path)

    assert load_joblib(path) == obj


def test_copy_file_optional_missing_returns_false(tmp_path) -> None:
    source = tmp_path / "missing.txt"
    target = tmp_path / "target.txt"

    assert copy_file(source, target, required=False) is False


def test_copy_file_required_existing_file(tmp_path) -> None:
    source = tmp_path / "source.txt"
    target = tmp_path / "nested" / "target.txt"

    source.write_text("hello", encoding="utf-8")

    assert copy_file(source, target, required=True) is True
    assert target.read_text(encoding="utf-8") == "hello"
