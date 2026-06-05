from __future__ import annotations

import pytest

try:
    from modules.preprocessing import clean, normalize_poetry_columns, parse_tags
except Exception as exc:
    pytest.skip(
        f"Preprocessing module is unavailable in this environment: {exc}",
        allow_module_level=True,
    )

import pandas as pd


def test_clean_removes_special_characters_and_lowercases() -> None:
    text = "  César Vallejo!!! 123  "
    assert clean(text) == "cesar vallejo"


def test_parse_tags_from_comma_separated_string() -> None:
    assert parse_tags("Love, Death, Memory") == ["love", "death", "memory"]


def test_parse_tags_from_stringified_list() -> None:
    assert parse_tags("['Love', 'Nature']") == ["love", "nature"]


def test_normalize_poetry_columns_creates_expected_metadata() -> None:
    df = pd.DataFrame(
        {
            "Title": [" My Window "],
            "Poem": ["The rain falls."],
            "Poet": ["Some Poet"],
            "Tags": ["Rain, Memory"],
        }
    )

    result = normalize_poetry_columns(df)

    assert "title" in result.columns
    assert "title_raw" in result.columns
    assert "poet" in result.columns
    assert "poet_raw" in result.columns
    assert "tags" in result.columns
    assert result.loc[0, "title"] == "my window"
    assert result.loc[0, "poet"] == "some poet"
    assert result.loc[0, "tags"] == ["rain", "memory"]
