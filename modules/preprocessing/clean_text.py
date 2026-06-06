from __future__ import annotations

__all__ = [
    "clean",
    "remove_stopwords",
    "lematize",
    "preprocess",
    "parse_tags",
    "preprocess_tags",
    "normalize_poetry_columns",
]

import ast
import os
import re
from typing import List

import pandas as pd
import spacy
import unidecode
from nltk.stem import PorterStemmer
from spacy.language import Language

from utils import Constants


_SPACY_MODEL_NAME: str = os.getenv("VERSOVECTOR_SPACY_MODEL", "en_core_web_lg")
_ALLOW_BLANK_SPACY: bool = os.getenv(
    "VERSOVECTOR_ALLOW_BLANK_SPACY",
    "false",
).lower() in {"1", "true", "yes"}

_nlp: Language | None = None
ps: PorterStemmer = PorterStemmer()


def get_nlp() -> Language:
    """
    Lazily load the spaCy model.

    The model is loaded only when a function needs tokenization, stopword
    filtering, or lemmatization. This keeps lightweight imports such as
    `modules.io` from requiring `en_core_web_lg`.
    """
    global _nlp

    if _nlp is not None:
        return _nlp

    try:
        _nlp = spacy.load(_SPACY_MODEL_NAME)
    except OSError as exc:
        if _ALLOW_BLANK_SPACY:
            _nlp = spacy.blank("en")
            return _nlp

        raise OSError(
            f"spaCy model '{_SPACY_MODEL_NAME}' is not installed. "
            "Install it with: python -m spacy download en_core_web_lg. "
            "For lightweight tests that do not require the real NLP model, "
            "set VERSOVECTOR_ALLOW_BLANK_SPACY=true."
        ) from exc

    return _nlp


def clean(text: str) -> str:
    """Clean text by removing URLs, special characters, digits and extra spaces."""
    text = unidecode.unidecode(str(text))
    text = re.sub(r"http\S+", Constants.EMPTY_STR, text)
    text = re.sub(r"www\S+", Constants.EMPTY_STR, text)
    text = re.sub(r"[^A-Za-z0-9\s]", Constants.EMPTY_STR, text)
    text = re.sub(r"[\[\]\"]", Constants.EMPTY_STR, text)
    text = re.sub(r"\d+", Constants.EMPTY_STR, text)
    text = re.sub(r"\s+", Constants.SPACE_STR, text).strip()
    return text.lower()


def remove_stopwords(text: str) -> str:
    """Remove stopwords from text using spaCy and stemming."""
    doc = get_nlp()(text)
    tokens = [
        ps.stem(token.text)
        for token in doc
        if not token.is_stop
        and token.is_alpha
        and not token.is_punct
    ]
    return Constants.SPACE_STR.join(tokens)


def lematize(text: str) -> str:
    """Lemmatize text using spaCy."""
    doc = get_nlp()(text)
    tokens = [token.lemma_ for token in doc]
    return Constants.SPACE_STR.join(tokens)


def preprocess(text: str) -> str:
    """Run the full text preprocessing pipeline."""
    text = clean(text)
    text = remove_stopwords(text)
    text = lematize(text)
    return text


def parse_tags(value) -> List[str]:
    """
    Convert a tag value into a normalized list.

    Supports:
    - real lists: ["Love", "Nature"]
    - stringified lists: "['Love', 'Nature']"
    - comma-separated strings: "Love, Nature"
    - null values: NaN / None
    """
    if isinstance(value, list):
        return [
            clean(str(tag))
            for tag in value
            if str(tag).strip()
        ]

    if pd.isna(value):
        return []

    try:
        parsed = ast.literal_eval(str(value))

        if isinstance(parsed, list):
            return [
                clean(str(tag))
                for tag in parsed
                if str(tag).strip()
            ]

        return [clean(str(parsed))] if str(parsed).strip() else []

    except (ValueError, SyntaxError):
        return [
            clean(tag)
            for tag in str(value).split(Constants.COMMA_STR)
            if str(tag).strip()
        ]


def preprocess_tags(tags_column: pd.Series) -> pd.Series:
    """Process a full tags column into normalized tag lists."""
    return tags_column.apply(parse_tags)


def normalize_poetry_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize expected poetry dataset columns.

    Guarantees:
    - snake_case column names;
    - title and title_raw;
    - poet and poet_raw;
    - tags normalized as list[str].

    Notes:
    - title and poet are cleaned with clean(), not preprocess(),
      to avoid removing stopwords from metadata.
    """
    df = df.copy()

    df = df.rename(
        columns={
            col: (
                col
                .strip()
                .lower()
                .replace(Constants.SPACE_STR, Constants.UNDERLINE)
            )
            for col in df.columns
        }
    )

    if Constants.TITLE not in df.columns:
        df[Constants.TITLE] = Constants.UNKNOWN

    df["title_raw"] = df[Constants.TITLE].astype(str)
    df[Constants.TITLE] = df["title_raw"].apply(clean)

    if Constants.POET not in df.columns:
        df[Constants.POET] = Constants.UNKNOWN

    df["poet_raw"] = df[Constants.POET].astype(str)
    df[Constants.POET] = df["poet_raw"].apply(clean)

    if Constants.TAGS not in df.columns:
        df[Constants.TAGS] = [[] for _ in range(len(df))]
    else:
        df[Constants.TAGS] = preprocess_tags(df[Constants.TAGS])

    return df