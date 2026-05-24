from __future__ import annotations

__all__ = [
    'clean',
    'remove_stopwords',
    'lematize',
    'preprocess',
    'parse_tags',
    'preprocess_tags',
    'normalize_poetry_columns'
]

import re
import ast

import spacy
import unidecode
import pandas as pd

from nltk.stem import PorterStemmer
from spacy.language import Language

from utils import Constants
from typing import List

# nlp load
# python -m spacy download en_core_web_sm
# python -m spacy download en_core_web_lg
nlp: Language = spacy.load("en_core_web_lg")
ps: PorterStemmer = PorterStemmer()


def clean(text:str) -> str:
    """Limpia el texto de caracteres especiales y espacios extras."""
    text = unidecode.unidecode(text)
    text = re.sub(r'http\S+', Constants.EMPTY_STR, text)            # remove URLs
    text = re.sub(r'www\S+', Constants.EMPTY_STR, text)             # remove URLs
    text = re.sub(r'[^A-Za-z0-9\s]', Constants.EMPTY_STR, text)     # remove special chars
    text = re.sub(r'[\[\]\"]', Constants.EMPTY_STR, text)           # remove brackets and quotes
    text = re.sub(r'\d+', Constants.EMPTY_STR, text)                # remove digits
    text = re.sub(r'\s+', Constants.SPACE_STR, text).strip()        # remove extra spaces
    return text.lower()


def remove_stopwords(text:str) -> str:
    """Remueve las stopwords del texto."""
    doc = nlp(text)
    tokens = [
        ps.stem(token.text)
        for token in doc
        if not token.is_stop 
        and token.is_alpha 
        and not token.is_punct
    ]
    return Constants.SPACE_STR.join(tokens)


def lematize(text:str) -> str:
    """Lematiza el texto."""
    doc = nlp(text)
    tokens = [token.lemma_ for token in doc]
    return Constants.SPACE_STR.join(tokens)


def preprocess(text:str) -> str:
    """Preprocesa el texto."""
    text = clean(text)
    text = remove_stopwords(text)
    text = lematize(text)
    return text


def parse_tags(value) -> List[str]:
    """
    Convierte un valor de tags a lista normalizada.

    Soporta:
    - listas reales: ["Love", "Nature"]
    - strings tipo lista: "['Love', 'Nature']"
    - strings separados por coma: "Love, Nature"
    - valores nulos: NaN / None

    Returns:
        list[str]: lista de tags limpios.
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
            
    except (ValueError, SyntaxError):
        return [
            clean(tag) 
            for tag in str(value).split(Constants.COMMA_STR)
            if str(tag).strip()
        ]


def preprocess_tags(tags_column: pd.Series) -> pd.Series:
    """
    Procesa una columna completa de etiquetas.

    Cada fila queda convertida en una lista de tags normalizados.
    """
    return tags_column.apply(parse_tags)


def normalize_poetry_columns(df: pd.DataFrame) -> pd.DataFrame:
    """"
    Normaliza columnas esperadas del dataset Poetry Foundation.

    Garantiza:
    - nombres de columnas en minúsculas y con snake_case
    - columna title
    - columna title_raw
    - columna poet
    - columna poet_raw
    - columna tags normalizada como lista

    Nota:
    - title y poet se limpian con clean(), no con preprocess(),
      para no eliminar stopwords ni alterar demasiado la metadata.
    """
    df = df.copy()
    
    df = df.rename(columns={
        col: (
            col
            .strip()
            .lower()
            .replace(Constants.SPACE_STR, Constants.UNDERLINE)
        )
        for col in df.columns
    })
    
    # Title
    if Constants.TITLE not in df.columns:
        df[Constants.TITLE] = Constants.UNKNOWN

    if "title_raw" not in df.columns:
        df["title_raw"] = df[Constants.TITLE].astype(str)
        
    df[Constants.TITLE] = df["title_raw"].apply(clean)

    # Poet
    if Constants.POET not in df.columns:
        df[Constants.POET] = Constants.UNKNOWN

    if "poet_raw" not in df.columns:
        df["poet_raw"] = df[Constants.POET].astype(str)
    
    df[Constants.POET] = df["poet_raw"].apply(clean)
    
    # Poem
    if Constants.POEM not in df.columns:
        df[Constants.POEM] = Constants.EMPTY_STR

    if "poem_raw" not in df.columns:
        df["poem_raw"] = df[Constants.POEM].astype(str)

    df[Constants.POEM] = df["poem_raw"].astype(str).apply(clean)

    # Tags
    if Constants.TAGS not in df.columns:
        df[Constants.TAGS] = [[] for _ in range(len(df))]
    else:
        df[Constants.TAGS] = preprocess_tags(df[Constants.TAGS])

    return df
