from __future__ import annotations

__all__ = ['preprocess', 'preprocess_tags']

import re
import pandas as pd
import spacy
import unidecode
from utils import Constants
from nltk.stem import PorterStemmer

# nlp load
# python -m spacy download en_core_web_sm
# python -m spacy download en_core_web_lg
nlp = spacy.load("en_core_web_lg")
ps = PorterStemmer()

def clean(text:str)->str:
    """Limpia el texto de caracteres especiales y espacios extras."""
    text = unidecode.unidecode(text)
    text = re.sub(r'http\S+', Constants.EMPTY_STR, text)            # remove URLs
    text = re.sub(r'www\S+', Constants.EMPTY_STR, text)             # remove URLs
    text = re.sub(r'[^A-Za-z0-9\s]', Constants.EMPTY_STR, text)     # remove special chars
    text = re.sub(r'[\[\]\"]', Constants.EMPTY_STR, text)           # remove brackets and quotes
    text = re.sub(r'\d+', Constants.EMPTY_STR, text)                # remove digits
    text = re.sub(r'\s+', Constants.SPACE_STR, text).strip()        # remove extra spaces
    return text.lower()

def remove_stopwords(text:str)->str:
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

def lematize(text:str)->str:
    """Lematiza el texto."""
    doc = nlp(text)
    tokens = [token.lemma_ for token in doc]
    return Constants.SPACE_STR.join(tokens)

def preprocess(text:str)->str:
    """Preprocesa el texto."""
    text = clean(text)
    text = remove_stopwords(text)
    text = lematize(text)
    return text

def preprocess_tags(tags_column: pd.Series) -> pd.Series:
    """Procesa la columna completa de etiquetas."""
    list_of_lists = tags_column.apply(
        lambda x: [clean(tag) for tag in x.split(Constants.COMMA_STR)]
    )
    
    # 2. Opcional: Eliminar cualquier cadena vacía resultante de la limpieza
    list_of_lists = list_of_lists.apply(
        lambda tags: [tag for tag in tags if tag]
    )
    
    return list_of_lists
