from __future__ import annotations

import re
import numpy as np
from collections import Counter

from modules.preprocesing.clean_text import preprocess
from sklearn.base import BaseEstimator, TransformerMixin

from typing import Iterable, List


class TokenText(BaseEstimator, TransformerMixin):
    def __init__(self):
        super().__init__()
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X: Iterable[str]) -> List[str]:
        if not isinstance(X, (list)):
            X = list(X.split())
            
        return [preprocess(text) for text in X]
    
    
class TextToDictTransformer(BaseEstimator, TransformerMixin):
    def __init__(self):
        super().__init__()
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X: Iterable[str]) -> List[dict]:
        return [
            Counter(re.findall(r'\b\w+\b', text))
            for text in X
        ]
    
class ToDense: ...
class Normalize: ...