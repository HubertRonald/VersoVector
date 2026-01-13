from __future__ import annotations

__all__ = [
    'TokenText',
    'TextToDictTransformer',
    'ToDense',
    'Normalize'
]

import re
import numpy as np
from collections import Counter

from modules.preprocesing.clean_text import preprocess
from sklearn.preprocessing import Normalizer
from sklearn.base import BaseEstimator, TransformerMixin

from typing import Iterable, List
from typing_extensions import Self

class TokenText(BaseEstimator, TransformerMixin):
    def __init__(self):
        super().__init__()
    
    def fit(self, X, y=None) -> Self:
        return self
    
    def transform(self, X: Iterable[str]) -> List[str]:
        if not isinstance(X, (list)):
            X = list(X.split())
            
        return [preprocess(text) for text in X]


class TextToDictTransformer(BaseEstimator, TransformerMixin):
    def __init__(self):
        super().__init__()
    
    def fit(self, X, y=None) -> Self:
        return self
    
    def transform(self, X: Iterable[str]) -> List[dict]:
        return [
            Counter(re.findall(r'\b\w+\b', text))
            for text in X
        ]
    
class ToDense(BaseEstimator, TransformerMixin):
    
    def fit(self, X, y=None) -> Self:
        return self
    
    def transform(self, X):
        return X.toarray() if hasattr(X, "toarray") else np.array(X)


class Normalize(BaseEstimator, TransformerMixin):
    def __init__(self, norm='l2'):
        self.norm = norm
        self._normalizer = Normalizer()
        super().__init__()
    
    def fit(self, X, y=None) -> Self:
        self._normalizer.fit(X)
        return self
    
    def transform(self, X):
        return self._normalizer.transform(X)
