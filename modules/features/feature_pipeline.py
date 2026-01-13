from __future__ import annotations

__all__ = ['build_feature_pipeline']

from sklearn.pipeline import Pipeline
from sklearn.pipeline import FeatureUnion

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction import DictVectorizer

from modules.features.transformers import TokenText
from modules.features.transformers import TextToDictTransformer
from modules.features.transformers import ToDense
from modules.features.transformers import Normalize

def build_feature_pipeline() -> Pipeline:
    """Construye una pipeline de características combinando varias transformaciones."""
    dict_vect = Pipeline([
        ('TextToDictTransformer', TextToDictTransformer()),
        ('DictVectorizer', DictVectorizer(sparse=True))
    ])
    
    feature_union = FeatureUnion([
        ('CountVect', CountVectorizer(
            stop_words="english", 
            analyzer='word', 
            ngram_range=(1, 1), 
            max_df=1.0, 
            min_df=1, 
            max_features=None)
        ),
        ('Tfid', TfidfVectorizer(
            stop_words='english', 
            smooth_idf=False, 
            sublinear_tf=False, 
            norm=None, 
            analyzer='word',
            max_features=5000)
        ),
        ('DictVect', dict_vect)
    ])
    
    return Pipeline([
        ('CleanTokenText', TokenText()),
        ('Features', feature_union),
        ('ToDense', ToDense()),
        ('Norm', Normalize())
    ])
