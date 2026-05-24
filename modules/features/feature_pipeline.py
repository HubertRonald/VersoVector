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


def build_feature_pipeline(
        input_is_processed: bool = False
    ) -> Pipeline:
    """
    Construye una pipeline de características combinando varias transformaciones.

    Args:
        input_is_processed:
            Si False, aplica TokenText() y por tanto ejecuta preprocess().
            Si True, asume que el texto ya viene limpio en poem_processed
            y evita reprocesar con spaCy.
    """
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
    
    steps = []
    
    if not input_is_processed:
        steps.append(('CleanTokenText', TokenText()))
    
    steps.extend([
        ('Features', feature_union),
        ('ToDense', ToDense()),
        ('Norm', Normalize())
    ])

    return Pipeline(steps=steps)
