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
        input_is_processed: bool = False,
        to_dense: bool = True,
        normalize: bool = True,
    ) -> Pipeline:
    """
    Construye una pipeline de características combinando varias transformaciones.

    Args:
        input_is_processed:
            Si False, aplica TokenText() y ejecuta preprocess().
            Si True, asume que el texto ya viene limpio en poem_processed.

        to_dense:
            Si True, convierte la matriz sparse a dense.
            Para corpus grandes conviene False.

        normalize:
            Si True, aplica normalización L2.
            Útil para similitud coseno y muchos modelos lineales.
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
    
    steps.append(("Features", feature_union))
    
    if to_dense:
        steps.append(("ToDense", ToDense()))

    if normalize:
        steps.append(("Norm", Normalize()))

    return Pipeline(steps=steps)
