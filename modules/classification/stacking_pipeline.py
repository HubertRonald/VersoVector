from __future__ import annotations

__all__ = [
    "build_stacking_classifier",
    "build_supervised_pipeline",
]

from sklearn.ensemble import StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold
from sklearn.multiclass import OneVsRestClassifier
from sklearn.naive_bayes import ComplementNB, MultinomialNB
from sklearn.pipeline import Pipeline

from modules.features import build_feature_pipeline


def build_stacking_classifier(seed: int = 42) -> OneVsRestClassifier:
    """Build the multilabel stacking classifier used in the supervised notebook."""
    return OneVsRestClassifier(
        StackingClassifier(
            estimators=[
                ("cnb", ComplementNB(alpha=0.1)),
                ("mnb", MultinomialNB(alpha=0.099)),
            ],
            cv=KFold(n_splits=5, shuffle=True, random_state=seed),
            passthrough=True,
            final_estimator=LogisticRegression(
                random_state=seed,
                max_iter=1000,
            ),
        )
    )


def build_fast_multilabel_classifier(
    model_type: str = "complement_nb",
    seed: int = 42,
    n_jobs: int = -1,
) -> OneVsRestClassifier:
    """
    Build a fast multilabel classifier.

    model_type:
        - complement_nb: recomendado para el flujo principal.
        - multinomial_nb: baseline clásico para texto.
        - logreg: baseline más fuerte, pero más lento.
    """
    if model_type == "complement_nb":
        estimator = ComplementNB(alpha=0.1)

    elif model_type == "multinomial_nb":
        estimator = MultinomialNB(alpha=0.099)

    elif model_type == "logreg":
        estimator = LogisticRegression(
            solver="liblinear",
            max_iter=1000,
            random_state=seed,
        )

    else:
        raise ValueError(f"Unsupported model_type: {model_type}")

    return OneVsRestClassifier(
        estimator=estimator,
        n_jobs=n_jobs,
    )
    

def build_supervised_pipeline(
    input_is_processed: bool = True,
    seed: int = 42,
) -> Pipeline:
    """
    Build the full supervised pipeline: features + multilabel classifier.

    Nota:
        Esta función es util para experimentos end-to-end.
        En el flujo principal del proyecto, el notebook 03 carga el
        feature pipeline ya ajustado por el notebook 02 y entrena solo
        el clasificador.
    """
    return Pipeline([
        ("features", build_feature_pipeline(input_is_processed=input_is_processed)),
        ("classifier", Pipeline([
            ("Stack", build_stacking_classifier(seed=seed))
        ])),
    ])
