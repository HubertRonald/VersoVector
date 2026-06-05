from __future__ import annotations

import numpy as np

from conftest import DummyClassifier, DummyMultiLabelBinarizer
from versovector.inference.tag_predictor import TagPredictor


def test_tag_predictor_returns_top_ranked_tags() -> None:
    predictor = TagPredictor(
        classifier=DummyClassifier(),
        multilabel_binarizer=DummyMultiLabelBinarizer(),
    )

    predictions = predictor.predict(
        X=np.ones((1, 3), dtype=float),
        top_k=2,
    )

    assert [prediction.tag for prediction in predictions] == ["sadness", "light"]
    assert predictions[0].score == 0.8


def test_tag_predictor_applies_threshold() -> None:
    predictor = TagPredictor(
        classifier=DummyClassifier(),
        multilabel_binarizer=DummyMultiLabelBinarizer(),
    )

    predictions = predictor.predict(
        X=np.ones((1, 3), dtype=float),
        top_k=3,
        threshold=0.60,
    )

    assert [prediction.tag for prediction in predictions] == ["sadness"]
