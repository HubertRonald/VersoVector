from __future__ import annotations

from versovector.inference.artifact_loader import load_model_bundle


def test_load_model_bundle_from_minimal_test_bundle(minimal_model_bundle_dir) -> None:
    bundle = load_model_bundle(minimal_model_bundle_dir)

    assert bundle.config["project"]["name"] == "versovector"
    assert bundle.model_metadata["registered_model_name"] == "versovector-test-model"
    assert bundle.reference_metadata.shape[0] == 2
    assert bundle.feature_pipeline is not None
    assert bundle.supervised_classifier is not None
    assert bundle.multilabel_binarizer is not None
    assert bundle.nearest_neighbors is not None
