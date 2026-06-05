from __future__ import annotations

import numpy as np

from conftest import DummyNearestNeighbors
from versovector.inference.similarity_search import SimilaritySearcher


def test_similarity_search_returns_expected_results(sample_reference_metadata) -> None:
    searcher = SimilaritySearcher(
        nearest_neighbors=DummyNearestNeighbors(n_reference=2),
        reference_metadata=sample_reference_metadata,
    )

    results = searcher.find_similar(
        X=np.ones((1, 3), dtype=float),
        top_n=2,
    )

    assert len(results) == 2
    assert results[0].title == "first poem"
    assert results[0].score == 0.9
    assert results[1].title == "second poem"
