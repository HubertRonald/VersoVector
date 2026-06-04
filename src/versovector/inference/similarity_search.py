from __future__ import annotations

from typing import Any

import pandas as pd

from .schemas import SimilarPoem


class SimilaritySearcher:
    """Find semantically similar poems using a precomputed nearest-neighbor index."""

    def __init__(
            self,
            nearest_neighbors: Any,
            reference_metadata: pd.DataFrame,
        ) -> None:
        self.nearest_neighbors = nearest_neighbors
        self.reference_metadata = reference_metadata.reset_index(drop=True)

    @staticmethod
    def _safe_value(row: pd.Series, column: str) -> str | None:
        """Return a string value from a row when available."""
        if column not in row.index:
            return None

        value = row[column]

        if pd.isna(value):
            return None

        return str(value)

    def find_similar(
            self,
            X,
            top_n: int = 5,
        ) -> list[SimilarPoem]:
        """Return the top-N nearest poems by cosine similarity."""
        n_neighbors = min(top_n, len(self.reference_metadata))

        distances, indices = self.nearest_neighbors.kneighbors(
            X,
            n_neighbors=n_neighbors,
        )

        results: list[SimilarPoem] = []

        for neighbor_idx, distance in zip(indices[0], distances[0]):
            row = self.reference_metadata.iloc[int(neighbor_idx)]
            score = round(float(1.0 - distance), 6)

            title = (
                self._safe_value(row, "title")
                or self._safe_value(row, "title_raw")
                or "unknown"
            )

            results.append(
                SimilarPoem(
                    poem_id=self._safe_value(row, "poem_id"),
                    title=title,
                    title_raw=self._safe_value(row, "title_raw"),
                    poet=self._safe_value(row, "poet"),
                    poet_raw=self._safe_value(row, "poet_raw"),
                    source=self._safe_value(row, "source"),
                    corpus_role=self._safe_value(row, "corpus_role"),
                    score=score,
                )
            )

        return results
