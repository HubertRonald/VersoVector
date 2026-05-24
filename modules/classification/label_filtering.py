from __future__ import annotations

__all__ = [
    "count_multilabel_tags",
    "filter_rare_multilabel_tags",
]

from collections import Counter
from typing import Any

import pandas as pd

def count_multilabel_tags(
        df: pd.DataFrame,
        tags_col: str = "tags",
    ) -> pd.DataFrame:
    """
    Count the frequency of each tag in a multilabel column.
    
    Args:
        df: DataFrame con una columna de listas de etiquetas.
        tags_col: Nombre de la columna con etiquetas.

    Returns:
        DataFrame con columnas: tag, count.
    """
    counter = Counter(
        tag
        for tags in df[tags_col]
        for tag in tags
    )
    
    return (
        pd.DataFrame(
            counter.items(),
            columns=["tag", "count"]
        )
        .sort_values("count", ascending=False)
        .reset_index(drop=True)
    )


def filter_rare_multilabel_tags(
    df: pd.DataFrame,
    tags_col: str = "tags",
    output_col: str = "filtered_tags",
    min_label_count: int = 10,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """
    Filtra etiquetas raras en un problema multilabel.

    Mantiene únicamente etiquetas cuya frecuencia sea >= min_label_count.
    Luego elimina filas que quedan sin ninguna etiqueta.

    Args:
        df: DataFrame con columna multilabel.
        tags_col: Columna original de etiquetas.
        output_col: Columna nueva con etiquetas filtradas.
        min_label_count: Frecuencia mínima para conservar una etiqueta.

    Returns:
        filtered_df: DataFrame filtrado.
        label_summary_df: resumen de etiquetas y si fueron conservadas.
        metadata: diccionario con estadísticas del filtro.
    """
    df = df.copy()

    label_summary_df = count_multilabel_tags(
        df=df,
        tags_col=tags_col,
    )

    kept_labels = (
        label_summary_df
        .loc[label_summary_df["count"] >= min_label_count, "tag"]
        .sort_values()
        .tolist()
    )

    kept_label_set = set(kept_labels)

    label_summary_df["kept"] = label_summary_df["tag"].isin(kept_label_set)

    df[output_col] = df[tags_col].apply(
        lambda tags: [
            tag
            for tag in tags
            if tag in kept_label_set
        ]
    )

    n_rows_before = len(df)

    filtered_df = (
        df
        .loc[df[output_col].map(len) > 0]
        .reset_index(drop=True)
    )

    metadata = {
        "min_label_count": int(min_label_count),
        "n_rows_before": int(n_rows_before),
        "n_rows_after": int(len(filtered_df)),
        "n_rows_removed": int(n_rows_before - len(filtered_df)),
        "n_labels_before": int(label_summary_df.shape[0]),
        "n_labels_after": int(len(kept_labels)),
        "kept_labels": kept_labels,
        "dropped_labels": (
            label_summary_df
            .loc[~label_summary_df["kept"], "tag"]
            .sort_values()
            .tolist()
        ),
    }

    return filtered_df, label_summary_df, metadata
