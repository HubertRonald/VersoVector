from __future__ import annotations

__all__ = [
    "parse_json_list",
    "json_dumps_list",
    "build_integrated_results",
    "cluster_tag_crosstab",
    "cluster_topic_crosstab",
    "cluster_poet_crosstab",
    "cluster_source_crosstab",
    "poet_tag_crosstab",
    "top_tags_by_cluster",
    "build_vallejo_view",
]

import ast
import json
from typing import Any

import pandas as pd


DEFAULT_KEY_COLS: list[str] = ["poem_id"]


def parse_json_list(value: Any) -> list:
    """
    Convierte strings JSON/list-like a lista.

    Soporta:
    - listas reales
    - strings JSON: '["a", "b"]'
    - strings tipo Python: "['a', 'b']"
    - nulos
    """
    if isinstance(value, list):
        return value

    if pd.isna(value):
        return []

    try:
        parsed = json.loads(str(value))
        return parsed if isinstance(parsed, list) else [parsed]
    except Exception:
        pass

    try:
        parsed = ast.literal_eval(str(value))
        return parsed if isinstance(parsed, list) else [parsed]
    except Exception:
        return []


def json_dumps_list(value: Any) -> str:
    """Serializa listas a JSON seguro para CSV."""
    if not isinstance(value, list):
        value = parse_json_list(value)

    return json.dumps(value, ensure_ascii=False)


def _available_columns(
        df: pd.DataFrame,
        columns: list[str],
    ) -> list[str]:
    """Retorna columnas disponibles preservando orden."""
    return [col for col in columns if col in df.columns]


def _validate_key_columns(
        df: pd.DataFrame,
        key_cols: list[str],
        df_name: str,
    ) -> None:
    """Valida que un DataFrame contenga las columnas llave."""
    missing = [col for col in key_cols if col not in df.columns]

    if missing:
        raise ValueError(
            f"{df_name} no contiene las columnas llave requeridas: {missing}"
        )


def _warn_duplicate_keys(
        df: pd.DataFrame,
        key_cols: list[str],
        df_name: str,
    ) -> None:
    """
    Imprime advertencia si hay llaves duplicadas.

    No detiene la ejecución porque algunos corpus pueden tener títulos repetidos,
    pero ayuda a detectar uniones potencialmente ambiguas.
    """
    duplicated = df.duplicated(subset=key_cols).sum()

    if duplicated > 0:
        print(
            f"Advertencia: {df_name} tiene {duplicated} filas con llaves "
            f"duplicadas usando {key_cols}."
        )


def _validate_unique_keys(
    df: pd.DataFrame,
    key_cols: list[str],
    df_name: str,
) -> None:
    duplicated = df.duplicated(subset=key_cols).sum()

    if duplicated > 0:
        sample = (
            df.loc[df.duplicated(subset=key_cols, keep=False), key_cols]
            .head(10)
        )
        raise ValueError(
            f"{df_name} tiene {duplicated} filas con llaves duplicadas "
            f"usando {key_cols}. Muestra:\n{sample}"
        )


def build_integrated_results(
        base_df: pd.DataFrame,
        unsupervised_df: pd.DataFrame,
        supervised_predictions_df: pd.DataFrame,
        key_cols: list[str] | None = None,
    ) -> pd.DataFrame:
    """
    Integra metadata base, salidas no supervisadas y predicciones supervisadas.

    Args:
        base_df:
            DataFrame del corpus procesado, normalmente data/poems_processed.csv.

        unsupervised_df:
            DataFrame producido por 04_embeddings_unsupervised.ipynb.

        supervised_predictions_df:
            DataFrame producido por 03_embeddings_supervised.ipynb.

        key_cols:
            Columnas usadas para alinear resultados. Por defecto:
            title, poet, source, corpus_role.

    Returns:
        DataFrame integrado.
    """
    key_cols = key_cols or DEFAULT_KEY_COLS

    _validate_key_columns(base_df, key_cols, "base_df")
    _validate_key_columns(unsupervised_df, key_cols, "unsupervised_df")
    _validate_key_columns(
        supervised_predictions_df,
        key_cols,
        "supervised_predictions_df",
    )
    
    _validate_unique_keys(base_df, key_cols, "base_df")
    _validate_unique_keys(unsupervised_df, key_cols, "unsupervised_df")
    _validate_unique_keys(supervised_predictions_df, key_cols, "supervised_predictions_df")

    _warn_duplicate_keys(base_df, key_cols, "base_df")
    _warn_duplicate_keys(unsupervised_df, key_cols, "unsupervised_df")
    _warn_duplicate_keys(
        supervised_predictions_df,
        key_cols,
        "supervised_predictions_df",
    )

    # -------------------------
    # Base metadata
    # -------------------------
    base_cols = _available_columns(
        base_df,
        key_cols
        + [
            "title",
            "title_raw",
            "poet",
            "poet_raw",
            "source",
            "corpus_role",
            "tags",
            "poem",
            "poem_raw",
            "poem_processed",
        ],
    )

    base = base_df[base_cols].copy()

    if "tags" in base.columns:
        base["original_tags"] = base["tags"].apply(parse_json_list)
        base = base.drop(columns=["tags"])

    # -------------------------
    # Supervised predictions
    # -------------------------
    supervised = supervised_predictions_df.copy()

    if "predicted_tags" not in supervised.columns:
        if "predicted_tags_json" in supervised.columns:
            supervised["predicted_tags"] = supervised[
                "predicted_tags_json"
            ].apply(parse_json_list)
        else:
            supervised["predicted_tags"] = [[] for _ in range(len(supervised))]
    else:
        supervised["predicted_tags"] = supervised["predicted_tags"].apply(
            parse_json_list
        )

    supervised_cols = _available_columns(
        supervised,
        key_cols + ["predicted_tags", "predicted_tags_json"],
    )

    supervised = supervised[supervised_cols].copy()

    # -------------------------
    # Unsupervised outputs
    # -------------------------
    unsupervised = unsupervised_df.copy()

    for col in [
        "nearest_titles_cosine_json",
        "nearest_scores_cosine_json",
    ]:
        if col in unsupervised.columns:
            unsupervised[col.replace("_json", "")] = unsupervised[col].apply(
                parse_json_list
            )

    # Evitar columnas duplicadas de metadata al unir.
    unsupervised_extra_cols = [
        col
        for col in unsupervised.columns
        if col not in base.columns or col in key_cols
    ]

    integrated = base.merge(
        unsupervised[unsupervised_extra_cols],
        on=key_cols,
        how="left",
        validate="one_to_one",
    )

    integrated = integrated.merge(
        supervised,
        on=key_cols,
        how="left",
        validate="one_to_one",
    )

    if "predicted_tags" in integrated.columns:
        integrated["predicted_tags"] = integrated["predicted_tags"].apply(
            parse_json_list
        )
    else:
        integrated["predicted_tags"] = [[] for _ in range(len(integrated))]

    return integrated


def cluster_tag_crosstab(
        df_integracion: pd.DataFrame,
        cluster_col: str = "cluster_km",
        normalize: str | bool = "index",
    ) -> pd.DataFrame:
    """Cruza clusters con tags supervisados predichos."""
    exploded = (
        df_integracion
        .explode("predicted_tags")
        .dropna(subset=["predicted_tags"])
    )

    return pd.crosstab(
        exploded[cluster_col],
        exploded["predicted_tags"],
        normalize=normalize,
    ).round(3)


def cluster_topic_crosstab(
        df_integracion: pd.DataFrame,
        cluster_col: str = "cluster_km",
        normalize: str | bool = "index",
    ) -> pd.DataFrame:
    """Cruza clusters con tópicos LDA dominantes."""
    return pd.crosstab(
        df_integracion[cluster_col],
        df_integracion["lda_topic_id"],
        normalize=normalize,
    ).round(3)


def cluster_poet_crosstab(
        df_integracion: pd.DataFrame,
        cluster_col: str = "cluster_km",
        normalize: str | bool = "index",
    ) -> pd.DataFrame:
    """Cruza clusters con poetas."""
    return pd.crosstab(
        df_integracion[cluster_col],
        df_integracion["poet"],
        normalize=normalize,
    ).round(3)


def cluster_source_crosstab(
        df_integracion: pd.DataFrame,
        cluster_col: str = "cluster_km",
        normalize: str | bool = "index",
    ) -> pd.DataFrame:
    """Cruza clusters con fuente/corpus."""
    return pd.crosstab(
        df_integracion[cluster_col],
        df_integracion["source"],
        normalize=normalize,
    ).round(3)


def poet_tag_crosstab(
        df_integracion: pd.DataFrame,
        normalize: str | bool = "index",
    ) -> pd.DataFrame:
    """Cruza poetas con tags predichos."""
    exploded = (
        df_integracion
        .explode("predicted_tags")
        .dropna(subset=["predicted_tags"])
    )

    return pd.crosstab(
        exploded["poet"],
        exploded["predicted_tags"],
        normalize=normalize,
    ).round(3)


def top_tags_by_cluster(
        df_integracion: pd.DataFrame,
        cluster_col: str = "cluster_km",
        top_n: int = 5,
    ) -> pd.DataFrame:
    """Retorna los tags predichos más frecuentes por cluster."""
    exploded = (
        df_integracion
        .explode("predicted_tags")
        .dropna(subset=["predicted_tags"])
    )

    summary = (
        exploded
        .groupby([cluster_col, "predicted_tags"])
        .size()
        .reset_index(name="count")
        .sort_values([cluster_col, "count"], ascending=[True, False])
    )

    return (
        summary
        .groupby(cluster_col)
        .head(top_n)
        .reset_index(drop=True)
    )


def build_vallejo_view(
        df_integracion: pd.DataFrame,
    ) -> pd.DataFrame:
    """Vista compacta de César Vallejo dentro del corpus integrado."""
    poet = df_integracion.get(
        "poet",
        pd.Series(index=df_integracion.index, dtype=str),
    ).astype(str)

    poet_raw = df_integracion.get(
        "poet_raw",
        pd.Series(index=df_integracion.index, dtype=str),
    ).astype(str)

    source = df_integracion.get(
        "source",
        pd.Series(index=df_integracion.index, dtype=str),
    ).astype(str)

    title = df_integracion.get(
        "title",
        pd.Series(index=df_integracion.index, dtype=str),
    ).astype(str)

    mask = (
        poet.str.contains("vallejo", case=False, na=False)
        | poet_raw.str.contains("vallejo", case=False, na=False)
        | source.eq("cesar_vallejo")
        | title.str.contains("black herald|vallejo", case=False, na=False)
    )

    columns = [
        "title",
        "title_raw",
        "poet",
        "poet_raw",
        "source",
        "corpus_role",
        "cluster_km",
        "cluster_gmm",
        "cluster_agg",
        "cluster_dbscan",
        "lda_topic_id",
        "lda_topic_prob",
        "lda_topic_terms",
        "predicted_tags",
        "nearest_titles_cosine",
        "nearest_scores_cosine",
        "embedding_x",
        "embedding_y",
    ]

    columns = [col for col in columns if col in df_integracion.columns]

    return df_integracion.loc[mask, columns].reset_index(drop=True)
