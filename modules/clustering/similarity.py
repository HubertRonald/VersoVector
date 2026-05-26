from __future__ import annotations

__all__ = [
    "cosine_similarity_matrix",
    "recommend_by_cosine",
    "get_top_neighbors_by_cosine",
    "recommendation_pearson_fast",
]

import numpy as np
import pandas as pd
from utils import Constants

from sklearn.metrics.pairwise import cosine_similarity


def cosine_similarity_matrix(X) -> np.ndarray:
    """Compute pairwise cosine similarity."""
    return cosine_similarity(X)


def recommend_by_cosine(
        title: str,
        cosine_sim: np.ndarray,
        df: pd.DataFrame,
        top_n: int = 10,
    ) -> pd.DataFrame:
    """Recommend nearest poems by cosine similarity."""
    df = df.reset_index(drop=True)
    indices = pd.Series(df.index, index=df.title).drop_duplicates()
    
    if title not in indices:
        raise ValueError(f"El título '{title}' no existe en el corpus.")

    idx = int(indices[title])
    
    sim_scores = sorted(enumerate(cosine_sim[idx]), key=lambda item: item[1], reverse=True)
    sim_scores = sim_scores[1 : top_n + 1]
    
    poem_indices = [i for i, _ in sim_scores]
    scores = [float(score) for _, score in sim_scores]

    return pd.DataFrame({
        "puesto": range(1, len(poem_indices) + 1),
        "title": df.loc[poem_indices, "title"].values,
        "poet": df.loc[poem_indices, "poet"].values,
        "source": df.loc[poem_indices, "source"].values,
        "corpus_role": df.loc[poem_indices, "corpus_role"].values,
        "similarity_cosine": scores,
    })


def get_top_neighbors_by_cosine(
        cosine_sim: np.ndarray,
        df: pd.DataFrame,
        top_n: int = 5,
    ) -> tuple[list[list[str]], list[list[float]]]:
    """Return top cosine neighbors and scores for every poem."""
    df = df.reset_index(drop=True)
    titles = df["title"].tolist()
    
    nearest_titles: list[list[str]] = []
    nearest_scores: list[list[float]] = []

    for idx in range(len(df)):
        sim_scores = sorted(enumerate(cosine_sim[idx]), key=lambda item: item[1], reverse=True)
        
         # Excluir el mismo poema
        sim_scores = sim_scores[1 : top_n + 1]
        
        nearest_titles.append([titles[i] for i, _ in sim_scores])
        nearest_scores.append([round(float(score), 4) for _, score in sim_scores])

    return nearest_titles, nearest_scores


def recommendation_pearson_fast(
        query_title: str,
        X: np.ndarray,
        df: pd.DataFrame,
        top_n: int = 5,
    ) -> pd.DataFrame:
    """Pearson recommendation for one query without building the full correlation matrix."""
    df = df.reset_index(drop=True)
    indices = pd.Series(df.index, index=df.title).drop_duplicates()
    
    if query_title not in indices:
        raise ValueError(f"El título '{query_title}' no existe en el corpus.")

    idx = int(indices[query_title])
    
    X = np.asarray(X, dtype=np.float32)
    q = X[idx]
    
    n_features = X.shape[1]

    row_means = X.mean(axis=1)
    row_sumsq = np.square(X).sum(axis=1)
    
    q_mean = q.mean()
    q_sumsq = np.square(q).sum()

    numerator = X @ q - n_features * row_means * q_mean
    
    denominator = np.sqrt(
        row_sumsq - n_features * np.square(row_means)
    ) * np.sqrt(
        q_sumsq - n_features * q_mean**2
    )
    
    pearson_scores = np.divide(
        numerator,
        denominator,
        out=np.zeros_like(numerator, dtype=np.float32),
        where=denominator != 0,
    )
    
    pearson_scores[idx] = -np.inf

    top_idx = np.argpartition(pearson_scores, -top_n)[-top_n:]
    top_idx = top_idx[np.argsort(pearson_scores[top_idx])[::-1]]

    return pd.DataFrame({
        "puesto": range(1, len(top_idx) + 1),
        "title": df.loc[top_idx, "title"].values,
        "similarity_pearson": pearson_scores[top_idx],
    })


def recommendation_corr(
        query_title: str,
        corr_matrix: pd.DataFrame,
        top_n: int = 5
    ) -> pd.DataFrame:
    """
    Genera recomendaciones basadas en correlación de Pearson entre documentos.

    Args:
        query_title (str): Documento de referencia.
        corr_matrix (pd.DataFrame): Matriz de correlaciones.
        top_n (int): Número de recomendaciones a devolver.

    Returns:
        pd.DataFrame: Recomendaciones con puesto, título y similitud.
        
    Nota:   La correlación de Pearson se mantiene como comparación exploratoria offline.
            Para recomendaciones en línea se usa similitud coseno por eficiencia y
            estabilidad en espacios vectoriales de texto.
            
            O(n_poemas² × n_features)
    """
    if query_title not in corr_matrix.columns:
        raise ValueError(f"El título '{query_title}' no está en la matriz de correlación.")

    # Extraer correlaciones con el documento de referencia
    sim_scores = corr_matrix[query_title].drop(query_title)

    # Ordenar de mayor a menor correlación
    sim_scores = sim_scores.sort_values(ascending=False).head(top_n)

    return pd.DataFrame({
        "Puesto": range(Constants.ONE, len(sim_scores) + Constants.ONE),
        "Recomendación": sim_scores.index,
        "Similitud": sim_scores.values
    })
    

def get_top_neighbors_by_corr(
        corr_matrix: pd.DataFrame,
        top_n: int = 5
    ) -> tuple[list[list[str]], list[list[float]]]:
    """
    Retorna vecinos más similares por correlación de Pearson para cada poema.
    """
    nearest_titles = []
    nearest_scores = []

    for title in corr_matrix.columns:
        sim_scores = corr_matrix[title].drop(title)
        sim_scores = sim_scores.sort_values(ascending=False).head(top_n)

        nearest_titles.append(sim_scores.index.tolist())
        nearest_scores.append(sim_scores.round(4).tolist())

    return nearest_titles, nearest_scores
