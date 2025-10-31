"""
Visualización 2D de poemas (UMAP/t-SNE → KMeans)
------------------------------------------------
Genera un scatter plot con:
- puntos (poemas)
- centroides destacados
- etiquetas de clúster
- guardado automático en ./figs/
"""

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

plt.style.use("seaborn-v0_8")

def plot_poem_clusters_2d(
    X2: np.ndarray,
    df_poems: pd.DataFrame,
    cluster_col: str = "cluster_km",
    reducer_name: str = "UMAP",
    save: bool = True,
    save_name: str = "umap_kmeans_clusters.png"
):
    """
    Renderiza visualización 2D de embeddings reducidos + clusters.
    
    Args:
        X2 (np.ndarray): matriz 2D resultante de UMAP o t-SNE
        df_poems (pd.DataFrame): dataframe con la columna cluster asignada
        cluster_col (str): nombre de la columna donde está el cluster
        reducer_name (str): 'UMAP' o 't-SNE' usado para el título
        save (bool): si True guarda la imagen
        save_name (str): nombre del archivo a guardar
    """

    if cluster_col not in df_poems.columns:
        raise KeyError(f"La columna '{cluster_col}' no está en el DataFrame.")

    clusters = df_poems[cluster_col].to_numpy()
    n_clusters = len(np.unique(clusters))

    fig, ax = plt.subplots(figsize=(8, 7))
    sc = ax.scatter(
        X2[:, 0], X2[:, 1],
        c=clusters,
        cmap="viridis_r",
        s=18, alpha=0.25
    )

    ax.set_title(f"Poemas en 2D ({reducer_name} → KMeans, k={n_clusters})")
    ax.set_xlabel("comp-1"); ax.set_ylabel("comp-2")

    # === Calcular centroides ===
    centroids = np.vstack([
        X2[clusters == c].mean(axis=0) for c in range(n_clusters)
    ])

    ax.scatter(
        centroids[:, 0],
        centroids[:, 1],
        c="black", s=250, marker="X", label="Centroides"
    )

    # === Agregar etiquetas cerca de cada centroide ===
    for idx, (x, y) in enumerate(centroids):
        ax.text(
            x + 0.3, y + 0.3,
            f"Cluster {idx}",
            fontsize=10, weight="bold"
        )

    cbar = plt.colorbar(sc, ax=ax, label="cluster_km")
    ax.legend(loc="upper right")

    plt.tight_layout()

    if save:
        figs_path = Path(__file__).resolve().parents[2] / "figs"
        figs_path.mkdir(parents=True, exist_ok=True)
        file_out = figs_path / save_name
        plt.savefig(file_out, dpi=300)
        print(f"✅ Imagen guardada en: {file_out}")

    plt.show()


# === Ejemplo de uso (si ejecutas el script directamente) ===
if __name__ == "__main__":
    print("Este script está pensado para ser importado desde un notebook o módulo.")
