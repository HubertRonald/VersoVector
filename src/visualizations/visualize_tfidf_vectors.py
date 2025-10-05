
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def plot_tfidf_vectors(save: bool = True):
    """
    Visualiza en 3D los vectores TF-IDF correspondientes a dos versos de
    'Los Heraldos Negros' de César Vallejo y guarda la imagen en ./figs/.
    """

    # === Vectores TF-IDF ===
    # Primer verso: "Hay golpes en la vida, tan fuertes... ¡Yo no sé!"
    v1 = np.array([0.215, 0.282, 0.282])

    # Segundo verso: "Golpes como del odio de Dios."
    v2 = np.array([0.178, 0.315, 0.201])

    # === Visualización ===
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')

    # Origen
    origin = np.zeros(3)

    # Dibujar vectores
    ax.quiver(*origin, *v1, color='blue', arrow_length_ratio=0.1, linewidth=2, label="Verso 1")
    ax.quiver(*origin, *v2, color='red', arrow_length_ratio=0.1, linewidth=2, label="Verso 2")

    # Configuración de ejes
    ax.set_xlim(0, 0.35)
    ax.set_ylim(0, 0.35)
    ax.set_zlim(0, 0.35)

    ax.set_xlabel('TF-IDF: "golpes"')
    ax.set_ylabel('TF-IDF: "vida"')
    ax.set_zlabel('TF-IDF: "fuertes"')

    ax.set_title('Vectores TF-IDF\n"Los Heraldos Negros"\n(César Vallejo)')
    ax.legend()

    plt.tight_layout()

    # === Guardar figura ===
    if save:
        figs_dir = Path(__file__).resolve().parents[1].parent / "figs"
        figs_dir.mkdir(exist_ok=True)
        output_path = figs_dir / "vallejo_tfidf_vectors.png"
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        print(f"✅ Figura guardada en: {output_path}")

    plt.show()

if __name__ == "__main__":
    plot_tfidf_vectors()
