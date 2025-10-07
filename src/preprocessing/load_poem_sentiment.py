from __future__ import annotations

# from datasets import load_dataset
import pandas as pd

# Cargar dataset de poemas con emociones
dataset = ... #load_dataset("poem_sentiment")

# Convertir el split de entrenamiento en DataFrame
df = pd.DataFrame(dataset["train"])
df = df.rename(columns={"verse_text": "text", "label": "emotion"})

# Mapear etiquetas numéricas a nombres legibles
label_map = {
    0: "anger",
    1: "fear",
    2: "joy",
    3: "love",
    4: "sadness",
    5: "surprise"
}
df["emotion"] = df["emotion"].map(label_map)

# Guardar dataset
df.to_csv("data/labeled_poems_emotion.csv", index=False)
print("✅ Archivo generado: data/labeled_poems_emotion.csv")
print(df.head())
