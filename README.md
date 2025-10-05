<p align="left">
    <a href="https://www.python.org/" target="_blank">
        <img src="https://img.shields.io/badge/python-3670A0?style=flat-square&logo=python&logoColor=ffdd54" />
        </a>
    <a href="https://pytorch.org/" target="_blank">
        <img src="https://img.shields.io/badge/PyTorch-red.svg?logo=pytorch?style=flat-square&logoColor=white" />
    </a>
    <a href="https://huggingface.co/" target="_blank">
        <img src="https://img.shields.io/badge/Transformers-yellow.svg?logo=huggingface?style=flat-square&logoColor=white" />
    </a>
    <a href="https://scikit-learn.org/" target="_blank">
        <img src="https://img.shields.io/badge/scikit--learn-orange.svg?logo=scikit-learn?style=flat-square&logoColor=white" />
    </a>
    <a href="https://pandas.pydata.org/" target="_blank">
        <img src="https://img.shields.io/badge/Pandas-DataFrames-green.svg?logo=pandas?style=flat-square&logoColor=white" />
    </a>
    <a href="https://hub.docker.com/r/google/cloud-sdk" target="_blank">
        <img src="https://img.shields.io/badge/docker-%230db7ed.svg?style=flat-square&logo=docker&logoColor=white" />
    </a>
    <a href="https://code.visualstudio.com/download" target="_blank">
        <img src="https://img.shields.io/badge/Visual%20Studio%20Code-007ACC?logo=visualstudiocode&logoColor=fff&flat-square=plastic" />
    </a>
    <img src="https://img.shields.io/github/last-commit/HubertRonald/PoesiaEmbeddingsClusteringClassification?style=flat-square" />
    <img src="https://img.shields.io/github/commit-activity/t/HubertRonald/PoesiaEmbeddingsClusteringClassification?style=flat-square&color=dodgerblue" />
</p>

# Poesia Embeddings Clustering Classification
Exploración de poesía mediante machine learning: generación de embeddings, clustering y clasificación emocional usando textos de César Vallejo y otros poetas traducidos al inglés.


## 📖 Descripción
Este proyecto explora la relación entre el **significado semántico y emocional** de la poesía a través de modelos de *embeddings* modernos.  
Combina dos enfoques de aprendizaje:

- **Aprendizaje no supervisado:** Agrupamiento (clustering) de poemas por estilo o tono.  
- **Aprendizaje supervisado:** Clasificación de poemas por emoción o tema.

Se busca responder:  
> “¿Puede un modelo de lenguaje percibir la emoción detrás de un poema, como lo hace un lector humano?”

## 🧠 Flujo general del proyecto

Cómo se presentará los modelos a emplear en este repositorio

```mermaid
graph TD
    A[Poemas originales<br>Vallejo + otros] --> B[Preprocesamiento<br>Limpieza y Tokenización]
    B --> C[Extracción de características<br>TF-IDF / BERT embeddings]
    C --> D1[Clustering no supervisado<br>KMeans / GaussianMixture / UMAP]
    C --> D2[Clasificación supervisado<br>LogReg / SVM / BERT]
    D1 --> E1[Análisis de temas y emociones emergentes]
    D2 --> E2[Predicción de emoción o tono poético]
```
