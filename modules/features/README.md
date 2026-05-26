# Dense

Algunos modelos sí necesitan matriz densa, pero en el notebook 04 eso se resuelve después, no en el feature pipeline.

| Parte                     | Usa feature pipeline normalizado sparse | Necesita dense después |
| ------------------------- | --------------------------------------: | ---------------------: |
| Similitud coseno          |                                      Sí |      No necesariamente |
| Vecinos por coseno        |                                      Sí |      No necesariamente |
| ComplementNB supervisado  |                                      Sí |                     No |
| MultinomialNB supervisado |                                      Sí |                     No |
| LogisticRegression        |                                      Sí |                     No |
| LDA                       |        No, usa `CountVectorizer` propio |                     No |
| MiniBatchKMeans           |                     Puede usar reducido |     Sí, después de SVD |
| GMM                       |                 No directo sobre sparse |     Sí, después de SVD |
| Agglomerative             |          No directo sobre sparse grande |     Sí, después de SVD |
| UMAP/t-SNE                |                    Mejor sobre reducido | Sí, después de SVD/PCA |
| DBSCAN                    |                 Mejor sobre 2D/reducido |                     Sí |


```text
feature_pipeline sparse + normalized
        ↓
X_reference / X_external
        ↓
TruncatedSVD o PCA
        ↓
X_cluster dense reducido
        ↓
KMeans / GMM / Agg / UMAP / t-SNE / DBSCAN
```

No se hace denso en el espacio completo, sino en una representación reducida.
