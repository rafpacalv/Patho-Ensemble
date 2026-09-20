# Qué llega exactamente al meta-clasificador

## Flujo de datos (simplificado)

```
Parches de slide
    ↓
[Embeddings 768/2560-dim de cada parche]  ← esto SÍ son embeddings
    ↓ ABMIL (per fold, per modelo base)
[Attention pooling + FC linear layer]
    ↓
preds.npy: [N_samples, num_classes]  ← esto es lo que llega al metaclasificador
    (probabilidades, ya reducidas a dimensión común)
    ↓
ensemble4.py: concatena los preds.npy de K modelos
    ↓
X_train: [N_train, K * num_classes]  ← features concatenadas
    ↓
(OPCIONAL: reweight rescala cada bloque)
    ↓
Meta-clasificador entrena sobre X_train
    (logreg / svm / knn / nb / mlp / etc.)
```

## Dimensiones concretas

Ejemplo: `cptac_brca/TP53_mutation` (tarea binaria, `num_classes=2` sin `--drop_redundant_class`):

### ABMIL: salida de cada modelo
- Entrada a ABMIL: N parches × 768-dim (ViT), 1536-dim (CTransPath), 2560-dim (Virchow), etc.
- Salida de ABMIL: `[N, 2]` (probabilidades por clase)

### ensemble4.py: entrada al meta-clasificador
- 3 modelos base: Virchow (2560-dim parche), UNI v2 (768-dim), CTransPath (1536-dim)
- Salida de cada ABMIL: `[N, 2]` (probabilidades P(TP53_neg), P(TP53_wt))
- Concatenadas: `[N, 6]` ← **esto es lo que ve el meta-clasificador**
  - Columnas 0-1: Virchow (P(neg), P(wt))
  - Columnas 2-3: UNI (P(neg), P(wt))
  - Columnas 4-5: CTransPath (P(neg), P(wt))

### Reweight (si `--reweight_by_model_importance`)
Si el bloque de Virchow tiene peso 1.2, UNI peso 0.9, CTransPath peso 1.1:
- Columnas 0-1 se multiplican por 1.2
- Columnas 2-3 se multiplican por 0.9
- Columnas 4-5 se multiplican por 1.1
- Resultado: `[N, 6]` rescalado

**Lo crucial:** El reweight **no** cambia las dimensiones. Solo reescala en lugar.

## Por qué el reweight ha empeoraba en nuestras pruebas

- El meta-clasificador entrena con pesos ya aprendidos por los ABMILs base
  (las probabilidades que salen de ellos reflejan la confianza del modelo)
- Volver a reescalar esos pesos basándose en los coeficientes de una LR
  auxiliar introduce un segundo nivel de ponderación que mete ruido
- `norm_ratio` y `softmax` empeoraban AUC de forma significativa (p<0.01)
- `signed` fue neutro (ns, p~0.65)

## El sbatch que acabo de crear

`run_svm_knn_nb_reweight.sbatch` corre:

1. **SVM (sin reweight)** + **SVM (con reweight)**
2. **KNN (sin reweight)** + **KNN (con reweight)**
3. **NB (sin reweight)** + **NB (con reweight)**

Cada uno sobre los mismos 50 folds de `cptac_brca`, con test pareado
(Wilcoxon signed-rank + Holm correction) para medir si el reweight ayuda o
empeora.

### Cómo usarlo

```bash
# Ejecutar sobre cptac_brca / TP53_mutation (por defecto):
sbatch run_svm_knn_nb_reweight.sbatch

# O especificar otro dataset/tarea:
DATASET=cervical_subtype TASK=subtype sbatch run_svm_knn_nb_reweight.sbatch
```

El script:
- Comprueba que existan las bases entrenadas (`_train_eval/val_outputs/fold_0`)
- Corre los 6 entrenamientos secuencialmente (~5 min cada uno, ~30 min total)
- Genera tabla con Δ medio, IC 95%, wins/ties/losses, y p-values Holm-corregidos
- Guarda logs en `logs/cptac_brca_TP53_mutation/`
- Salva resultados en `PARADIS/datos/patches/.../abmil/ensemble4_{svm,knn,nb}_reweighted/`

## Referencias

- Informe detallado: `INFORME_REWEIGHT_METACLASIFICADOR.md` (sección 2 y 3)
- Implementación de reweight: `src/ensemble4.py` (_compute_reweight_scales, forward)
- Cálculo de importancia: `src/utils.py` (model_importance_from_coefs)
