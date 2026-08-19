# Reweight por importancia de modelo en el meta-clasificador

Fecha: 2026-08-19
Rama: `pruebas_rafa`
Código relevante: [`src/ensemble4.py`](src/ensemble4.py) (`Ensemble._compute_reweight_scales`,
`Ensemble.forward`), [`src/utils.py`](src/utils.py) (`model_importance_from_coefs`),
[`src/analyze_meta_importance.py`](src/analyze_meta_importance.py)

## Contexto

La idea original: usar Logistic Regression para identificar qué modelo base
(ctranspath, uni_v2, virchow_v1, …) domina la decisión del meta-clasificador,
y usar esa información para reponderar las features de entrada de **cualquier**
meta-modelo (SVM, KNN, NB, MLP…), no solo de la propia LR.

Esto ya está implementado (commits `8948a2d`, `5ab90b3`, `3ca14c7`). Este
informe documenta (1) cómo funciona el cálculo actual paso a paso, (2) la duda
planteada sobre las dimensiones de los modelos base, y (3) dos opciones para
refinar el cálculo del reweight, sin implementarlas todavía.

---

## 1. Aclaración importante: qué es lo que se reponderar

**No se reponderan los embeddings crudos de los modelos base** (768-dim para
la mayoría, 2560 para Virchow). Esos embeddings nunca llegan a `ensemble4.py`.

El pipeline en dos etapas es:

```
patch embeddings (768/2560/... dim) → ABMIL de cada modelo → preds.npy [N, num_classes]
                                                                      │
                                                        ensemble4.py concatena
                                                        los preds.npy de cada modelo
```

Cada ABMIL (`train_abmil.py`) ya reduce el embedding de su propio modelo
fundacional a un vector de probabilidades `[N, num_classes]` (2 clases, 3
clases, según la tarea) — es justo esa reducción lo que hace que los modelos
con distinta dimensión de embedding (768 vs 2560) sean combinables en primer
lugar.

`ensemble4.py::forward()` ([líneas 235-257](src/ensemble4.py#L235-L257)) carga
`preds.npy` de cada modelo y comprueba si todas tienen la misma dimensión:

```python
dims = [p.shape[1] for p in xpreds]
all_same_dim = len(set(dims)) == 1

if all_same_dim:
    # Caso típico y el único usado en la práctica: todos los preds.npy
    # tienen num_classes columnas → concatenación por bloques
    X_train = xstacked.permute(1, 0, 2).reshape(N, -1)   # [N, num_models*num_classes]
else:
    # Fallback defensivo, no usado en el flujo real: concatenación directa
    X_train = torch.cat(xpreds, dim=1)
```

**No es una suma ponderada.** Es concatenación por columnas:

```
X = [ modelo_0: P(clase_0), P(clase_1), ... | modelo_1: P(clase_0), P(clase_1), ... | ... ]
```

En el caso típico (`all_same_dim=True`), cada bloque mide exactamente
`num_classes` columnas, igual para todos los modelos — de ahí que el reweight
pueda reescalar bloques completos con un único escalar por modelo sin
ambigüedad. El bloque `else` (dimensiones distintas) existe como salvaguarda,
pero en la práctica no se alcanza: todos los modelos de un mismo ensemble
comparten tarea y por tanto `num_classes`.

**Conclusión a la duda planteada**: la distinta dimensión de los embeddings
(768/2560/...) de los modelos base es irrelevante en esta etapa — ya fue
absorbida por cada ABMIL individual. Lo que se concatena y reescala son
vectores de probabilidad de tamaño `num_classes`, idéntico entre modelos.

---

## 2. Cómo se calcula el reweight (implementación actual)

### Paso 1 — LR auxiliar desechable

`_compute_reweight_scales(X_train, y_train, num_models)`
([ensemble4.py:149-186](src/ensemble4.py#L149-L186)) entrena una
`LogisticRegression(max_iter=1000)` **solo para medir**, sobre el mismo
`X_train` ya transformado (`_transform`, prob o logit según
`--feature_space`). Nunca se usa para predecir — es independiente del
`meta_model` que el usuario eligió con `--meta_model`.

### Paso 2 — Agregación de coeficientes por bloque de modelo

`model_importance_from_coefs()` ([utils.py](src/utils.py)) trocea
`lr_aux.coef_[0]` en `num_models` bloques de `num_classes` columnas y calcula,
por bloque:

```
norm_i = ||coef[i*num_classes : (i+1)*num_classes]||_2
```

Luego normaliza a media 1.0, con un exponente de temperatura:

```
weight_i = (norm_i / mean(norms)) ** temperature
weight_i /= mean(weight)          # renormaliza tras aplicar el exponente
```

- `temperature = 0` → todos los pesos = 1.0 (no-op)
- `temperature = 1` (default) → lineal
- `temperature > 1` → acentúa la diferencia entre modelos

### Paso 3 — Expansión a vector de escalas y aplicación

El peso escalar de cada modelo se repite en todas las columnas de su bloque:

```python
scales = np.ones(X_train.shape[1])
scales[i*num_classes:(i+1)*num_classes] = weight_i
```

y se aplica por multiplicación elemento a elemento:

```python
X_train = X_train * scales
X_val   = X_val   * scales
X_test  = X_test  * scales   # mismas scales que train, no recalculadas
```

El meta-modelo real (SVM/KNN/NB/MLP/logreg/...) se entrena **después**, sobre
`X_train`/`X_val` ya reescalados.

---

## 3. Actualización: las dos opciones, implementadas y probadas

Implementadas en `--reweight_method {norm_ratio, softmax, signed}` (nuevo flag
de `ensemble4.py`; `norm_ratio` = comportamiento anterior, es el default).

### Corrección encontrada al probar "signed"

La fórmula de la Opción B tal como se planteó abajo (§4) tiene un fallo real,
descubierto reproduciéndola con datos sintéticos antes de usarla: **sin
`--drop_redundant_class`, cada bloque tiene `num_classes` columnas
complementarias** (softmax suma 1, así que `P(clase_1) = 1 - P(clase_0)`).
Sus coeficientes en la LR auxiliar salen con signos opuestos y magnitud
similar — el escalado firmado terminaba amplificando una columna y **anulando
la otra columna del mismo modelo** (se observó `scale=0.0` exactamente sobre
la columna de mayor coeficiente en valor absoluto de todo el vector, es decir,
la más informativa). No es una diferencia entre modelos, es un artefacto de
la colinealidad interna de cada bloque.

**Fix aplicado**: `_compute_reweight_scales` ahora exige `num_classes == 1`
(es decir, `--drop_redundant_class` activo) para `reweight_method="signed"`,
y lanza `ValueError` explicando por qué si no lo está. Verificado con datos
sintéticos: con 1 columna/modelo, el modelo dominante recibe `scale≈2.0` y
los demás `≈1.0`, sin artefactos.

### Resultados empíricos (smoke test real, no solo compilación)

`cptac_brca/TP53_mutation`, `ctranspath uni_v2 virchow_v1`, features
in-sample, `--drop_redundant_class`, 50 folds, comparación **pareada** por
fold (mismo protocolo que `run_3_experiments_v2_improved.sbatch`: Δ medio,
wins/ties/losses, Wilcoxon):

**Meta-modelo `logreg`** (baseline AUC = 0.792 ± 0.015):

| método | ΔAUC medio | wins | ties | losses | Wilcoxon p |
|---|---:|---:|---:|---:|---:|
| norm_ratio | −0.0085 | 12 | 8 | 30 | 0.0128 |
| softmax | −0.0123 | 11 | 5 | 34 | 0.0021 |
| signed | −0.0105 | 17 | 7 | 26 | 0.3373 |

**Meta-modelo `svm`** (baseline AUC = 0.7895, el caso que motivó la idea —
SVM es sensible a escala):

| método | ΔAUC medio | wins | ties | losses | Wilcoxon p |
|---|---:|---:|---:|---:|---:|
| norm_ratio | −0.0091 | 6 | 17 | 27 | 0.0009 |
| softmax | −0.0106 | 6 | 17 | 27 | 0.0009 |
| signed | −0.0064 | 20 | 3 | 27 | 0.6567 |

**Lectura, con Holm sobre los 6 tests (α=0.05, m=6)**:

- `norm_ratio` y `softmax` **empeoran significativamente** el AUC en ambos
  meta-modelos (los 4 p-valores sobreviven Holm: el mayor, 0.0128, es menor
  que su umbral 0.025). Consistente además en dirección: `losses` domina
  claramente sobre `wins` en las 4 filas.
- `signed` **no es distinguible de no reponderar** (p=0.34 y p=0.66, no
  sobreviven ninguna corrección) — ni ayuda ni perjudica de forma medible.

**Interpretación**: reescalar features *antes* de entrenar un modelo que ya
va a aprender su propia combinación lineal (logreg, y en menor medida SVM
lineal) es casi redundante con lo que el modelo optimiza — la LR auxiliar y
el meta-modelo final ven la misma señal, así que forzar una escala previa no
añade información nueva, solo cambia el punto de partida de la
regularización, y aquí lo empeora. Esto es un único dataset/tarea con
features in-sample (conocido optimista, ver "Meta-features: in-sample vs
out-of-fold" en `CLAUDE.md`) — no es una conclusión definitiva multi-tarea,
pero sí evidencia real y pareada en contra de activar
`--reweight_by_model_importance` por defecto, al menos con `norm_ratio` o
`softmax`.

Artefactos de esta corrida (fuera de `experiments.yaml`, nombrados ad hoc con
`--results_subdir_suffix` para esta comparación):
`PARADIS/datos/patches/cptac_brca/TP53_mutation/abmil/ensemble4*_rw_*` —
quedan en disco por si se quieren inspeccionar (`model_importance_weights.json`,
`coefs.npy`, `test_metrics/fold_*/metrics.json`); no están limpiados
automáticamente porque no sé si los quieres conservar como evidencia.

---

## 4. Diseño original de las dos opciones (referencia)

El diseño actual tiene dos límites conocidos, útiles como puntos de mejora si
los resultados empíricos lo justifican.

### Opción A — Softmax en vez de ratio-a-la-media

**Problema actual**: `weight_i = (norm_i / mean(norms))^temperature` puede
crecer sin cota cuando un modelo tiene una norma mucho mayor que el resto (o
colapsar a casi 0 si `mean(norms)` está dominada por un outlier), y la
temperatura funciona "al revés" de lo habitual (subir T acentúa, en vez de
suavizar).

**Alternativa**:

```python
raw = norms / temperature          # temperature baja → distribución más "dura"
weights = num_models * softmax(raw)  # softmax clásico, reescalado a media 1.0
```

- `temperature → 0`: casi one-hot sobre el modelo dominante (winner-take-all)
- `temperature → ∞`: uniforme (equivalente al `temperature=0` actual)
- Acotado en `(0, num_models]` por construcción — nunca hay un peso que
  explote, incluso con normas muy dispares entre modelos.
- Contras: el sentido de la temperatura se invierte respecto a la
  convención actual (`temperature=1` deja de ser el caso lineal "neutro"),
  lo que rompería la semántica ya documentada en `--reweight_temperature` y
  obligaría a rehacer cualquier barrido de temperatura ya hecho.

### Opción B — Reweight con signo, no solo magnitud

**Problema actual**: la norma L2 usada para medir importancia por bloque
descarta el signo. Un modelo cuyos coeficientes son grandes pero
contradictorios entre clases (ruido) recibe el mismo peso que un modelo cuyos
coeficientes son igual de grandes pero consistentemente alineados con la
clase correcta.

**Alternativa** (aplicable sobre todo a clasificación binaria, donde hay una
columna de "clase positiva" clara): en vez de una escala uniforme por bloque
de modelo, calcular una escala **por columna** dentro del bloque, basada en el
coeficiente con signo de esa columna específica:

```python
# en vez de repetir weight_i en todo el bloque:
scales[start:end] = weight_i

# escalar cada columna según su propio coeficiente (con signo, normalizado):
scales[start:end] = 1 + temperature * (coef[start:end] / max(|coef|))
```

- Distingue "este modelo empuja fuerte y consistente hacia la clase positiva"
  de "este modelo tiene mucha norma pero es ambiguo entre clases".
- Contras: para tareas multiclase (`num_classes > 2`) no hay una única
  "dirección positiva" por columna, así que la interpretación deja de ser
  tan directa; y las escalas ya no serían necesariamente positivas, lo que
  puede invertir el signo de una feature — hay que decidir si eso es
  deseable o si conviene un `clip` a `[0, ...]`.

### Recomendación (actualizada tras §3)

Con datos reales ya no es "sin datos": en el smoke test pareado, `softmax` no
resultó más robusto que `norm_ratio` — ambos empeoran de forma significativa,
consistente con que el problema no es la fórmula de agregación sino la
premisa (reescalar antes de un modelo lineal que ya optimiza esa combinación
es redundante-o-peor). `signed` sí evita el daño, pero tampoco ayuda — se
comporta como un no-op ligeramente ruidoso una vez arreglado el bug de la
colinealidad. **Ninguna de las tres justifica activar reweight por defecto**
en este dataset/tarea. Antes de descartarlo del todo, valdría la pena
probarlo donde tiene más sentido teórico: `meta_features=oof` (sin la
inflación in-sample) y con un meta-modelo verdaderamente no lineal (`knn`,
donde la escala de las features determina directamente la distancia), que no
se ha probado todavía.

---

## 5. ¿Suma ponderada en vez de concatenación?

Pregunta planteada: *"¿Sería factible igualar las dimensiones de todos los
clasificadores (mediante transformación, no truncamiento) y hacer una suma
ponderada mediante el reweight antes de pasarlo al metaclasificador?"*

**Aclaración previa**: las dimensiones ya están igualadas — como se explica en
§1, cada bloque de `X` mide `num_classes` columnas para todos los modelos, sin
necesidad de ninguna transformación adicional. No hay un problema de
dimensiones distintas que resolver en esta etapa.

**Lo que de verdad cambiaría con una suma ponderada**: no la dimensión, sino
la operación — sustituir la concatenación (`[N, num_models*num_classes]`,
todas las columnas visibles al meta-modelo) por:

```python
weighted_sum = sum(weight_i * P_i for i in range(num_models)) / sum(weights)
# P_i: [N, num_classes] del modelo i → weighted_sum: [N, num_classes]
```

Es decir, colapsar `X` a un único vector de `num_classes` columnas **antes**
de que el meta-modelo lo vea. Esto es técnicamente factible (es una línea de
código), pero conviene ser explícito sobre qué se pierde:

- **Deja de ser stacking.** Con `num_classes` columnas de entrada, el
  meta-clasificador ya no tiene margen para aprender combinaciones
  específicas por clase o por par de modelos (p. ej. "el modelo A es mejor
  clasificando la clase 2, el modelo B mejor en la 0") — esa es precisamente
  la información que la concatenación preserva y que motiva usar `ensemble4.py`
  en vez de un promedio simple.
- **El proyecto ya tiene esta variante, con otra fuente de pesos.**
  `ensemble.py` hace exactamente esta suma ponderada, con pesos de
  `MetricDistance` derivados de AUC/F1 de validación (ver CLAUDE.md,
  "Ensemble Weighting") en vez de coeficientes de una LR auxiliar. Lo que se
  propone aquí sería, en la práctica, una variante de `ensemble.py` con una
  fuente de pesos distinta — no una variante nueva de `ensemble4.py`.
- **Es más agresivo que el reweight actual, que ya probó perjudicar.** El
  reweight ya implementado en §3 es una versión *suave* de esta misma idea:
  reescala las columnas pero se las deja todas al meta-modelo, que conserva
  libertad para deshacer o ajustar el escalado si no le sirve. Colapsar a
  suma ponderada quita esa libertad por completo. Dado que incluso la versión
  suave empeoró el AUC de forma significativa en el smoke test de §3, la
  versión dura (que es estrictamente menos expresiva) es de esperar que
  empeore igual o más — no hay evidencia en este informe que la respalde.

**Recomendación**: no implementarlo como parte de `ensemble4.py`. Si quieres
comparar específicamente "pesos derivados de LR" vs. "pesos derivados de AUC
de validación" en un esquema de promedio ponderado, el camino más barato es
añadir una fuente de pesos alternativa a `MetricDistance`/`ensemble.py`
(reutilizando `model_importance_from_coefs`), no tocar `ensemble4.py` — son
scripts con roles distintos y ya separados en el proyecto.

---

## 6. Próximos pasos

1. Probar reweight (los 3 métodos) con `meta_features=oof` — el smoke test de
   §3 usó in-sample, conocido optimista; falta ver si la conclusión se
   sostiene con OOF, que es el escenario realista.
2. Probar con `meta_model=knn` específicamente — es el caso donde la escala
   de las features importa más directamente (distancias), y no se ha
   probado todavía.
3. Si en OOF/KNN tampoco ayuda, no añadir filas `_reweighted` a
   `experiments.yaml` por defecto — dejar el flag disponible pero apagado,
   documentado como "probado, no ayuda en este dataset" en vez de asumir que
   sí.
4. Limpiar o conservar deliberadamente los directorios ad hoc de esta corrida
   (`ensemble4*_rw_*` en `cptac_brca/TP53_mutation/abmil/`) — quedan en disco,
   fuera de la convención `experiments.yaml`/`SUBDIR`.

## Referencias

- `IMPLEMENTACION_SVM_KNN_NB_REWEIGHT.md` — resumen de la implementación
  original (Parte A: SVM/KNN/NB, Parte B: reweight)
- Plan aprobado:
  `/home/JKP6679/.claude/plans/estaba-pensando-en-dos-sparkling-treehouse.md`
- CLAUDE.md — sección "Analysis protocol" (comparaciones pareadas, MDE,
  corrección de Holm)
