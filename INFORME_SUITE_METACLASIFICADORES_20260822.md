# Informe — Suite de metaclasificadores sobre el trío ganador

**Fecha:** 22 de agosto de 2026
**Tarea:** `cptac_brca / TP53_mutation`
**Modelos base:** `ctranspath + uni_v2 + conch_v1_5`, ABMIL independiente por modelo
**Folds:** 50, comparación pareada
**Jobs:** 70291 (embeddings + suite de metaclasificadores) · 70292 (snapshot ensemble a nivel base)

---

## Resumen ejecutivo

Se han construido y evaluado **14 configuraciones** sobre el trío ganador —
diez metaclasificadores en dos espacios de features (§1-§7) y cuatro sobre bases
con snapshot ensemble (§8)— cada grupo en **una sola pasada**, con su baseline
recalculado dentro de ella y sobre **las mismas filas de train/val/test**.

| | |
|---|---|
| **Mejor por AUC** | `MLP Snapshot` sobre probabilidades — **0.8018** |
| **Mejor por métricas de decisión** | `MLP Snapshot` sobre embeddings — bacc **0.7429**, kappa **0.4911** |
| **Referencia (LogReg sobre probabilidades)** | AUC 0.7986 · bacc 0.7302 · kappa 0.4629 |
| **Mejor global por métricas de decisión** | `LogReg` sobre bases FGE — bacc **0.7422**, kappa **0.4845** |
| **Configuraciones que baten a LogReg tras corrección de Holm** | **ninguna de las 13** |

**Las tres conclusiones que ordenan el resto:**

1. **Ninguna de las trece alternativas bate a la regresión logística de forma
   significativa.** Todos los Δ positivos medidos caen dentro del suelo de
   reproducibilidad del pipeline (≈ 0.004 AUC) o no superan Holm. LogReg sobre
   las 6 probabilidades sigue siendo la elección correcta por defecto.
2. **El hundimiento de los MLP no es un fallo de ordenación, es un fallo de
   corte.** `MLP Snapshot` tiene el AUC más alto de todo el barrido (0.8018) y
   a la vez una kappa de 0.2536, la segunda peor. Ajustando el umbral en
   validación, esa pérdida de −0.2093 de kappa se convierte en −0.0066: se
   evapora entera. Es un artefacto del desbalance 40/60 contra un umbral fijo
   en 0.5, no una incapacidad del modelo.
3. **La comparación pendiente de §8.1 se cierra: los embeddings de 1536 D no
   baten a las 6 probabilidades.** `emb:logreg` pierde −0.0160 de AUC frente a
   `prob:logreg`. El único brazo que apunta hacia arriba en las métricas de
   decisión es `emb:mlp_snapshot` (+0.0281 kappa), y no supera Holm.
4. **El snapshot ensemble a nivel base es un nulo, ahora también sobre el trío
   ganador.** `fge_trio:logreg` es numéricamente la mejor configuración de toda
   la campaña y aun así **pierde en 26 de 50 folds**: media positiva, mediana en
   contra. Su MDE (0.026 AUC) es 4 × el de la suite, así que el nulo es más
   ancho de lo que parece.

---

## 1. Qué se ha construido

### 1.1 Las siete tareas

| # | Configuración | Espacio de features | Estado |
|---:|---|---|---|
| 1 | Stacking **LogReg** | probabilidades (6 D) | ✅ |
| 2 | Stacking **MLP** | probabilidades (6 D) | ✅ |
| 3 | Stacking **MLP + Snapshot Ensemble** | probabilidades (6 D) | ✅ |
| 4 | Stacking **SVM** (lineal, C = 1) | probabilidades (6 D) | ✅ |
| 5 | Stacking **KNN** (k = 5, pesos por distancia) | probabilidades (6 D) | ✅ |
| 6 | Stacking **Naive Bayes** (gaussiano) | probabilidades (6 D) | ✅ |
| 7 | §8.1 — **embeddings vs. probabilidades** | embeddings (1536 D) | ✅ |

La tarea 7 se ha instrumentado con cuatro metaclasificadores sobre el espacio de
embeddings (`logreg`, `mlp`, `mlp_snapshot`, `mlp_fge`), que son exactamente las
cuatro filas de la tabla obsoleta del 18-ago que §8.1 pedía re-medir.

### 1.2 Los dos espacios de features

**Probabilidades (el pipeline vigente).** Cada ABMIL colapsa su slide a un
softmax de 2 clases; se concatenan a lo ancho.

```
X_meta = [p₁(c0), p₁(c1), p₂(c0), p₂(c1), p₃(c0), p₃(c1)]   → [N, 6]
```

**Embeddings.** El vector de 512 D que el ABMIL produce tras la atención, justo
antes de su clasificador.

```
X_meta = [emb₁(512), emb₂(512), emb₃(512)]                   → [N, 1536]
```

### 1.3 Los embeddings hubo que regenerarlos

Los que había en disco eran del **18-ago 19:00**, anteriores al checkpoint de
`ctranspath` (20-ago 12:52), y **`conch_v1_5` no tenía ninguno** — nunca había
formado parte de ese experimento. Se han regenerado los tres desde los
checkpoints vigentes con `test_abmil.py --use_embeddings`, y el job verifica
explícitamente que el `mtime` de cada `preds.npy` es posterior al de su
`model.pt` antes de continuar.

Esto es lo que hace que la tarea 7 sea, por primera vez, una medición válida.

---

## 2. Cómo se garantiza que la comparación es limpia

### 2.1 Mismo train, mismo val, mismo test — verificado, no asumido

Los 10 brazos comparten proceso, splits y predicciones base. `run_meta_suite.py`
aborta si algo no cuadra, en dos niveles:

- **Entre modelos base**: para cada fold y cada uno de los tres splits, compara
  el vector de etiquetas de cada modelo contra el del primero. Si difieren en
  longitud o en contenido, `SystemExit`.
- **Entre espacios de features**: la misma comprobación entre el espacio de
  probabilidades y el de embeddings, para que `prob:logreg` y `emb:mlp_snapshot`
  no puedan estar evaluándose sobre poblaciones distintas.

Esta comprobación no es teórica. En este proyecto ya ocurrió que una campaña
evaluase a nivel paciente (103 filas) y otra a nivel slide (112), con AUC que
parecían comparables sin serlo.

Salida del job:

```
Cargando espacio 'prob' ...
  ✓ alineado. Columnas por modelo: {'ctranspath': 2, 'uni_v2': 2, 'conch_v1_5': 2}
  ✓ filas fold_0: train=75 val=15 test=22
Cargando espacio 'emb' ...
  ✓ alineado. Columnas por modelo: {'ctranspath': 512, 'uni_v2': 512, 'conch_v1_5': 512}
  ✓ filas fold_0: train=75 val=15 test=22
✓ Todos los brazos comparten exactamente los mismos folds y filas.
```

### 2.2 El baseline se recalcula en la misma pasada

Regla operativa del proyecto: un directorio `ensemble4_*` antiguo no es
comparable con nada nuevo, porque las predicciones base de las que salió pueden
haber sido regeneradas. Aquí el baseline **no se cita, se recalcula**, en el
mismo proceso que sus contendientes.

**Control de validez:** `prob:logreg` da AUC **0.7986**, bacc **0.7302**,
macro-F1 **0.7294**, kappa **0.4629** — idéntico a los cuatro decimales al valor
publicado en §9.1 del estado del arte. El runner reproduce `ensemble4.py`.

### 2.3 Los constructores son los de `ensemble4.py`

Cada metaclasificador se instancia con los mismos hiperparámetros y la misma
semilla (`42 + fold`) que en `ensemble4.py`, incluida la resolución del `None`
de `--meta_epochs` a 120 para `mlp_snapshot` y 100 para el resto. No se ha
reajustado nada: la pregunta es qué rinde la configuración de referencia, no qué
rendiría si se tuneara.

---

## 3. Métricas y la referencia degenerada

La muestra está desbalanceada: **67 wild-type / 45 mutado**, un 40.2 % de
positivos globales (39.4 % dentro de los splits de test, con 21.5 slides de test
por fold de media). Ningún fold de test tiene una sola clase, así que las 50
observaciones son válidas para las siete métricas.

Con ese desbalance, **la accuracy sola es engañosa**. Un clasificador que
prediga siempre la clase mayoritaria obtiene:

| Métrica | Clasificador degenerado |
|---|---:|
| accuracy | **0.598** |
| balanced accuracy | 0.500 |
| macro-F1 | 0.374 |
| F1 de la clase mutada | 0.000 |
| kappa | 0.000 |

Ésa es la vara: cualquier accuracy que no supere claramente 0.598 no dice nada.
Por eso se reportan siete métricas en tres bloques:

| Bloque | Métricas | Qué mide |
|---|---|---|
| **A — ordenación** | `auc`, `ap` | Capacidad de ordenar, independiente del umbral. `ap` (precisión media / PR-AUC) es la que más pesa bajo desbalance porque ignora los verdaderos negativos |
| **B — decisión a 0.5** | `acc`, `bacc`, `macro_f1`, `f1_pos`, `kappa` | Lo que hace `ensemble4.py`: argmax del softmax |
| **C — decisión con umbral ajustado** | las mismas, sufijo `_t` | Umbral elegido **en validación** maximizando balanced accuracy |

El umbral del bloque C se elige en `val`, nunca en `test`. `val` ya se usa para
el early stopping del ABMIL y del metaclasificador, así que no introduce ninguna
fuga nueva; ajustarlo en `test` destruiría la única superficie de reporte
honesta.

---

## 4. Resultados

### 4.1 Bloque A — capacidad de ordenación

| Configuración | AUC | AP (PR-AUC) |
|---|---:|---:|
| `prob:mlp_snapshot` | **0.8018** | **0.7556** |
| **`prob:logreg`** (referencia) | 0.7986 | 0.7494 |
| `emb:mlp_snapshot` | 0.7988 | 0.7448 |
| `prob:svm` | 0.7965 | 0.7476 |
| `prob:nb` | 0.7955 | 0.7378 |
| `emb:mlp_fge` | 0.7937 | 0.7457 |
| `emb:mlp` | 0.7848 | 0.7370 |
| `emb:logreg` | 0.7826 | 0.7289 |
| `prob:knn` | 0.7665 | 0.6711 |
| `prob:mlp` | 0.6948 | 0.6591 |

### 4.2 Bloque B — decisión con umbral 0.5 (el criterio actual)

| Configuración | acc | bacc | macro-F1 | F1 mutado | kappa |
|---|---:|---:|---:|---:|---:|
| `emb:mlp_snapshot` | **0.7605** | 0.7429 | **0.7434** | 0.6816 | **0.4911** |
| `prob:nb` | 0.7486 | **0.7421** | 0.7381 | **0.6918** | 0.4800 |
| **`prob:logreg`** (referencia) | 0.7446 | 0.7302 | 0.7294 | 0.6721 | 0.4629 |
| `prob:svm` | 0.7417 | 0.7287 | 0.7273 | 0.6712 | 0.4584 |
| `emb:logreg` | 0.7277 | 0.7179 | 0.7146 | 0.6594 | 0.4332 |
| `prob:knn` | 0.7223 | 0.7144 | 0.7096 | 0.6565 | 0.4245 |
| `emb:mlp_fge` | 0.7066 | 0.6577 | 0.6387 | 0.4924 | 0.3284 |
| `prob:mlp_snapshot` | 0.6803 | 0.6223 | 0.5827 | 0.3955 | 0.2536 |
| `emb:mlp` | 0.6720 | 0.6053 | 0.5490 | 0.3349 | 0.2187 |
| `prob:mlp` | 0.5405 | 0.5172 | 0.3836 | 0.2605 | 0.0340 |

`prob:mlp` queda **por debajo del clasificador degenerado en accuracy** (0.5405
frente a 0.598) y con kappa 0.0340, es decir, prácticamente sin acuerdo por
encima del azar.

### 4.3 Bloque C — decisión con umbral ajustado en validación

| Configuración | acc_t | bacc_t | macro-F1_t | F1 mutado_t | kappa_t |
|---|---:|---:|---:|---:|---:|
| `emb:mlp_snapshot` | **0.7494** | **0.7438** | **0.7364** | **0.6941** | **0.4826** |
| `emb:mlp_fge` | 0.7367 | 0.7292 | 0.7206 | 0.6726 | 0.4548 |
| `prob:nb` | 0.7268 | 0.7206 | 0.7117 | 0.6616 | 0.4351 |
| **`prob:logreg`** (referencia) | 0.7261 | 0.7198 | 0.7105 | 0.6627 | 0.4344 |
| `prob:mlp_snapshot` | 0.7241 | 0.7148 | 0.7067 | 0.6531 | 0.4278 |
| `prob:svm` | 0.7205 | 0.7144 | 0.7064 | 0.6572 | 0.4237 |
| `prob:knn` | 0.7148 | 0.7048 | 0.6967 | 0.6401 | 0.4075 |
| `emb:mlp` | 0.7138 | 0.7108 | 0.6999 | 0.6559 | 0.4148 |
| `emb:logreg` | 0.6914 | 0.6873 | 0.6697 | 0.6206 | 0.3688 |
| `prob:mlp` | 0.6638 | 0.6655 | 0.6314 | 0.5934 | 0.3282 |

Umbral medio elegido en validación: entre **0.411** y **0.518** según el brazo,
con mediana en torno a 0.45–0.50. Es decir, el óptimo está sistemáticamente
**por debajo de 0.5**, como cabe esperar con un 40 % de positivos.

---

## 5. El hallazgo transversal: fallo de corte ≠ fallo de ordenación

Comparar los bloques B y C separa dos cosas que a 0.5 quedan confundidas.

| Configuración | kappa @ 0.5 | kappa @ umbral ajustado | recuperación |
|---|---:|---:|---:|
| `prob:mlp` | 0.0340 | 0.3282 | **+0.294** |
| `prob:mlp_snapshot` | 0.2536 | 0.4278 | **+0.174** |
| `emb:mlp` | 0.2187 | 0.4148 | **+0.196** |
| `emb:mlp_fge` | 0.3284 | 0.4548 | **+0.126** |
| `prob:logreg` | 0.4629 | 0.4344 | −0.029 |
| `prob:svm` | 0.4584 | 0.4237 | −0.035 |
| `prob:nb` | 0.4800 | 0.4351 | −0.045 |
| `emb:mlp_snapshot` | 0.4911 | 0.4826 | −0.009 |

**Los cuatro brazos que se hundían son exactamente los cuatro que se recuperan.**
En términos pareados, el desplome de `prob:mlp_snapshot` frente a LogReg pasa de
**Δ kappa = −0.2093** (IC [−0.2911, −0.1318], p Holm = 0.0002, significativo) a
**Δ kappa_t = −0.0066** (IC [−0.0451, +0.0335], p Holm = 1.0, nulo). Lo mismo en
las cuatro métricas de decisión.

La lectura correcta es que **un MLP entrenado con 75 filas sobre clases al 40/60
no aprende a poner su frontera en 0.5**, aunque ordene bien. Su AUC (0.8018, el
más alto del barrido) siempre lo dijo; el argmax lo escondía.

**El efecto simétrico también importa:** para los brazos que ya funcionaban
(LogReg, SVM, NB), ajustar el umbral **empeora** el resultado en 0.03–0.045 de
kappa. Con ~15 slides de validación por fold, el umbral estimado es ruidoso, y
0.5 resulta ser una elección más robusta que optimizarlo. **No se recomienda
adoptar el ajuste de umbral en el pipeline**: sólo sirve como diagnóstico para
saber si un brazo falla al ordenar o al cortar.

---

## 6. Contraste estadístico

### 6.1 Protocolo

Todas las configuraciones se evalúan sobre los mismos 50 folds, así que las
comparaciones son **pareadas**. Por métrica se reporta: Δ medio pareado, IC 95 %
por bootstrap (10 000 remuestreos), ganadas/empatadas/perdidas con los empates
contados de forma explícita, p de Wilcoxon, **p corregida por Holm** sobre la
familia de métricas, y el **efecto mínimo detectable (MDE)** al 80 % de potencia,
para que un resultado no significativo se lea como *acotado* y no como
*indeterminado*.

### 6.2 Tareas 1-6 — stacking sobre probabilidades

Δ frente a `prob:logreg`, métricas de ordenación y kappa a 0.5:

| Brazo | Δ AUC | IC 95 % | Δ kappa | p Holm (kappa) | Veredicto |
|---|---:|---|---:|---:|---|
| **SVM** | −0.0021 | [−0.0067, +0.0024] | −0.0045 | 1.00 | ⚖️ **Empate exacto** |
| **NB** | −0.0031 | [−0.0093, +0.0028] | +0.0170 | 0.88 | ⚖️ Empate, sesgo favorable |
| **MLP Snapshot** | +0.0031 | [−0.0020, +0.0084] | −0.2093 | **0.0002** | ❌ Se hunde a 0.5 |
| **KNN** | −0.0321 | [−0.0467, −0.0183] | −0.0384 | 0.38 | ❌ **Peor** (AUC y AP significativos) |
| **MLP** | −0.1038 | [−0.1485, −0.0635] | −0.4280 | **< 0.0001** | ❌ **Colapsa** |

Lectura brazo a brazo:

- **SVM es literalmente el mismo modelo.** Las 12 métricas empatan, con
  **40 de 50 folds decidiendo idénticamente**. Un SVM lineal con C = 1 sobre 6
  columnas de probabilidad es una frontera lineal en el mismo espacio que LogReg;
  que coincidan no es sorprendente, es la confirmación de que no hay estructura
  no lineal que explotar.
- **NB es un empate con sesgo favorable en las métricas del desbalance:** bacc
  +0.0119, F1 de la clase mutada **+0.0197** (p = 0.0115 sin corregir, 0.14 tras
  Holm), kappa +0.0170. Pero su AUC y su AP son **peores** (−0.0031, −0.0117), y
  con el umbral ajustado la ventaja desaparece por completo (Δ bacc_t = +0.0007).
  **La pequeña ventaja de NB es enteramente un efecto de umbral**, no una mejor
  ordenación: sus probabilidades caen mejor situadas respecto al 0.5, nada más.
- **KNN es el único claramente peor en ordenación:** Δ AUC −0.0321 y Δ AP
  −0.0784, ambos significativos tras Holm. Con 75 filas en 6 dimensiones, 5
  vecinos son demasiado pocos para estimar la densidad.
- **MLP colapsa de verdad**, no sólo en el corte: su AUC cae −0.1038 (43 de 50
  folds perdidos). El ajuste de umbral lo recupera parcialmente (kappa_t 0.3282)
  pero sigue significativamente por debajo de LogReg.

### 6.3 Tarea 7 (§8.1) — embeddings vs. probabilidades

Δ frente a `prob:logreg`:

| Brazo | Δ AUC | IC 95 % | Δ kappa @ 0.5 | Δ kappa_t | p Holm mínima |
|---|---:|---|---:|---:|---:|
| `emb:logreg` | −0.0160 | [−0.0305, −0.0020] | −0.0297 | −0.0656 | 0.56 |
| `emb:mlp` | −0.0138 | [−0.0294, +0.0005] | −0.2442 | −0.0196 | < 0.0001 |
| **`emb:mlp_snapshot`** | **+0.0002** | [−0.0077, +0.0084] | **+0.0281** | **+0.0482** | 0.58 |
| `emb:mlp_fge` | −0.0049 | [−0.0139, +0.0043] | −0.1346 | +0.0204 | 0.0022 |

**La comparación abierta más importante del proyecto se cierra en negativo.**
El espacio de 1536 D no bate al de 6 columnas:

1. **A igualdad de metaclasificador, las probabilidades ganan.** `emb:logreg`
   pierde −0.0160 de AUC y −0.0205 de AP frente a `prob:logreg`, con 29 y 30
   folds perdidos de 50. Es la comparación limpia — mismo combinador, sólo cambia
   la entrada — y va en contra de los embeddings.
2. **El orden interno de la tabla obsoleta del 18-ago sí replica.** Entonces:
   MLP Snapshot 0.7995 > MLP FGE 0.7947 > MLP 0.7901 ≈ LogReg 0.7896. Ahora:
   0.7988 > 0.7937 > 0.7848 > 0.7826. La conclusión *relativa* dentro del espacio
   de embeddings se sostiene con los checkpoints actuales, con una fidelidad
   notable en los valores absolutos.
3. **Pero el nivel entero está en o por debajo de `prob:logreg`.** Lo que en su
   día pareció una mejora era el mejor punto de un espacio peor.

**El matiz honesto:** `emb:mlp_snapshot` es el **único brazo del barrido que
apunta hacia arriba en las cinco métricas de decisión**, tanto a 0.5 (kappa
+0.0281, acc +0.0160 con IC que excluye el cero) como con umbral ajustado (kappa_t
+0.0482, bacc_t +0.0240). **Ninguno supera Holm** (p entre 0.58 y 0.88) y el AUC
es plano (+0.0002), así que no es un resultado, es una dirección. Si algo de este
barrido merece una réplica, es esto — pero el MDE de kappa está en 0.046, muy
cerca del efecto medido, así que 50 folds no bastan para resolverlo.

---

## 7. Suelo de reproducibilidad — medido otra vez, sin querer

La suite se ejecutó dos veces: primero en GPU (job 70291) y después en CPU al
añadir el análisis de umbral. Los brazos deterministas (`logreg`, `svm`, `knn`,
`nb`) dieron resultados **idénticos**. Los que usan PyTorch se movieron:

| Brazo | GPU | CPU | Δ |
|---|---:|---:|---:|
| `prob:mlp` | 0.6884 | 0.6948 | +0.0064 |
| `prob:mlp_snapshot` | 0.8011 | 0.8018 | +0.0007 |
| `emb:mlp_snapshot` | 0.7957 | 0.7988 | +0.0031 |
| `emb:mlp_fge` | 0.7946 | 0.7937 | −0.0009 |

Es una confirmación independiente del suelo de ≈ 0.004 de AUC que ya documenta
el estado del arte, y tiene una consecuencia directa para leer este informe:
**todos los Δ positivos medidos aquí (+0.0031 de `prob:mlp_snapshot`, +0.0002 de
`emb:mlp_snapshot`) están por debajo del ruido de volver a lanzar el mismo job.**

Los contrastes *dentro* de una pasada siguen siendo válidos, porque todos los
brazos comparten proceso y dispositivo. Lo que no es válido es tratar +0.003 como
una mejora.

---

## 8. Snapshot ensemble a nivel base (job 70292)

Distinto del `mlp_snapshot` de las secciones anteriores: allí los snapshots son
del MLP del stacking; aquí lo son **de los propios ABMIL**. Se generaron desde
cero con tags nuevos (`fge_trio`, `se_trio`) porque los snapshots que había en
disco eran del 14-15 ago, anteriores a los checkpoints base actuales, y
`conch_v1_5` no tenía ninguno.

Hiperparámetros: los validados en `experiments.yaml`.
**FGE** (Garipov et al., ICLR 2018) `lr_1 = 2e-4, lr_2 = 2e-5, cycle_length = 4`;
**SE** (Huang et al., ICLR 2017) `lr_1 = 1e-3, lr_2 = 1e-6, cycle_length = 15`.
Duración total: **2 h 13 min**, 300 corridas de fold (3 modelos × 2 esquemas × 50
folds).

### 8.1 Resultados

| Configuración | AUC | AP | bacc | macro-F1 | kappa |
|---|---:|---:|---:|---:|---:|
| **`prob:logreg`** (referencia) | 0.7986 | 0.7494 | 0.7302 | 0.7294 | 0.4629 |
| `prob:mlp_snapshot` | **0.8011** | 0.7554 | 0.6229 | 0.5809 | 0.2551 |
| **`fge_trio:logreg`** | 0.8006 | **0.7575** | **0.7422** | **0.7403** | **0.4845** |
| `fge_trio:mlp_snapshot` | 0.8007 | 0.7561 | 0.6970 | 0.6865 | 0.4037 |
| `se_trio:logreg` | 0.8000 | 0.7511 | 0.7262 | 0.7250 | 0.4528 |
| `se_trio:mlp_snapshot` | 0.7986 | 0.7550 | 0.7024 | 0.6950 | 0.4120 |

`fge_trio:logreg` es **numéricamente la mejor configuración de toda la campaña**
en cuatro de las cinco métricas de la tabla, incluido el mejor AP jamás medido
aquí (0.7575). Los contrastes pareados dicen que no significa nada:

| Brazo | Δ AUC | IC 95 % | G/E/P | Δ kappa | p Holm | MDE (AUC) |
|---|---:|---|---|---:|---:|---:|
| `fge_trio:logreg` | +0.0020 | [−0.0161, +0.0196] | **22/2/26** | +0.0215 | 1.00 | 0.0261 |
| `se_trio:logreg` | +0.0013 | [−0.0166, +0.0194] | 24/4/22 | −0.0102 | 1.00 | 0.0259 |
| `fge_trio:mlp_snapshot` | +0.0021 | [−0.0163, +0.0207] | 24/2/24 | −0.0592 | 0.34 | 0.0266 |
| `se_trio:mlp_snapshot` | −0.0001 | [−0.0200, +0.0192] | 24/4/22 | −0.0510 | 0.41 | 0.0283 |

**Ninguna p corregida por Holm baja de 0.2 en ninguna métrica.**

### 8.2 Por qué el +0.0020 de FGE no es una mejora

Tres razones, en orden de peso:

1. **Pierde en más folds de los que gana.** `fge_trio:logreg` tiene media
   positiva (+0.0020) y **G/E/P de 22/2/26** en AUC: la mediana va en contra del
   signo de la media. Es la firma de unos pocos folds con desviaciones grandes,
   no la de un efecto sistemático. Compárese con el cambio de modelo base, que
   dio 34/4/12.
2. **Está muy por debajo del suelo de reproducibilidad** (≈ 0.004, §7). Volver a
   lanzar el mismo job mueve el AUC más que esto.
3. **El nulo aquí es más débil que el de la ablación previa.** El MDE es de
   **0.026 de AUC**, unas **4 × el de la suite de metaclasificadores** (0.0065
   para SVM, 0.0074 para MLP Snapshot). Las predicciones de los snapshots son más
   ruidosas fold a fold, así que esta comparación sólo puede excluir mejoras por
   encima de ≈ 0.026, frente al ≈ 0.017 que acotaba el estado del arte.

Dicho de forma honesta: **este experimento no descarta una mejora pequeña de
FGE; descarta una grande.** Pero tampoco aporta ninguna evidencia a favor, y su
mejor lectura —cuatro métricas arriba— se desmonta al mirar el reparto por folds.

### 8.3 Dos observaciones secundarias

**La diversidad de snapshots satura, otra vez.** De las 300 corridas de fold, la
distribución de snapshots aceptados fue:

| Snapshots/fold | 2 | 3 | 4 | 5 | 6 |
|---|---:|---:|---:|---:|---:|
| Corridas | **174** | 66 | 38 | 11 | 11 |

Con `cycle_patience = 2`, el 58 % de los folds se corta en dos snapshots. Es el
mismo mecanismo que ya documentaba la ablación previa: los snapshots de un mismo
modelo correlacionan demasiado como para que añadir ciclos aporte, y el corte por
paciencia lo detecta.

**El patrón corte-vs-ordenación se repite sobre bases de snapshot.** Los dos
brazos `mlp_snapshot` sobre bases FGE/SE pierden en decisión a 0.5 (Δ kappa
−0.059 y −0.051) y **recuperan por completo con umbral ajustado** (+0.0019 y
−0.0029). Es exactamente §5 otra vez, ahora con las bases cambiadas: confirma que
el fenómeno es del metaclasificador MLP, no de las features que recibe.

Nota lateral: `se_trio:logreg` elige un umbral de validación notablemente más
bajo (**0.364** de media, frente a 0.442 del baseline). Los reinicios agresivos
de SE desplazan la calibración de las probabilidades base, aunque su ordenación
quede igual.

---

## 9. Conclusiones

1. **LogReg sigue sin ser batido.** De nueve alternativas, ninguna consigue un Δ
   positivo significativo tras Holm en ninguna métrica. Dos empatan (SVM, NB),
   una empata con dirección favorable en decisión (`emb:mlp_snapshot`), y seis
   son peores, cuatro de ellas de forma significativa.
2. **Con 75 filas de entrenamiento y 6 columnas de features, la flexibilidad no
   se paga.** El orden del barrido lo dice sin ambigüedad: los dos modelos
   lineales (LogReg, SVM) empatan en cabeza, el generativo simple (NB) empata, y
   los flexibles (MLP, KNN) pierden. Es el mismo patrón que el estado del arte ya
   documentaba para la atención: **el cuello de botella es el tamaño del
   conjunto, no la capacidad del modelo.**
3. **El desbalance se manifiesta como un problema de umbral, y hay que
   diagnosticarlo con métricas de los dos bloques.** Un brazo con AUC 0.8018 y
   kappa 0.2536 no es un mal clasificador: es un buen ordenador con el corte mal
   puesto. Reportar sólo AUC lo habría declarado ganador; reportar sólo kappa lo
   habría declarado inútil. Ambas lecturas serían falsas.
4. **El snapshot ensemble a nivel base tampoco aporta** (§8). Ni FGE ni SE
   producen un cambio significativo sobre el trío ganador — es la primera vez
   que se mide sobre este trío, y replica el nulo que la ablación previa había
   encontrado sobre el histórico. La configuración numéricamente mejor de toda
   la campaña (`fge_trio:logreg`, AUC 0.8006 y el mejor AP medido) pierde en 26
   de 50 folds: media positiva, mediana en contra.
5. **§8.1 queda cerrada en negativo.** Los embeddings de 1536 D no baten a las 6
   probabilidades. Queda una dirección abierta y acotada — `emb:mlp_snapshot` en
   métricas de decisión — que no alcanza significación y cuyo MDE (0.046 en kappa)
   dice que 50 folds no la resolverán.

### 9.1 Recomendación operativa

**No cambiar nada.** La configuración recomendada sigue siendo:

```bash
python src/ensemble4.py \
    --foundational_models ctranspath uni_v2 conch_v1_5 \
    --work_dir /home/JKP6679/Patho-Ensemble/PARADIS/datos/patches \
    --train_source cptac_brca \
    --tissue_patching 20x_224px_0px_overlap \
    --task_name TP53_mutation \
    --meta_model logreg
```

En particular, **no adoptar el ajuste de umbral**: con ~15 slides de validación
por fold empeora los brazos que ya funcionaban. Es una herramienta de
diagnóstico, no una mejora del pipeline.

Este barrido refuerza la conclusión central del proyecto: cambiar **qué modelos
base entran** valió +0.037 de AUC; cambiar **cómo se combinan** no ha valido nada
medible en nueve intentos más.

---

## Anexo A — Reproducción

```bash
# Pasada completa (regenera embeddings + los 10 brazos + estadística)
sbatch run_meta_suite_winning_trio.sbatch

# Sólo el análisis, con los outputs ya en disco
python src/run_meta_suite.py \
    --foundational_models ctranspath uni_v2 conch_v1_5 \
    --n_folds 50 --baseline prob:logreg \
    --configs prob:logreg prob:mlp prob:mlp_snapshot prob:svm prob:knn prob:nb \
              emb:logreg emb:mlp emb:mlp_snapshot emb:mlp_fge \
    --out results_meta_suite_cptac_brca_TP53_mutation.json
```

| Fichero | Contenido |
|---|---|
| `src/run_meta_suite.py` | El runner: carga, verificación de alineamiento, 10 brazos, 3 bloques de métricas, estadística pareada |
| `run_meta_suite_winning_trio.sbatch` | Job completo, incluida la regeneración de embeddings y su verificación de vigencia |
| `results_meta_suite_cptac_brca_TP53_mutation.json` | Métricas por fold, agregados, umbrales elegidos y contrastes pareados |
| `logs/meta_suite_trio_70291.out` | Salida del job (pasada GPU) |
| `logs/meta_suite_rerun.log` | Salida de la pasada con análisis de umbral (CPU, la reportada) |

## Anexo B — El barrido completo

Catorce configuraciones medidas en total, en tres pasadas, cada una con su
baseline recalculado dentro de la propia pasada:

| Pasada | Job | Brazos | Resultado |
|---|---|---:|---|
| Suite de metaclasificadores | 70291 | 10 | Ninguno bate a LogReg |
| Snapshot ensemble a nivel base | 70292 | 4 (+2 de control) | Nulo, acotado a ≈ 0.026 AUC |

| Fichero adicional | Contenido |
|---|---|
| `run_snapshot_base_trio.sbatch` | Job de snapshots base: FGE + SE sobre los tres ABMIL, tags `fge_trio` / `se_trio` |
| `results_snapshot_base_cptac_brca_TP53_mutation.json` | Métricas por fold y contrastes pareados de §8 |
| `logs/snapshot_base_trio_70292.out` | Salida del job (2 h 13 min) |
