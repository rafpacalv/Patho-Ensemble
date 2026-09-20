# Estado del arte — Patho-Ensemble

**Fecha:** 22 de agosto de 2026
**Tarea evaluada:** `cptac_brca / TP53_mutation`
**Alcance:** todo lo medido hasta el cierre de los jobs 70271 y 70285 (21-ago 23:50)

---

## Resumen ejecutivo

| | |
|---|---|
| **Mejor configuración medida** | `ctranspath + uni_v2 + conch_v1_5` → ABMIL independiente por modelo → stacking con LogReg |
| **AUC (macro-OvR, 50 folds)** | **0.7986** |
| **bacc / macro-F1 / kappa ponderada** | 0.7302 / 0.7294 / 0.4629 |
| **Estimación honesta sobre los 8 modelos** | **0.8011** (selección leave-one-fold-out entre 255 subconjuntos) |
| **Baseline histórico que sustituye** | `ctranspath + uni_v2 + virchow_v1` = 0.7616 |

**La conclusión que ordena todo lo demás:** el rendimiento lo determina **qué modelos base entran**, no **cómo se combinan**.

- Cambiar un modelo base vale **+0.0395 AUC** (p = 2 · 10⁻⁵).
- El mejor resultado jamás medido de tocar la regla de combinación vale **+0.0052** (p = 0.058, no significativo).

Son casi **8×** de diferencia. Cualquier trabajo futuro sobre metaclasificadores debe justificar por qué no se está gastando ese esfuerzo en selección de modelos base.

---

## 1. Datos y protocolo experimental

### 1.1 Conjunto

| | |
|---|---|
| Dataset | `cptac_brca` (CPTAC breast carcinoma) |
| Tarea | `TP53_mutation`, clasificación binaria a nivel de slide |
| Slides | 112 |
| Pacientes (`case_id`) | 103 |
| Distribución de clases | 67 wild-type / 45 mutado (40.2 % positivos) |
| Folds | 50 |
| Patching | `20x_224px_0px_overlap` |

### 1.2 Particiones

Cada columna `fold_*` del `k=all.tsv` asigna cada slide a `train`, `val` o `test`. En `fold_0`:

| Split | n | Uso |
|---|---:|---|
| `train` | 75 | entrena el ABMIL base; **y** genera las meta-features del metaclasificador |
| `val` | 15 | early stopping del ABMIL y del metaclasificador |
| `test` | 22 | única superficie de reporte |

El tamaño de `test` varía entre 15 y 24 según el fold. **La estratificación es a nivel de paciente**: todas las slides de un `case_id` caen en el mismo split, vía `StratifiedGroupKFold`.

### 1.3 Arquitectura base (ABMIL)

Fija en `abmil_engine.py`, idéntica para los 8 modelos:

| Parámetro | Valor |
|---|---|
| `in_dim` | autodetectado del `.h5` (768 / 1024 / 1280 / 2560 según modelo) |
| Cabezas de atención | 1 |
| `head_dim` | 512 |
| Dropout | 0.25 |
| Gated attention | No |
| Bag size | 2048 parches |
| Épocas | 100, con early stopping sobre `val` |

Que la arquitectura sea idéntica es lo que hace comparable el barrido de modelos base: **la única variable es el extractor de características**.

---

## 2. Modelos base evaluados

Se han entrenado y evaluado **8 modelos fundacionales**, los 8 sobre las mismas 112 slides y los mismos 50 folds. AUC de test del modelo **en solitario** (ABMIL + su propia predicción, sin ensemble):

| # | Modelo | AUC test | AUC val | Δ vs. el mejor |
|---:|---|---:|---:|---:|
| 1 | **conch_v1_5** | **0.7902** | 0.8536 | — |
| 2 | virchow2 | 0.7771 | 0.8245 | −0.0131 |
| 3 | uni_v1 | 0.7732 | 0.8011 | −0.0170 |
| 4 | phikon_v2 | 0.7685 | 0.8093 | −0.0217 |
| 5 | uni_v2 | 0.7634 | 0.8369 | −0.0268 |
| 6 | ctranspath | 0.7571 | 0.8117 | −0.0331 |
| 7 | hoptimus1 | 0.7450 | 0.7998 | −0.0452 |
| 8 | virchow_v1 | 0.6823 | 0.7319 | −0.1079 |

Tres lecturas:

1. **`conch_v1_5` en solitario (0.7902) supera al ensemble del trío histórico (0.7616).** Un único modelo bien elegido bate a tres modelos mal elegidos combinados con un metaclasificador.
2. **`virchow_v1` es prescindible, no simplemente peor.** Con 0.6823 está a 11 puntos del mejor y arrastra el ensemble.
3. La correlación entre AUC de val y de test es alta pero no perfecta, lo que justifica la selección LOFO (§6.3) en lugar de elegir por val a ojo.

### 2.1 Grupos de coordenadas de parche

Verificado sobre las 112 slides. Los modelos de un mismo grupo tienen `coords` byte-idénticos y **pueden fusionarse a nivel de parche**; los de grupos distintos no, sin registro espacial previo.

| Grupo | Parches/slide | Modelos |
|---:|---:|---|
| 1 | 4068 | hoptimus1, phikon_v2, uni_v1, **uni_v2**, virchow2 |
| 2 | 2100 | **conch_v1_5**, **ctranspath**, virchow_v1 |
| 3 | 3148 | uni_v1_256px, uni_v2_256px |
| — | 857 / 734 | conch512, conch512_monai (cada uno solo) |

Esto tiene una consecuencia directa: el trío ganador (`ctranspath` + `uni_v2` + `conch_v1_5`) **cruza los grupos 1 y 2**, así que la fusión temprana solo puede aplicarse a dos de sus tres miembros. Es la misma limitación que tenía el trío histórico.

---

## 3. Fusión: temprana vs. tardía

### 3.1 Qué es cada una

**Fusión tardía (la arquitectura actual).** Cada modelo fundacional entrena su propio ABMIL, que colapsa sus parches a una predicción de slide. El metaclasificador ve solo esas predicciones finales.

```
ctranspath  → ABMIL → p₁ ┐
uni_v2      → ABMIL → p₂ ├→ LogReg → predicción
conch_v1_5  → ABMIL → p₃ ┘
```

**Fusión temprana (`ensemble5.py`).** Los embeddings de parche se concatenan *antes* del ABMIL, de modo que la atención puede ponderar el acuerdo/desacuerdo entre modelos parche a parche.

```
[ctranspath ⊕ virchow_v1] (concat por parche, in_dim = 768+2560 = 3328)
    → un solo ABMIL → predicción
```

La hipótesis era que la fusión tardía pierde información al colapsar cada modelo por separado, y que la temprana la recuperaría.

### 3.2 Resultado: la hipótesis es falsa

| Configuración | Δ AUC | IC 95 % | G/E/P | p | Veredicto |
|---|---:|---|---|---:|---|
| Fusión temprana (3 modelos) vs. tardía | **−0.0056** | [−0.0250, +0.0124] | 21/3/26 | 0.55 | Sin efecto |
| Fusión temprana ponderada vs. Nelder-Mead | **−0.0182** | [−0.0421, +0.0061] | 20/1/29 | 0.15 | Empeora |

Concatenar a nivel de parche **no recupera** la información que se pierde al colapsar cada modelo. Además, el vector fusionado (3328–4096 D) desborda un mecanismo de atención diseñado para entradas de 768–2560 D. Escalar los embeddings antes de concatenar (fusión temprana ponderada) lo empeora todavía más.

**La estructura de fusión tardía del pipeline es aproximadamente óptima** para los modelos y la arquitectura actuales.

> ⚠️ Existe un `results_early_fusion_cptac_brca_TP53_mutation.json` con Δ = **+0.0319** favorable a la fusión temprana. Es de una comparación **no pareada con distinto conjunto de modelos** y está superado por la comparación justa de arriba (`results_opcion_c_fair_comparison.json`, misma pasada, mismos folds). No usar ese número.

---

## 4. El metaclasificador

### 4.1 Qué recibe como entrada — la pregunta clave

**En el pipeline vigente, el metaclasificador recibe la salida del ABMIL, no los embeddings.**

Concretamente, para cada slide y cada modelo base se carga `preds.npy`, que contiene el vector **softmax de 2 clases**. Se concatenan a lo ancho:

```
X_meta = [p₁(clase0), p₁(clase1), p₂(clase0), p₂(clase1), p₃(clase0), p₃(clase1)]
       → [N, 2 × n_modelos]     # con 3 modelos: [N, 6]
```

Confirmado en `src/ensemble4.py` (construcción de `X_train` en las líneas 500–525) y en los logs de ejecución, donde las meta-features de entrenamiento salen como `(75, 2)` para un modelo fusionado y `(N, 6)` para el trío.

No es una salida binaria dura: son **probabilidades continuas**, y esa distinción importa — el metaclasificador puede leer la confianza, no solo el voto. Opciones disponibles sobre ese espacio:

| Opción | Efecto |
|---|---|
| `--feature_space prob` (por defecto) | Combina en espacio de probabilidad |
| `--feature_space logit` | Combina en log-odds |
| `--drop_redundant_class` | Elimina una columna softmax por modelo (cada par suma 1, así que es redundante: en binario, 6 columnas tienen rango 3) |
| `--extra_features` | Añade columnas derivadas: desacuerdo (Jensen-gap), entropía, margen, estadísticos de atención |
| `--reweight_by_model_importance` | Reescala cada bloque de columnas por la importancia del modelo (4 métodos) |

### 4.2 La variante con embeddings — sí se probó

Existe una rama alternativa que alimenta al metaclasificador con el **embedding de slide de 512 D posterior a la atención** (la representación que el ABMIL produce justo antes de su clasificador), en lugar de las 2 probabilidades:

```
X_meta = [emb₁(512), emb₂(512), emb₃(512)] → [N, 1536]
```

Resultados medidos el 18-ago sobre 50 folds:

| Meta-modelo | AUC | Accuracy | F1 | Kappa |
|---|---:|---:|---:|---:|
| LogReg | 0.7896 ± 0.0907 | 0.7377 | 0.7366 | 0.4531 |
| MLP | 0.7901 ± 0.1019 | 0.7052 | 0.6589 | 0.3223 |
| **MLP Snapshot** | **0.7995 ± 0.1027** | 0.7512 | 0.7498 | 0.4812 |
| MLP FGE | 0.7947 ± 0.1011 | 0.7319 | 0.7142 | 0.4095 |

> 🔴 **Estos números están obsoletos.** Se calcularon el 18-ago 19:03, y los checkpoints de los tres modelos implicados cambiaron después (ctranspath 20-ago 12:52, uni_v2 19-ago 21:37, virchow_v1 19-ago 21:48). Los embeddings de los que salen ya no existen en disco. Ver §6.5.

Aun así, la comparación **interna** entre esas cuatro filas sigue siendo válida (mismo momento, mismos embeddings), y dice algo útil: en el espacio de 1536 D el MLP simple se degrada (kappa 0.32) y solo se recupera con regularización por snapshots. Es el mismo patrón que la fuga in-sample produce en el espacio de probabilidades (§6.2).

**Esta es la comparación abierta más importante del proyecto** (§8.1): embeddings 1536 D vs. probabilidades 6 D, con los checkpoints actuales y el trío ganador. Nunca se han medido juntos.

### 4.3 Metaclasificadores implementados

15 registrados en `src/meta_models.py`, seleccionables con `--meta_model`:

| Familia | Variantes |
|---|---|
| Lineales | `logreg`, `logit_avg` |
| MLP | `mlp`, `deep_mlp`, `mlp_snapshot`, `deep_mlp_snapshot`, `mlp_fge`, `mlp_deepens`, `gating` |
| Bayesianos / de instancias | `nb`, `knn` |
| Márgenes | `svm` (lineal y RBF) |
| Árboles | `lightgbm` |
| Transformer tabular | `tabpfn`, `tabpfn_snapshot` |

**`logreg` sigue siendo el que se usa por defecto y el que no ha sido batido de forma significativa.** Con 75 filas de entrenamiento y 6 columnas hay poco margen para cualquier cosa más flexible, y encima el espacio es casi separable por la fuga in-sample (§6.2), lo que da a un meta-modelo flexible aún más ocasiones de ajustar ruido.

---

## 5. Métricas

`utils.Metrics` calcula, por fold y agregadas:

| Métrica | Papel |
|---|---|
| **`macro-ovr-auc`** | Métrica primaria de reporte |
| `macro-ovo-auc` | Variante one-vs-one |
| **`bacc`** (balanced accuracy) | Corrige el desbalance 67/45 |
| **`macro-f1`** | Media no ponderada por clase |
| **`weighted_kappa`** | Acuerdo corregido por azar; la más sensible del conjunto |
| Log-loss | Calibración (calculada, no usada para decidir) |

Salidas: métricas por fold en `{split}_metrics/fold_X/`, agregadas con **IC 95 % por bootstrap (1000 iteraciones)**, curvas ROC, matrices de confusión y curvas precisión-recall.

En la comparación principal (§7) las cuatro métricas principales se mueven en el mismo sentido y todas superan su efecto mínimo detectable, lo cual es la señal de que el efecto es real y no un artefacto de una métrica concreta.

---

## 6. Cómo se evalúa el rendimiento

Esta sección es la que decide si los números de arriba significan algo.

### 6.1 Contra qué conjunto se reporta

**Siempre contra `test`.** `val` se usa exclusivamente para early stopping y para la selección de subconjuntos; nunca se reporta como resultado.

### 6.2 La brecha entre entrenamiento y test — de quién es

El metaclasificador se entrena con las predicciones de los modelos base **sobre el propio split de entrenamiento de ese fold**, es decir, sobre slides que los ABMIL ya han visto. Eso produce una brecha grande entre el AUC de entrenamiento y el de test. Medido sobre el trío ganador, 50 folds (rango sobre los tres modelos):

| Superficie | Papel | AUC base |
|---|---|---:|
| `_train_eval/val_outputs` | meta-entrenamiento (**in-sample**) | **0.849 – 0.949** |
| `val_outputs` | early stopping | 0.812 – 0.854 |
| `test_outputs` | reporte | 0.757 – 0.790 |
| `_train_eval_oof/val_outputs_oof` | meta-entrenamiento (**out-of-fold**) | 0.722 – 0.730 |

#### La brecha es del ABMIL, no del metaclasificador

Descompuesta, la respuesta es inequívoca:

| | meta-train | test | brecha |
|---|---:|---:|---:|
| Metaclasificador (LogReg) | 0.9495 | 0.7986 | **+0.1509** |
| `conch_v1_5` solo, su mejor entrada | 0.9493 | 0.7902 | **+0.1591** |
| **aporte del metaclasificador** | **+0.0003** | **+0.0084** | |

Dos lecturas, ambas en contra de culpar al metaclasificador:

1. **No crea la brecha: la hereda.** Su 0.9495 sobre sus propias filas es prácticamente idéntico al 0.9493 que `conch_v1_5` ya traía — aporta **+0.0003**. Una regresión logística con 6 columnas y 75 filas apenas puede sobreajustar; lo que hace es reproducir su mejor entrada.
2. **No falla en test: mejora ahí.** Sobre test aporta **+0.0084** y gana al mejor modelo base fijo en **27 de 50 folds**. Un combinador incapaz de evaluar test rendiría *por debajo* de su mejor entrada, no por encima.

De hecho la brecha del metaclasificador (+0.1509) es **menor** que la de su entrada (+0.1591): combinar la reduce ligeramente.

El origen real es el ABMIL: `head_dim` 512, ~0.5 M parámetros, entrenado con **75 slides**. Memoriza parte de su conjunto de entrenamiento.

| Modelo | in-sample | test | brecha |
|---|---:|---:|---:|
| hoptimus1 | 0.9411 | 0.7450 | +0.1962 |
| phikon_v2 | 0.9580 | 0.7685 | +0.1895 |
| uni_v1 | 0.9432 | 0.7732 | +0.1700 |
| **conch_v1_5** | 0.9493 | **0.7902** | +0.1591 |
| virchow2 | 0.9163 | 0.7771 | +0.1392 |
| virchow_v1 | 0.7972 | 0.6823 | +0.1149 |
| uni_v2 | 0.8686 | 0.7634 | +0.1052 |
| ctranspath | 0.8489 | 0.7571 | +0.0918 |

**Memorizar más no predice mejor.** `phikon_v2` es el que más memoriza (0.9580) y queda cuarto en test; `ctranspath` es el que menos y queda sexto. El AUC in-sample no solo es inútil como estimador — no ordena a los modelos.

Un detalle sobre `val`: tampoco está limpio. Queda entre in-sample y test porque **el early stopping selecciona el modelo que maximiza val**, así que también está sesgado al alza. Solo `test` es una superficie honesta.

#### Arreglar la fuga no mejora el resultado

`build_oof_features.py` genera meta-features sin fuga mediante CV anidada agrupada por paciente (`--meta_features oof`). Medido sobre el trío histórico, el único con OOF completo, mismo test, 50 folds:

| Metaclasificador entrenado con | AUC test |
|---|---:|
| features in-sample (con fuga) | 0.7616 |
| features out-of-fold (sin fuga) | 0.7571 |

**Δ = −0.0045, IC 95 % [−0.0115, +0.0021], G/E/P 18/11/21.** No significativo, y si acaso ligeramente peor.

La razón es que con 6 features y un combinador lineal el desplazamiento apenas importa: la fuga infla las tres columnas de forma **aproximadamente pareja**, así que el peso *relativo* que aprende LogReg sobrevive. Y el OOF trae su propio coste — sus predicciones vienen de modelos entrenados con menos datos, así que las features son más ruidosas. Las dos distorsiones se cancelan.

Donde la fuga *sí* debería morder es con meta-modelos flexibles, que pueden explotar un espacio 6-D casi separable de formas que no transfieren. Eso explicaría por qué el MLP simple colapsa, TabPFN rinde por debajo de lo esperado, y los Snapshot Ensembles «arreglan» el MLP **regularizándolo de vuelta hacia una regresión logística**. Sigue siendo una hipótesis: para `logreg` está medido y es un empate.

### 6.3 Selección LOFO (leave-one-fold-out)

Se evaluaron **255 subconjuntos** (2⁸ − 1) de los 8 modelos base. Elegir el mejor mirando el mismo test que luego se reporta produce un número optimista: con 255 candidatos, algo gana por ruido.

La estimación honesta usa selección leave-one-fold-out. Para cada fold *k*:

1. Se calcula el AUC de **validación** en los otros 49 folds.
2. Se elige el subconjunto que lo maximiza — sin haber tocado el fold *k*.
3. Se reporta el AUC de **test** de ese subconjunto en el fold *k*.

Estima el rendimiento del **procedimiento completo** (elegir + aplicar), que es lo que se desplegaría.

| Estimación | AUC |
|---|---:|
| Naive (mejor en test — **optimista**) | 0.8057 |
| **Honesta (LOFO por val)** | **0.8011** |
| Baseline (trío histórico) | 0.7616 |
| **Sesgo de selección** (naive − honesta) | **+0.0046** |

El sesgo es pequeño, y la selección es **estable**: 49 de 50 folds eligieron `ctranspath+uni_v2+conch_v1_5`. Si cada fold hubiera elegido algo distinto, el procedimiento estaría siguiendo ruido.

### 6.4 Protocolo de comparación pareada

Todas las configuraciones se evalúan sobre **los mismos folds**, así que las comparaciones son pareadas. Se reporta, por métrica:

- Δ medio pareado e **IC 95 % por bootstrap**
- **Ganadas / Empatadas / Perdidas** — los empates se cuentan explícitamente: con 22 slides de test por fold, kappa empató en hasta 22 de 50 folds
- **Efecto mínimo detectable (MDE) al 80 % de potencia** — un resultado no significativo se reporta *con su cota*, para que se lea como **acotado**, no como indeterminado
- **p corregida por Holm** sobre las 4 métricas

Dos modos de fallo que este protocolo existe para evitar, ambos ya sufridos:

1. Comparar IC **independientes** ocultó un efecto sistemático en `cervical_subtype`.
2. Con 5 folds, un «5 de 5 folds, p = 0.024» sin corregir **no replicó** a 50 folds, y el signo se invirtió.

#### El suelo de reproducibilidad: ≈ 0.004 AUC

Medido el 22-ago reentrenando **la misma configuración exacta** — trío ganador, bolsa completa, mismo código, mismas semillas — y comparándola con los checkpoints originales:

| | AUC |
|---|---:|
| Checkpoints originales (19-21 ago) | 0.7986 |
| Reentrenado, configuración idéntica | 0.8026 |

**Δ = +0.0040, IC 95 % [−0.0107, +0.0180], G/E/P 22/11/17.** Desviación por fold **0.0518**; el fold típico se mueve **±0.032** solo por volver a entrenar.

Es no determinismo de GPU, no un cambio de código: el camino con `bag_size=0` es byte a byte el de siempre.

**Consecuencia para leer cualquier resultado de este informe:** un efecto por debajo de ≈ 0.01 de AUC es indistinguible de volver a lanzar el mismo job. Eso sitúa en zona de ruido dos resultados que aparecen como positivos en §7:

- Nelder-Mead, **+0.0052** (§7, fila 4)
- Atención GNN espacial, **+0.0077** (§7.2)

Ninguno de los dos supera el ruido de reentrenamiento por un margen que permita distinguirlos de él. **No basta con recalcular el baseline en la misma pasada** (§6.5): el efecto tiene que superar además el ruido de reentrenar, no solo el MDE del test. Los resultados grandes — el cambio de modelo base, +0.037 — están un orden de magnitud por encima de este suelo y no se ven afectados.

### 6.5 Auditoría de vigencia — qué resultados siguen siendo válidos

Un directorio de resultados solo significa algo frente a los checkpoints con los que se calculó. **El disparador de obsolescencia es la fecha del checkpoint**, no la del `preds.npy`: la re-predicción es determinista, verificado porque el baseline del trío reproduce a 16 dígitos (`0.7616233719983719`) entre dos pasadas distintas separadas por casi dos horas.

Fechas de los checkpoints (`fold_0/model.pt`):

| Modelo | Checkpoint |
|---|---|
| uni_v2 | 19-ago 21:37 |
| virchow_v1 | 19-ago 21:48 |
| ctranspath | 20-ago 12:52 |
| conch_v1_5 | **21-ago 12:31** ← última modificación relevante |
| phikon_v2 / uni_v1 / hoptimus1 / virchow2 | 21-ago 22:19 – 23:11 |

Todo lo calculado **después del 21-ago 12:31** es mutuamente comparable:

| Resultado | Fecha | Estado |
|---|---|---|
| Barrido de 255 subconjuntos | 21-ago 23:47 | ✅ **vigente** |
| Trío vs. swap conch vs. cuatro | 21-ago 22:00–22:03 | ✅ **vigente** |
| Early vs. late fusion (opción C) | 21-ago 23:50 | ✅ **vigente** |
| Nelder-Mead (opción A) | 21-ago 14:39 | ✅ **vigente** |
| Fusión temprana ponderada (opción D) | 21-ago 18:45 | ✅ **vigente** |
| Escalera de features (`ensemble4_fea_*`) | 19-ago 21:16 | 🔴 obsoleto |
| Idea 3 y atención (`ensemble4_idea3_*`) | 20-ago 12:10–12:43 | 🔴 obsoleto |
| SVM / KNN / NB y reponderación | 19-ago 13:03–14:23 | 🔴 obsoleto |
| Metaclasificador con embeddings | 18-ago 19:03 | 🔴 obsoleto |

> **`E0 = 0.7918` en `results_ladder_*.json` no reproduce.** El valor vigente para ese mismo trío y el mismo código es **0.7616**.

Las comparaciones *dentro* de un mismo grupo obsoleto siguen siendo válidas (mismos checkpoints, mismo momento), así que sus conclusiones **relativas** se mantienen. Lo que no vale es citar sus números absolutos ni comparar entre grupos.

**Regla operativa: para cualquier contraste nuevo, recalcular el baseline en la misma pasada.** Nunca reutilizar un `ensemble4_*` antiguo.

---

## 7. Alternativas probadas

Todas sobre `cptac_brca / TP53_mutation`, 50 folds, comparación pareada.

| # | Alternativa | Δ AUC | IC 95 % | G/E/P | p | Veredicto |
|---:|---|---:|---|---|---:|---|
| 1 | **Selección LOFO sobre 255 subconjuntos** | **+0.0395** | [+0.0253, +0.0540] | 35/4/11 | 2·10⁻⁵ | ✅ **Funciona** |
| 2 | **Cambiar `virchow_v1` → `conch_v1_5`** | **+0.0370** | [+0.0230, +0.0514] | 34/4/12 | 4·10⁻⁵ | ✅ **Funciona** |
| 3 | Añadir `conch_v1_5` sin quitar `virchow_v1` | +0.0370 | [+0.0230, +0.0515] | 35/2/13 | 3·10⁻⁵ | ✅ Empata con #2 |
| 4 | Pesos por Nelder-Mead vs. LogReg | +0.0052 | [+0.0007, +0.0106] | 26/11/13 | 0.058 | ⚠️ Marginal |
| 5 | Escalera de features (desacuerdo, margen, entropía) | −0.0004 | [−0.0011, +0.0003] | — | 0.29 | ❌ Nulo (obsoleto) |
| 6 | Fusión temprana a nivel de parche | −0.0056 | [−0.0250, +0.0124] | 21/3/26 | 0.55 | ❌ Nulo |
| 7 | Fusión temprana ponderada | −0.0182 | [−0.0421, +0.0061] | 20/1/29 | 0.15 | ❌ Empeora |
| 8 | Snapshot Ensembles (FGE / SE) | +0.001 … +0.006 | MDE 0.013–0.017 | — | ns | ❌ Nulo **acotado** |
| 9 | Atención GNN espacial (§7.2) | +0.0077 | [−0.0069, +0.0218] | 24/9/17 | ns | ❌ Nulo |
| 10 | Atención multihead, 4 cabezas (§7.2) | −0.0240 | *(5 folds)* | 2/0/3 | — | ❌ Empeora |
| 11 | Augmentación de bolsa, mejor variante (§7.3) | +0.0077 | [−0.0043, +0.0201] | 22/5/23 | ns | ❌ Nulo |

Notas por fila:

- **#2 y #3 empatan.** Añadir `conch_v1_5` sin quitar `virchow_v1` da lo mismo que sustituirlo (0.7986 en ambos casos), así que se prefiere el trío por coste. Las **cuatro** métricas de #2 superan Holm (p ≈ 1.4·10⁻⁴) y son 1.7–1.8× su MDE.
- **#4 es el techo de la optimización de la regla de combinación**, y no llega a significativo. Es el mejor resultado que ha dado nunca tocar el combinador — pero cae **por debajo del suelo de reproducibilidad** de §6.4, así que no se distingue de reentrenar el mismo job.
- **#4, #9 y #11 están todos en zona de ruido de reentrenamiento** (≈ 0.004, §6.4). Solo #1, #2 y #3 —los tres sobre *qué modelos entran*— superan ese suelo con margen.
- **#8 es un nulo con cota, no una ausencia de evidencia.** Los IC excluyen cualquier mejora de AUC superior a ≈ +0.017. En kappa el MDE ronda 0.048–0.051, así que ahí el nulo es más débil. El hallazgo original en `cervical_subtype` (5/5 folds, p = 0.024) **no replicó** a 50 folds y el signo se invirtió.

### 7.1 Rendimiento por tamaño del subconjunto

Del barrido de 255 subconjuntos, mejor y media de AUC de test por número de modelos:

| k | Subconjuntos | Mejor | Media | Mejor combinación |
|---:|---:|---:|---:|---|
| 1 | 8 | 0.7902 | 0.7571 | conch_v1_5 |
| 2 | 28 | 0.8009 | 0.7759 | conch_v1_5+uni_v1 |
| 3 | 56 | 0.8040 | 0.7833 | conch_v1_5+uni_v1+virchow2 |
| 4 | 70 | 0.8054 | 0.7883 | ctranspath+conch_v1_5+phikon_v2+virchow2 |
| 5 | 56 | 0.8055 | 0.7920 | +uni_v1 |
| 6 | 28 | **0.8057** | 0.7949 | ctranspath+uni_v2+virchow_v1+conch_v1_5+phikon_v2+virchow2 |
| 7 | 8 | 0.8054 | 0.7970 | (todos menos hoptimus1) |
| 8 | 1 | 0.7988 | 0.7988 | todos |

**Satura en k = 3–4.** Pasar de 3 a 6 modelos compra +0.0017 de AUC al doble de coste de inferencia, y con 8 el rendimiento **baja**. La media sube monótonamente mientras el máximo se aplana: añadir modelos protege contra una mala elección, pero no supera a una buena.

### 7.2 El brazo de atención — probado, con control negativo

Un tercer escalón, distinto de «qué modelos entran» y de «cómo se combinan»: **cómo se agregan los parches dentro de cada slide**. Se han entrenado variantes del mecanismo de atención del ABMIL.

| Variante (sobre `uni_v2`) | AUC | Δ vs. baseline | IC 95 % | G/E/P |
|---|---:|---:|---|---|
| ABMIL, 1 cabeza (baseline) | 0.7634 | — | | |
| + GNN espacial | 0.7663 | +0.0029 | [−0.0115, +0.0162] | 22/14/14 |
| + GNN espacial (p = 0.3) | 0.7711 | +0.0077 | [−0.0069, +0.0218] | 24/9/17 |
| + GNN con **grafo barajado** | 0.7592 | −0.0042 | [−0.0165, +0.0074] | 17/12/21 |
| `ctranspath` con 4 cabezas ⚠ | 0.7362 | **−0.0240** | *(5 folds)* | 2/0/3 |

**Ninguna variante bate al baseline de forma significativa**, y la multihead — la forma más obvia de «mejorar la atención» — empeora.

#### El control negativo: la señal espacial es real, pero diminuta

El barrido incluye una rama con **el mismo grafo barajado**: misma arquitectura, mismos parámetros, misma capacidad, con la estructura espacial destruida. Es la comparación limpia, porque ambas ramas se entrenaron en el mismo momento.

| | AUC |
|---|---:|
| GNN con grafo real | 0.7711 |
| GNN con grafo barajado | 0.7592 |

**Δ = +0.0119, IC 95 % [+0.0004, +0.0238], G/E/P 26/8/16.**

Significativo, por poco. **La vecindad espacial entre parches sí lleva información sobre TP53.** Es la única evidencia positiva de que la atención tiene margen real.

Pero fíjese en la aritmética: el mecanismo espacial aporta **+0.0119** sobre su propio control y solo **+0.0077** sobre el baseline. La diferencia es lo que **cuesta la arquitectura** — el GNN añade parámetros, y con 75 slides de entrenamiento esos parámetros se comen la ganancia (§6.2).

#### Por qué una atención «mejor» tiende a empeorar aquí

Es la consecuencia directa de §6.2. La restricción que manda son **75 slides contra 0.4–1.4 M de parámetros**. Casi todo lo que se entiende por mejorar la atención — más cabezas, transformers entre parches, GAT — significa **más capacidad**, y con esa n la capacidad extra se convierte en memorización. El −0.0240 de la multihead es exactamente eso.

**El cuello de botella no es el diseño de la atención: es el tamaño del conjunto.** Esta conclusión caduca si algún día se entrena sobre un cohorte mayor.

#### El techo, para calibrar

Oráculos que **miran las etiquetas de test** — cotas superiores inalcanzables, no objetivos:

| | AUC |
|---|---:|
| Oráculo: mejor modelo individual por fold | 0.8423 |
| Oráculo: mejor promedio de subconjunto por fold | 0.8608 |
| **Alcanzable hoy** (LOFO honesto) | **0.8011** |

Con 21 slides de test por fold, tomar el máximo de 8 estimaciones ruidosas infla mucho el oráculo: el margen real es bastante menor que esos 0.04–0.06.

> ⚠️ **Dos avisos.** Los GNN se entrenaron el 17-18 ago y el checkpoint de `uni_v2` es del 19-ago 21:37, así que la comparación *contra baseline* no cumple la regla de §6.5 — el **control barajado sí es limpio**, porque ambas ramas son del mismo momento. Y la multihead son **5 folds**, justo el tamaño que en `cervical_subtype` no replicó.
>
> Además, el +0.0077 del GNN está **por debajo del suelo de reproducibilidad** de §6.4 (≈ 0.004, con folds moviéndose ±0.03). El control barajado sobrevive a esa objeción porque compara dos ramas entrenadas en el mismo momento; el Δ contra baseline, no.

### 7.3 Augmentación de bolsa — probado, nulo

Si el cuello de botella son 75 slides (§6.2, §7.2), atacar *n* directamente tiene más sentido que cualquier cambio de arquitectura. La vía más barata es muestrear parches: **con `bag_size` parches por forward, cada slide es una bolsa distinta en cada época**, sin recomputar embeddings.

Hubo que implementarlo — no existía. `_apply` pasaba *todos* los parches en cada forward, sin muestreo. (CLAUDE.md afirma «Bag size: 2048»; **es falso**.) El muestreo actúa **solo en el bucle de entrenamiento**: evaluación, predicción y meta-features usan la bolsa completa, porque muestrear ahí metería varianza en la métrica y contaminaría §6.2.

Barrido de 50 folds sobre el trío ganador, con el baseline **reentrenado en la misma pasada** (job 70289, 1 h 53 min, 12 corridas de 50/50 folds):

| Configuración | AUC | bacc | macro-F1 | kappa |
|---|---:|---:|---:|---:|
| **Bolsa completa** (baseline reentrenado) | 0.8026 | 0.7448 | 0.7419 | 0.4886 |
| bag 256 | 0.8040 | 0.7428 | 0.7414 | 0.4872 |
| bag 512 | 0.8053 | 0.7442 | 0.7412 | 0.4870 |
| bag 1024 | **0.8103** | 0.7376 | 0.7349 | 0.4737 |

| Δ vs. bolsa completa | Δ AUC | IC 95 % | G/E/P | MDE |
|---|---:|---|---|---:|
| bag 256 | +0.0014 | [−0.0115, +0.0139] | 25/5/20 | 0.0185 |
| bag 512 | +0.0027 | [−0.0091, +0.0144] | 25/6/19 | 0.0169 |
| bag 1024 | +0.0077 | [−0.0043, +0.0201] | 22/5/23 | 0.0177 |

**Ninguna de las 12 comparaciones es significativa**, en ninguna de las cuatro métricas.

Dos razones para no rescatar el +0.0077 de bag 1024:

1. Está **por debajo del doble del suelo de reproducibilidad** de §6.4 (+0.0040).
2. **Sube el AUC pero baja las otras tres métricas** (bacc −0.0072, F1 −0.0070, kappa −0.0149). Un efecto real se mueve en el mismo sentido en las cuatro, como hizo el cambio de modelo base. Este no.

**Por qué falla:** la augmentación de bolsa **no crea pacientes nuevos**. Siguen siendo 103 personas, y la varianza que domina el error de generalización es la de *entre* pacientes, no la de entre submuestras del mismo paciente. Dos bolsas de la misma slide comparten etiqueta, tinción, escáner y morfología: aportan casi la misma información.

Implementación en `abmil_engine.py` (`_apply`) y `train_abmil.py` (`--bag_size N`, defecto 0 = desactivado). Se conserva: el coste es nulo y deja la puerta abierta si algún día cambia el tamaño del cohorte.

---

## 8. Qué falta por probar

Ordenado por relación entre valor esperado y coste.

### 8.1 🔴 Prioridad alta — embeddings vs. probabilidades con checkpoints actuales

La única comparación con evidencia previa de mejora que **sigue sin medirse en condiciones válidas**. Los datos obsoletos sugieren que MLP Snapshot sobre embeddings de 1536 D llegaba a 0.7995, frente a 0.7896 de LogReg sobre el mismo espacio. Con el trío ganador y los checkpoints actuales podría o no sostenerse.

```bash
# extraer embeddings para el trío ganador y re-comparar en una sola pasada
for m in ctranspath uni_v2 conch_v1_5; do
  python src/test_abmil.py --foundational_model $m --output_embeddings ...
done
python src/ensemble4.py --foundational_models ctranspath uni_v2 conch_v1_5 \
    --meta_model logreg --results_subdir_suffix _emb_vs_prob ...
```

### 8.2 🟡 Prioridad media — meta-features out-of-fold **con un meta-modelo flexible**

**Ya no es prioridad alta: con `logreg` está medido y no aporta** (Δ = −0.0045, IC [−0.0115, +0.0021], ver §6.2). Corregir la fuga no mejora el test porque un combinador lineal sobre 6 columnas apenas la sufre.

Lo que queda por probar es la única versión con fundamento: **OOF + un meta-modelo flexible** (`mlp`, `lightgbm`, `tabpfn`). La hipótesis es que esos modelos fracasan precisamente porque explotan la fuga, y que sin ella podrían superar a `logreg`. Requiere generar OOF para `conch_v1_5`, que no lo tiene.

### 8.3 🟡 Prioridad media — validación en un segundo dataset

**Todo lo de este informe es `cptac_brca / TP53_mutation`.** La lección de `cervical_subtype` es precisamente que un hallazgo de un dataset puede no replicar. Hasta que `conch_v1_5 > virchow_v1` se reproduzca en otra tarea, es un resultado de un solo conjunto. Disponibles: `cervical_subtype` (5 folds), `hancook` (5), `imp` (1).

### 8.4 🟡 Prioridad media — fusión temprana dentro del grupo 1

La fusión temprana solo se ha probado sobre el grupo 2 (2100 parches: ctranspath, virchow_v1, conch_v1_5). El **grupo 1 tiene cinco modelos con coordenadas idénticas** (hoptimus1, phikon_v2, uni_v1, uni_v2, virchow2) y nunca se ha fusionado. Dado el −0.0056 del grupo 2 la expectativa es baja, pero es la única versión de la hipótesis que puede probarse a 5 bandas.

### 8.5 🟡 Prioridad media — recomputar la escalera de features y la idea 3

Sus conclusiones relativas (las features derivadas no aportan) probablemente se mantengan, pero sus números absolutos no son citables. Es barato: minutos de CPU.

### 8.6 🟢 Prioridad baja

- **Gated attention.** Está en `gated=False` y **nunca se ha probado**. Es un cambio de una línea en `abmil_engine.py` y añade muy pocos parámetros, así que es la única variante de atención que no cae en la trampa de capacidad de §7.2. No por lo que se espere de ella, sino porque cierra la casilla en una tarde.
- **Regularizar la atención en vez de ampliarla** — penalización de entropía o top-k de parches. Reduce capacidad efectiva, que es lo que sobra con n = 75.
- **Calibración.** El log-loss se calcula pero nunca se ha usado para decidir. Si la salida va a informar decisión clínica, la calibración importa tanto como el AUC.
- **Evaluación a nivel de paciente.** 112 slides ↔ 103 pacientes: 9 pacientes aportan más de una slide. La opción B quedó incompleta (3 jobs fallidos, 70267–70269).
- **Reponderación por importancia del modelo.** Cuatro métodos implementados, todos medidos solo sobre resultados obsoletos.

### 8.7 ⛔ Cerrado — no reabrir sin motivo nuevo

- **Fusión temprana por concatenación** (−0.0056) y **ponderada** (−0.0182).
- **Snapshot Ensembles / FGE** como fuente de ganancia: nulo acotado a ≈ +0.017 de AUC.
- **Atención con más capacidad**: multihead (−0.0240), transformers entre parches, y **GAT** (8 jobs, 70272–70283, ninguno llegó a entrenar; código eliminado el 22-ago). Ver §7.2: con 75 slides de entrenamiento la capacidad extra se convierte en memorización.
- **Augmentación de bolsa** (§7.3): 12 comparaciones, ninguna significativa. La augmentación que no crea pacientes nuevos no ataca la varianza que domina el error. El código (`--bag_size`) se conserva desactivado por si cambia el cohorte.

> **Salvedad sobre lo cerrado.** El control negativo de §7.2 demuestra que la señal espacial **existe** (+0.0119, significativo). Lo que no funciona es pagarla con parámetros. Si alguna vez se entrena sobre un cohorte sustancialmente mayor que 112 slides, esta casilla debe reabrirse.

---

## 9. Mejor modelo

### 9.1 La configuración recomendada

```bash
python src/ensemble4.py \
    --foundational_models ctranspath uni_v2 conch_v1_5 \
    --work_dir /home/JKP6679/Patho-Ensemble/PARADIS/datos/patches \
    --train_source cptac_brca \
    --tissue_patching 20x_224px_0px_overlap \
    --task_name TP53_mutation \
    --meta_model logreg
```

| Métrica | Trío histórico | **Trío ganador** | Δ pareado | IC 95 % | p (Holm) |
|---|---:|---:|---:|---|---:|
| macro-OvR AUC | 0.7616 | **0.7986** | +0.0370 | [+0.0230, +0.0514] | 1.4·10⁻⁴ |
| bacc | 0.6702 | **0.7302** | +0.0600 | [+0.0363, +0.0838] | 1.4·10⁻⁴ |
| macro-F1 | 0.6687 | **0.7294** | +0.0608 | [+0.0362, +0.0857] | 1.4·10⁻⁴ |
| kappa ponderada | 0.3502 | **0.4629** | +0.1127 | [+0.0657, +0.1601] | 1.4·10⁻⁴ |

Las cuatro métricas son significativas tras Holm y todas superan su MDE por un factor de 1.7–1.8.

### 9.2 Cuánto aporta realmente el metaclasificador

Conviene ser honesto sobre de dónde viene la ganancia:

| Configuración | AUC |
|---|---:|
| `conch_v1_5` en solitario, sin ensemble | 0.7902 |
| Trío ganador + stacking LogReg | 0.7986 |
| **Aportación del metaclasificador** | **+0.0084** |
| **Aportación de cambiar el modelo base** | **+0.0370** |

El metaclasificador vale **+0.008** sobre el mejor modelo individual. Cambiar qué modelo entra vale **+0.037**. Esa proporción es el resultado central del proyecto.

### 9.3 Acción pendiente en el repositorio

`defaults.models` en `experiments.yaml` sigue apuntando al trío histórico. La selección LOFO elige `ctranspath+uni_v2+conch_v1_5` en 49 de 50 folds; la configuración por defecto debería reflejarlo.

---

## Anexo — Procedencia de los datos

| Fuente | Contenido |
|---|---|
| `results_base_model_sweep_cptac_brca_TP53_mutation.json` | Barrido de 255 subconjuntos, selección LOFO, AUC individuales (job 70271) |
| `results_model_selection_sweep.json` | Trío vs. swap vs. cuatro, 4 métricas con Holm y MDE |
| `results_opcion_c_fair_comparison.json` | Fusión temprana vs. tardía, comparación justa (job 70285) |
| `results_opcion_a_nelder_mead.json` | Optimización de pesos por Nelder-Mead |
| `results_opcion_d_weighted_early_fusion.json` | Fusión temprana ponderada |
| `results_bag_size_sweep_cptac_brca_TP53_mutation.json` | Barrido de augmentación de bolsa y suelo de reproducibilidad (job 70289) |
| `results_ladder_cptac_brca_TP53_mutation.json` | Escalera de features (🔴 obsoleto) |
| `INFORME_ENSEMBLE_EMBEDDINGS.md` | Metaclasificador sobre embeddings (🔴 obsoleto) |
| `INFORME_ABLATION_FGE_cervical_subtype.md` | Ablación FGE/SE y su réplica a 50 folds |
| `logs/sweep_base_models_70271.out`, `logs/opcion_c_fair_comparison_70285.out` | Salidas de los dos últimos jobs |

Fechas de checkpoint y de resultados verificadas con `stat` sobre el árbol
`PARADIS/datos/patches/cptac_brca/TP53_mutation/abmil/` el 22-ago-2026.
