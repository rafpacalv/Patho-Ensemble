# Apagado dinámico de modelos fundación: cuatro pruebas, cuatro negativos

**Fecha** 1-sep-2026 · **Tareas** 13 tareas binarias en 3 cohortes · **Folds** 50, pareados
**Origen** la pregunta que plantea PathBench (arXiv 2505.20202): si el mejor modelo
fundación cambia según el órgano, ¿por qué fijar uno en vez de poner el comité entero
y decidir sobre la marcha a quién escuchar?

---

## Resumen ejecutivo

El apagado dinámico de modelos fundación **es implementable y está implementado**:
el enrutador construido elige comités distintos según el órgano — en colon descarta
`uni_v1`, en mama conserva los cuatro modelos. **Pero no compensa.** Cuatro pruebas
independientes lo acotan:

| prueba | qué mide | Δ AUC | p |
|---|---|---:|---:|
| 1 | ampliar el comité de 3 a 8 modelos | +0.0002 | 0.99 |
| 2 | elegir el comité por tarea (conociendo la tarea) | +0.0016 | 0.46 |
| 3 | enrutar el comité por órgano (escenario ciego real) | +0.0015 | 0.31 |
| 4 | cambiar el criterio de selección | +0.0000 | — |

El cuello de botella **no es el mecanismo de apagado ni la granularidad del enrutado**:
es que ningún criterio disponible sin etiquetas de test sabe a quién apagar. El oráculo
—elegir el comité mirando el test— dejaría +0.0071 sobre la mesa en `cptac_brca/TP53_mutation`
y +0.0141 en el conjunto de 13 tareas. La información existe; el selector no la lee.

**Recomendación**: cerrar la línea. El único mecanismo con efecto grande medido en este
repositorio sigue siendo la **calidad del modelo base** (+0.037 al cambiar uno), frente a
+0.008 de la regla de combinación y menos de +0.003 de todo lo medido aquí.

---

## 1. Datos y protocolo

**Verificación previa.** La restricción #10 de `CLAUDE.md` afirmaba que `hoptimus1`,
`phikon_v2`, `uni_v1` y `virchow2` no eran comparables con el resto por provenir de una
campaña a nivel paciente (103 filas). **Está obsoleta**: se reentrenaron el 21-ago-2026 y
hoy comparten split y etiquetas exactas con los demás — verificado fila a fila, 75 filas de
meta-train / 15 de validación / 22 de test por fold. Sigue viva, en cambio, la advertencia
sobre los `_train_eval/test_outputs` de dic-2025, que conservan 20 filas.

**Comparaciones pareadas.** Todas las configuraciones se evalúan sobre los mismos folds,
con Δ pareado, IC95 % bootstrap (10 000 remuestreos) y Wilcoxon. Ningún directorio
`ensemble4_*` previo se reutiliza.

**Meta-learner.** Regresión logística sobre probabilidades concatenadas, la configuración
por defecto de `ensemble4.py`. Meta-features in-sample.

### Los 8 modelos base — AUC test individual, 50 folds

| modelo | AUC |
|---|---:|
| conch_v1_5 | 0.7902 |
| virchow2 | 0.7771 |
| uni_v1 | 0.7732 |
| phikon_v2 | 0.7685 |
| uni_v2 | 0.7634 |
| ctranspath | 0.7571 |
| hoptimus1 | 0.7450 |
| virchow_v1 | 0.6823 |

### El peor no es el prescindible

Correlación de Pearson entre las probabilidades de clase 1, agrupando los 50 folds:

- `ctranspath` ↔ `uni_v2` = **0.84** — casi redundantes, y **ambos están en el trío histórico**.
- `virchow_v1`, el peor en solitario (0.6823), es el **más decorrelacionado** de
  `conch_v1_5` (0.35) y de `hoptimus1` (0.35).
- `hoptimus1` es el más independiente del comité en conjunto (0.35–0.66).

El criterio de AUC individual que justificó el cambio de modelo base de +0.037 no es el
que maximiza un comité: la redundancia importa tanto como la calidad.

---

## 2. Prueba 1 — Ampliar el comité no aporta

`cptac_brca/TP53_mutation`, los **255 subconjuntos** de 8 modelos, 50 folds.
Baseline: el trío ganador actual `conch_v1_5+ctranspath+uni_v2`.

| comité | AUC test | Δ vs trío | IC95 % | W/T/L | p |
|---|---:|---:|---|---|---:|
| trío actual (baseline) | 0.7986 | — | | | |
| **los 8 completos** | 0.7988 | **+0.0002** | [−0.0160, +0.0163] | 23/2/25 | 0.99 |
| mejor en test (6 modelos) | 0.8057 | +0.0071 | [−0.0074, +0.0215] | 29/5/16 | 0.30 |
| selección honesta LOFO | 0.8011 | +0.0025 | [+0.0000, +0.0075] | 1/49/0 | 0.32 |

Poner **todos** los modelos en paralelo es indistinguible de no hacer nada: 23 victorias
frente a 25 derrotas. El MDE del diseño ronda **±0.016**, así que incluso el +0.0071 del
mejor subconjunto queda por debajo de lo detectable con 50 folds.

### El óptimo está en 4–6 modelos, no en 8

| tamaño | nº subconj. | AUC test medio | AUC test mejor |
|---:|---:|---:|---:|
| 1 | 8 | 0.7571 | 0.7902 |
| 2 | 28 | 0.7759 | 0.8009 |
| 3 | 56 | 0.7833 | 0.8040 |
| 4 | 70 | 0.7883 | 0.8054 |
| 5 | 56 | 0.7920 | 0.8055 |
| 6 | 28 | 0.7949 | **0.8057** |
| 7 | 8 | 0.7970 | 0.8054 |
| 8 | 1 | 0.7988 | 0.7988 |

La media sube monótonamente con el tamaño —añadir modelos protege del mal azar— pero el
**mejor** subconjunto satura en k=4 y el comité completo es peor que el mejor de 6.

### Ningún trío alternativo bate al actual

| trío | AUC test | Δ | IC95 % | W/T/L | p |
|---|---:|---:|---|---|---:|
| `conch_v1_5+uni_v1+virchow2` | 0.8040 | +0.0054 | [−0.0076, +0.0191] | 23/2/25 | 0.73 |
| `conch_v1_5+phikon_v2+virchow2` | 0.8030 | +0.0044 | [−0.0121, +0.0201] | 29/4/17 | 0.45 |

### La selección honesta elige el trío 49 de 50 veces

La estimación *leave-one-fold-out* —para cada fold, el subconjunto con mejor AUC de
validación en los otros 49— **converge al trío baseline en 49 de 50 folds**, porque el
trío gana en validación (0.8681) por amplio margen sobre los comités de 6 (0.8535).
Y pierde en test. El sesgo de selección (naive − honesta) es **+0.0046**.

---

## 3. Prueba 2 — Elegir el comité por tarea tampoco

13 tareas binarias en 3 cohortes, con los 4 modelos comunes a todas
(`hoptimus1`, `uni_v1`, `uni_v2`, `virchow2`), 15 subconjuntos, 50 folds cada tarea.

| tarea | órgano | fijo(4) | honesta | oráculo | Δ honesta |
|---|---|---:|---:|---:|---:|
| cptac_brca/PIK3CA_mutation | mama | 0.6064 | 0.6044 | 0.6125 | −0.0020 |
| cptac_brca/TP53_mutation | mama | 0.7752 | 0.7618 | 0.7864 | −0.0134 |
| bc_therapy/er_status | mama | 0.6756 | 0.6756 | 0.6801 | +0.0000 |
| bc_therapy/grade | mama | 0.7471 | 0.7522 | 0.7655 | +0.0051 |
| bc_therapy/her2_status | mama | 0.6942 | 0.6942 | 0.7012 | +0.0000 |
| cptac_coad/APC_mutation | colon | 0.7332 | 0.7318 | 0.7382 | −0.0014 |
| cptac_coad/KRAS_mutation | colon | 0.6231 | 0.6210 | 0.6468 | −0.0021 |
| cptac_coad/MSI_H | colon | 0.8800 | 0.8943 | 0.8943 | +0.0143 |
| cptac_coad/TP53_mutation | colon | 0.7142 | 0.7219 | 0.7356 | +0.0078 |
| cptac_coad/ARID1A_mutation | colon | 0.7649 | 0.7191 | 0.7827 | −0.0458 |
| cptac_coad/PIK3CA_mutation | colon | 0.6655 | 0.6698 | 0.6662 | +0.0043 |
| cptac_coad/ACVR2A_mutation | colon | 0.8204 | 0.8596 | 0.8596 | +0.0393 |
| cptac_coad/SETD1B_mutation | colon | 0.8129 | 0.8276 | 0.8276 | +0.0147 |

**Selección honesta por tarea vs comité fijo: Δ = +0.0016, gana en 6/13, Wilcoxon p = 0.46.**
**Oráculo: +0.0141.** La selección recupera el 11 % del margen; el resto es sesgo de selección.

Esta prueba **conocía la tarea** y usaba su propia validación — información que un enrutado
sobre una WSI ciega no tiene. Es por tanto el **techo** del enrutado dinámico, y el techo
está dentro del ruido. Basta para falsar la hipótesis por la vía barata.

### La señal de órgano existe, en el oráculo

| órgano | comité óptimo por tarea |
|---|---|
| mama | `uni_v1+uni_v2+virchow2` · `uni_v1+virchow2` · `uni_v1` · `uni_v2` · `uni_v1+uni_v2+virchow2` |
| colon | `uni_v1+uni_v2` · `uni_v2` · `hoptimus1+uni_v2` · `hoptimus1+uni_v2` · `hoptimus1+uni_v1` · `hoptimus1+uni_v2+virchow2` · `hoptimus1` · `hoptimus1+uni_v2` |

**`hoptimus1` aparece en 6 de los 8 comités óptimos de colon y en 0 de los 5 de mama.**
Es exactamente el patrón de PathBench, donde H-Optimus-1 es el mejor modelo en colorrectal
y cae al 7.º puesto en mama, reproducido aquí con datos propios. La señal es real. La
prueba 3 comprueba si es explotable.

---

## 4. Prueba 3 — Enrutar por órgano tampoco

*Leave-one-task-out*: para cada tarea se elige el comité usando el AUC de validación de las
**otras** tareas, y se evalúa en la tarea retenida. Tres políticas:

- **A · fijo** — los 4 modelos, siempre. Sin selección.
- **B · global** — un único comité para todo, elegido por las otras tareas.
- **C · órgano** — comité elegido por las otras tareas **del mismo órgano**.

C es el escenario ciego realizable: el órgano se lee del embedding de la WSI, no de la
etiqueta del dataset.

| política | Δ vs fijo | gana | pierde | p |
|---|---:|---:|---:|---:|
| B global vs A fijo | −0.0001 | 6/13 | 4/13 | 0.85 |
| **C órgano vs A fijo** | **+0.0015** | 6/13 | 2/13 | 0.31 |
| C órgano vs B global | +0.0016 | 5/13 | 3/13 | 0.46 |

**El enrutador sí discrimina**: en colon selecciona `hoptimus1+uni_v2+virchow2` (descarta
`uni_v1`), en mama conserva los cuatro. El mecanismo funciona. El beneficio no existe.

Nótese que el enrutador **no reprodujo el patrón del oráculo**: éste indica que `hoptimus1`
sobra en mama, pero la validación lo prefiere, así que el enrutador lo mantuvo en ambos
órganos. Esa discrepancia es el objeto de la prueba 4.

---

## 5. Prueba 4 — El selector no se arregla cambiando de métrica

Sobre los 255 subconjuntos ya evaluados de `cptac_brca/TP53_mutation`, ocho criterios
calculables **sin etiquetas de test**. Se mide la correlación de rangos con el AUC de test
y, lo que decide, el AUC que obtiene la selección honesta guiada por cada criterio.

| criterio | Spearman | sel. honesta | recupera |
|---|---:|---:|---:|
| `auc_val_penaliz` (media − desviación entre folds) | **+0.814** | 0.8009 | 32 % |
| `auc_val_medio` (el actual) | +0.750 | **0.8011** | 35 % |
| `auc_val_agrupado` (folds concatenados) | +0.712 | 0.7992 | 8 % |
| `brier_val` | +0.685 | 0.8006 | 28 % |
| `logloss_val` | +0.682 | 0.8006 | 28 % |
| `auc_metatrain` (in-sample, control negativo) | +0.623 | 0.7960 | — |
| `avg_precision` | +0.559 | 0.7992 | 8 % |
| `n_modelos` (control: "usa más modelos") | +0.449 | 0.7988 | 3 % |

**Sí existe un criterio que ordena mejor.** Penalizar la inestabilidad entre folds
(media − desviación) sube la correlación de +0.750 a **+0.814**: la varianza del comité
entre folds informa sobre su rendimiento en test. Es el único hallazgo positivo del estudio.

**Y no sirve para seleccionar.** Elige 0.8009 frente a 0.8011 del criterio actual. Ordenar
mejor los 255 en conjunto no mueve el argmax, y para elegir sólo importa quién queda arriba.
Los ocho criterios caen en un rango de 0.005 (0.7960–0.8011), y el mejor recupera el 35 % de
un margen que ya era +0.0071 y no significativo.

Dos resultados contraintuitivos que conviene registrar:

- **`auc_val_agrupado` era el favorito teórico y salió peor** (+0.712 vs +0.750). El
  razonamiento —15 slides por fold es poco, agrupar 750 filas estima mejor— ignora que
  agrupar mezcla folds con modelos y poblaciones distintas; esa heterogeneidad ensucia más
  de lo que limpia el ruido. Explicación *a posteriori*, no prevista.
- **`n_modelos` no es competitivo** (0.7988): la heurística "usa más modelos" equivale al
  comité fijo, lo que confirma que la prueba 1 no se explica por un simple efecto de tamaño.

---

## 6. Diagnóstico

En las cuatro pruebas el obstáculo es el mismo, y no es el mecanismo de apagado:

> **El AUC de validación no ordena los comités como lo hace el test.**

- Spearman(validación, test) = **+0.750** sobre 255 subconjuntos.
- El mejor comité en validación (el trío, 0.8681) queda **16.º** en test.
- El mejor en test (6 modelos, 0.8057) marca 0.8535 en validación.
- El enrutador por órgano hereda el problema y no reproduce el patrón del oráculo.
- Ningún criterio alternativo mejora la selección, ni siquiera el que correlaciona mejor.

La consecuencia práctica es que **apagar modelos dinámicamente es implementable pero no
accionable**: se puede construir el enrutador —está construido— pero no hay señal disponible
en producción que le diga a quién apagar. El oráculo demuestra que la información existe;
lo que no existe es la forma de leerla sin las etiquetas que en producción no se tienen.

---

## 7. Qué sí funciona

Ordenado por efecto medido en este repositorio, sobre `cptac_brca/TP53_mutation`:

| palanca | Δ AUC | ¿significativo? |
|---|---:|---|
| cambiar un modelo base (`virchow_v1` → `conch_v1_5`) | **+0.037** | sí, p ≈ 1.4e-4 |
| stacking frente al mejor modelo suelto | +0.008 | marginal |
| optimización de pesos Nelder-Mead | +0.005 | marginal |
| **ampliar / enrutar / seleccionar el comité (este informe)** | **< +0.003** | **no** |
| early fusion a nivel de patch | −0.006 | no |

`conch_v1_5` en solitario da 0.7902 frente a 0.7986 del stacking del trío completo. La
recomendación operativa es entrenar los modelos que PathBench sitúa arriba y que aquí no
están entrenados —UNI2, H-Optimus-0, Prov-GigaPath— antes que seguir refinando cómo se
combinan los existentes.

---

## 8. Correcciones de código aplicadas

Cuatro defectos encontrados durante el estudio, tres de ellos preexistentes:

- **`src/ensemble4.py` · `--extra_features` estaba roto por dos motivos.**
  `_compute_extra_features` devuelve `(array, names)` y los tres *call-sites* trataban la
  tupla como array (`AttributeError` antes de entrenar); además `num_classes` se derivaba de
  la `X` **ya transformada**, de modo que con `--drop_redundant_class` el reshape era
  imposible en tarea binaria. Ahora se deriva de `X_raw`.
- **`src/utils.py` · `MetricDistance` apagaba modelos sin criterio.** Normalizaba min-max y
  reescalaba a suma 1 —no un softmax, pese al comentario—, lo que asigna **peso exactamente 0
  al peor modelo de cada fold, siempre**. Documentado y acotable con `weight_floor`
  (default 0.0: no altera resultados previos).
- **`src/sweep_base_models.py` · `--baseline` era sensible al orden.** Un baseline correcto
  escrito en otro orden que `--models` abortaba el barrido. Ahora se reordena.
- **`src/ensemble4.py` · añadido `logreg_l1`** con `--meta_l1_c`. Con
  `--feature_space logit --drop_redundant_class` en tarea binaria hay una columna por modelo,
  así que L1 sobre ellas es *group-lasso* sobre modelos: coeficiente 0 = modelo apagado.
  **Escrito pero no ejecutado**: perdió sentido al dar negativo el enrutado.

---

## 9. Limitaciones

- **Dos órganos.** Los resultados acotan mama (`cptac_brca`, `bc_therapy`) y colon
  (`cptac_coad`). PathBench observa los saltos de ranking mayores entre órganos más
  distantes —pulmón, cerebro, estómago—. Con `cptac_gbm` hay 3 tareas de cerebro, pero el
  núcleo común de modelos cae a 3 y el experimento pierde potencia. **No está descartado que
  el enrutado pague con más órganos y mayor separación entre ellos.**
- **Cuatro modelos comunes** en las pruebas 2 y 3, frente a los 8 de la prueba 1.
- **Meta-features in-sample.** Los modelos base valen AUC 0.934–0.944 en meta-train frente a
  0.764–0.795 en test. Las comparaciones son relativas y aguantan, pero cualquier selección
  *aprendida* debería repetirse sobre features out-of-fold, que existen sólo para 3 de los 8
  modelos.
- **MDE ≈ ±0.016** con 50 folds y 22 slides de test por fold. Efectos menores que eso son
  indetectables por diseño; los negativos de este informe se reportan acotados, no como
  ausencia de efecto.

---

## 10. Reproducción

```bash
# Prueba 1 — los 255 subconjuntos de 8 modelos
python src/sweep_base_models.py \
    --work_dir /home/JKP6679/Patho-Ensemble/PARADIS/datos/patches \
    --train_source cptac_brca --task_name TP53_mutation \
    --tissue_patching 20x_224px_0px_overlap \
    --models conch_v1_5 ctranspath hoptimus1 phikon_v2 uni_v1 uni_v2 virchow2 virchow_v1 \
    --baseline ctranspath uni_v2 conch_v1_5 \
    --out results_sweep_8models_cptac_brca_TP53_mutation.json
```

Pruebas 2 y 3: `scripts/routing_ceiling.py`. Prueba 4: `scripts/selector_search.py`
(cachea los 255×50 ajustes en `selector_cache.npz`; el primer pase tarda ~3 min).

**Salidas**: `results_sweep_8models_cptac_brca_TP53_mutation.json` (121 KB, incluye el
ranking completo de los 255 subconjuntos con Δ pareados e IC95 %).
