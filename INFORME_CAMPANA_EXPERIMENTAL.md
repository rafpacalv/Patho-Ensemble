# Campaña experimental Patho-Ensemble: diversificación de ensembles y meta-clasificación

> **⚠️ CORRECCIÓN (2026-08-21) — los valores absolutos están obsoletos.** Se
> calcularon antes de regenerarse las predicciones base de
> `cptac_brca/TP53_mutation` (21-ago 19:26–19:28). El baseline **0.7918 no
> reproduce**; el vigente es **0.7616**. Las comparaciones relativas entre
> meta-modelos siguen valiendo. Y la conclusión de fondo cambia de sitio: toda
> esta campaña optimiza la *regla de combinación* (mejor resultado: +0.0052)
> sobre un trío base que no se revisó; cambiar `virchow_v1` por `conch_v1_5`
> vale **+0.037**. Ver
> [INFORME_SELECCION_MODELOS_BASE_20260821.md](INFORME_SELECCION_MODELOS_BASE_20260821.md).

**Dataset principal:** `cptac_brca / TP53_mutation` (50 folds)
**Dataset exploratorio:** `cervical_subtype / subtype` (5 folds)
**Periodo:** agosto 2026 · **Última actualización:** 17 de agosto de 2026

Este documento consolida **toda** la campaña experimental: la ablación de variantes FGE
sobre los modelos base, la extensión a Snapshot Ensembles, el diagnóstico de las
meta-features y el barrido de meta-clasificadores. Sustituye a la lectura de los logs de
SLURM como fuente de referencia.

El detalle fold a fold de la primera ablación se conserva en
[INFORME_ABLATION_FGE_cervical_subtype.md](INFORME_ABLATION_FGE_cervical_subtype.md), que
este informe resume y continúa.

---

## Conclusión

**Sobre `cptac_brca / TP53_mutation`, la regresión logística sobre probabilidades
in-sample es óptima dentro de lo estadísticamente detectable.** Ninguna de las 39
configuraciones evaluadas la supera en ninguna de las cinco métricas tras corrección de
Holm.

El resultado no es «no encontramos nada»: está **acotado**. Los intervalos de confianza
pareados sobre 50 folds sitúan el efecto mínimo detectable en ≈0.016 de AUC, y el mejor
contendiente (`m4_logreg_fge_lowlr`) queda en +0.0060 con p = 0.291. Si alguna de estas
técnicas aportara una mejora apreciable, se habría visto.

Se probaron cuatro ejes de mejora, y los cuatro están cerrados con explicación mecánica:

| Eje | Qué se probó | Resultado | Por qué |
|---|---|---|---|
| **Diversidad de snapshots** | 4 variantes FGE + 2 SE sobre modelos base | Nulo acotado (±0.017 AUC) | Saturación: los snapshots correlacionan r = 0.92–0.98, muy por encima del suelo inter-modelo r = 0.797. El presupuesto de diversidad ya lo gasta el eje multi-modelo. |
| **Capacidad del meta-learner** | TabPFN, LightGBM, MLP, MLP+SE, MLP+FGE, deep ensemble, gating | Degrada, monótonamente con la capacidad | 75 muestras de meta-train sobre una matriz de rango efectivo 4. No hay estructura que aprender más allá de una combinación lineal. |
| **Calidad de las meta-features** | Features out-of-fold con CV anidada, en 3 tamaños | Degrada (contraintuitivo) | El coste muestral de retener datos supera al sesgo in-sample que elimina. Verificado con `n_inner` ∈ {3, 5, 10}. |
| **Espacio de combinación** | log-odds, eliminación de columna redundante, media ponderada de 3 parámetros | Empate exacto | `logit_avg` iguala a la logreg con 3 parámetros en vez de 7 (MDE 0.008). |

La lectura para la tesis: **el techo no está en el meta-clasificador, está en el
presupuesto muestral.** Con 75 slides de meta-entrenamiento y 22 de test por fold, la
combinación lineal de tres modelos agota lo que los datos soportan.

---

## Mapa de la campaña

| # | Fase | Jobs SLURM | Configs | Coste GPU | Estado |
|---|---|---|---:|---:|---|
| 1 | Ablación FGE — `cervical_subtype` | 69936 | 8 | ~1 h | ✅ |
| 2 | Réplica FGE — `cptac_brca` 50 folds | 69938–69943 | 10 | ~4 h | ✅ |
| 3 | Snapshot Ensemble en modelos base | 69951, 69952 | 2 variantes | ~2 h | ✅ |
| 4 | Fase 0 — features out-of-fold (`n_inner=3`) | 69950 | — | 39 min | ✅ |
| 5 | Barrido de meta-clasificadores | 69953 | 31 | 25 min | ✅ (superado por #7) |
| 6 | Test discriminante de `n_inner` (5 y 10) | 69956, 69957 | — | 4 h 17 min | ✅ |
| 7 | **Barrido final consolidado** | **69958** | **39** | **28 min** | ✅ |

El job **69958** es la corrida de referencia: reevalúa las 39 configuraciones sobre los
mismos 50 folds, condición necesaria para que el análisis pareado sea válido.

---

## Configuración experimental

### Datasets

| | `cptac_brca` | `cervical_subtype` |
|---|---|---|
| Tarea | `TP53_mutation` (binaria) | `subtype` (4 clases) |
| Folds | **50** | 5 |
| Slides de meta-train por fold | 75 | ~180 |
| Slides de test por fold | 22 | ~45 |
| Papel | Corrida con potencia real | Exploración |

La diferencia de potencia estadística entre ambos es el hilo conductor de toda la
campaña: lo que parecía un efecto en `cervical_subtype` con 5 folds no sobrevivió a los
50 folds de `cptac_brca`.

### Modelos base

Tres ABMIL entrenados de forma independiente sobre embeddings de tres modelos
fundacionales, con `20x_224px_0px_overlap`:

| Modelo | AUC en test (media, 50 folds) |
|---|---:|
| `ctranspath` | 0.7948 |
| `uni_v2` | 0.7658 |
| `virchow_v1` | 0.7639 |

Correlación media entre sus predicciones de test: **r = 0.797**. Este número es el suelo
de referencia de toda la sección de diversidad.

### Protocolo de análisis

Todas las configuraciones se evalúan sobre los **mismos** folds, así que los contrastes
son **pareados**. Cada tabla reporta:

- **Δ medio** frente al baseline, restado fold a fold.
- **IC 95 % pareado** — no intervalos independientes, que en la primera ablación
  ocultaron un efecto sistemático.
- **G/E/P** — folds ganados / empatados / perdidos. Los empates se reportan
  explícitamente: con 22 slides de test por fold, muchas configuraciones producen la
  misma matriz de confusión, y en kappa los empates llegaron a 22 de 50.
- **p Holm** — corrección por comparaciones múltiples dentro de cada métrica.
- **MDE** — efecto mínimo detectable al 80 % de potencia, `(1.96 + 0.842)·sd/√n`. Es lo
  que convierte un resultado nulo en un resultado **acotado**.

Este protocolo se adoptó tras dos fallos concretos documentados en la primera ablación:
comparar IC independientes ocultó un efecto real, y un «5 de 5 folds, p = 0.024» con
n = 5 resultó ser un falso positivo que no replicó.

---

## Nomenclatura de las configuraciones

Los nombres de las 39 configuraciones son sistemáticos, no arbitrarios. La fuente
autoritativa es `experiments.yaml`; esto es su lectura.

### Anatomía de un nombre

```
m4_logreg_fge_lowlr
│  │      └────────── fuente de los modelos base  (base_source)
│  └───────────────── meta-clasificador           (meta_model)
└──────────────────── bloque experimental + índice

f15_tabpfn_oof
│   │      └───────── régimen de meta-features    (meta_features)
│   └──────────────── meta-clasificador
└──────────────────── bloque experimental + índice
```

Cada configuración varía **un solo eje** respecto de su pareja, para que el contraste
pareado aísle ese eje. Por eso los nombres van casi siempre en parejas `_ins` / `_oof`.

### Prefijos de bloque

| Prefijo | Bloque | Qué eje varía | Configs |
|---|---|---|---|
| `m` | Ablación de snapshots FGE en modelos base | `base_source` | m0–m9 |
| `s` | Snapshot Ensembles en modelos base | `base_source` | s0–s1 |
| `f` | Meta-clasificadores × régimen de features | `meta_model`, `meta_features` | f0–f18 |
| `g` | Coste muestral de la CV interna | `n_inner` de las features | g0–g7 |

`m0_logreg_std` y `f0_logreg_ins` son **la misma configuración**, evaluada como baseline
en ambos bloques. Igual ocurre con `m1_tabpfn_std` y `f14_tabpfn_ins`.

### Meta-clasificadores (`meta_model`)

Implementados en `src/meta_models.py` salvo los dos externos.

| Etiqueta | Clase | Qué hace |
|---|---|---|
| `logreg` | `LogisticRegression` (sklearn) | **Baseline.** 7 parámetros sobre las 6 columnas de features |
| `logit_avg` | `LogitAveragingMetaClassifier` | Un peso por modelo base sobre log-odds, con softmax sobre los pesos y temperatura aprendida. **3 parámetros** |
| `mlp` | `MLPMetaClassifier` | MLP pequeña, un solo ciclo de entrenamiento |
| `mlp_snapshot` | `SnapshotMLPMetaClassifier` | MLP + Snapshot Ensemble: coseno con reinicios, promedio de snapshots |
| `mlp_fge` | `FGEMLPMetaClassifier` | MLP + FGE: ciclos piecewise-linear. Implementado desde hacía tiempo pero **nunca ejecutado** antes de esta campaña |
| `mlp_deepens` | `DeepEnsembleMLPMetaClassifier` | 6 MLP con semillas independientes, promediando softmax. Es el **control científico** de `mlp_snapshot` |
| `gating` | `GatingMLPMetaClassifier` | Mixture-of-experts: una cabeza emite 3 pesos por muestra y combina las salidas base |
| `tabpfn` | TabPFN 8.1.0 | Transformer preentrenado para datos tabulares pequeños |
| `lightgbm` | `LGBMClassifier` 4.7.0 | Gradient boosting. Referencia tabular fuerte |

### Fuente de los modelos base (`base_source`)

Determina de qué predicciones base se construyen las features. Los hiperparámetros
completos están en la tabla de [A.1](#a1-primera-ablación-cervical_subtype-5-folds) y en
`experiments.yaml`.

| Etiqueta | Significado |
|---|---|
| `std` / `standard` | Modelo convergido, sin snapshots |
| `fge_orig` | FGE de referencia: lr₁ 2e-4, ciclos de 4 épocas |
| `fge_wbase` | Igual, pero el modelo convergido **entra en el promedio** (`include_base: true`) |
| `fge_lowlr` | Ciclos más suaves y largos: lr₁ 5e-5, 8 épocas |
| `fge_pat3` | Igual que `fge_orig` con paciencia inter-ciclo 3 en vez de 2 |
| `se_orig` | Snapshot Ensemble canónico: reinicio agresivo a lr 1e-3, ciclos de 15 épocas |
| `se_mild` | Mismo *schedule* coseno pero reiniciando a 2e-4, el techo de `fge_orig`. Aísla la **forma** del ciclo de su **amplitud** |

### Régimen de meta-features (`meta_features`)

Determina cómo se generaron las predicciones sobre las que aprende el meta-learner.

| Etiqueta | Sufijo | Origen | Train interno | AUC de las features |
|---|---|---|---:|---:|
| `insample` | `_ins` | `test_abmil.py` — el modelo predice sobre su **propio** train | 75 de 75 | 0.934–0.944 |
| `oof` | `_oof` | `build_oof_features.py`, `n_inner=3` | 50 de 75 | 0.699 |
| `oof5` | `_oof5` | ídem, `n_inner=5` | 60 de 75 | 0.710 |
| `oof10` | `_oof10` | ídem, `n_inner=10` | 67 de 75 | 0.719 |

Referencia: los modelos base valen **0.775** de AUC en test, entrenados con las 75.

### Transformaciones de las features

Sólo aparecen en `f2` y `f3`, marcadas con `Logit` en el nombre. Ambas se aplican juntas:

| Campo | Efecto |
|---|---|
| `feature_space: logit` | Transforma las probabilidades a log-odds, `log(p/(1−p))` con recorte a [1e-6, 1−1e-6] |
| `drop_redundant_class: true` | Elimina una columna softmax por modelo base, corrigiendo el rango deficiente (6 columnas → 4 dimensiones efectivas) |

### Ejemplos leídos

| Etiqueta | Lectura |
|---|---|
| `m0_logreg_std` | Logreg sobre bases convergidas, features in-sample. **El baseline de todo el informe** |
| `m4_logreg_fge_lowlr` | Logreg sobre bases diversificadas con FGE de ciclos suaves |
| `s0_logreg_se_orig` | Logreg sobre bases diversificadas con Snapshot Ensemble agresivo |
| `f3_logregLogit_oof` | Logreg en espacio de log-odds y sin columna redundante, con features out-of-fold |
| `f12_mlpdeepens_ins` | Seis MLP con semillas distintas, promediadas, sobre features in-sample |
| `g3_tabpfn_oof10` | TabPFN sobre features out-of-fold generadas con CV interna de 10 folds |

---

## Bloque A — Diversificación de los modelos base

### A.1 Primera ablación: `cervical_subtype` (5 folds)

Cuatro variantes FGE, todas partiendo del mismo checkpoint convergido:

| Variante | lr₁ | lr₂ | Long. ciclo | Ciclos | Paciencia | Base incluido |
|---|---|---|---:|---:|---:|:---:|
| `fge_orig` | 2e-4 | 2e-5 | 4 | 6 | 2 | no |
| `fge_wbase` | 2e-4 | 2e-5 | 4 | 6 | 2 | **sí** |
| `fge_lowlr` | 5e-5 | 5e-6 | 8 | 6 | 2 | no |
| `fge_pat3` | 2e-4 | 2e-5 | 4 | 6 | **3** | no |

**Resultado aparente:** FGE degradaba el AUC de forma sistemática (`fge_wbase` 0/5 folds,
p = 0.009) pero mejoraba la kappa ponderada (`fge_lowlr` +0.0477, 5/5 folds, p = 0.024).

Dos hallazgos secundarios que condicionaron la interpretación:

- **La ventaja en kappa dependía del esquema de ponderación.** Recalculada con los tres:
  sin ponderar +0.0149 (3/5, p = 0.316), lineal +0.0286 (5/5, p = 0.005), cuadrática
  +0.0477 (5/5, p = 0.024). La ventaja crece con la distancia y desaparece sin ponderar.
  Como la clase 3 es «otros», el supuesto ordinal sólo se cumple a medias.
- **Con 5 folds, un 5/5 no es evidencia fuerte.** Ese fue el motivo para replicar.

### A.2 Réplica: `cptac_brca` (50 folds)

**El hallazgo no replica.** En kappa, `fge_lowlr` queda en +0.0066 (p = 0.703) frente al
+0.0477 de cervical. El signo del efecto sobre el AUC además se **invierte** entre ambos
datasets, lo que es propio del ruido y no de un efecto real.

### A.3 Snapshot Ensembles

Se generalizó la ruta FGE existente añadiendo el coseno con reinicios de Huang et al.
(`se_cycle_lr` en `src/fge_utils.py`), con dos variantes:

| Variante | schedule | restart_lr | Long. ciclo | Propósito |
|---|---|---|---:|---|
| `se_orig` | se | 1e-3 | 15 | SE canónico: reinicios agresivos |
| `se_mild` | se | 2e-4 | 15 | Reinicio al nivel de `fge_orig`, para aislar el efecto del *schedule* del de la amplitud |

Resultado sobre AUC: `se_orig` +0.0022 (p = 0.785), `se_mild` −0.0097 (p = 0.281).
Igual que FGE: nulo acotado.

### A.4 La explicación mecánica: saturación de diversidad

Correlación media entre snapshots del mismo modelo, calculada sobre
`preds_per_snapshot.npy` en los 50 folds de test:

| Variante | ctranspath | uni_v2 | virchow_v1 | ΔAUC del ensemble |
|---|---:|---:|---:|---:|
| `fge_lowlr` | 0.983 | 0.980 | 0.976 | **+0.0060** |
| `se_mild` | 0.967 | 0.969 | 0.961 | −0.0097 |
| `fge_orig` | 0.962 | 0.961 | 0.960 | +0.0032 |
| `fge_pat3` | 0.959 | 0.959 | 0.952 | +0.0051 |
| `se_orig` | 0.935 | 0.941 | 0.940 | +0.0022 |
| `fge_wbase` | 0.941 | 0.922 | 0.921 | +0.0011 |
| **suelo inter-modelo** | **0.797** | | | — |

Dos lecturas, ambas relevantes:

1. **Ninguna variante se acerca al suelo inter-modelo.** El rango completo de snapshots
   (0.92–0.98) está muy por encima del 0.797 que ya aportan tres modelos fundacionales
   distintos. SE reinicia la LR cinco veces más arriba que FGE y sólo baja la correlación
   de 0.96 a 0.94.
2. **Diversidad y rendimiento no correlacionan.** `fge_lowlr` es la variante **menos**
   diversa (r = 0.98) y la mejor en AUC; `fge_wbase` es la más diversa (r = 0.92) y la
   penúltima. Con las diferencias dentro del ruido esto no es concluyente por sí solo,
   pero descarta la hipótesis de que bastara con forzar más diversidad.

El early stopping inter-ciclo corta además en **2.7–3.0 snapshots de media** sin llegar
nunca a los 6 ciclos configurados: los snapshots posteriores son peores que el modelo
convergido.

**Conclusión del bloque A:** aplicar FGE o SE a los modelos base antes del ensamblado no
mejora el rendimiento, y la causa medida es que el eje de diversidad ya está saturado por
el eje multi-modelo.

---

## Bloque B — El meta-clasificador

### B.1 El diagnóstico: las meta-features eran in-sample

Al preparar el bloque B se encontró que `test_abmil.py` genera las features del
meta-learner prediciendo con el modelo de cada fold sobre su **propio split de train**.
AUC de los modelos base sobre cada split, medido:

| Split | Papel | AUC |
|---|---|---:|
| `_train_eval/val_outputs` | meta-train | **0.934–0.944** |
| `val_outputs` | meta-val (early stopping) | 0.795–0.859 |
| `test_outputs` | meta-test | 0.764–0.795 |

El meta-learner aprende a combinar señales que valen 0.94 y se aplica donde valen 0.77.
Es el fallo clásico del stacking que Wolpert (1992) y Breiman (1996) resuelven exigiendo
predicciones out-of-fold, y penaliza sobre todo a los meta-modelos flexibles.

Un problema estructural adicional: cada vector softmax suma 1, así que la matriz de
features tiene **6 columnas pero rango 4**. Con 75 muestras de meta-train, la logreg
ajusta 7 parámetros sobre 4 dimensiones efectivas.

### B.2 Fase 0: features out-of-fold

`src/build_oof_features.py` genera, para cada fold externo, predicciones de los modelos
base sobre su propio train mediante CV anidada **agrupada por paciente**
(`StratifiedGroupKFold` sobre `case_id`), de modo que ninguna muestra recibe la
predicción de un modelo que la haya visto.

Verificación bloqueante superada: cero fugas de paciente en los 50 folds, y el AUC de
meta-train cae de 0.94 al rango esperado.

### B.3 Los siete meta-clasificadores × dos regímenes

Baseline: `f0_logreg_ins` = logreg sobre probabilidades in-sample, AUC 0.7918.

| Meta-modelo | Parámetros | AUC in-sample | AUC out-of-fold | Δ del régimen |
|---|---|---:|---:|---:|
| `logit_avg` | 3 + temperatura | 0.7921 | 0.7932 | +0.0011 |
| `logreg` | 7 | **0.7918** | 0.7826 | −0.0092 |
| `mlp_snapshot` (SE) | MLP + SE | 0.7936 | 0.7851 | −0.0086 |
| `tabpfn` | no paramétrico | 0.7842 | 0.7350 | −0.0492 |
| `lightgbm` | boosting | 0.7622 | 0.6270 | −0.1352 |
| `mlp_deepens` | 6 × MLP | 0.6690 | 0.6042 | −0.0648 |
| `mlp_fge` (FGE) | MLP + FGE | 0.6647 | 0.5679 | −0.0968 |
| `mlp` | MLP | 0.6563 | 0.5635 | −0.0928 |
| `gating` | mixture-of-experts | — | 0.7899 | — |

Cinco hallazgos, todos medidos:

1. **La corrección out-of-fold empeora todo salvo `logit_avg`**, y la degradación escala
   con la capacidad del modelo. Es lo contrario de lo que predecía el diagnóstico.
2. **La MLP colapsa en ambos regímenes** (0.656 in-sample, 0.564 OOF). Las features
   in-sample **no** eran la causa de su sobreajuste.
3. **SE sobre el meta-learner regulariza, no ensambla.** `mlp_snapshot` (0.7936) recupera
   exactamente el baseline, mientras que `mlp_deepens` (0.6690) —seis MLP con semillas
   independientes, el control científico correcto— queda 12 puntos por debajo. Si SE
   funcionara como ensemble, el deep ensemble debería igualarlo o superarlo. Lo que hace
   SE es devolver la MLP a una función tan simple como la regresión logística, no llevarla
   más allá.
4. **TabPFN nunca fue significativamente peor que la logreg** con features in-sample
   (−0.0076, p = 0.150). Su mal rendimiento previo era del régimen de features, no del
   modelo.
5. **`logit_avg` iguala a la logreg con 3 parámetros** (Δ +0.0003, MDE 0.008) y es
   indiferente al régimen de features. Es la opción preferible por interpretabilidad, y
   unifica conceptualmente con la media ponderada de `ensemble.py`.

### B.4 El test discriminante de `n_inner`

Si el daño out-of-fold fuera coste muestral, debería encogerse al aumentar el tamaño del
train interno. Se generaron features con `n_inner` ∈ {3, 5, 10}.

**Calidad de las features** (AUC de los modelos base sobre ellas):

| Train interno | ctranspath | uni_v2 | virchow_v1 | media |
|---|---:|---:|---:|---:|
| 50 de 75 (`n_inner=3`) | 0.7040 | 0.7097 | 0.6819 | 0.699 |
| 60 de 75 (`n_inner=5`) | 0.7303 | 0.7225 | 0.6778 | 0.710 |
| 67 de 75 (`n_inner=10`) | 0.7325 | 0.7254 | 0.6980 | 0.719 |
| **75 de 75 (modelo real)** | **0.7948** | **0.7658** | **0.7639** | **0.775** |

**Efecto sobre el meta-learner** (AUC):

| Meta-modelo | in-sample | oof (50) | oof5 (60) | oof10 (67) | recuperado |
|---|---:|---:|---:|---:|---:|
| `logit_avg` | 0.7921 | 0.7932 | 0.7914 | 0.7926 | sin daño |
| `logreg` | 0.7918 | 0.7826 | 0.7825 | 0.7872 | 50 % |
| `tabpfn` | 0.7842 | 0.7350 | 0.7518 | 0.7645 | 52 % |
| `lightgbm` | 0.7622 | 0.6270 | 0.6893 | 0.6869 | 36 %, estancado |

**La hipótesis del coste muestral resulta cierta pero insuficiente.** Cierta porque
`tabpfn` recupera la mitad de lo perdido con sólo 17 muestras más. Insuficiente porque no
basta para cruzar: ajustando la recta sobre los tres puntos, la pendiente de `tabpfn` es
≈0.0017 de AUC por muestra, que extrapolada a las 75 da **0.777**, todavía por debajo del
baseline. Ni con leave-one-out —75 × 50 × 3 = 11 250 entrenamientos, del orden de días de
GPU— llegaría.

`lightgbm` se aplana entre 60 y 67 muestras: su daño no es muestral, sino que boosting
sobre una matriz de 75 × 6 con rango 4 es el sesgo inductivo equivocado.

### B.5 El régimen out-of-fold rompe la calibración, no el orden

Un hallazgo que sólo aparece al comparar métricas entre sí. Para `logreg` con features
out-of-fold (`n_inner=3`):

| Métrica | Δ | p Holm | ¿Significativo? |
|---|---:|---:|---|
| AUC-ROC | −0.0092 | 1.000 | no |
| Balanced accuracy | −0.0540 | 0.028 | **sí** |
| Macro-F1 | −0.0627 | 0.042 | **sí** |
| Kappa | −0.1029 | 0.046 | **sí** |

El AUC es un estadístico de **rango** y apenas se mueve; las cuatro métricas que dependen
del umbral de decisión se desploman. La interpretación es directa: las features
out-of-fold provienen de modelos entrenados con 50 muestras y son **menos confiadas** que
las que el meta-learner encontrará en test, donde los modelos base se entrenaron con 75.
El meta-learner aprende un umbral calibrado para una distribución de confianza que no es
la que va a ver. El **orden** de las predicciones sobrevive; el **punto de corte** no.

Esto refuerza el resultado de B.4, porque la recuperación es monótona también aquí:
kappa pasa de −0.1029 (`n_inner=3`) a −0.0809 (5) y −0.0599 (10), a medida que la
confianza de las features se acerca a la real.

---

## Predicciones registradas y su desenlace

Las predicciones se anotaron **antes** de ejecutar, en
[SNAPSHOT_ENSEMBLE_ROADMAP.md](SNAPSHOT_ENSEMBLE_ROADMAP.md), para que el resultado fuera
informativo y no una racionalización posterior.

| # | Predicción | Desenlace |
|---|---|---|
| 1 | Con features OOF, `tabpfn`, `mlp` y `lightgbm` mejoran claramente; `logreg` se mueve poco. **Si TabPFN no mejora, el diagnóstico es falso.** | ❌ **Falsada.** Todos empeoran, y la degradación escala con la capacidad. |
| 2 | Con features OOF, la ventaja de `mlp_snapshot` sobre `mlp` se reduce o desaparece | ❌ **Falsada.** La ventaja *aumenta*: +0.137 in-sample → +0.222 OOF. |
| 3 | `mlp_deepens` ≥ `mlp_snapshot` | ❌ **Falsada, y es el hallazgo más informativo.** 0.6690 frente a 0.7936. SE actúa como regularizador, no como ensemble. |
| 4 | SE baja la correlación a ≈0.88–0.93, seguirá por encima de 0.797, y el AUC quedará dentro de ±0.017 | ◐ **Parcial.** La correlación quedó en 0.935–0.941, ligeramente por encima del rango previsto; el resto se cumplió exactamente. |
| 5 | `logit_avg` queda dentro de ±0.01 de la logreg | ✅ **Confirmada.** Δ +0.0003 con MDE 0.008. |

Tres de cinco predicciones falsadas. El valor metodológico está precisamente ahí: el
diagnóstico in-sample era correcto como descripción del problema y **equivocado** como
predicción de su remedio, y sin registro previo esa distinción se habría perdido.

---

## Tablas completas — job 69958, 50 folds

Baseline en todas: `f0_logreg_ins` (= `m0_logreg_std`).
`*` = significativo tras Holm · `~` = significativo sin corregir, no concluyente.

### Medias por métrica

| Configuración | AUC | Kappa | Macro-F1 | Bal. acc | Accuracy |
|---|---:|---:|---:|---:|---:|
| `f0_logreg_ins` *(baseline)* | 0.7918 | 0.4718 | 0.7335 | 0.7363 | 0.7472 |
| `m4_logreg_fge_lowlr` | **0.7978** | **0.4784** | **0.7369** | **0.7383** | **0.7510** |
| `m5_logreg_fge_pat3` | 0.7969 | 0.4616 | 0.7288 | 0.7312 | 0.7414 |
| `m2_logreg_fge_orig` | 0.7949 | 0.4676 | 0.7313 | 0.7338 | 0.7442 |
| `s0_logreg_se_orig` | 0.7940 | 0.4507 | 0.7239 | 0.7251 | 0.7381 |
| `f8_mlpsnap_ins` | 0.7936 | 0.3493 | 0.6458 | 0.6722 | 0.7038 |
| `f5_logitavg_oof` | 0.7932 | 0.4461 | 0.7136 | 0.7301 | 0.7236 |
| `m3_logreg_fge_wbase` | 0.7929 | 0.4671 | 0.7310 | 0.7334 | 0.7444 |
| `g7_logitavg_oof10` | 0.7926 | 0.4357 | 0.7088 | 0.7243 | 0.7195 |
| `f4_logitavg_ins` | 0.7921 | 0.4619 | 0.7224 | 0.7368 | 0.7336 |
| `g6_logitavg_oof5` | 0.7914 | 0.4467 | 0.7144 | 0.7300 | 0.7249 |
| `f18_gating_oof` | 0.7899 | 0.4424 | 0.7122 | 0.7280 | 0.7228 |
| `g1_logreg_oof10` | 0.7872 | 0.4119 | 0.6987 | 0.7022 | 0.7275 |
| `f2_logregLogit_ins` | 0.7864 | 0.4677 | 0.7313 | 0.7335 | 0.7454 |
| `m7_tabpfn_fge_lowlr` | 0.7855 | 0.4446 | 0.7184 | 0.7224 | 0.7328 |
| `f9_mlpsnap_oof` | 0.7851 | 0.2087 | 0.5448 | 0.6000 | 0.6696 |
| `m1_tabpfn_std` / `f14_tabpfn_ins` | 0.7842 | 0.4655 | 0.7301 | 0.7332 | 0.7424 |
| `f1_logreg_oof` | 0.7826 | 0.3689 | 0.6707 | 0.6823 | 0.7108 |
| `g0_logreg_oof5` | 0.7825 | 0.3909 | 0.6863 | 0.6923 | 0.7171 |
| `s1_logreg_se_mild` | 0.7821 | 0.4600 | 0.7281 | 0.7298 | 0.7418 |
| `m9_tabpfn_fge_pat3` | 0.7794 | 0.4537 | 0.7240 | 0.7273 | 0.7365 |
| `m8_tabpfn_fge_orig` | 0.7791 | 0.4537 | 0.7234 | 0.7269 | 0.7364 |
| `m6_tabpfn_fge_wbase` | 0.7781 | 0.4678 | 0.7309 | 0.7344 | 0.7426 |
| `f3_logregLogit_oof` | 0.7737 | 0.3400 | 0.6485 | 0.6676 | 0.7008 |
| `g3_tabpfn_oof10` | 0.7645 | 0.3815 | 0.6808 | 0.6891 | 0.7126 |
| `f16_lightgbm_ins` | 0.7622 | 0.3873 | 0.6902 | 0.6949 | 0.7048 |
| `g2_tabpfn_oof5` | 0.7518 | 0.3375 | 0.6469 | 0.6665 | 0.6887 |
| `f15_tabpfn_oof` | 0.7350 | 0.2984 | 0.6247 | 0.6485 | 0.6834 |
| `g4_lightgbm_oof5` | 0.6893 | 0.2647 | 0.6208 | 0.6317 | 0.6515 |
| `g5_lightgbm_oof10` | 0.6869 | 0.2542 | 0.6152 | 0.6282 | 0.6439 |
| `f12_mlpdeepens_ins` | 0.6690 | 0.0993 | 0.4436 | 0.5510 | 0.5922 |
| `f10_mlpfge_ins` | 0.6647 | 0.0519 | 0.4024 | 0.5252 | 0.5399 |
| `f6_mlp_ins` | 0.6563 | 0.0484 | 0.4005 | 0.5235 | 0.5379 |
| `f17_lightgbm_oof` | 0.6270 | 0.1609 | 0.5609 | 0.5807 | 0.5962 |
| `f13_mlpdeepens_oof` | 0.6042 | 0.0610 | 0.4163 | 0.5329 | 0.5789 |
| `f11_mlpfge_oof` | 0.5679 | 0.0202 | 0.3863 | 0.5102 | 0.5268 |
| `f7_mlp_oof` | 0.5635 | 0.0178 | 0.3822 | 0.5080 | 0.5229 |

### AUC-ROC — contrastes pareados

| Configuración | Δ | IC 95 % pareado | G/E/P | p Holm | MDE | |
|---|---:|---|---:|---:|---:|:--:|
| `m4_logreg_fge_lowlr` | +0.0060 | [−0.0050, +0.0171] | 25/4/21 | 1.000 | 0.0158 | |
| `m5_logreg_fge_pat3` | +0.0051 | [−0.0068, +0.0171] | 25/5/20 | 1.000 | 0.0171 | |
| `m2_logreg_fge_orig` | +0.0032 | [−0.0086, +0.0149] | 25/5/20 | 1.000 | 0.0168 | |
| `s0_logreg_se_orig` | +0.0022 | [−0.0136, +0.0180] | 26/2/22 | 1.000 | 0.0226 | |
| `f8_mlpsnap_ins` | +0.0018 | [−0.0037, +0.0073] | 19/13/18 | 1.000 | 0.0079 | |
| `f5_logitavg_oof` | +0.0014 | [−0.0048, +0.0077] | 20/12/18 | 1.000 | 0.0090 | |
| `m3_logreg_fge_wbase` | +0.0011 | [−0.0082, +0.0104] | 24/4/22 | 1.000 | 0.0133 | |
| `g7_logitavg_oof10` | +0.0009 | [−0.0054, +0.0071] | 17/13/20 | 1.000 | 0.0090 | |
| `f4_logitavg_ins` | +0.0003 | [−0.0054, +0.0060] | 17/14/19 | 1.000 | 0.0081 | |
| `g6_logitavg_oof5` | −0.0004 | [−0.0058, +0.0051] | 19/12/19 | 1.000 | 0.0078 | |
| `f18_gating_oof` | −0.0019 | [−0.0075, +0.0037] | 17/9/24 | 1.000 | 0.0080 | |
| `g1_logreg_oof10` | −0.0046 | [−0.0119, +0.0027] | 16/11/23 | 1.000 | 0.0105 | |
| `f2_logregLogit_ins` | −0.0054 | [−0.0143, +0.0036] | 19/7/24 | 1.000 | 0.0128 | |
| `m7_tabpfn_fge_lowlr` | −0.0063 | [−0.0243, +0.0116] | 21/1/28 | 1.000 | 0.0257 | |
| `f9_mlpsnap_oof` | −0.0067 | [−0.0190, +0.0055] | 19/10/21 | 1.000 | 0.0175 | |
| `m1_tabpfn_std` / `f14_tabpfn_ins` | −0.0076 | [−0.0177, +0.0026] | 20/4/26 | 1.000 | 0.0145 | |
| `f1_logreg_oof` | −0.0092 | [−0.0198, +0.0015] | 18/4/28 | 1.000 | 0.0152 | |
| `g0_logreg_oof5` | −0.0093 | [−0.0186, +0.0000] | 11/16/23 | 1.000 | 0.0133 | |
| `s1_logreg_se_mild` | −0.0097 | [−0.0271, +0.0077] | 26/0/24 | 1.000 | 0.0249 | |
| `m9_tabpfn_fge_pat3` | −0.0124 | [−0.0299, +0.0052] | 22/2/26 | 1.000 | 0.0251 | |
| `m8_tabpfn_fge_orig` | −0.0127 | [−0.0298, +0.0044] | 21/3/26 | 1.000 | 0.0245 | |
| `m6_tabpfn_fge_wbase` | −0.0137 | [−0.0309, +0.0035] | 19/7/24 | 1.000 | 0.0246 | |
| `f3_logregLogit_oof` | −0.0181 | [−0.0382, +0.0020] | 18/5/27 | 1.000 | 0.0287 | |
| `g3_tabpfn_oof10` | −0.0273 | [−0.0466, −0.0079] | 18/2/30 | 0.208 | 0.0276 | ~ |
| `f16_lightgbm_ins` | −0.0296 | [−0.0499, −0.0092] | 16/3/31 | 0.173 | 0.0291 | ~ |
| `g2_tabpfn_oof5` | −0.0400 | [−0.0655, −0.0145] | 14/2/34 | 0.095 | 0.0364 | ~ |
| `f15_tabpfn_oof` | −0.0568 | [−0.0856, −0.0280] | 12/2/36 | 0.009 | 0.0412 | * |
| `g4_lightgbm_oof5` | −0.1025 | [−0.1407, −0.0643] | 9/1/40 | 0.000 | 0.0546 | * |
| `g5_lightgbm_oof10` | −0.1048 | [−0.1364, −0.0733] | 9/1/40 | 0.000 | 0.0451 | * |
| `f12_mlpdeepens_ins` | −0.1227 | [−0.1806, −0.0649] | 13/1/36 | 0.004 | 0.0827 | * |
| `f10_mlpfge_ins` | −0.1271 | [−0.1772, −0.0769] | 9/3/38 | 0.000 | 0.0717 | * |
| `f6_mlp_ins` | −0.1354 | [−0.1875, −0.0834] | 10/1/39 | 0.000 | 0.0744 | * |
| `f17_lightgbm_oof` | −0.1648 | [−0.2132, −0.1164] | 7/1/42 | 0.000 | 0.0692 | * |
| `f13_mlpdeepens_oof` | −0.1876 | [−0.2517, −0.1235] | 7/2/41 | 0.000 | 0.0917 | * |
| `f11_mlpfge_oof` | −0.2238 | [−0.2909, −0.1567] | 7/0/43 | 0.000 | 0.0959 | * |
| `f7_mlp_oof` | −0.2283 | [−0.2954, −0.1611] | 7/0/43 | 0.000 | 0.0961 | * |

**Ninguna configuración supera al baseline con significación.** Las 12 significativas tras
Holm son todas degradaciones.

### Kappa — contrastes pareados (extracto)

Sólo las configuraciones con Δ significativo o cercano:

| Configuración | Δ | IC 95 % pareado | G/E/P | p Holm | MDE | |
|---|---:|---|---:|---:|---:|:--:|
| `m4_logreg_fge_lowlr` | +0.0066 | [−0.0273, +0.0405] | 15/15/20 | 1.000 | 0.0484 | |
| `g7_logitavg_oof10` | −0.0361 | [−0.0727, +0.0004] | 15/15/20 | 1.000 | 0.0523 | |
| `g1_logreg_oof10` | −0.0599 | [−0.1054, −0.0144] | 15/12/23 | 0.258 | 0.0650 | ~ |
| `g0_logreg_oof5` | −0.0809 | [−0.1313, −0.0306] | 12/11/27 | 0.058 | 0.0720 | ~ |
| `f16_lightgbm_ins` | −0.0845 | [−0.1263, −0.0426] | 14/4/32 | 0.006 | 0.0599 | * |
| `g3_tabpfn_oof10` | −0.0903 | [−0.1445, −0.0361] | 15/7/28 | 0.046 | 0.0775 | * |
| `f1_logreg_oof` | −0.1029 | [−0.1647, −0.0411] | 9/13/28 | 0.046 | 0.0884 | * |
| `f8_mlpsnap_ins` | −0.1225 | [−0.1823, −0.0627] | 12/9/29 | 0.005 | 0.0855 | * |
| `f3_logregLogit_oof` | −0.1318 | [−0.2017, −0.0618] | 10/8/32 | 0.013 | 0.1000 | * |
| `g2_tabpfn_oof5` | −0.1342 | [−0.1941, −0.0744] | 9/9/32 | 0.002 | 0.0856 | * |
| `f15_tabpfn_oof` | −0.1734 | [−0.2486, −0.0981] | 13/6/31 | 0.001 | 0.1075 | * |
| `g4_lightgbm_oof5` | −0.2070 | [−0.2837, −0.1304] | 8/3/39 | 0.000 | 0.1096 | * |
| `g5_lightgbm_oof10` | −0.2176 | [−0.2833, −0.1519] | 9/1/40 | 0.000 | 0.0939 | * |
| `f9_mlpsnap_oof` | −0.2631 | [−0.3432, −0.1829] | 6/2/42 | 0.000 | 0.1146 | * |
| `f17_lightgbm_oof` | −0.3109 | [−0.3959, −0.2258] | 5/3/42 | 0.000 | 0.1216 | * |
| `f6_mlp_ins` … `f11_mlpfge_oof` | −0.37 a −0.45 | — | ≤2 ganados | 0.000 | — | * |

Los MDE de kappa (0.05–0.12) son 5–10 veces mayores que los de AUC (0.008–0.03): con 22
slides de test por fold, kappa es una métrica mucho más ruidosa, y una diferencia por
debajo de ≈0.05 sencillamente no era detectable con 50 folds.

---

## Limitaciones

1. **Un solo dataset con potencia real.** Todas las conclusiones cuantitativas son de
   `cptac_brca / TP53_mutation`. `cervical_subtype` sirvió para mostrar que 5 folds
   producen falsos positivos, no para sostener conclusiones.
2. **Tres modelos base.** La saturación de diversidad se mide con r = 0.797 entre tres
   modelos fundacionales. Con más modelos, o con modelos más correlacionados entre sí, el
   balance snapshot↔multi-modelo podría cambiar.
3. **El presupuesto muestral es la restricción dominante y no se ha variado.** Todos los
   resultados son con 75 slides de meta-train. La conclusión «el techo es muestral»
   predice que con cohortes mayores la jerarquía cambiaría, y eso **no está medido**.
4. **`leave-one-out` no se ejecutó.** La extrapolación de `tabpfn` a 75 muestras es una
   recta sobre tres puntos, no una medición.
5. **TabM no se evaluó** — no está instalado. `mlp_deepens` cubre la misma idea
   (ensemble de MLP para tabular pequeño) sin la dependencia.
6. **Arquitectura ABMIL fija** en todos los experimentos (1 cabeza, dim 512, dropout 0.25,
   bag 2048). No se ablacionó.

---

## Índice de artefactos

### Documentos

| Fichero | Contenido |
|---|---|
| `INFORME_CAMPANA_EXPERIMENTAL.md` | **Este documento.** Campaña completa, fuente canónica. |
| `INFORME_CAMPANA_EXPERIMENTAL.html` | El mismo informe maquetado, autocontenido (sin dependencias externas). Se abre con doble clic; también publicado en `claude.ai/code/artifact/25eb867a-94a5-4c95-9b9c-41e62179277e`. |
| `INFORME_ABLATION_FGE_cervical_subtype.md` | Detalle fold a fold del bloque A, definición de kappa, discusión de AUC y desbalance |
| `SNAPSHOT_ENSEMBLE_ROADMAP.md` | Plan previo con las predicciones registradas |
| `PARADIS_DATA_REFERENCE.md` | Árbol de datos, datasets, tareas, número de folds |
| `experiments.yaml` | Matriz de experimentos; los comentarios registran por qué existe cada variante |

### Logs de SLURM (`logs/`)

| Job | Fichero | Contenido |
|---|---|---|
| 69936 | `run_meta_experiments_69936.out` | Ablación FGE cervical, 8 configs |
| 69938–69943 | `train_base_models_*.out`, `run_meta_experiments_69943.out` | Réplica brca |
| 69950 | `build_oof_69950.out` | Features OOF `n_inner=3` + verificación de fugas |
| 69951, 69952 | `train_base_models_fge_6995*.out` | `se_orig`, `se_mild` |
| 69953 | `run_meta_experiments_69953.out` | Primer barrido, 31 configs |
| 69956, 69957 | `build_oof_6995*.out` | Features OOF `n_inner` = 5 y 10 |
| **69958** | `run_meta_experiments_69958.out` | **Barrido final, 39 configs × 5 métricas** |

### Resultados numéricos

```
PARADIS/datos/patches/cptac_brca/TP53_mutation/abmil/ensemble4*/
```

Un directorio por configuración de meta-learner, con métricas por fold y agregadas.
Además, en la raíz del repositorio: `grid_search_results.json` y
`ablation_study_results.json` (barridos previos de la MLP, anteriores a esta campaña).

### Reproducción

```bash
# Fase 0 — features out-of-fold (parametrizable por n_inner)
N_INNER=10 OOF_TAG=oof10 DATASET=cptac_brca TASK=TP53_mutation \
    sbatch build_oof_features.sbatch

# Barrido de meta-clasificadores + análisis pareado
DATASET=cptac_brca TASK=TP53_mutation sbatch run_3_experiments_v2_improved.sbatch
```

---

## Referencias

- Wolpert, D. (1992). *Stacked generalization.* Neural Networks 5(2).
- Breiman, L. (1996). *Stacked regressions.* Machine Learning 24(1).
- Garipov, T. et al. (2018). *Loss surfaces, mode connectivity, and fast ensembling of DNNs.* ICLR — FGE.
- Huang, G. et al. (2017). *Snapshot ensembles: train 1, get M for free.* ICLR — SE.
- Davis, J. & Goadrich, M. (2006). *The relationship between precision-recall and ROC curves.* ICML.
- Saito, T. & Rehmsmeier, M. (2015). *The precision-recall plot is more informative than the ROC plot on imbalanced datasets.* PLoS ONE.