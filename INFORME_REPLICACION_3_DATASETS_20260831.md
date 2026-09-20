# Informe — Replicación en tres datasets

**Fecha:** 31 de agosto de 2026
**Modelos base:** `ctranspath + uni_v2 + conch_v1_5` (el trío óptimo), ABMIL independiente por modelo
**Jobs:** 70495 (`cptac_gbm`), 70496 (`bc_therapy`), 70497 (`cervical_subtype`), 70501-70502 (atención)

---

## Resumen ejecutivo

Se repitió la campaña completa —14 configuraciones de metaclasificador, comparación
pareada, corrección de Holm— sobre **tres datasets nuevos con más datos** que
`cptac_brca`. El resultado obliga a **revisar dos conclusiones** del informe del 22-ago.

| | |
|---|---|
| **Lo que se mantiene** | LogReg sigue sin ser batido *como regla de combinación*; el MLP simple colapsa en los cuatro datasets; promediar atenciones no aporta |
| **Lo que se INVIERTE** | Los embeddings **sí** baten a las probabilidades en `cptac_gbm` y `bc_therapy` (§8.1 no estaba cerrada: era un resultado de `cptac_brca`) |
| **Lo que se INVIERTE** | Los snapshot ensembles a nivel base **sí** funcionan en `cptac_gbm` (no eran un nulo, eran un nulo *sin potencia*) |
| **Lo que queda ABIERTO** | **Por qué** depende del dataset. No es el tamaño del train (§2.2, falsado); el candidato es la selectividad de la atención (§2.3) |
| **Lo que NO replica** | La transferencia de atención de conch, que en `cptac_brca` apuntaba a +0.0145 de AP |

**La lectura que ordena todo:** el signo de estos resultados **depende del
dataset, y no del tamaño del conjunto de entrenamiento**. La explicación
intuitiva —«con más slides el espacio de 1536 D deja de sobreajustar»— se
sometió a prueba directa (§2.2) y **quedó falsada**: dentro de `cptac_gbm`, los
embeddings ganan a *todos* los tamaños de train, incluido n=40. Lo que cambia
entre datasets es otra cosa, y §2.3 propone un candidato con evidencia.

> **Nota de revisión (1-sep).** La primera versión de este informe atribuía el
> vuelco al tamaño del train. El experimento de submuestreo del §2.2 refutó esa
> explicación. Las mediciones no han cambiado; la interpretación sí.

---

## 1. Los cuatro conjuntos

| Dataset / tarea | Slides | Pacientes | Folds | Clases | Train/fold | Test/fold | Positivos |
|---|---:|---:|---:|---:|---:|---:|---:|
| `cptac_brca/TP53_mutation` | 112 | 103 | 50 | 2 | **75** | 21.5 | 40.2 % |
| `bc_therapy/er_status` | 166 | 166 | 50 | 2 | **113** | 33 | 69.3 % |
| `cptac_gbm/TP53_mutation` | 243 | 99 | 50 | 2 | **164** | 45.3 | 33.7 % |
| `cervical_subtype/subtype` | 599 | 599 | **5** | **4** | **407** | 119.8 | — |

Descartados por no tener el trío completo o ser más pequeños: `cptac_ccrcc`,
`cptac_hnsc`, `cptac_lscc` (sólo `uni_v2`), `tcga_brca` (sin `uni_v2`),
`hancook` y `crc_outcomes` (sin features), `cptac_coad` (98 slides), `imp`
(un solo fold).

### 1.1 Qué es un fold aquí — y por qué importa

**Los 50 folds no son 50 experimentos independientes.** Medido: la suma de los
tamaños de test es **9.6 veces** el número de slides, y cada slide cae en test
entre 3 y 17 veces. No es un k-fold clásico sino **remuestreo aleatorio repetido
y solapado**. `cervical_subtype`, con 5 folds, sí es una partición real (cada
slide en test exactamente una vez).

| Dataset | Folds | Apariciones en test por slide | Diseño |
|---|---:|---|---|
| `cptac_brca` | 50 | 9.59 (3–16) | remuestreo repetido |
| `cptac_gbm` | 50 | 9.32 (2–17) | remuestreo repetido |
| `bc_therapy` | 50 | 9.94 (4–17) | remuestreo repetido |
| `cervical_subtype` | 5 | **1.00** | partición real |

**Consecuencia, y es una limitación que ningún informe anterior menciona:** los
IC por bootstrap sobre folds **subestiman la incertidumbre** de generalizar a
pacientes nuevos, porque los folds comparten datos masivamente. La unidad
realmente independiente es el **paciente**, no el fold. Los contrastes pareados
sufren menos —buena parte de la varianza compartida se cancela en la
diferencia— pero un IC de este informe es más estrecho de lo que sería con 50
muestras independientes.

---

## 2. El resultado central: §8.1 se invierte

`emb:logreg` vs. `prob:logreg` — el metaclasificador sobre embeddings de 1536 D
frente al mismo metaclasificador sobre 6 probabilidades. **Misma comparación,
mismo combinador, sólo cambia la entrada.**

| Dataset | Train/fold | Δ AUC | IC 95 % | G/E/P | p Holm |
|---|---:|---:|---|---|---:|
| `cptac_brca` | 75 | **−0.0160** | [−0.0305, −0.0020] | 19/2/29 | 0.60 |
| `bc_therapy` | 113 | **+0.0296** | [+0.0073, +0.0530] | 30/1/19 | 0.35 |
| `cptac_gbm` | 164 | **+0.0230** | [+0.0079, +0.0384] | 34/0/16 | **0.031** ✱ |
| `cervical_subtype` | 407 | −0.0009 | [−0.0042, +0.0029] | 2/0/3 | 1.00 |

Y las cinco métricas se mueven **en el mismo sentido dentro de cada dataset**,
que es el criterio que el informe anterior fijó para distinguir un efecto real
de un artefacto de métrica:

| Dataset | AUC | AP | bacc | macro-F1 | kappa |
|---|---:|---:|---:|---:|---:|
| `cptac_brca` (75) | −0.0160 | −0.0205 | −0.0123 | −0.0148 | −0.0297 |
| `bc_therapy` (113) | +0.0296 | +0.0203 | +0.0163 | +0.0215 | +0.0265 |
| `cptac_gbm` (164) | +0.0230✱ | +0.0235 | +0.0117 | +0.0127 | +0.0275 |
| `cervical` (407) | −0.0009 | −0.0056 | −0.0168 | −0.0129 | −0.0215 |

Confirmación por una vía independiente: en `cptac_gbm`, la sonda de atención
—que también trabaja en espacio de embeddings— da `attn_own:logreg` = 0.7787
frente a `prob:logreg` = 0.7556, **Δ = +0.0231, p Holm = 0.030**. Dos rutas de
código distintas, el mismo hallazgo.

**El efecto es real y replicado. Lo que falta es la causa** — y la explicación
intuitiva resultó ser falsa.

### 2.2 El submuestreo falsa la explicación por tamaño del train

La lectura evidente era: «una logreg sobre 1536 columnas con 75 filas
sobreajusta; con 164 no». Se sometió a prueba **dentro de `cptac_gbm`**,
recortando las filas de entrenamiento del metaclasificador y dejando fijo todo
lo demás —modelos base, folds, val, test—. Cinco submuestreos estratificados por
fold, promediados (job 70614).

| n meta-train | `prob:logreg` | `emb:logreg` | Δ AUC | IC 95 % | G/E/P | p Holm |
|---:|---:|---:|---:|---|---|---:|
| 40 | 0.7557 | 0.7610 | **+0.0053** | [−0.0037, +0.0138] | 28/0/22 | 0.86 |
| 55 | 0.7555 | 0.7724 | **+0.0169** | [+0.0072, +0.0269] | 31/0/19 | **0.013** ✱ |
| **75** | 0.7558 | 0.7694 | **+0.0137** | [+0.0018, +0.0254] | 34/0/16 | 0.090 |
| 95 | 0.7545 | 0.7746 | **+0.0201** | [+0.0088, +0.0317] | 34/0/16 | **0.008** ✱ |
| 120 | 0.7548 | 0.7718 | **+0.0170** | [+0.0042, +0.0297] | 33/0/17 | **0.047** ✱ |
| 145 | 0.7560 | 0.7758 | **+0.0198** | [+0.0057, +0.0335] | 32/1/17 | **0.030** ✱ |
| 169 (completo) | 0.7556 | 0.7786 | **+0.0230** | [+0.0077, +0.0380] | 34/0/16 | **0.013** ✱ |

**El Δ es positivo en los siete tamaños, incluido n=40** — muy por debajo de los
75 de `cptac_brca`. Y en el punto decisivo, **n=75 exactamente**, `cptac_gbm` da
**+0.0137 ganando 34 de 50 folds**, mientras `cptac_brca` da **−0.0160 ganando
19 de 50**. Mismo tamaño de train, mismo código, signos opuestos.

**Conclusión: el tamaño del entrenamiento no explica el vuelco.** Sí modula la
magnitud —el Δ crece de +0.0053 a +0.0230 al pasar de 40 a 169 filas— pero
**nunca cambia el signo**. La diferencia es una propiedad del dataset.

### 2.3 Un candidato con evidencia: la selectividad de la atención

Un embedding es `z = Σ (a · proj(h))`. Si la atención `a` es plana, `z` es poco
más que la media de los parches; si es selectiva, `z` concentra el tejido que el
ABMIL considera informativo. **La calidad del embedding debería depender de lo
selectiva que sea la atención que lo produjo** — y ahí los dos datasets difieren
mucho:

| Modelo | entropía brca | entropía gbm | logit_sd brca | logit_sd gbm | retiene brca | retiene gbm |
|---|---:|---:|---:|---:|---:|---:|
| ctranspath | 0.9282 | **0.7751** | 0.98 | **1.74** | 66 % | **78 %** |
| uni_v2 | 0.9942 | **0.9699** | 0.30 | **0.70** | 66 % | **79 %** |
| conch_v1_5 | 0.8879 | 0.8736 | 1.29 | 1.23 | 58 % | 58 % |

*(entropía 1.0 = uniforme. En `cptac_gbm` los ABMIL atienden de forma
sensiblemente más selectiva y sus proyecciones retienen más estructura.)*

La predicción que se sigue: **si se aplana la atención, los embeddings deberían
perder su ventaja sobre las probabilidades.** Está medido, con el régimen
`attn_mean` de la sonda:

| Dataset | `attn_own` vs `prob` | `attn_mean` vs `prob` | Vuelco |
|---|---:|---:|---|
| `cptac_gbm` | **+0.0231** ✱ | **−0.0249** | **+0.048 de swing** |
| `cptac_brca` | −0.0101 | **−0.0434** ✱ | −0.033 |

En `cptac_gbm`, aplanar la atención **invierte el signo**: los embeddings pasan
de batir a las probabilidades por +0.023 a perder por −0.025. En `cptac_brca`,
donde la atención ya era casi plana de partida, aplanarla del todo sólo
profundiza una pérdida que ya existía.

> ⚠️ **Esto es una hipótesis compatible con los datos, no un resultado
> establecido.** La evidencia son dos datasets y una única manipulación (el
> extremo uniforme). El contraste decisivo sería **graduar** la selectividad de
> la atención —no sólo llevarla al extremo plano— y comprobar si el Δ
> embeddings-probabilidades la sigue de forma monótona. No está hecho.
### 2.4 Por qué `cervical_subtype` no arbitra

Con 407 slides de entrenamiento debería ser el caso más favorable a los
embeddings, y sale ligeramente negativo. **No se puede usar para decidir**,
porque difiere del resto en cuatro cosas a la vez: sólo **5 folds** (el propio
proyecto documenta un hallazgo a 5 folds que no replicó y cambió de signo), **4
clases** en vez de 2, un AUC de **0.943** que deja poco margen (efecto techo), y
un tejido y una tarea distintos. Es un dato, no un contraejemplo limpio.

---

## 3. Segundo vuelco: los snapshot ensembles sí funcionan

El estado del arte cerraba los Snapshot Ensembles como **«nulo acotado a
≈ +0.017 de AUC»**. En `cptac_gbm` dejan de ser nulos.

*(Aquí la explicación NO es el tamaño del train —§2.2 lo descarta para los
embeddings— sino la potencia del contraste, como se detalla abajo.)*

`fge_trio:logreg` vs. `prob:logreg` en `cptac_gbm`, **las cinco métricas
significativas tras Holm**:

| Métrica | Δ | p Holm |
|---|---:|---:|
| AUC | +0.0223 | ✱ |
| AP | +0.0264 | ✱ |
| bacc | +0.0314 | ✱ |
| macro-F1 | +0.0327 | ✱ |
| **kappa** | **+0.0653** | ✱ |

En `bc_therapy` van en la misma dirección con IC que excluye el cero
(`fge_trio:logreg` +0.0263; `se_trio:mlp_snapshot` +0.0296, p Holm 0.016) y en
`cervical_subtype` son negativos.

**Por qué el nulo anterior era engañoso:** en `cptac_brca` el MDE de estas
comparaciones era de 0.026 de AUC —unas 4 × el de la suite de
metaclasificadores— porque las predicciones de los snapshots son mucho más
ruidosas fold a fold. Aquel nulo no decía «no hay efecto», decía «no puedo ver
un efecto menor de 0.026». El efecto real ronda +0.022, justo por debajo de esa
cota. **Era un problema de potencia, no una ausencia.**

---

## 4. Lo que se mantiene

### 4.1 LogReg sigue sin ser batido como regla de combinación

Ningún metaclasificador alternativo *en el espacio de probabilidades* bate a
LogReg en ningún dataset. Y en `cptac_gbm` el margen se amplía: **SVM y NB, que
empataban en `cptac_brca`, ahora pierden significativamente** (−0.0103 y
−0.0119, ambos p Holm ≤ 0.05).

| Brazo | `cptac_brca` | `bc_therapy` | `cptac_gbm` | `cervical` |
|---|---:|---:|---:|---:|
| `prob:svm` | −0.0021 | +0.0000 | **−0.0103** ✱ | −0.0027 |
| `prob:nb` | −0.0031 | −0.0041 | **−0.0119** ✱ | −0.0145 |
| `prob:knn` | **−0.0321** ✱ | **−0.0225** ✱ | **−0.0421** ✱ | −0.0409 |
| `prob:mlp` | **−0.1038** ✱ | **−0.1061** ✱ | **−0.0803** ✱ | −0.0409 |

**Todas las ganancias reales de esta campaña vienen de cambiar la ENTRADA del
metaclasificador (embeddings, snapshots), no de cambiar el metaclasificador.**
Es la misma lección que dio el barrido de modelos base, un escalón más arriba.

### 4.2 El MLP colapsa en los cuatro datasets

`prob:mlp` pierde entre −0.08 y −0.11 de AUC en todos. Y el hundimiento en las
métricas de decisión sigue siendo, como se documentó, **un fallo de corte**: en
`cervical_subtype` su bacc es 0.4461 con 4 clases, por debajo del azar (0.25 es
el degenerado), mientras su AUC es 0.9021.

---

## 5. La atención: respuesta definitiva a tres preguntas

### 5.1 ¿Merece la pena promediar las atenciones? — **No, confirmado en dos datasets**

| Dataset | Δ AUC (`attn_avg` vs `attn_own`) | IC 95 % | p Holm |
|---|---:|---|---:|
| `cptac_brca` (par) | −0.0009 | [−0.0083, +0.0061] | 1.00 |
| `cptac_gbm` (trío) | −0.0061 | [−0.0157, +0.0029] | 1.00 |

Nulo en ambos, con MDE de 0.010 y 0.014: son nulos **estrechos**, no ausencia de
evidencia. Promediar aplana la atención selectiva sin ganar nada.

### 5.2 ¿Y transferir la atención del mejor modelo? — **No replica**

En `cptac_brca` fue la señal más limpia del barrido: AP **+0.0145** con IC
[+0.0034, +0.0258] y 31/4/15. En `cptac_gbm`, con el trío completo:

| | Δ AUC | Δ AP | G/E/P (AP) | p Holm |
|---|---:|---:|---|---:|
| `attn_conch` vs `attn_own` | +0.0079 | +0.0075 | **25/0/25** | 1.00 |

**Empate exacto, 25 contra 25.** La dirección se mantiene pero el efecto
desaparece. Es el desenlace que el propio protocolo del proyecto anticipa: un
hallazgo marginal en un dataset que no sobrevive a la réplica. **Queda cerrado.**

### 5.3 ¿Aporta algo la atención? — **Sí, y más cuanto más datos**

El control negativo (atención uniforme = mean pooling) es el resultado más
robusto de toda la sonda:

| Dataset | Δ AUC (`attn_mean` vs `attn_own`) | IC 95 % | p Holm |
|---|---:|---|---:|
| `cptac_brca` | **−0.0333** | [−0.0460, −0.0205] | **0.0005** ✱ |
| `cptac_gbm` | **−0.0479** | [−0.0699, −0.0260] | **0.0009** ✱ |

Forzar mean pooling cuesta 0.033 y 0.048 de AUC. La atención aporta, y **aporta
más en el dataset con más datos**, lo que encaja con todo lo demás de este
informe.

---

## 6. Por qué `uni_v2` no aprende a atender — no es la dimensión

Diagnóstico sobre los 8 modelos entrenados en `cptac_brca` (25 slides, fold 0):

| modelo | in_dim | compr. | parches | entropía | logit_sd | retiene | AUC |
|---|---:|---:|---:|---:|---:|---:|---:|
| conch_v1_5 | 768 | 1.5× | 3719 | **0.8879** | **1.2856** | 58 % | **0.7902** |
| **hoptimus1** | **1536** | **3.0×** | **6582** | **0.9279** | 0.8277 | 37 % | 0.7450 |
| ctranspath | 768 | 1.5× | 3719 | 0.9282 | 0.9797 | 66 % | 0.7571 |
| phikon_v2 | 1024 | 2.0× | 6582 | 0.9764 | 0.6196 | 71 % | 0.7685 |
| **uni_v2** | **1536** | **3.0×** | **6582** | **0.9942** | 0.3000 | 66 % | 0.7634 |
| uni_v1 | 1024 | 2.0× | 6582 | 0.9974 | 0.2009 | 59 % | 0.7732 |
| virchow2 | 2560 | 5.0× | 6582 | 0.9981 | 0.1755 | **15 %** | 0.7771 |
| virchow_v1 | 2560 | 5.0× | 3719 | 0.9993 | 0.1013 | **5 %** | 0.6823 |

*(entropía normalizada: 1.0 = uniforme = mean pooling. `retiene` = varianza de
coseno entre parches que sobrevive a la proyección.)*

**El control natural que zanja la hipótesis de la dimensión:** `hoptimus1` y
`uni_v2` tienen la **misma** `in_dim` (1536), la **misma** compresión (3×), el
**mismo** número de parches (6582) y la misma arquitectura — y entropías de
0.928 frente a 0.994. Con todo lo demás constante, el comportamiento diverge.
La dimensión no es la causa.

Tampoco lo es la compresión: `uni_v2` **retiene el 66 %** de su estructura entre
parches, igual que ctranspath, y su `cos_var_proj` (0.0182) es de los más altos
de la tabla. **Tiene señal disponible y no la usa.** Lo que sí localiza el fallo
es `logit_sd`: su cabeza de atención emite logits casi constantes (0.30 frente a
1.29 de conch). Es un resultado de optimización, no un límite arquitectónico —
y añadir capacidad no lo arreglaría, porque su módulo de atención ya tiene los
mismos 262 400 parámetros que los demás (opera sobre el espacio proyectado de
512 D, no sobre `in_dim`) y su proyección ya tiene el doble.

**Donde la dimensión sí muerde:** los modelos de 2560 D. `virchow_v1` retiene un
**5 %** de su geometría y `virchow2` un 15 %. Y `virchow_v1` —el que menos
retiene, el de atención más plana— **es el peor modelo del proyecto (0.6823)**.
Es la primera explicación mecánica de por qué es prescindible.

**El giro:** atender no predice rendir. `uni_v1` (0.9974, casi uniforme) es el
3.º mejor y `virchow2` (0.9981) el 2.º. Ordenados por entropía, los AUC salen
0.7902, 0.7450, 0.7571, 0.7685, 0.7634, 0.7732, 0.7771, 0.6823 — sin relación
monótona. La atención plana **no es un defecto que arreglar**, salvo cuando va
acompañada de una proyección que destruye el 95 % de la geometría.

---

## 7. Corrección a CLAUDE.md: los grupos de coordenadas

CLAUDE.md presenta los «grupos de coordenadas de parche» como una propiedad de
los **modelos**, con cifras fijas (2100 y 4068 parches). **Ambas cosas son
falsas.**

Verificado sobre 30 slides por dataset:

| Dataset | ctranspath == conch_v1_5 | los tres iguales | ejemplo de parches |
|---|---|---|---|
| `cptac_brca` | 30/30 | **0/30** | ct 4101 · conch 4101 · uni_v2 5487 |
| `cptac_gbm` | 30/30 | **30/30** | los tres 734, los tres 1434 |
| `bc_therapy` | 30/30 | **30/30** | los tres 1754, los tres 3323 |
| `cervical_subtype` | 30/30 | **30/30** | los tres 10374, los tres 9027 |

Los grupos son una propiedad **del parcheado de cada dataset**, no de los
modelos, y los recuentos varían mucho por slide (662, 4087, 4101 en `cptac_brca`).

**Consecuencia práctica:** en tres de los cuatro datasets **se puede fusionar el
trío entero a nivel de parche**. La limitación que el informe atribuía a los
modelos —«el trío cruza los grupos 1 y 2, así que la fusión temprana sólo puede
aplicarse a dos de sus tres miembros»— es específica de `cptac_brca`. La fusión
temprana a tres bandas nunca se ha probado y ahora es posible.

---

## 8. Qué cambia en las recomendaciones

### 8.1 Lo que hay que dejar de afirmar

| Afirmación anterior | Estado |
|---|---|
| «§8.1 queda cerrada en negativo: los embeddings no baten a las probabilidades» | ❌ **Es un resultado de `cptac_brca`.** En gbm y bc_therapy se invierte |
| «Snapshot Ensembles: nulo acotado a ≈+0.017» | ❌ **Era falta de potencia.** En gbm dan +0.022 en las 5 métricas |
| «El trío cruza grupos de coordenadas, sólo se pueden fusionar dos modelos» | ❌ **Sólo en `cptac_brca`** |
| «`emb:mlp_snapshot` es la dirección a replicar» | ⚠️ Replica en gbm (+0.0175✱) pero no en bc_therapy (+0.0011) |
| «La transferencia de atención de conch merece réplica» | ❌ **No replica** (25/0/25) |

### 8.2 Lo que se refuerza

- **LogReg como combinador.** No batido en cuatro datasets; el margen crece con
  los datos.
- **La ganancia está en la entrada, no en el combinador.**
- **Promediar atenciones no aporta** (nulo estrecho en dos datasets).
- **La atención aporta** (mean pooling pierde 0.033–0.048, significativo en ambos).

### 8.3 Configuración recomendada, por dataset

Con ≥ 113 slides de entrenamiento, **la configuración por defecto debería dejar
de ser `prob:logreg`**:

| Dataset | Recomendada | AUC | vs `prob:logreg` |
|---|---|---:|---:|
| `cptac_brca` (75) | `prob:logreg` | 0.7986 | — |
| `bc_therapy` (113) | `emb:logreg` | 0.6999 | +0.0296 |
| `cptac_gbm` (164) | `se_trio:logreg` | 0.7825 | +0.0269 |
| `cervical_subtype` (407) | `prob:logreg` | 0.9430 | — |

---

## 9. Qué falta

1. **Fusión temprana a tres bandas** en `gbm`/`bc_therapy`/`cervical`, ahora que
   se sabe que es posible. El −0.0056 que la cerró se midió sobre dos modelos en
   `cptac_brca`, con 75 slides — el mismo régimen en el que los embeddings
   también parecían no funcionar.
2. **Repetir `cervical_subtype` con más folds.** Con 5 no arbitra nada. Se pueden
   generar splits nuevos con `create_val_splits.py`.
3. ~~Confirmar la relación con `n`~~ — **hecho (§2.2): falsado.** Lo que
   sustituye a esta pregunta es **graduar la selectividad de la atención** (no
   sólo llevarla al extremo plano) y comprobar si el Δ embeddings-probabilidades
   la sigue de forma monótona. Es el contraste que decidiría §2.3.
4. **Reevaluar `virchow_v1` con `proj_dim` mayor.** Retiene un 5 % de su
   geometría; es la única hipótesis mecánica que explica su AUC de 0.6823.

---

## Anexo — Procedencia

| Fichero | Contenido |
|---|---|
| `results_campaign_cptac_gbm_TP53_mutation.json` | 14 brazos, 50 folds (job 70495) |
| `results_campaign_bc_therapy_er_status.json` | 14 brazos, 50 folds (job 70496) |
| `results_campaign_cervical_subtype_subtype.json` | 14 brazos, 5 folds, 4 clases (job 70497) |
| `results_shared_attention_cptac_gbm_TP53_mutation.json` | 4 regímenes de atención, trío (job 70501) |
| `results_attention_diagnostics_cptac_brca_TP53_mutation.json` | Diagnóstico de 8 modelos (job 70499) |
| `results_train_size_curve_cptac_gbm_TP53_mutation.json` | Curva de tamaño de meta-train (job 70614) |
| `run_full_campaign.sbatch` | La campaña completa, parametrizada por `DATASET`/`TASK` |
| `run_train_size_curve.sbatch`, `src/train_size_curve.py` | El experimento de submuestreo de §2.2 |
| `src/run_meta_suite.py` | Runner: alineamiento, 3 bloques de métricas, estadística pareada (ahora multiclase) |
| `src/shared_attention_probe.py` | Sonda de regímenes de atención |
| `src/attention_diagnostics.py` | Diagnóstico de por qué unos modelos atienden y otros no |

Duraciones: `cptac_gbm` 4 h 41, `cervical_subtype` 4 h 10, `bc_therapy` 13 h 11
(la más cara por parches/slide, no por número de slides).
