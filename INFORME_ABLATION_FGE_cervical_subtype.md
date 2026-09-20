# Ablación de variantes FGE: `cervical_subtype` y réplica en `cptac_brca`

**Fecha:** 14 de agosto de 2026
**Jobs SLURM:** 69936 (cervical_subtype, 5 folds) · 69938-69943 (cptac_brca, 50 folds)
**Datos:** `/home/JKP6679/Patho-Ensemble/PARADIS/datos/patches/{dataset}/{task}/abmil/`

---

## Conclusión

**Aplicar FGE a los modelos base antes del ensamblado no mejora el rendimiento.**

En la réplica sobre `cptac_brca / TP53_mutation` con 50 folds —la corrida con potencia
estadística real— ninguna de las 9 configuraciones se separa del baseline en ninguna
métrica. Los intervalos de confianza pareados acotan el efecto de FGE sobre el AUC por
debajo de **+0.017**, con un efecto mínimo detectable de ≈0.016: si FGE aportara algo
apreciable, se habría visto.

La señal favorable que apareció en `cervical_subtype` (+0.048 de kappa ponderada en 5/5
folds, p = 0.024) **no replica**: en brca queda en +0.007 (p = 0.70). El signo del efecto
sobre el AUC además se invierte entre ambos datasets, lo que es propio del ruido y no de
un efecto real.

El detalle de la primera ablación se conserva en las secciones siguientes; la réplica que
zanja el asunto está en [Réplica en `cptac_brca`](#réplica-en-cptac_brca-50-folds).

---

## Primera ablación: `cervical_subtype / subtype`

**Job SLURM:** 69936 · **Duración:** 2 min 54 s (meta-learners). Las secciones siguientes,
hasta [Réplica en `cptac_brca`](#réplica-en-cptac_brca-50-folds), describen
esta primera corrida.

Se comparan 8 configuraciones de meta-learner sobre los mismos 5 folds, cruzando dos
meta-modelos (regresión logística y TabPFN) con las bases estándar y con cuatro
variantes de Fast Geometric Ensembles.

El resultado depende de **qué se mida**, y la discrepancia es el hallazgo principal:

- En **AUC-ROC**, todas las variantes FGE quedan por debajo del baseline, y lo hacen
  de forma sistemática — tres de ellas pierden en los 5 folds de 5.
- En **kappa ponderada**, dos variantes FGE superan al baseline de forma significativa
  (`fge_lowlr` +0.048 en 5/5 folds, p = 0.024; `fge_pat3` +0.042 en 4/5, p = 0.034).

Es decir: promediar snapshots **empeora el ordenamiento de las probabilidades y a la vez
mejora la decisión final**. No es contradictorio — son propiedades distintas del mismo
clasificador — pero impide resumir la ablación con una sola cifra.

**Matiz importante sobre la ventaja en kappa:** la métrica que reporta el pipeline es
kappa con ponderación *cuadrática*, que penaliza cada error en proporción a la distancia
entre la clase predicha y la real. Al recalcularla sin ponderar, la ventaja de `fge_lowlr`
cae de +0.048 (5/5 folds, p = 0.024) a **+0.015 (3/5 folds, p = 0.316)**. Lo que mejora
FGE no es acertar más, sino **equivocarse menos lejos**. Ver
[Qué es la kappa y qué mide aquí](#qué-es-la-kappa-y-qué-mide-aquí).

La tabla automática del pipeline concluía «ninguna configuración supera al baseline» y
«nada es concluyente». Ambas afirmaciones son artefactos del método de comparación, no
del resultado; se explica en [Análisis pareado](#análisis-pareado).

---

## Configuración experimental

### Dataset

`cervical_subtype`, tarea `subtype`: 599 pacientes, 1 slide por paciente, 4 clases.

| Clase | Etiqueta | n | % |
|---|---|---:|---:|
| 0 | NNeo | 204 | 34.1 % |
| 1 | LSIL | 249 | 41.6 % |
| 2 | HSIL | 80 | 13.4 % |
| 3 | otros | 66 | 11.0 % |

Ratio mayoritaria/minoritaria = **3.8×** (desbalance moderado).

Validación cruzada de **5 folds** con estratificación por paciente. El `k=all.tsv` sólo
tiene 5 columnas `fold_*`, de modo que el `max_folds: 10` de `experiments.yaml` no aplica:
`EXPECTED_FOLDS = min(10, 5) = 5`. Partición del fold 0: train 407 / val 72 / test 120.
La composición de clases del test es estable entre folds (33-34 % / 41-42 % / 13 % / 11-12 %).

### Modelos base

Tres ABMIL entrenados de forma independiente, 100 épocas, patching `20x_224px_0px_overlap`:

| Modelo fundacional | Dimensión de embedding |
|---|---:|
| `ctranspath` | 768 |
| `uni_v2` | 1536 |
| `virchow_v1` | 2560 |

La dimensión se auto-detecta del `.h5`; no se declara en la configuración.

### Variantes FGE evaluadas

Todas parten del mismo checkpoint convergido y aplican ciclos de LR piecewise-linear,
guardando un snapshot al final de cada ciclo. Cada variante escribe en su propio árbol
(`checkpoints_<name>/`, `val_outputs_<name>/`, `_train_eval_<name>/`), así que conviven
sin pisarse.

| Variante | lr₁ | lr₂ | Long. ciclo | Ciclos | Paciencia | Base incluido | Motivación |
|---|---|---|---:|---:|---:|:---:|---|
| `fge_orig` | 2e-4 | 2e-5 | 4 | 6 | 2 | no | Referencia |
| `fge_wbase` | 2e-4 | 2e-5 | 4 | 6 | 2 | **sí** | (1) Si los snapshots son peores que el base, promediar sin él sólo puede empeorar |
| `fge_lowlr` | **5e-5** | **5e-6** | **8** | 6 | 2 | no | (2) Ciclos más suaves y largos perturban menos el óptimo |
| `fge_pat3` | 2e-4 | 2e-5 | 4 | 6 | **3** | no | (3) Más paciencia permite explorar más ciclos antes de cortar |

Las tres últimas responden al diagnóstico previo sobre `cptac_brca / TP53_mutation`
(50 folds × 3 modelos): con `fge_orig` los snapshots quedaron ≈0.04 AUC **por debajo**
del modelo convergido en el 70-76 % de los folds, y el early stopping cortaba en 2.5
snapshots de media sin alcanzar nunca los 6 ciclos.

### Configuraciones de meta-learner

Cada configuración es un **ensemble por apilamiento** (*stacking*): los tres ABMIL base
producen una distribución de probabilidad sobre las 4 clases para cada slide, esas salidas
se concatenan en un vector de 3 × 4 = 12 características, y un segundo modelo —el
meta-learner— aprende a combinarlas. Se entrena sobre las predicciones del conjunto de
train y se evalúa sobre las del test, fold a fold.

Las 8 pruebas cruzan **dos factores independientes**:

- **Meta-modelo** — cómo se combinan las salidas de los modelos base:
  - *Regresión logística*: combinación lineal, un coeficiente por característica. Es el
    apilamiento clásico; con 407 muestras de entrenamiento y 12 características es
    difícil de sobreajustar, lo que la hace una referencia sólida.
  - *TabPFN*: un transformer preentrenado sobre datos tabulares sintéticos que clasifica
    sin ajuste de hiperparámetros. Puede capturar interacciones no lineales entre las
    salidas de los modelos base que la regresión logística no ve.

- **Bases** — de qué predicciones parte el meta-learner: las del checkpoint convergido
  (*estándar*) o las de una de las cuatro variantes FGE, donde cada modelo base aporta el
  promedio de sus snapshots en lugar de un único punto.

| ID | Meta-modelo | Bases | Qué pregunta responde |
|---|---|---|---|
| `m0_logreg_std` | Regresión logística | estándar | **Referencia.** El apilamiento clásico, sin FGE. Todo lo demás se mide contra esto. |
| `m1_tabpfn_std` | TabPFN | estándar | ¿Aporta algo un meta-modelo más expresivo, manteniendo las bases fijas? Aísla el efecto del meta-modelo. |
| `m2_logreg_fge_orig` | Regresión logística | `fge_orig` | ¿Aporta algo FGE tal y como se publicó, manteniendo el meta-modelo fijo? Aísla el efecto de FGE. |
| `m3_logreg_fge_wbase` | Regresión logística | `fge_wbase` | ¿Corrige la degradación **incluir el modelo convergido** en el promedio de snapshots? (sugerencia 1) |
| `m4_logreg_fge_lowlr` | Regresión logística | `fge_lowlr` | ¿La corrigen **ciclos más suaves y largos**, que perturban menos el óptimo? (sugerencia 2) |
| `m5_logreg_fge_pat3` | Regresión logística | `fge_pat3` | ¿La corrige **más paciencia** en el early stopping, que permite explorar más ciclos? (sugerencia 3) |
| `m6_tabpfn_fge_wbase` | TabPFN | `fge_wbase` | ¿Sobrevive el efecto de la sugerencia 1 al cambiar de meta-modelo, o es un artefacto de la regresión logística? |
| `m7_tabpfn_fge_lowlr` | TabPFN | `fge_lowlr` | Lo mismo para la sugerencia 2, la variante más prometedora. |

El bloque `m2`-`m5` es la ablación propiamente dicha: cuatro configuraciones idénticas en
todo salvo en los hiperparámetros del ciclo FGE, de modo que cualquier diferencia entre
ellas es atribuible a esa modificación y sólo a esa. Los pares `m3`/`m6` y `m4`/`m7`
comprueban si esas diferencias son **robustas al meta-modelo**: un efecto que aparece con
regresión logística y desaparece con TabPFN no es una propiedad de las bases FGE, sino de
cómo la regresión logística en particular las aprovecha.

**Cobertura del diseño.** El cruce completo serían 2 meta-modelos × 5 conjuntos de bases
= 10 celdas; se ejecutan 8:

| | estándar | `fge_orig` | `fge_wbase` | `fge_lowlr` | `fge_pat3` |
|---|:---:|:---:|:---:|:---:|:---:|
| **Regresión logística** | `m0` | `m2` | `m3` | `m4` | `m5` |
| **TabPFN** | `m1` | — | `m6` | `m7` | — |

Faltan TabPFN × `fge_orig` y TabPFN × `fge_pat3`. La primera ausencia es la que más
limita: sin ella no hay referencia FGE-sin-modificar dentro del bloque TabPFN, así que las
mejoras de `m6` y `m7` sólo pueden medirse contra las bases estándar, no contra FGE
original.

Ambas celdas se han añadido a `experiments.yaml` como `m8_tabpfn_fge_orig` y
`m9_tabpfn_fge_pat3`, de modo que la réplica sobre `cptac_brca` ejecuta el cruce completo
de 10 configuraciones. Los resultados de este informe corresponden a las 8 originales.

---

## Resultados

Medias sobre los 5 folds. El mejor valor de cada columna en **negrita**.

| Configuración | AUC-ROC | Bal. acc. | Kappa pond. | Macro-F1 | Accuracy |
|---|---:|---:|---:|---:|---:|
| `m0_logreg_std` | **0.9471** | 0.8037 | 0.7609 | 0.8118 | 0.8280 |
| `m1_tabpfn_std` | 0.9456 | 0.8048 | 0.7867 | 0.8153 | 0.8264 |
| `m2_logreg_fge_orig` | 0.9384 | 0.7938 | 0.7562 | 0.8073 | 0.8247 |
| `m3_logreg_fge_wbase` | 0.9360 | 0.8118 | 0.7867 | 0.8231 | 0.8297 |
| `m4_logreg_fge_lowlr` | 0.9306 | **0.8228** | **0.8085** | **0.8327** | **0.8381** |
| `m5_logreg_fge_pat3` | 0.9316 | 0.8150 | 0.8029 | 0.8255 | 0.8331 |
| `m6_tabpfn_fge_wbase` | 0.9395 | 0.8098 | 0.8022 | 0.8243 | 0.8298 |
| `m7_tabpfn_fge_lowlr` | 0.9339 | 0.7954 | 0.8007 | 0.8146 | 0.8180 |

`m0` gana en AUC y pierde en las otras cuatro columnas. `m4_fge_lowlr` es exactamente lo
contrario: peor AUC del bloque logreg y mejor en todo lo demás.

### Análisis pareado

Las 8 configuraciones se evalúan sobre los **mismos** 5 folds, así que la comparación
correcta es pareada: se resta fold a fold contra `m0_logreg_std` y se aplica una t
pareada. «Gana» = número de folds en los que la configuración supera al baseline.

#### AUC-ROC

| Configuración | Δ medio | Gana | p |
|---|---:|:---:|---:|
| `m1_tabpfn_std` | −0.0015 | 3/5 | 0.758 |
| `m2_logreg_fge_orig` | −0.0087 | 1/5 | 0.230 |
| `m3_logreg_fge_wbase` | −0.0111 | **0/5** | **0.009** |
| `m4_logreg_fge_lowlr` | −0.0165 | **0/5** | **0.029** |
| `m5_logreg_fge_pat3` | −0.0155 | **0/5** | **0.018** |
| `m6_tabpfn_fge_wbase` | −0.0076 | 1/5 | **0.037** |
| `m7_tabpfn_fge_lowlr` | −0.0132 | 1/5 | 0.060 |

La magnitud es pequeña (1-2 puntos), pero el signo es el mismo en todos los folds.

#### Kappa ponderada

| Configuración | Δ medio | Gana | p |
|---|---:|:---:|---:|
| `m1_tabpfn_std` | +0.0258 | 2/5 | 0.478 |
| `m2_logreg_fge_orig` | −0.0047 | 3/5 | 0.841 |
| `m3_logreg_fge_wbase` | +0.0259 | 3/5 | 0.148 |
| `m4_logreg_fge_lowlr` | **+0.0477** | **5/5** | **0.024** |
| `m5_logreg_fge_pat3` | +0.0420 | 4/5 | **0.034** |
| `m6_tabpfn_fge_wbase` | +0.0414 | 5/5 | 0.061 |
| `m7_tabpfn_fge_lowlr` | +0.0398 | 5/5 | 0.054 |

#### Macro-F1

| Configuración | Δ medio | Gana | p |
|---|---:|:---:|---:|
| `m1_tabpfn_std` | +0.0035 | 2/5 | 0.801 |
| `m2_logreg_fge_orig` | −0.0045 | 4/5 | 0.731 |
| `m3_logreg_fge_wbase` | +0.0113 | 4/5 | 0.365 |
| `m4_logreg_fge_lowlr` | +0.0209 | **5/5** | 0.068 |
| `m5_logreg_fge_pat3` | +0.0137 | 4/5 | 0.277 |
| `m6_tabpfn_fge_wbase` | +0.0125 | 3/5 | 0.371 |
| `m7_tabpfn_fge_lowlr` | +0.0028 | 3/5 | 0.759 |

#### Balanced accuracy y accuracy cruda

Ninguna diferencia alcanza significación (p mínimo 0.202 y 0.237 respectivamente). El
mejor en ambas es `m4_fge_lowlr` (+0.019 bacc, +0.010 acc), pero sólo gana en 3/5 y 2/5
folds: la mejora viene de folds concretos, no de un efecto consistente.

### Detalle por fold: `m0` frente a `m4_fge_lowlr`

El contraste entre las dos métricas se ve mejor sin promediar.

| Fold | AUC `m0` | AUC `m4` | Δ | Kappa `m0` | Kappa `m4` | Δ |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | 0.9337 | 0.9016 | −0.0321 | 0.8195 | 0.8240 | +0.0045 |
| 1 | 0.9518 | 0.9356 | −0.0162 | 0.7501 | 0.7772 | +0.0271 |
| 2 | 0.9660 | 0.9507 | −0.0153 | 0.7534 | 0.8204 | +0.0670 |
| 3 | 0.9600 | 0.9421 | −0.0179 | 0.7575 | 0.8264 | +0.0689 |
| 4 | 0.9240 | 0.9229 | −0.0011 | 0.7240 | 0.7947 | +0.0707 |

Cinco de cinco en ambas direcciones. No es ruido de un fold desafortunado.

---

## Qué es la kappa y qué mide aquí

La **kappa de Cohen** mide el acuerdo entre dos evaluadores —aquí, el modelo y la
verdad de referencia— **descontando el acuerdo que cabría esperar por azar**:

$$\kappa = \frac{p_o - p_e}{1 - p_e}$$

donde $p_o$ es el acuerdo observado (que coincide con la accuracy) y $p_e$ el acuerdo
esperado si ambos asignaran clases al azar respetando sus distribuciones marginales.

La escala se lee así: **κ = 1** es acuerdo perfecto, **κ = 0** significa que el modelo no
supera lo que se obtendría acertando por casualidad, y valores negativos indican acuerdo
peor que el azar. Dicho de otro modo, κ es la fracción del margen de mejora *disponible
por encima del azar* que el modelo efectivamente captura.

**Por qué importa en un dataset desbalanceado.** El denominador $1 - p_e$ es justamente
la corrección que le falta a la accuracy cruda. Aquí `m0` obtiene 0.828 de accuracy, pero
con marginales de 34/42/13/11 % una parte de esos aciertos se explica por la frecuencia de
las clases; la kappa los descuenta y deja 0.761. Esa diferencia entre 0.828 y 0.761 *es*
el efecto del desbalance sobre la accuracy, cuantificado.

En la literatura de patología es además la métrica convencional para medir concordancia
entre patólogos, lo que facilita comparar el modelo con la variabilidad inter-observador
publicada.

### La ponderación, y por qué aquí condiciona el resultado

El pipeline usa `cohen_kappa_score(y_true, y_pred, weights="quadratic")`
(`patho_bench/experiments/utils/ClassificationMixin.py:159`). La **ponderación** hace que
no todos los errores cuenten igual: se penalizan en proporción al cuadrado de la distancia
entre la clase predicha y la real. Confundir NNeo (0) con «otros» (3) pesa 9 veces más que
confundir NNeo con LSIL (1).

Eso **presupone que las etiquetas son ordinales** y que su distancia numérica es
significativa. En `cervical_subtype` la suposición se cumple a medias: NNeo → LSIL → HSIL
es una progresión real de severidad, pero la clase 3 es «otros», y colocarla como la más
alejada de NNeo es una convención del fichero de etiquetas, no un hecho clínico.

Merece comprobar cuánto de la ventaja de FGE depende de esa convención. Recalculando la
kappa de `m4_fge_lowlr` frente a `m0_logreg_std` con los tres esquemas:

| Ponderación | `m0` | `m4_fge_lowlr` | Δ | Gana | p |
|---|---:|---:|---:|:---:|---:|
| Sin ponderar | 0.7455 | 0.7604 | +0.0149 | 3/5 | 0.316 |
| Lineal | 0.7600 | 0.7886 | +0.0286 | **5/5** | **0.005** |
| Cuadrática *(la reportada)* | 0.7609 | 0.8085 | +0.0477 | **5/5** | **0.024** |

La ventaja **crece con el peso que se da a la distancia entre clases**, y sin ponderar
deja de ser significativa. La lectura correcta no es «FGE acierta más», sino **«FGE se
equivoca menos lejos»**: cuando falla, tiende a caer en una clase adyacente en la escala
de severidad en vez de en una lejana. Para un uso clínico de gradación eso es una mejora
genuina y relevante; para un uso donde sólo importa acertar la clase exacta, la ventaja
prácticamente desaparece.

En la réplica sobre `cptac_brca / TP53_mutation` esta ambigüedad no existe: la tarea es
binaria, y con dos clases la kappa cuadrática coincide exactamente con la no ponderada
(sólo hay una distancia posible fuera de la diagonal). El resultado de allí será, por
tanto, más limpio de interpretar.

---

## Sobre el AUC y el desbalance de clases

Conviene precisar por qué el AUC discrepa aquí, porque la objeción habitual —«el AUC
engaña si la muestra está desbalanceada»— no es exactamente lo que ocurre.

**El AUC-ROC no se infla con el desbalance.** Se calcula a partir de TPR y FPR, y cada
una se normaliza dentro de su propia clase, de modo que es invariante a la proporción de
clases. La métrica que sí engaña con desbalance es la **accuracy cruda**: en este dataset,
un clasificador que responda siempre «LSIL» acierta el 41.6 %. Por eso la accuracy es la
columna menos informativa de la tabla de resultados.

**La crítica real al ROC-AUC es otra** (Davis & Goadrich 2006; Saito & Rehmsmeier 2015):
cuando la clase positiva es rara, el denominador del FPR es enorme, así que un número
absoluto grande de falsos positivos apenas mueve el FPR y la curva ROC luce optimista
donde la precisión es mala. La curva precision-recall es más sensible en ese régimen.
Aquí ese efecto está atenuado: con un ratio de 3.8× el desbalance es moderado, y las
clases minoritarias (HSIL 13 %, otros 11 %) no son raras en el sentido que hace fallar al
ROC.

**Lo que sí conviene tener presente** es que la métrica reportada es *macro-OVR*: promedia
los AUC de las 4 clases dando el mismo peso a cada una, de modo que HSIL y «otros»
—66 y 80 casos— pesan tanto como LSIL con 249. Para una tarea de subtipado eso es
razonable, pero significa que el AUC agregado está dominado por lo bien que se ordenan
las clases minoritarias.

**La discrepancia observada no se explica por el desbalance, sino por qué mide cada
métrica.** El AUC evalúa el *ordenamiento* de las probabilidades e ignora el umbral; la
kappa y el F1 evalúan la *decisión* tras aplicar el argmax. Promediar snapshots FGE
degrada ligeramente el ordenamiento y a la vez desplaza la frontera de decisión a una
posición más favorable — un efecto de calibración. Ambas cosas pueden ser ciertas a la
vez y aquí lo son.

Consecuencia práctica: **la métrica que se reporte debe elegirse por el uso clínico
previsto**, no por conveniencia. Si el modelo va a producir un ranking o un score continuo
(triaje, priorización de lecturas), manda el AUC y FGE perjudica. Si va a emitir un
subtipo concreto, mandan kappa y F1 y FGE ayuda. Reportar sólo una de las dos
oculta la mitad del resultado.

---

## Verificación mecánica de las variantes

Antes de interpretar las métricas conviene comprobar si cada variante hizo lo que
pretendía. Snapshots por fold (media sobre 5 folds):

| Variante | `ctranspath` | `uni_v2` | `virchow_v1` | ¿Cumplió su mecanismo? |
|---|---:|---:|---:|---|
| `fge_orig` | 2.60 | 2.80 | 2.40 | — (referencia) |
| `fge_wbase` | 3.20 | 3.60 | 3.80 | **Sí** — el +1 es el modelo base como snapshot 0 |
| `fge_lowlr` | 2.40 | 2.20 | 2.40 | **No** — no desbloqueó más ciclos |
| `fge_pat3` | 3.80 | 3.60 | 3.80 | **Sí** — +1.2 snapshots sobre `fge_orig` |

Dos observaciones:

1. **`fge_orig` replica el comportamiento de `cptac_brca`**: 2.6 snapshots/fold aquí frente
   a 2.5 allí, muy lejos de los 6 ciclos configurados. El corte prematuro del early
   stopping no era una peculiaridad de aquel dataset.

2. **`fge_lowlr` falló en su propio mecanismo y aun así fue la mejor variante.** La
   hipótesis era que un lr₁ 4× menor y ciclos de 8 épocas permitirían aceptar más ciclos;
   el recuento (2.4 vs 2.6) dice que no ocurrió. Su ventaja en kappa y F1 viene, por tanto,
   de la *calidad* de los snapshots —ciclos más suaves que se alejan menos del óptimo—
   y no de su cantidad. Es un resultado útil: sugiere que la palanca es la magnitud de la
   perturbación, no el número de snapshots. `fge_pat3`, que sí generó más snapshots,
   quedó por detrás en kappa (+0.042 vs +0.048).

---

## Conclusiones (primera ablación)

> **Aviso de lectura.** Las conclusiones 2 y 3 **no sobrevivieron a la réplica con 50
> folds**; se conservan como registro de lo que mostró la primera corrida y de por qué
> resultaba convincente. Ver [El hallazgo de cervical no replica](#el-hallazgo-de-cervical-no-replica).

1. **Ninguna de las tres modificaciones recupera el AUC.** Las tres empeoran el
   ordenamiento respecto al baseline, y `fge_wbase` —la que más prometía tras el
   diagnóstico de `cptac_brca`— es de las peores (0/5 folds, p = 0.009). Incluir el
   modelo convergido en el promedio no compensa la degradación de los snapshots.

2. **Dos variantes mejoran la kappa ponderada, pero por reducir la *gravedad* del error,
   no su frecuencia.** `fge_lowlr` (+0.048, 5/5 folds) y `fge_pat3` (+0.042, 4/5) superan
   al baseline con significación nominal. Sin ponderación por distancia entre clases la
   ventaja de `fge_lowlr` cae a +0.015 en 3/5 folds (p = 0.316). Es la primera señal a
   favor de FGE en todo el trabajo, pero es más estrecha de lo que sugiere la cifra
   principal: FGE no acierta más casos, los falla más cerca.

3. **La palanca parece ser la suavidad del ciclo, no su número.** `fge_lowlr` mejoró sin
   generar más snapshots; `fge_pat3` generó más y mejoró menos.

4. **El efecto de FGE es de calibración, no de discriminación.** Reordena peor y decide
   mejor. Cualquier conclusión sobre «FGE funciona / no funciona» que no especifique la
   métrica es incompleta.

5. **TabPFN no aporta sobre las bases estándar** (`m1` vs `m0`: −0.0015 AUC, +0.026 kappa,
   ambos no significativos), pero sí interactúa favorablemente con las bases FGE en kappa
   (`m6`, `m7`: ≈+0.040 en 5/5 folds, p ≈ 0.055-0.061).

### Limitaciones

- **n = 5 folds.** Es la limitación dominante. Una t pareada sobre 5 observaciones es
  frágil, y los resultados 5/5 tienen p = 0.031 por test de signos de una cola, ya en el
  límite. Los valores p de esta ablación deben leerse como indicios, no como evidencia.
- **Sin corrección por comparaciones múltiples.** Se han realizado 35 contrastes
  (7 configuraciones × 5 métricas). A α = 0.05 cabe esperar ≈1.75 falsos positivos, y se
  han observado 5 resultados significativos. La consistencia direccional (0/5 y 5/5) es
  aquí más informativa que el valor p aislado.
- **La kappa reportada asume clases ordinales.** La ponderación cuadrática penaliza los
  errores por su distancia en la escala 0-3, pero la clase 3 es «otros» y su posición es
  una convención del fichero de etiquetas, no un orden clínico. La ventaja de FGE es
  sensible a esa elección (+0.048 cuadrática → +0.015 sin ponderar).
- **Un solo dataset y una sola tarea.** La discrepancia AUC/kappa podría ser específica
  del subtipado cervical.
- **Test de 120 muestras por fold**, con 13-16 casos de las clases minoritarias. Las
  métricas macro son inestables con ese tamaño.

---

## Réplica en `cptac_brca` (50 folds)

**Jobs SLURM:** 69938 (bases) → 69939-69942 (4 variantes FGE en paralelo) → 69943 (meta)
**Duración:** 2 min 54 s + 57-79 min por variante + 14 min 21 s

Se repite la ablación sobre `cptac_brca / TP53_mutation`, que dispone de **50 columnas
`fold_*`** frente a las 5 de `cervical_subtype`. El objetivo es resolver si la ventaja en
kappa medida en la primera corrida es real o un artefacto de n = 5.

Tres razones hacen de este dataset el contraste adecuado:

- **Potencia.** Con 50 observaciones pareadas el contraste detecta efectos pequeños y la
  corrección por comparaciones múltiples deja de ser prohibitiva.
- **Tarea binaria.** Con dos clases la kappa cuadrática coincide exactamente con la no
  ponderada, de modo que desaparece la dependencia del orden de las clases que condiciona
  el resultado en cervical.
- **Tarea de mutación, no de gradación.** La noción de «equivocarse menos lejos» no aplica
  aquí. Si la ventaja de `fge_lowlr` procedía sólo de la severidad del error, no debería
  reproducirse.

### Dataset

112 slides de 103 pacientes, clasificación binaria del estado mutacional de TP53:
67 wild-type (59.8 %) y 45 mutados (40.2 %). Desbalance leve (1.5×). Test de **22 slides
por fold**, con 50 folds de validación cruzada estratificada por paciente. Mismos 3 modelos
base, mismo patching y mismas 4 variantes FGE que en la primera ablación.

El diseño se amplía a **10 configuraciones**, completando el cruce 2 meta-modelos × 5
conjuntos de bases con las dos celdas TabPFN que faltaban (`m8_tabpfn_fge_orig`,
`m9_tabpfn_fge_pat3`).

### Resultados

Medias sobre los 50 folds:

| Configuración | AUC-ROC | Bal. acc. | Kappa | Macro-F1 | Accuracy |
|---|---:|---:|---:|---:|---:|
| `m0_logreg_std` | 0.7918 | 0.7363 | 0.4718 | 0.7335 | 0.7472 |
| `m1_tabpfn_std` | 0.7845 | 0.7270 | 0.4535 | 0.7237 | 0.7367 |
| `m2_logreg_fge_orig` | 0.7949 | 0.7338 | 0.4676 | 0.7313 | 0.7442 |
| `m3_logreg_fge_wbase` | 0.7929 | 0.7334 | 0.4671 | 0.7310 | 0.7444 |
| `m4_logreg_fge_lowlr` | **0.7978** | **0.7383** | **0.4784** | **0.7369** | **0.7510** |
| `m5_logreg_fge_pat3` | 0.7969 | 0.7312 | 0.4616 | 0.7288 | 0.7414 |
| `m6_tabpfn_fge_wbase` | 0.7781 | 0.7339 | 0.4670 | 0.7307 | 0.7424 |
| `m7_tabpfn_fge_lowlr` | 0.7862 | 0.7204 | 0.4407 | 0.7162 | 0.7311 |
| `m8_tabpfn_fge_orig` | 0.7804 | 0.7272 | 0.4549 | 0.7238 | 0.7374 |
| `m9_tabpfn_fge_pat3` | 0.7801 | 0.7264 | 0.4507 | 0.7221 | 0.7344 |

El rendimiento absoluto es muy inferior al de cervical (AUC 0.79 frente a 0.95): predecir
el estado mutacional de TP53 desde morfología es una tarea sustancialmente más difícil que
el subtipado cervical.

### Análisis pareado

Δ medio contra `m0_logreg_std`, con intervalo de confianza al 95 % de la diferencia
pareada y el reparto **gana/empata/pierde** sobre los 50 folds.

#### AUC-ROC

| Configuración | Δ medio | IC 95 % pareado | G/E/P | p |
|---|---:|---|:---:|---:|
| `m1_tabpfn_std` | −0.0073 | [−0.0177, +0.0032] | 22/2/26 | 0.18 |
| `m2_logreg_fge_orig` | +0.0032 | [−0.0086, +0.0149] | 25/5/20 | 0.60 |
| `m3_logreg_fge_wbase` | +0.0011 | [−0.0082, +0.0104] | 24/4/22 | 0.82 |
| `m4_logreg_fge_lowlr` | +0.0060 | [−0.0050, +0.0171] | 25/4/21 | 0.29 |
| `m5_logreg_fge_pat3` | +0.0051 | [−0.0068, +0.0171] | 25/5/20 | 0.41 |
| `m6_tabpfn_fge_wbase` | −0.0136 | [−0.0299, +0.0026] | 19/6/25 | 0.11 |
| `m7_tabpfn_fge_lowlr` | −0.0056 | [−0.0231, +0.0119] | 22/1/27 | 0.53 |
| `m8_tabpfn_fge_orig` | −0.0114 | [−0.0282, +0.0055] | 23/4/23 | 0.19 |
| `m9_tabpfn_fge_pat3` | −0.0117 | [−0.0291, +0.0057] | 21/3/26 | 0.19 |

#### Kappa

| Configuración | Δ medio | IC 95 % pareado | G/E/P | p |
|---|---:|---|:---:|---:|
| `m1_tabpfn_std` | −0.0183 | [−0.0524, +0.0158] | 13/17/20 | 0.30 |
| `m2_logreg_fge_orig` | −0.0042 | [−0.0391, +0.0308] | 14/15/21 | 0.82 |
| `m3_logreg_fge_wbase` | −0.0047 | [−0.0379, +0.0285] | 10/22/18 | 0.78 |
| `m4_logreg_fge_lowlr` | +0.0066 | [−0.0273, +0.0405] | 15/15/20 | 0.70 |
| `m5_logreg_fge_pat3` | −0.0102 | [−0.0460, +0.0256] | 14/11/25 | 0.58 |
| `m6_tabpfn_fge_wbase` | −0.0048 | [−0.0501, +0.0406] | 16/17/17 | 0.84 |
| `m7_tabpfn_fge_lowlr` | −0.0311 | [−0.0743, +0.0121] | 14/7/29 | 0.16 |
| `m8_tabpfn_fge_orig` | −0.0169 | [−0.0580, +0.0242] | 16/15/19 | 0.42 |
| `m9_tabpfn_fge_pat3` | −0.0211 | [−0.0603, +0.0181] | 13/11/26 | 0.30 |

En macro-F1 y balanced accuracy el patrón es el mismo: p mínimo de 0.13 y 0.16
respectivamente, ningún contraste significativo. **Ninguno de los 36 contrastes
(9 configuraciones × 4 métricas) alcanza p < 0.05**, ni siquiera antes de corregir por
comparaciones múltiples.

Conviene reparar en la columna de **empates**, que en kappa llega a 22 de 50. Con 22
muestras de test por fold la kappa toma pocos valores distintos, así que muchos folds
producen exactamente la misma matriz de confusión con ambas configuraciones. Ignorar los
empates al leer el reparto exagera la desventaja: `m3` es 10 victorias frente a 18
derrotas, no «10 de 50».

### Un resultado nulo acotado, no una ausencia de evidencia

Que ningún contraste sea significativo no basta por sí solo: podría deberse a falta de
potencia. Lo relevante es el **tamaño del efecto que la corrida era capaz de detectar**.

| Variante (logreg) | Δ AUC observado | IC 95 % pareado | Efecto mín. detectable (80 %) |
|---|---:|---|---:|
| `m2_fge_orig` | +0.0032 | [−0.0086, +0.0149] | 0.0168 |
| `m3_fge_wbase` | +0.0011 | [−0.0082, +0.0104] | 0.0133 |
| `m4_fge_lowlr` | +0.0060 | [−0.0050, +0.0171] | 0.0158 |
| `m5_fge_pat3` | +0.0051 | [−0.0068, +0.0171] | 0.0171 |

Los intervalos **excluyen cualquier mejora superior a ≈+0.017 de AUC**. No es «no hay
evidencia por falta de datos»: es que un efecto apreciable habría sido detectado. Es un
resultado nulo con cota, que es la forma informativa de un resultado nulo.

En kappa la situación es distinta y conviene decirlo: el efecto mínimo detectable ronda
0.048-0.051, justo del tamaño del que se observó en cervical, y con 11-22 empates por
comparación el contraste apenas discrimina. **La réplica no refuta con fuerza la hipótesis
de la calibración; lo que hace es no reproducirla.**

### El hallazgo de cervical no replica

| | cervical (5 folds) | brca (50 folds) |
|---|---|---|
| `fge_lowlr` Δ kappa | **+0.048**, 5/5, p = 0.024 | +0.007, 15/15/20, p = 0.70 |
| `fge_lowlr` Δ AUC | **−0.017**, 0/5, p = 0.029 | +0.006, 25/4/21, p = 0.29 |
| `fge_pat3` Δ kappa | **+0.042**, 4/5, p = 0.034 | −0.010, 14/11/25, p = 0.58 |
| `fge_wbase` Δ AUC | **−0.011**, 0/5, p = 0.009 | +0.001, 24/4/22, p = 0.82 |

**El signo se invierte en las cuatro comparaciones.** Un efecto real no cambia de dirección
al cambiar de dataset; el ruido, sí. Los resultados «5/5 folds» de la primera corrida,
que resultaban tan persuasivos, eran precisamente la clase de patrón que n = 5 produce con
facilidad: la probabilidad de que cinco diferencias independientes compartan signo por azar
es 1/16 = 6.3 % en dos colas, y con 35 contrastes se esperan un par de ellos.

Queda también descartada la explicación que se había propuesto para la discrepancia
AUC/kappa —que FGE «se equivoca menos lejos»—. Era comprobable: en una tarea binaria esa
noción no tiene sentido, y en efecto la ventaja no aparece.

### Consistencia mecánica entre datasets

Los recuentos de snapshots reproducen casi exactamente los de cervical, lo que confirma
que las variantes se comportan igual en ambos datasets y que la diferencia de resultados
no procede de una ejecución distinta de FGE:

| Variante | Snapshots/fold en brca | En cervical |
|---|---:|---:|
| `fge_orig` | 2.64 | 2.60 |
| `fge_wbase` | 3.64 | 3.53 |
| `fge_lowlr` | 2.32 | 2.33 |
| `fge_pat3` | 3.77 | 3.73 |

Se confirman las dos observaciones de la primera corrida: `fge_orig` corta en ≈2.6
snapshots pese a tener 6 ciclos configurados, y `fge_lowlr` **no** desbloquea más ciclos
(2.32, incluso menos que la referencia). La diferencia es que en brca esa menor cantidad
de snapshots tampoco viene acompañada de ninguna ventaja en calidad.

### Conclusión de la réplica

1. **FGE no mejora el ensamblado.** Con 50 folds, ninguna variante se separa del baseline
   en ninguna métrica, y el efecto sobre el AUC está acotado por debajo de +0.017.

2. **Ninguna de las tres modificaciones propuestas cambia nada**, ni respecto al baseline
   ni entre sí. El diagnóstico que las motivó —snapshots peores que el modelo convergido,
   early stopping prematuro— era correcto como descripción, pero corregirlo no produce
   ganancia.

3. **TabPFN empeora sistemáticamente sobre bases FGE.** Las cuatro configuraciones TabPFN
   con FGE (`m6`-`m9`) quedan por debajo del baseline en AUC (−0.006 a −0.014), mientras
   que las cuatro logreg con FGE quedan ligeramente por encima (+0.001 a +0.006). Ninguna
   diferencia es significativa, pero la separación por meta-modelo es consistente y sugiere
   que TabPFN no aprovecha la redundancia añadida por el promediado de snapshots.

4. **El coste no se justifica.** FGE añade ≈1 hora de GPU por variante y triplica el
   almacenamiento de checkpoints a cambio de un efecto por debajo del umbral de detección.

Como resultado para la tesis, el hallazgo negativo está bien establecido: dos datasets, dos
tareas (subtipado multiclase y mutación binaria), cuatro configuraciones de ciclo y dos
meta-modelos.

### Continuación (en ejecución)

El análisis de diversidad que explica este resultado nulo abrió dos líneas que se están
ejecutando y que se documentarán por separado:

1. **Snapshot Ensembles en los modelos base** (variantes `se_orig` y `se_mild`). SE
   reinicia la LR mucho más arriba que FGE, así que baja la correlación entre snapshots;
   la pregunta es si baja lo suficiente para acercarse al suelo inter-modelo de 0.797.
   Predicción registrada: r ≈ 0.88-0.93 y AUC dentro de ±0.017 del baseline.

2. **Corrección de las meta-features.** Al preparar esa línea se encontró que el
   meta-learner se entrena con predicciones *in-sample* de los modelos base: alcanzan
   AUC 0.94 sobre las features de meta-entrenamiento frente a 0.77 en test. Es el fallo
   clásico del stacking, y penaliza sobre todo a los meta-modelos flexibles, lo que
   explica el sobreajuste de la MLP y el mal rendimiento de TabPFN observados en este
   proyecto. `src/build_oof_features.py` genera la alternativa out-of-fold mediante CV
   anidada agrupada por paciente; verificado sobre `cptac_brca`, el AUC de meta-train cae
   de 0.94 a 0.68-0.71.

   Las 31 configuraciones de meta-learner se ejecutan sobre **ambos regímenes de
   features**, de modo que el diagnóstico quede medido y no supuesto. Predicción
   registrada: `tabpfn` y `mlp` mejoran claramente al pasar a out-of-fold mientras
   `logreg` apenas se mueve. Si TabPFN no mejora, el diagnóstico es falso.

Ambas líneas usan el protocolo de análisis pareado descrito arriba, con reparto
gana/empata/pierde, efecto mínimo detectable y corrección de Holm.

---

## Anexo: validez de la ejecución

Los resultados aquí recogidos provienen del job **69936**. Las dos ejecuciones previas
sobre la misma matriz **no son válidas** y se documentan para evitar que se reutilicen:

| Job | Estado | Causa |
|---|---|---|
| 69934 | Inválido | `m0` abortó por `coefs.npy` de una corrida anterior (5 modelos base, shape `(5,20)`) incompatible con la actual (`(5,12)`). Además la tabla comparativa asignaba mal los directorios. |
| 69935 | Falló al inicio | `CHECKED_BASES` sin inicializar con `set -u` activo. |
| 69936 | **Válido** | 8/8 configuraciones completadas, cada una leyendo su directorio. |
| 69938-69943 | **Válido** | Réplica en `cptac_brca`: 10/10 configuraciones, 50 folds, con las correcciones ya aplicadas. |

El fallo más grave del job 69934 no fue el aborto de `m0`, que era visible, sino que **la
tabla comparativa mostró cifras plausibles y falsas**: la condición que seleccionaba el
directorio de resultados comparaba `base_source` con el literal `"fge"`, que no coincide
con ningún nombre de variante (`fge_orig`, `fge_wbase`, …). En consecuencia las cuatro
filas logreg leyeron todas `ensemble4/` y las tres TabPFN leyeron `ensemble4_tabpfn/`,
produciendo grupos de filas idénticas. El «baseline» que se imprimió era además un
`test_metrics_summary.json` de enero, correspondiente a una corrida con 5 modelos base.

Correcciones aplicadas:

| Fichero | Corrección |
|---|---|
| `src/ensemble4.py` | Borrar `coefs.npy`/`odds_ratios.npy` al inicio de cada corrida en vez de acumular con `np.vstack` sobre una forma heredada |
| `run_3_experiments_v2_improved.sbatch` | Sufijo de directorio `_{base_source}` para cualquier base ≠ `standard`, igual que hace `ensemble4.py` |
| `run_3_experiments_v2_improved.sbatch` | La tabla sólo lee resultados de configuraciones completadas **en la ejecución en curso** (`RAN`), para que un JSON antiguo no se cuele como actual |
| `run_3_experiments_v2_improved.sbatch` | La comprobación previa itera sobre las variantes reales de `META_CONFIGS` en lugar de buscar el literal `_train_eval_fge` |

El `ensemble4/` de enero se conservó en
`ensemble4_backup_2026-01-08_5modelos/` antes de regenerarlo.

**Pendiente:** la tabla comparativa del pipeline sigue reportando únicamente medias con
intervalos de confianza independientes. En brca ese criterio coincide con el pareado
—ambos concluyen que no hay diferencia—, pero en cervical archivaba como «nada
concluyente» un patrón que el contraste pareado sí destacaba. Incorporar el pareado a la
salida automática, con el reparto gana/empata/pierde y el efecto mínimo detectable, evitaría
tanto pasar por alto una señal como tomar por señal lo que es ruido de pocos folds.
