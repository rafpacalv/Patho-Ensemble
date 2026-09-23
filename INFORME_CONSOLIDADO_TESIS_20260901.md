# Patho-Ensemble — Informe consolidado de la campaña experimental

**Ensembles de modelos fundacionales para clasificación en patología digital (WSI)**

| | |
|---|---|
| **Fecha de cierre** | 1 de septiembre de 2026 |
| **Periodo cubierto** | 11 de agosto – 1 de septiembre de 2026 |
| **Datasets evaluados** | `cptac_brca`, `cptac_gbm`, `bc_therapy`, `cervical_subtype` (+13 tareas en 3 cohortes para el estudio de enrutado) |
| **Modelos fundacionales** | 8 entrenados y comparados |
| **Metaclasificadores** | 15 registrados en el código, 6 evaluados en la campaña vigente |
| **Experimentos consolidados** | 8 positivos · 20 negativos o nulos · 2 incompletos |

> Este documento sustituye a todos los informes parciales previos. Recoge **únicamente
> resultados vigentes**: donde una medición fue posteriormente corregida o invalidada, aquí
> figura solo el valor final. Todas las cifras se han verificado contra los ficheros
> `results_*.json` que las produjeron.

---

## Resumen en una página

La campaña tenía una pregunta de fondo: **¿dónde está el margen de mejora de un ensemble de
modelos fundacionales — en qué modelos se eligen, o en cómo se combinan?**

La respuesta está medida y es inequívoca:

| Palanca | Δ AUC medido | ¿Significativo? |
|---|---:|---|
| **Cambiar un modelo base** (`virchow_v1` → `conch_v1_5`) | **+0.037** | ✅ Sí, p ≈ 1.4·10⁻⁴ |
| Cambiar el espacio de entrada del metaclasificador (probabilidades → embeddings), en datasets grandes | +0.023 | ✅ Sí, p = 0.031 (solo `cptac_gbm`) |
| Snapshot Ensembles sobre los modelos base, en datasets grandes | +0.022 | ✅ Sí, en las 5 métricas (solo `cptac_gbm`) |
| Pooling duro top-10 de atención (en vez de blando), en datasets grandes | +0.039 | ✅ Sí, p Holm < 0.001 (solo `cptac_gbm`) |
| El stacking, frente al mejor modelo individual | +0.008 | ⚠️ Marginal |
| Optimización de pesos (Nelder-Mead) | +0.005 | ❌ No |
| Ampliar, enrutar o seleccionar dinámicamente el comité | < +0.003 | ❌ No |
| Fusión temprana a nivel de parche | −0.006 | ❌ No |
| Cambiar el metaclasificador (9 alternativas a LogReg) | ≤ 0 | ❌ Ninguna gana |

**Elegir qué modelos entran vale ~7× más que elegir cómo se combinan.** Un único modelo bien
escogido (`conch_v1_5`, AUC 0.7902) supera al ensemble completo de tres modelos mal escogidos
(0.7616). Nueve intentos distintos de mejorar la regla de combinación no han producido ninguna
ganancia que sobreviva a la corrección por comparaciones múltiples.

El segundo hallazgo, más sutil y descubierto al final de la campaña: **varios resultados que se
habían cerrado como "nulos" en `cptac_brca` eran en realidad falta de potencia estadística.**
Al replicar en datasets con más pacientes (`cptac_gbm`, 164 slides de entrenamiento frente a 75),
los embeddings, los Snapshot Ensembles y el pooling duro top-K de atención pasan los tres a dar
ganancias significativas. El tamaño del cohorte es el factor que gobierna casi todo lo demás — y
en los tres casos el patrón es el mismo: nulo acotado en `cptac_brca`, significativo en `cptac_gbm`
(ver P2 y P7).

---

# 1. Métricas

## 1.1 Las métricas que se calculan

El pipeline calcula 12 métricas por brazo y por fold. Siete son primarias; cinco son las mismas
métricas de decisión recalculadas con un umbral reajustado (sección 1.3).

| Métrica | Qué mide | Rango | Cómo se lee |
|---|---|---|---|
| **`auc`** (macro-OvR AUC-ROC) | Capacidad de **ordenar**: probabilidad de que un positivo al azar reciba mayor puntuación que un negativo al azar. Es la métrica primaria de reporte. | 0–1 | 0.5 = azar. **No depende del umbral de decisión.** Es invariante a la proporción de clases, pero pierde sensibilidad si la clase positiva es muy rara (Davis & Goadrich 2006). En este proyecto los modelos rondan **0.68–0.80** en `cptac_brca`/`cptac_gbm`/`bc_therapy` y **0.94** en `cervical_subtype`. |
| **`ap`** (average precision, PR-AUC) | También ordenación, pero ignorando los verdaderos negativos. Es la que más pesa bajo desbalance. | 0–1 | Su suelo es la prevalencia de la clase positiva, no 0.5. Se exige que **`auc` y `ap` se muevan en el mismo sentido**: si no lo hacen, el efecto es sospechoso. |
| **`acc`** (accuracy) | Fracción de aciertos. | 0–1 | **La columna menos informativa.** Un clasificador que siempre prediga la clase mayoritaria ya saca 0.60 en `cptac_brca` y 0.70 en `bc_therapy`. Cualquier accuracy que no supere claramente ese suelo no dice nada. |
| **`bacc`** (balanced accuracy) | Media de la sensibilidad por clase; corrige el desbalance. | 0–1 | **Suelo degenerado exacto: 0.500** en tareas binarias y **0.250** en `cervical_subtype` (4 clases). Es la lectura honesta de "accuracy". |
| **`macro_f1`** | Media no ponderada del F1 por clase. | 0–1 | Suelo degenerado entre 0.15 y 0.41 según el dataset (tabla 1.2). |
| **`f1_pos`** | F1 de la clase positiva. | 0–1 | **Engañosa según qué clase sea mayoritaria.** En `bc_therapy` la clase positiva (er+) es la mayoritaria (69.3 %), y el clasificador degenerado ya saca `f1_pos` = 0.821. En `cptac_brca` y `cptac_gbm` el degenerado saca 0.000. No usar sola. |
| **`kappa`** (κ de Cohen, ponderación cuadrática) | Acuerdo entre predicción y verdad **corregido por el acuerdo esperado por azar**: κ = (p₀ − pₑ)/(1 − pₑ). | −1 a 1 | **Suelo degenerado: 0.000 siempre, por construcción.** Es **la métrica más sensible del conjunto** y la que mejor separa un fallo de ordenación de un fallo de umbral. Valores observados en el proyecto: 0.35 (trío histórico) a 0.49 (mejor configuración). |

**Advertencia sobre la ponderación de κ.** Por defecto se usa ponderación cuadrática, que penaliza
los errores según el cuadrado de la distancia entre clase predicha y real. Esto **cambia
conclusiones**: en la ablación FGE sobre `cervical_subtype`, la ventaja de una variante pasaba de
+0.0477 (5/5 folds, p = 0.024) con ponderación cuadrática a +0.0149 (3/5, p = 0.316) sin ponderar.
La ponderación solo tiene sentido si las clases son ordinales; en `cervical_subtype`
(NNeo → LSIL → HSIL → otros) ese supuesto es solo parcialmente válido.

## 1.2 El suelo real: el clasificador degenerado

Antes de juzgar cualquier cifra hay que saber qué saca un clasificador que **ignora por completo
las imágenes** y siempre predice la clase mayoritaria. Estos son los suelos verificados:

| Dataset | `acc` | `bacc` | `macro_f1` | `f1_pos` | `kappa` |
|---|---:|---:|---:|---:|---:|
| `cptac_brca` / TP53_mutation | 0.598 | **0.500** | 0.374 | 0.000 | 0.000 |
| `cptac_gbm` / TP53_mutation | 0.669 | **0.500** | 0.400 | 0.000 | 0.000 |
| `bc_therapy` / er_status | 0.697 | **0.500** | 0.411 | **0.821** | 0.000 |
| `cervical_subtype` / subtype (4 clases) | 0.416 | **0.250** | 0.147 | — | 0.000 |

Lectura para la tesis: **una accuracy de 0.70 en `bc_therapy` no es un resultado**, es exactamente
lo que da no hacer nada. Solo `bacc` y `kappa` tienen un suelo interpretable y constante.

## 1.3 Las métricas `_t` (umbral reajustado)

Las cinco métricas de decisión se recalculan también con el umbral que maximiza la balanced
accuracy **elegido en el conjunto de validación** (nunca en test): `acc_t`, `bacc_t`,
`macro_f1_t`, `f1_pos_t`, `kappa_t`.

**Existen como diagnóstico, no como configuración de producción.** Sirven para distinguir dos
fallos que se confunden:

- **Fallo de ordenación**: el modelo no sabe qué caso es más probable. Se ve en `auc` bajo.
- **Fallo de corte**: el modelo ordena bien pero el umbral 0.5 está mal puesto para el desbalance.
  Se ve en `auc` alto con `kappa` bajo, que se recupera al reajustar el umbral.

Ejemplo medido (`cptac_brca`, 50 folds): `prob:mlp` tiene κ = 0.034 con umbral 0.5 y κ = 0.328
con umbral reajustado (+0.294). En cambio LogReg, SVM y NB **empeoran** al reajustar (−0.029 a
−0.045) porque ya estaban bien calibrados. El umbral medio elegido cae en 0.41–0.52, coherente con
un 40 % de positivos.

**No se adopta el reajuste en el pipeline**: con ~15 slides de validación por fold el umbral
estimado es más ruidoso que fijar 0.5.

## 1.4 El aparato estadístico

Todas las configuraciones se evalúan sobre **los mismos folds**, de modo que las comparaciones son
**pareadas**. Por cada métrica se reporta:

| Elemento | Qué es | Por qué está |
|---|---|---|
| **Δ pareado + IC 95 %** | Diferencia media fold a fold, con intervalo bootstrap (1 000–10 000 remuestreos). | Comparar IC *independientes* en vez de pareados llegó a ocultar un efecto sistemático real. |
| **W/T/L** | Folds ganados / empatados / perdidos. **Los empates se cuentan explícitamente.** | Con ~22 slides de test por fold, κ puede empatar en hasta 22 de 50 folds. Un W/T/L sin empates visibles es sospechoso. |
| **p de Wilcoxon + corrección de Holm** | Sobre la familia de métricas de cada comparación. | Con 5 folds, un "5 de 5, p = 0.024" sin corregir **no replicó** a 50 folds y el signo se invirtió. |
| **MDE al 80 % de potencia** | Mínimo efecto detectable: `(t_crít + t_pot)·σ_dif/√n_folds`. Con 50 folds ronda **±0.016–0.026** de AUC. | Convierte un "no encontramos nada" en un **nulo acotado** ("no hay efecto mayor que X"), que es un resultado y no una indeterminación. |
| **Suelo de reproducibilidad ≈ 0.004 AUC** | Reentrenar la configuración idéntica (mismo código, mismas semillas) mueve el AUC agregado ~0.004, y cada fold individual hasta **±0.032**, solo por no determinismo de GPU. | **Criterio más exigente que el MDE.** Cualquier Δ por debajo de ≈0.01 de AUC es indistinguible de relanzar el mismo trabajo. Este criterio, por sí solo, descarta Nelder-Mead (+0.005), la GNN espacial (+0.008) y la augmentación de bolsa (+0.008). |

### Una advertencia que conviene llevar a la defensa

**Los 50 folds no son 50 experimentos independientes.** La suma de los tamaños de test es 9.6 veces
el número de slides del dataset, y cada slide cae en el conjunto de test entre 3 y 17 veces: es
remuestreo repetido y solapado, no una partición k-fold clásica.

| Dataset | Apariciones medias de cada slide en test | Rango |
|---|---:|---|
| `cptac_brca` | 9.59 | 3–16 |
| `cptac_gbm` | 9.32 | 2–17 |
| `bc_therapy` | 9.94 | 4–17 |
| `cervical_subtype` | **1.00** | partición real |

**Consecuencia:** los IC bootstrap sobre folds **subestiman** la incertidumbre real de generalizar
a pacientes nuevos. Los Δ pareados siguen siendo válidos como comparación entre configuraciones
(todas ven exactamente los mismos folds), pero los valores absolutos de AUC deben citarse con esta
cautela.

## 1.5 La entropía: qué es, por qué se usa y cómo se interpreta

La entropía aparece en este proyecto en **dos lugares completamente distintos** que conviene no
confundir. Y hay un tercer sitio donde mucha gente asume que está y **no está**.

### (a) Entropía de la atención — diagnóstico de selectividad del ABMIL

**Definición.** El ABMIL asigna un peso de atención *aᵢ* a cada uno de los *N* parches de una slide
(softmax, así que Σaᵢ = 1). Su entropía normalizada es:

```
entropy_norm = −Σ aᵢ·log(aᵢ) / log(N)
```

**Por qué se normaliza por `log(N)`:** para que el valor no dependa del número de parches, que
varía entre 3 700 y 6 600 según el modelo. Sin normalizar, un modelo con más parches saldría
"más entrópico" por construcción.

**Cómo se interpreta:**

| Valor | Significado |
|---|---|
| **1.0** | Atención perfectamente uniforme. El ABMIL **no está decidiendo nada**: es exactamente equivalente a *mean pooling* sobre todos los parches. |
| **0.85–0.95** | Atención moderadamente selectiva: concentra el peso en una fracción de los parches. |
| **→ 0** | Atención extremadamente concentrada en unos pocos parches. |

**Para qué se usa:** para responder *por qué* unos modelos rinden más que otros, y *por qué* el
espacio de embeddings a veces gana y a veces pierde. Es una variable **explicativa**, no un
criterio de decisión del pipeline.

Valores medidos sobre `cptac_brca` (fold 0, 25 slides), junto con dos diagnósticos que la
acompañan siempre:

| Modelo | `in_dim` | Compresión | Entropía | `logit_sd` | Retiene geometría | AUC test |
|---|---:|---:|---:|---:|---:|---:|
| `conch_v1_5` | 768 | 1.5× | **0.8879** | **1.286** | 58 % | **0.7902** |
| `hoptimus1` | 1536 | 3.0× | 0.9279 | 0.828 | 37 % | 0.7450 |
| `ctranspath` | 768 | 1.5× | 0.9282 | 0.980 | 66 % | 0.7571 |
| `phikon_v2` | 1024 | 2.0× | 0.9764 | 0.620 | 71 % | 0.7685 |
| `uni_v2` | 1536 | 3.0× | 0.9942 | **0.300** | 66 % | 0.7634 |
| `uni_v1` | 1024 | 2.0× | 0.9974 | 0.201 | 59 % | 0.7732 |
| `virchow2` | 2560 | 5.0× | 0.9981 | 0.176 | **15 %** | 0.7771 |
| `virchow_v1` | 2560 | 5.0× | **0.9993** | 0.101 | **5 %** | **0.6823** |

- `logit_sd` = desviación típica de los logits de atención **antes** del softmax. Si es ≈0, el
  modelo directamente no discrimina; si es grande y la entropía sigue alta, discrimina pero de
  forma difusa.
- "Retiene geometría" = porcentaje de la varianza de similitud coseno entre parches que sobrevive
  a la proyección `in_dim → 512` del ABMIL. Mide cuánta estructura destruye el cuello de botella.

**Tres lecturas importantes de esta tabla:**

1. **La entropía NO predice el AUC de forma monótona.** `uni_v1` (0.9974, casi uniforme) es el 3.º
   mejor modelo y `virchow2` (0.9981) el 2.º. Atender mucho no garantiza rendir. La entropía sola
   no basta: hay que cruzarla con la retención de geometría.
2. **La dimensión de entrada no es la causa de la atención plana.** `hoptimus1` y `uni_v2`
   comparten `in_dim` (1536), compresión (3×), número de parches (6 582) y arquitectura, y sus
   entropías divergen (0.928 vs 0.994). Lo que localiza el fallo de `uni_v2` es `logit_sd` = 0.30:
   su cabeza de atención emite logits casi constantes. **Es un problema de optimización, no un
   límite arquitectónico.**
3. **Donde la dimensión sí muerde es en 2560 D.** `virchow_v1` retiene solo el **5 %** de la
   geometría entre parches tras la proyección, y es el peor modelo del proyecto (0.6823). Ésta es
   la primera explicación *mecánica* de por qué es prescindible.

**Y la entropía se ha manipulado experimentalmente**, no solo observado. Forzar atención uniforme
(`attn_mean`, es decir *mean pooling*) cuesta **−0.0333 de AUC en `cptac_brca`** (p Holm = 0.00046)
y **−0.0479 en `cptac_gbm`** (p Holm = 0.00090). Es decir: **la atención sí aporta**, y aporta más
en el dataset con más datos.

### (b) Entropía de las predicciones — como meta-característica

Sobre las probabilidades de clase *p* que cada modelo base produce por slide se calculan:

```
Entropía de Shannon:  H(p) = −Σ_c p_c · log(p_c)          rango [0, log 2] ≈ [0, 0.69] en binaria
Jensen-gap:           H(media_m p_m) − media_m H(p_m)
```

**Por qué el Jensen-gap.** Es la descomposición estándar de la incertidumbre en *ensemble
learning*: la incertidumbre total del comité se separa en **incertidumbre media** (lo que ninguno
sabe) más **desacuerdo entre modelos** (lo que unos saben y otros no). Es la única característica
de una sola columna que resume información **entre** modelos.

**Por qué la entropía por modelo.** `H(p)` es simétrica: vale lo mismo en p = 0.1 que en p = 0.9.
Ninguna combinación lineal de `[p, 1−p]` puede reproducirla, así que **en teoría** añade
expresividad a un metaclasificador lineal.

**Resultado medido: nulo.** La escalera de características (E0 = 6 probabilidades → +desacuerdo →
+margen → +entropía por modelo) da Δ = −0.0014, no significativo, en `cptac_brca` con 50 folds.
La razón es sencilla: con 6 probabilidades y 75 filas de meta-entrenamiento, LogReg ya extrae
prácticamente toda la señal disponible; las transformaciones no lineales explícitas no aportan
información nueva, y sí consumen grados de libertad.

**Una trampa metodológica que costó un experimento.** Al probar la entropía *de la atención* como
columna extra del metaclasificador, el resultado salió **idéntico al baseline** (Δ = 0.000000,
p = nan). No era un nulo: la atención de ABMIL está tan colapsada hacia lo uniforme
(`entropy_norm` ≈ 0.999, sd ≈ 4·10⁻⁴) que la columna era **constante a efectos numéricos**, y una
columna constante la absorbe el intercepto del modelo. El código ahora aborta explícitamente ante
columnas con sd < 10⁻³ (`_reject_degenerate` en `src/ensemble4.py`), precisamente para que esto no
vuelva a leerse como "esta característica no aporta".

### (c) Dónde la entropía NO se usa — y conviene decirlo

**La entropía no es el criterio de apagado dinámico del comité.** El estudio de enrutado
(sección 3.3, N13–N14) probó ocho criterios de selección y **ninguno es entrópico**:
AUC de validación media, AUC penalizada por inestabilidad entre folds, AUC agrupada, Brier,
log-loss, average precision, AUC de meta-entrenamiento (control negativo) y número de modelos.

Si en algún borrador de la tesis se afirma que el comité se apaga por la incertidumbre del softmax,
**eso no es lo que se implementó ni lo que se midió**. En este proyecto la entropía vive
exclusivamente en el plano de "cómo de selectiva es la atención dentro de una slide" (uso a) y como
meta-característica descartada (uso b).

---

# 2. Estado vigente del pipeline

## 2.1 Los datos

Los cuatro datasets efectivamente usados en la campaña:

| Dataset / tarea | Clases | Slides | Pacientes | Folds | Train por fold | Test por fold | Balance |
|---|---:|---:|---:|---:|---:|---:|---|
| `cptac_brca` / TP53_mutation | 2 | 112 | 103 | 50 | 75 | 21.5 | 40.2 % positivos |
| `cptac_gbm` / TP53_mutation | 2 | 243 | 99 | 50 | 164 | 45.3 | 33.7 % positivos |
| `bc_therapy` / er_status | 2 | 166 | 166 | 50 | 113 | 33 | 69.3 % positivos |
| `cervical_subtype` / subtype | **4** | 599 | 599 | **5** | 407 | 119.8 | 34/42/13/11 % |

**El número de pacientes, no el de slides, es el tamaño muestral efectivo**: la estratificación es a
nivel de paciente (`StratifiedGroupKFold`), de modo que todas las slides de un `case_id` van juntas.

Datasets descartados y por qué: `cptac_ccrcc`, `cptac_hnsc`, `cptac_lscc` (solo tienen `uni_v2`),
`tcga_brca` (sin `uni_v2`), `hancook` y `crc_outcomes` (sin features), `cptac_coad` (98 slides,
el más pequeño), `imp` (**1 solo fold**).

`cervical_subtype` merece una nota: con 407 slides de entrenamiento debería ser el caso más
favorable, pero **no arbitra nada** porque difiere del resto en cuatro ejes a la vez — solo 5 folds,
4 clases en vez de 2, techo de AUC en 0.943 (efecto techo) y tejido/tarea distintos. Es un dato,
no un contraejemplo limpio.

## 2.2 El pipeline

```
WSI → parcheado 20x_224px_0px_overlap → ~3 700–6 600 parches/slide
  │
  ├─ modelo fundacional A → embeddings [N, in_dim] → ABMIL_A → p_A  ┐
  ├─ modelo fundacional B → embeddings [N, in_dim] → ABMIL_B → p_B  ├─→ metaclasificador → predicción
  └─ modelo fundacional C → embeddings [N, in_dim] → ABMIL_C → p_C  ┘
```

Un ABMIL independiente por modelo fundacional y fold (**fusión tardía**), y un metaclasificador
entrenado sobre las probabilidades concatenadas. Configuración del ABMIL: `in_dim` autodetectado
del `.h5`, 1 cabeza de atención, `head_dim` 512, dropout 0.25, 100 épocas con parada temprana y
partición interna de validación agrupada por paciente.

**Metaclasificadores registrados en el código (15) frente a evaluados (6).** Están implementados
`logreg`, `logit_avg`, `mlp`, `deep_mlp`, `mlp_snapshot`, `deep_mlp_snapshot`, `mlp_fge`,
`mlp_deepens`, `gating`, `nb`, `knn`, `svm`, `lightgbm`, `tabpfn` y `tabpfn_snapshot`. Los que se
han evaluado realmente sobre los checkpoints vigentes son seis: `logreg`, `mlp`, `mlp_snapshot`,
`svm`, `knn` y `nb` (sobre probabilidades), más `mlp_fge` (sobre embeddings). **`lightgbm`,
`tabpfn` y `tabpfn_snapshot` nunca se han evaluado en la campaña vigente** — solo existen cifras
suyas de la etapa previa a la regeneración de checkpoints, que no son reproducibles. Es
precisamente el hueco que señala la vía abierta 6.

**Una nota sobre `ensemble.py` (promediado ponderado).** Es la rama alternativa al stacking y no ha
sido objeto de la campaña, pero conviene saber que `MetricDistance` en `src/utils.py` normaliza los
pesos min-max y los reescala a suma 1 (no es un softmax, pese al comentario del código), lo que
asigna **peso exactamente 0 al peor modelo de cada fold, siempre**. El comportamiento está ahora
documentado y se puede acotar con `weight_floor` (por defecto 0.0, para no alterar resultados
previos). Cualquier cifra que provenga de esta rama debe leerse con ese detalle presente.

**Configuración recomendada:**

```bash
python src/ensemble4.py \
    --foundational_models ctranspath uni_v2 conch_v1_5 \
    --work_dir /home/JKP6679/Patho-Ensemble/PARADIS/datos/patches \
    --train_source cptac_brca --tissue_patching 20x_224px_0px_overlap \
    --task_name TP53_mutation --meta_model logreg
```

## 2.3 Los ocho modelos fundacionales

AUC individual sobre `cptac_brca`/TP53_mutation, 50 folds (cada uno con su propio ABMIL, sin
ensemble):

| Modelo | `in_dim` | AUC test | AUC validación | Brecha train→test |
|---|---:|---:|---:|---:|
| **`conch_v1_5`** | 768 | **0.7902** | 0.8536 | +0.1591 |
| `virchow2` | 2560 | 0.7771 | 0.8245 | +0.1392 |
| `uni_v1` | 1024 | 0.7732 | 0.8011 | +0.1700 |
| `phikon_v2` | 1024 | 0.7685 | 0.8093 | +0.1895 |
| `uni_v2` | 1536 | 0.7634 | 0.8369 | +0.1052 |
| `ctranspath` | 768 | 0.7571 | 0.8117 | +0.0918 |
| `hoptimus1` | 1536 | 0.7450 | 0.7998 | +0.1962 |
| `virchow_v1` | 2560 | **0.6823** | 0.7319 | +0.1149 |

**`conch_v1_5` en solitario (0.7902) supera al ensemble completo del trío histórico
`ctranspath+uni_v2+virchow_v1` (0.7616).** Es el dato que resume toda la campaña.

**Memorizar más no predice mejor:** `phikon_v2` es el que más memoriza (0.9580 en
meta-entrenamiento) y queda 4.º en test; `ctranspath` es el que menos memoriza y queda 6.º.

## 2.4 Redundancia entre modelos

Correlación de Pearson entre las probabilidades de clase 1, 50 folds:

- `ctranspath` ↔ `uni_v2` = **0.84** — casi redundantes, y ambos están en el trío ganador.
- `virchow_v1` (el peor en solitario) es el **más decorrelacionado** de `conch_v1_5` (0.35).
- `hoptimus1` es el más independiente del comité en conjunto (0.35–0.66).

**El criterio de AUC individual que justificó el cambio de modelo base no es el que maximiza un
comité: la redundancia importa tanto como la calidad.** Que `virchow_v1` sea el más diverso y aun
así prescindible indica que la diversidad, por sí sola, tampoco basta.

## 2.5 Grupos de coordenadas de parche

Para fusionar dos modelos a nivel de parche (o comparar sus atenciones) hacen falta coordenadas
idénticas. **Los grupos de coordenadas son una propiedad del parcheado de cada dataset, no de los
modelos** — verificado sobre 30 slides por dataset:

| Dataset | `ctranspath` == `conch_v1_5` | Los tres del trío iguales |
|---|---|---|
| `cptac_brca` | 30/30 | **0/30** |
| `cptac_gbm` | 30/30 | **30/30** |
| `bc_therapy` | 30/30 | **30/30** |
| `cervical_subtype` | 30/30 | **30/30** |

**Consecuencia práctica:** en tres de los cuatro datasets se puede fusionar el trío entero a nivel
de parche. La limitación que impidió probar la fusión temprana a tres bandas es específica de
`cptac_brca`. Esta fusión sigue sin probarse (vía abierta 2, sección 4).

## 2.6 De dónde viene la brecha entre entrenamiento y test

Una pregunta natural es si el metaclasificador sobreajusta. La respuesta medida es **no**:

| | Meta-entrenamiento | Test | Brecha |
|---|---:|---:|---:|
| LogReg (trío ganador) | 0.9495 | 0.7986 | +0.1509 |
| `conch_v1_5` solo (su mejor entrada) | 0.9493 | 0.7902 | +0.1591 |
| **Aportación del metaclasificador** | **+0.0003** | **+0.0084** | — |

El metaclasificador **no crea la brecha, la hereda** (aporta +0.0003 dentro de la muestra) y **no
falla en test** (aporta +0.0084 y gana al mejor modelo base fijo en 27 de 50 folds). El origen real
de la brecha es el ABMIL: ~0.5 M de parámetros entrenados con 75 slides.

---

# 3. Qué funcionó y qué no

Esta es la sección central. Cada experimento se ficha con: **dataset**, **modelos base**,
**metaclasificador**, **extras** (FGE, Snapshot, atención, fusión...), resultado numérico,
veredicto y una explicación en lenguaje llano de por qué salió así.

## 3.1 Tabla de una ojeada

### ✅ Lo que funcionó

| # | Experimento | Dataset | Modelos base | Meta | Extras | Δ AUC | Veredicto |
|---|---|---|---|---|---|---:|---|
| **P1** | Cambiar un modelo base | `cptac_brca` | ctranspath + uni_v2 + **conch_v1_5** | logreg | — | **+0.0370** | ✅ El mayor efecto del proyecto |
| **P2** | Embeddings en vez de probabilidades | `cptac_gbm` | trío ganador | logreg | espacio 1536 D | **+0.0230** | ✅ Solo en datasets grandes |
| **P3** | Snapshot Ensembles en modelos base | `cptac_gbm` | trío ganador | logreg | **FGE / SE** | **+0.0223** | ✅ Las 5 métricas |
| **P4** | La atención aporta (control negativo) | `cptac_brca` + `cptac_gbm` | par / trío | logreg | `attn_mean` | **−0.0333 / −0.0479** al quitarla | ✅ Aporta, y más con más datos |
| **P5** | El stacking bate al mejor modelo suelto | `cptac_brca` | 8 modelos | logreg | — | +0.0084 | ⚠️ Modesto pero real |
| **P6** | La vecindad espacial lleva señal | `cptac_brca` | uni_v2 | — | GNN vs grafo barajado | +0.0119 | ⚠️ Real pero no rentable |
| **P7** | Pooling duro top-10 de atención | `cptac_gbm` | trío ganador | logreg | top-K vs blando sobre toda la bolsa | **+0.0388** | ✅ Solo en datasets grandes (repite P2) |
| **P8** | Pooling duro top-K adaptativo por ESS (sin barrido) | `cptac_gbm` | trío ganador | logreg | K por slide = 1/Σaᵢ² (effective sample size) | **+0.0292** | ✅ Repite P7 sin elegir K a mano |

### ❌ Lo que no funcionó

Veinte experimentos. Las cuatro razones que los explican casi todos están en §3.4.

| # | Experimento | Dataset | Modelos base | Meta | Extras | Δ AUC | Veredicto |
|---|---|---|---|---|---|---:|---|
| **N1** | Metaclasificadores alternativos | 4 datasets | trío ganador | mlp, svm, knn, nb, mlp_snapshot | — | −0.104 … −0.002 | ❌ Ninguno bate a logreg |
| **N2** | MLP como metaclasificador | 4 datasets | trío ganador | mlp | — | −0.080 … −0.106 | ❌ Colapsa en todos |
| **N3** | Fusión temprana (concatenar parches) | `cptac_brca` | ctranspath ⊕ virchow_v1 ⊕ conch_v1_5 | logreg | ABMIL único 4096 D | −0.0056 | ❌ Sin efecto |
| **N4** | Fusión temprana ponderada | `cptac_brca` | ídem, escalados | — | pesos sobre embeddings | −0.0182 | ❌ Empeora |
| **N5** | Optimización de pesos Nelder-Mead | `cptac_brca` | ctranspath + virchow_v1 + conch_v1_5 | — | combinación lineal | +0.0052 | ❌ Bajo el suelo de reproducibilidad |
| **N6** | Escalera de características | `cptac_brca` | trío histórico | logreg | desacuerdo + margen + **entropía** | −0.0014 | ❌ Nulo |
| **N7** | Reponderación por importancia | `cptac_brca` | trío histórico | logreg, svm | 3 métodos | −0.006 … −0.012 | ❌ Empeora |
| **N8** | Atención multi-cabeza | `cptac_brca` | ctranspath | — | 4 cabezas | −0.0240 | ❌ Empeora |
| **N9** | Atención con GNN espacial | `cptac_brca` | uni_v2 | — | GraphSAGE | +0.0077 | ❌ No significativo |
| **N10** | Augmentación de bolsa (`bag_size`) | `cptac_brca` | trío ganador | logreg | 256/512/1024 | +0.0077 máx. | ❌ 12 comparaciones, 0 significativas |
| **N11** | Corregir la fuga con OOF | `cptac_brca` | trío histórico | logreg | CV anidada | −0.0045 | ❌ Empate |
| **N12** | Ampliar el comité de 3 a 8 | `cptac_brca` | 255 subconjuntos | logreg | — | +0.0002 | ❌ Nulo |
| **N13** | Enrutar el comité por órgano | 13 tareas, 3 cohortes | hoptimus1+uni_v1+uni_v2+virchow2 | logreg | apagado dinámico | +0.0015 | ❌ p = 0.31 |
| **N14** | Cambiar el criterio de selección | `cptac_brca` | 8 modelos | logreg | 8 criterios | ≤ 0 | ❌ Ninguno mejora el argmax |
| **N15** | Promediar las atenciones | `cptac_brca` + `cptac_gbm` | par / trío | logreg | `attn_avg` | −0.0009 / −0.0061 | ❌ Nulo estrecho |
| **N16** | Transferir la atención de conch | `cptac_gbm` | trío ganador | logreg | `attn_conch` | +0.0079 | ❌ No replica (25/0/25) |
| **N17** | FGE / SE en datasets pequeños | `cptac_brca`, `cervical_subtype` | trío | logreg | FGE, SE | +0.002 / −0.010 | ❌ Nulo / negativo |
| **N18** | Barrido de hiperparámetros del MLP | `cptac_brca` | trío histórico | mlp, mlp_snapshot | 369 + 73 configuraciones | sin contraste pareado | ❌ Ninguna gana en las 3 métricas |
| **N19** | Selección LOFO con bagging (stability selection) | `cptac_brca` | 255 subconjuntos | logreg | bootstrap=2000 sobre folds != k | −0.0025 vs honesta simple | ❌ Vuelve la selección más conservadora, no más certera |
| **N20** | CMA-ES para pesos del ensemble | `cptac_brca` | trío ganador + histórico | — | combinación lineal, CMA-ES vs Nelder-Mead vs logreg | ≤ +0.0021 | ❌ Ningún Δ significativo ni por encima del suelo |

### ⏸️ Incompletos

| # | Experimento | Dataset | Estado |
|---|---|---|---|
| **I1** | Agregación a nivel de paciente antes del metaclasificador | `cptac_brca` (9 pacientes con >1 slide) | 3 trabajos fallidos, nunca completado |
| **I2** | Atención con GAT / transformers entre parches | `cptac_brca` | 8 trabajos lanzados, ninguno llegó a entrenar; código eliminado |

---

## 3.2 Fichas — lo que funcionó

### P1 · Cambiar un modelo base: `virchow_v1` → `conch_v1_5`

| | |
|---|---|
| **Dataset** | `cptac_brca` / TP53_mutation — 50 folds pareados |
| **Modelos base** | `ctranspath` + `uni_v2` + **`virchow_v1`** → `ctranspath` + `uni_v2` + **`conch_v1_5`** |
| **Metaclasificador** | LogReg (configuración por defecto) |
| **Extras** | Ninguno |

| Métrica | Trío histórico | Trío con conch | Δ | IC 95 % | W/T/L | p Holm | MDE |
|---|---:|---:|---:|---|---|---:|---:|
| AUC | 0.7616 | **0.7986** | **+0.0370** | [+0.0230, +0.0514] | 34/4/12 | 1.4·10⁻⁴ | 0.0209 |
| bacc | 0.6702 | **0.7302** | **+0.0600** | [+0.0363, +0.0838] | 36/5/9 | 1.4·10⁻⁴ | 0.0345 |
| macro-F1 | 0.6687 | **0.7294** | **+0.0608** | [+0.0362, +0.0857] | 36/5/9 | 1.4·10⁻⁴ | 0.0365 |
| kappa | 0.3502 | **0.4629** | **+0.1127** | [+0.0657, +0.1601] | 36/5/9 | 1.4·10⁻⁴ | 0.0680 |

**Veredicto: ✅ FUNCIONA.** Las cuatro métricas se mueven en el mismo sentido, todas superan su MDE
por un factor de 1.7–1.8× y todas sobreviven a la corrección de Holm.

**Por qué funciona, en llano.** `virchow_v1` no era simplemente el modelo más débil: era un modelo
*dañino*. Produce embeddings de 2 560 dimensiones que el ABMIL debe comprimir a 512, y en esa
compresión **solo sobrevive el 5 % de la estructura que distinguía unos parches de otros**. Sin esa
estructura, su atención no tiene nada que mirar (entropía 0.9993, prácticamente uniforme) y el
modelo acaba promediando parches. El metaclasificador recibe entonces una tercera opinión que es
casi ruido, y como LogReg le asigna un peso no nulo, ese ruido contamina la decisión.

Tres controles confirman el diagnóstico:

- **Añadir `conch_v1_5` sin quitar `virchow_v1`** (cuatro modelos) da exactamente el mismo AUC,
  0.7986. `virchow_v1` no aporta nada que `conch_v1_5` no cubra ya.
- **Quitar `virchow_v1` sin sustituirlo** (`ctranspath+uni_v2` = 0.7650) ya bate al trío histórico.
- **En un barrido exhaustivo de los 255 subconjuntos posibles de 8 modelos**, la selección honesta
  (leave-one-fold-out, sin mirar el test) elige `ctranspath+uni_v2+conch_v1_5` en **49 de 50
  folds**. El sesgo de selección es de solo +0.0046.

**Lo que hay que llevarse:** el metaclasificador aporta +0.008 sobre el mejor modelo individual;
cambiar qué modelo entra aporta +0.037. Antes de invertir en cómo se combinan los modelos, hay que
comprobar cuáles entran.

---

### P2 · Embeddings en vez de probabilidades — depende del dataset

| | |
|---|---|
| **Dataset** | Los cuatro (`cptac_brca`, `bc_therapy`, `cptac_gbm`, `cervical_subtype`) |
| **Modelos base** | Trío ganador `ctranspath + uni_v2 + conch_v1_5` |
| **Metaclasificador** | LogReg en ambos brazos |
| **Extras** | Espacio de entrada: 6 probabilidades vs. **1 536 D** (3 × 512 D post-atención) |

| Dataset | Train por fold | Δ AUC | IC 95 % | W/T/L | p Holm |
|---|---:|---:|---|---|---:|
| `cptac_brca` | 75 | **−0.0160** | [−0.0305, −0.0020] | 19/2/29 | 0.60 |
| `bc_therapy` | 113 | **+0.0296** | [+0.0073, +0.0530] | 30/1/19 | 0.35 |
| **`cptac_gbm`** | **164** | **+0.0230** | [+0.0079, +0.0384] | 34/0/16 | **0.031** ✱ |
| `cervical_subtype` | 407 | −0.0009 | [−0.0042, +0.0029] | 2/0/3 | 1.00 |

**Veredicto: ✅ FUNCIONA en `cptac_gbm`** (única con significación clara tras Holm) y
direccionalmente en `bc_therapy` (el IC excluye el cero, pero no supera Holm). **No funciona en
`cptac_brca`**, donde el signo se invierte. En `cptac_gbm` las cinco métricas se mueven en el mismo
sentido (AUC +0.0230, AP +0.0235, bacc +0.0117, macro-F1 +0.0127, kappa +0.0275).

**Por qué — y por qué NO es lo que parece.** La explicación intuitiva sería el tamaño del
meta-entrenamiento: "una regresión logística sobre 1 536 columnas con 75 filas sobreajusta; con 164
no". **Esa hipótesis se probó explícitamente y quedó falsada.** Submuestreando el meta-train de
`cptac_gbm` a 40, 55, 75, 95, 120, 145 y 169 filas:

| n meta-train | Δ AUC (emb − prob) | IC 95 % | W/T/L | p Holm |
|---:|---:|---|---|---:|
| 40 | +0.0053 | [−0.0037, +0.0138] | 28/0/22 | 0.86 |
| 55 | +0.0169 | [+0.0072, +0.0269] | 31/0/19 | **0.013** ✱ |
| **75** | **+0.0137** | [+0.0018, +0.0254] | 34/0/16 | 0.090 |
| 95 | +0.0201 | [+0.0088, +0.0317] | 34/0/16 | **0.008** ✱ |
| 120 | +0.0170 | [+0.0042, +0.0297] | 33/0/17 | **0.047** ✱ |
| 145 | +0.0198 | [+0.0057, +0.0335] | 32/1/17 | **0.030** ✱ |
| 169 (completo) | +0.0230 | [+0.0077, +0.0380] | 34/0/16 | **0.013** ✱ |

**El Δ es positivo en los siete tamaños, incluido n = 40**, por debajo de los 75 de `cptac_brca`.
En el punto decisivo — n = 75 exactos, mismo código, mismos modelos — `cptac_gbm` da **+0.0137**
(34 de 50 folds) y `cptac_brca` da **−0.0160** (19 de 50). El tamaño modula la *magnitud* del
efecto, pero nunca cambia su *signo*: **la diferencia es una propiedad del dataset, no del tamaño
de la muestra.**

**La hipótesis candidata (aún no establecida).** La calidad del embedding depende de cuán selectiva
sea la atención que lo produjo, y los datasets difieren en selectividad:

| Modelo | Entropía en `cptac_brca` | Entropía en `cptac_gbm` | `logit_sd` brca | `logit_sd` gbm |
|---|---:|---:|---:|---:|
| `ctranspath` | 0.9282 | **0.7751** | 0.98 | **1.74** |
| `uni_v2` | 0.9942 | **0.9699** | 0.30 | **0.70** |
| `conch_v1_5` | 0.8879 | 0.8736 | 1.29 | 1.23 |

En `cptac_gbm` la atención es sistemáticamente más selectiva. Y al aplanarla a la fuerza
(`attn_mean`), **el signo del efecto se invierte** en `cptac_gbm` (de +0.023 a −0.025, un vuelco de
0.048), mientras que en `cptac_brca` — donde ya era casi plana — solo se profundiza una pérdida
preexistente.

⚠️ **Esto es una hipótesis compatible con los datos, no un resultado establecido.** La evidencia son
dos datasets y una única manipulación llevada al extremo uniforme. El contraste decisivo —
*graduar* la selectividad y comprobar si el Δ la sigue de forma monótona — está pendiente
(vía abierta 4, sección 4).

---

### P3 · Snapshot Ensembles (FGE / SE) sobre los modelos base

| | |
|---|---|
| **Dataset** | `cptac_gbm` (positivo), `bc_therapy` (misma dirección), `cervical_subtype` (negativo) |
| **Modelos base** | Trío ganador, con variantes de snapshot por modelo |
| **Metaclasificador** | LogReg (también `mlp_snapshot`) |
| **Extras** | **FGE** (Garipov et al., ICLR 2018): `lr` 2·10⁻⁴ → 2·10⁻⁵, ciclos de 4 épocas. **SE** (Huang et al., ICLR 2017): `lr` 10⁻³ → 10⁻⁶, ciclos de 15 épocas |

`cptac_gbm`, `fge_trio:logreg` frente a `prob:logreg`, 50 folds — **las cinco métricas
significativas tras Holm**:

| Métrica | Δ | IC 95 % | W/T/L | p Holm | MDE |
|---|---:|---|---|---:|---:|
| AUC | **+0.0223** | [+0.0090, +0.0381] | 31/0/19 | **0.049** ✱ | 0.0209 |
| AP | **+0.0264** | [+0.0083, +0.0454] | 33/0/17 | **0.049** ✱ | 0.0268 |
| bacc | **+0.0314** | [+0.0147, +0.0497] | 32/5/13 | **0.015** ✱ | 0.0253 |
| macro-F1 | **+0.0327** | [+0.0157, +0.0516] | 32/5/13 | **0.015** ✱ | 0.0259 |
| kappa | **+0.0653** | [+0.0315, +0.1013] | 31/5/14 | **0.011** ✱ | 0.0506 |

En `bc_therapy`: `se_trio:mlp_snapshot` Δ AUC = **+0.0296**, p Holm = **0.016** ✱.
En `cervical_subtype`: negativo (`fge_trio:logreg` −0.0098).

**Veredicto: ✅ FUNCIONA en `cptac_gbm` y `bc_therapy`.** Es el resultado más significativo de
toda la campaña después del cambio de modelo base.

**Por qué el resultado anterior era engañoso.** En `cptac_brca` esta misma comparación daba
+0.0020, sin significación, y se cerró como "nulo acotado a ≈+0.017". Pero el **MDE de esa
comparación era 0.026** — unas 4 veces el de la suite de metaclasificadores — porque las
predicciones de los snapshots son mucho más ruidosas fold a fold. Aquel nulo no decía "no hay
efecto": decía **"no puedo ver un efecto menor de 0.026"**. El efecto real ronda +0.022. **Era un
problema de potencia estadística, no una ausencia de efecto.**

Esta es la lección metodológica más transferible de la campaña: **un nulo solo es informativo si se
reporta junto a su MDE**, y un nulo cuyo MDE es mayor que el efecto plausible no es un nulo, es una
medición que no se hizo.

---

### P4 · La atención sí aporta (control negativo `attn_mean`)

| | |
|---|---|
| **Dataset** | `cptac_brca` (par `ctranspath+conch_v1_5`) y `cptac_gbm` (trío completo) |
| **Metaclasificador** | LogReg |
| **Extras** | `attn_mean` = sustituir la atención aprendida por *mean pooling* uniforme |

| Dataset | Δ AUC al forzar mean pooling | IC 95 % | p Holm |
|---|---:|---|---:|
| `cptac_brca` | **−0.0333** | [−0.0460, −0.0205] | **0.00046** ✱ |
| `cptac_gbm` | **−0.0479** | [−0.0699, −0.0260] | **0.00090** ✱ |

**Veredicto: ✅ La atención aporta**, y aporta **más** en el dataset con más datos, lo cual es
coherente con todo el resto del informe.

**Por qué importa.** Es el control que valida la arquitectura completa. Muchos de los resultados
negativos de la campaña son intentos de *mejorar* la atención (multi-cabeza, GNN espacial,
atención compartida) que fracasan; este control demuestra que el problema no es que la atención sea
inútil — es que **la que hay ya captura lo que se puede capturar con este tamaño de cohorte**.

---

### P5 · El stacking bate al mejor modelo individual

Aportación medida del metaclasificador sobre `cptac_brca`: **+0.0084 de AUC en test** frente al
mejor modelo base fijo, ganando en **27 de 50 folds**. Dentro de la muestra aporta solo +0.0003, lo
que confirma que no está memorizando: está combinando.

**Veredicto: ⚠️ Real pero modesto.** Es la referencia contra la que hay que medir cualquier idea
sobre la regla de combinación: **el techo de esa palanca es +0.008**, y ninguna de las nueve
alternativas probadas lo ha superado.

---

### P6 · La vecindad espacial entre parches lleva información

| | |
|---|---|
| **Dataset** | `cptac_brca` / TP53_mutation |
| **Modelo base** | `uni_v2` (elegido por tener el mejor vecindario: grado medio 7.78/8, 0 % nodos aislados) |
| **Extras** | GraphSAGE sobre grafo reticular de parches (hasta 8 vecinos, K = 2 rondas ≈ 560 µm de contexto), **antes** de la atención |

| Variante | AUC | Δ vs. baseline | IC 95 % | W/T/L |
|---|---:|---:|---|---|
| ABMIL 1 cabeza (baseline) | 0.7634 | — | — | — |
| + GNN espacial | 0.7663 | +0.0029 | [−0.0115, +0.0162] | 22/14/14 |
| + GNN espacial (p = 0.3) | 0.7711 | +0.0077 | [−0.0069, +0.0218] | 24/9/17 |
| + GNN con **grafo barajado** (control) | 0.7592 | −0.0042 | [−0.0165, +0.0074] | 17/12/21 |

**El contraste limpio** es GNN con grafo real (0.7711) frente a GNN con grafo barajado (0.7592) —
misma arquitectura, mismos parámetros, misma capacidad, solo destruida la estructura espacial:
**Δ = +0.0119, IC [+0.0004, +0.0238], W/T/L 26/8/16 — significativo, por poco.**

**Veredicto: ⚠️ La señal existe, pero no es rentable.** La vecindad espacial entre parches sí
lleva información sobre TP53. Pero frente al baseline sin GNN la ganancia neta cae a +0.0077 (no
significativo, y por debajo del suelo de reproducibilidad): **el resto se lo come el coste de
añadir 1.05 M de parámetros** en un problema con 103 pacientes.

Es el hallazgo que mejor ilustra el diagnóstico general del proyecto: **no falta información, falta
muestra para pagarla.**

---

### P7 · Pooling duro top-K de atención — refuerza P2, no rescata al MLP+snapshot

| | |
|---|---|
| **Dataset** | `cptac_brca` + `cptac_gbm` / TP53_mutation |
| **Modelos base** | Trío ganador `ctranspath + uni_v2 + conch_v1_5` |
| **Metaclasificador** | LogReg y `mlp_snapshot` (`SnapshotMLPMetaClassifier`), mismo espacio de entrada |
| **Extras** | `abmil_extract_topk_embeddings` (nuevo): en vez de agregar con atención blanda sobre TODA la bolsa, selecciona los K=10 parches de mayor peso y renormaliza su atención antes de agregar — línea "regularizar la atención con top-k de parches" que constaba como no probada (vía abierta 8, sección 4) |

Baseline recomputado en la misma pasada (`insample:logreg`, 50 folds pareados en cada dataset):

| Dataset | `topk10:logreg` Δ AUC | IC 95 % | W/T/L | p Holm | `topk10:mlp_snapshot` Δ AUC | p Holm |
|---|---:|---|---|---:|---:|---:|
| `cptac_brca` | +0.0140 | [−0.0034, +0.0315] | 24/7/19 | 0.243 (ns) | +0.0084 | 0.351 (ns) |
| **`cptac_gbm`** | **+0.0388** | [+0.0208, +0.0568] | 35/1/14 | **<0.001 ✱** | **+0.0326** | **<0.001 ✱** |

En `cptac_gbm` también sale significativa la balanced accuracy (+0.0250 ✱, sólo con `logreg`);
kappa y macro-F1 quedan en zona "significativo sin corregir, no tras Holm" (p Holm 0.064–0.090). En
`cptac_brca` las cinco métricas quedan nulas y acotadas (MDE 0.025–0.065).

**Veredicto: ✅ Refuerza P2, con una tercera manipulación de la atención.** Mismo patrón exacto que
embeddings-vs-probabilidades: gana en el dataset con atención más selectiva y más meta-train
(`cptac_gbm`), nulo acotado en `cptac_brca`. No es todavía el contraste *graduado* que pide la vía
abierta 4, sección 4 — aquí sólo se probó un punto (K=10), no una curva — pero es la tercera pieza de
evidencia independiente (tras `attn_mean` en P4 y los embeddings en P2) de que la selectividad de
la atención, no el tamaño de la muestra por sí solo, es lo que separa los dos datasets.

**El metaclasificador no aporta nada sobre el espacio de features.** Dentro del mismo `topk10`,
`logreg` gana en AUC a `mlp_snapshot` en **los dos** datasets (0.8127 vs. 0.8070 en `cptac_brca`;
0.7945 vs. 0.7882 en `cptac_gbm`). Confirma N1/N2 con una entrada distinta: la ganancia, donde la
hay, es la representación (top-K duro), no el combinador.

⚠️ Igual que P2: es una hipótesis compatible con los datos (dos datasets, un valor de K), no un
resultado establecido para "el pooling duro ayuda en datasets grandes" en general.

---

### P8 · Pooling duro top-K adaptativo por ESS — repite P7 sin elegir K a mano

| | |
|---|---|
| **Dataset** | `cptac_brca` + `cptac_gbm` / TP53_mutation |
| **Modelos base** | Trío ganador `ctranspath + uni_v2 + conch_v1_5` |
| **Metaclasificador** | LogReg y `mlp_snapshot`, mismo espacio de entrada que P7 |
| **Extras** | `abmil_extract_adaptive_topk_embeddings` (nuevo): en vez de un K=10 fijo elegido a mano (P7), K se deriva **por slide** del ESS (effective sample size) de la propia atención — `K = 1/Σaᵢ²`, el mismo índice de Simpson inverso que ya calcula `attention_stats_extended` para N15/N16, aquí usado para controlar el pooling en vez de sólo diagnosticarlo. Sin hiperparámetro libre, sin barrido: se computa en el mismo forward pass que ya hacía el pooling duro |

**K efectivo usado (mediana, agregado sobre folds y splits) — nada de esto se fijó a mano:**

| Modelo | `cptac_brca` | `cptac_gbm` |
|---|---:|---:|
| `conch_v1_5` | 200 | **81** |
| `ctranspath` | 3602 | 318 |
| `uni_v2` | 6953 | 1274 |

`conch_v1_5` —el mejor modelo individual del trío— es también, en los dos datasets, el que más
selecciona (K más bajo). Y los tres modelos sacan K más bajo en `cptac_gbm` que en `cptac_brca`: el
ESS reproduce, como número derivado del propio modelo, la misma "atención más selectiva en datasets
grandes" que hasta ahora sólo constaba como diagnóstico aparte (entropía de P2/P7).

Baseline recomputado en la misma pasada (`insample:logreg`, 50 folds pareados en cada dataset):

| Dataset | `topk_ess:logreg` Δ AUC | IC 95 % | p Holm | MDE | `topk_ess:mlp_snapshot` Δ AUC | p Holm |
|---|---:|---|---:|---:|---:|---:|
| `cptac_brca` | −0.0090 | [−0.0223, +0.0043] | 0.385 (ns) | 0.0191 | +0.0035 | 0.429 (ns) |
| **`cptac_gbm`** | **+0.0292** | [+0.0151, +0.0432] | **<0.001 ✱** | 0.0201 | +0.0256 | **<0.001 ✱** |

**Veredicto: ✅ Cuarta pieza de evidencia independiente de la selectividad de la atención** (tras
`attn_mean` en P4, los embeddings en P2 y el top-10 fijo en P7) — **y la primera lograda sin elegir
ningún hiperparámetro a mano.** Mismo patrón exacto: nulo acotado en `cptac_brca`, significativo y
por encima del MDE en `cptac_gbm` (+0.0292, 1.45× su MDE, sobrevive Holm).

**Frente a P7:** el K=10 fijo dio +0.0388 en `cptac_gbm`, algo más que el +0.0292 de aquí. Elegir K a
mano, cuando se acierta, puede superar a un criterio automático — pero P7 nunca demostró que K=10
fuera una buena elección *a priori*, sólo que funcionó una vez probada. **P8 llega casi al mismo
sitio sin haber mirado el AUC ni una sola vez para elegir K**, lo que descarta que el resultado de
P7 fuera fruto de haber acertado con K=10 por azar entre candidatos no probados.

**El metaclasificador tampoco aporta aquí.** `mlp_snapshot` no bate a `logreg` en `cptac_gbm`
(+0.0256 vs. +0.0292) — confirma N1/N2/P7 con una cuarta entrada distinta: la ganancia, donde la
hay, es la representación, no el combinador.

⚠️ Mismo caveat que P2 y P7: dos datasets, no una curva graduada de selectividad — sigue siendo la
vía abierta 4 (sección 4) la que falta para establecer la relación como monótona en vez de puntual.

---

## 3.3 Fichas — lo que no funcionó

### N1–N2 · Cambiar el metaclasificador: nueve alternativas, ninguna gana

| | |
|---|---|
| **Datasets** | `cptac_brca` (50 folds) + réplica en `cptac_gbm`, `bc_therapy`, `cervical_subtype` |
| **Modelos base** | Trío ganador `ctranspath + uni_v2 + conch_v1_5` |
| **Metaclasificadores** | LogReg (referencia), MLP, MLP+Snapshot, Deep MLP, SVM lineal (C=1), KNN (k=5, pesos por distancia), Naive Bayes gaussiano, TabPFN, TabPFN+Snapshot, MLP+FGE |
| **Extras** | Ninguno. Espacio de 6 probabilidades, semilla `42+fold` |

Δ AUC frente a LogReg en los cuatro datasets:

| Brazo | `cptac_brca` | `bc_therapy` | `cptac_gbm` | `cervical_subtype` |
|---|---:|---:|---:|---:|
| `prob:svm` | −0.0021 | +0.0000 | **−0.0103** ✱ | −0.0027 |
| `prob:nb` | −0.0031 | −0.0041 | **−0.0119** ✱ | −0.0145 |
| `prob:knn` | −0.0321 ✱ | −0.0225 ✱ | **−0.0421** ✱ | −0.0409 |
| `prob:mlp` | **−0.1038** ✱ | **−0.1061** ✱ | **−0.0803** ✱ | −0.0409 |

**Veredicto: ❌ Ninguno bate a LogReg en ningún dataset.** Y en `cptac_gbm` — el dataset con más
datos — el margen de LogReg **se amplía**: SVM y NB, que empataban en `cptac_brca`, pasan a perder
significativamente.

**Por qué, en llano.** Con **75 filas y 6 columnas**, la flexibilidad no se paga. Los modelos
lineales empatan en cabeza (SVM lineal decide idéntico a LogReg en 40 de 50 folds — es literalmente
el mismo modelo); el generativo simple (NB) empata; los flexibles (MLP, KNN) pierden. KNN falla
porque 5 vecinos en un espacio de 6 dimensiones con 75 puntos no bastan para estimar una densidad.
**El cuello de botella es el tamaño del conjunto, no la capacidad del modelo.**

**El caso del MLP merece un matiz importante.** El MLP pierde 0.08–0.11 de AUC en los cuatro
datasets, pero su fracaso tiene **dos componentes distintos** que conviene separar:

| Brazo | κ con umbral 0.5 | κ con umbral reajustado | Recuperación |
|---|---:|---:|---:|
| `prob:mlp` | 0.0340 | 0.3282 | **+0.294** |
| `prob:mlp_snapshot` | 0.2536 | 0.4278 | **+0.174** |
| `emb:mlp` | 0.2187 | 0.4148 | **+0.196** |
| LogReg / SVM / NB | — | — | **−0.029 a −0.045** (empeoran) |

Es decir: **buena parte del hundimiento del MLP en las métricas de decisión es un fallo de umbral,
no de ordenación.** `mlp_snapshot` llega a tener el AUC más alto de todo el barrido (0.8018) y a la
vez la segunda peor κ (0.2536): es un buen ordenador con el corte mal puesto. Aun así, el MLP
simple **sí** pierde también en AUC (−0.10), de modo que su colapso es real, no solo de umbral.

**No se adopta el reajuste de umbral en el pipeline** (sección 1.3): con ~15 slides de validación
por fold, el umbral estimado es más ruidoso que fijar 0.5.

---

### N3–N4 · Fusión temprana a nivel de parche

| | |
|---|---|
| **Dataset** | `cptac_brca` / TP53_mutation, 50 folds pareados |
| **Modelos base** | `ctranspath` (768 D) ⊕ `virchow_v1` (2560 D) ⊕ `conch_v1_5` (768 D) concatenados **a nivel de parche** → un único ABMIL con `in_dim` = 4096 |
| **Metaclasificador** | LogReg en ambos brazos (comparación justa) |
| **Extras** | `ensemble5.py`, con verificación dura de alineación de coordenadas |

| Variante | Δ AUC | IC 95 % | W/T/L | p |
|---|---:|---|---|---:|
| Fusión temprana vs. tardía | **−0.0056** | [−0.0250, +0.0124] | 21/3/26 | 0.55 |
| Fusión temprana **ponderada** (pesos sobre los embeddings crudos) | **−0.0182** | [−0.0421, +0.0061] | 20/1/29 | 0.15 |

**Veredicto: ❌ No funciona, y ponderar lo empeora.**

**Por qué, en llano.** La idea era razonable: la fusión tardía colapsa cada modelo a una
probabilidad por separado, así que el desacuerdo *parche a parche* entre modelos se pierde antes de
que nadie lo pueda usar. Concatenar los embeddings antes del ABMIL debería permitir que la atención
resolviera ese desacuerdo local.

No ocurre, por dos razones:

1. **El vector fusionado de 4 096 D desborda un mecanismo de atención diseñado para entradas de
   768–2 560 D de un solo modelo.** La proyección a 512 D pasa de comprimir 1.5× a comprimir 8×.
2. **La concatenación no da al modelo ninguna forma de saber qué dimensiones pertenecen a qué
   modelo base.** Ve 4 096 números sin estructura; no hay nada que le indique que las primeras 768
   y las siguientes 2 560 son dos opiniones sobre el mismo parche.

Y la variante ponderada es peor todavía porque **escalar embeddings crudos por pesos arbitrarios
(rango 0.04–0.99) rompe la estructura que el modelo fundacional aprendió**, sin sustituirla por
nada. La proyección del ABMIL se entrena asumiendo magnitudes sin escalar.

⚠️ **Salvedad honesta:** esta medición se hizo en `cptac_brca`, el único dataset donde el trío
completo **no** comparte coordenadas de parche, así que solo pudieron fusionarse dos de los tres
modelos. En los otros tres datasets sí se puede fusionar el trío entero, y eso **nunca se ha
probado** (vía abierta 2).

---

### N5 · Optimización de pesos con Nelder-Mead

| | |
|---|---|
| **Dataset** | `cptac_brca` / TP53_mutation, 50 folds |
| **Modelos base** | `ctranspath` + `virchow_v1` + `conch_v1_5` |
| **Metaclasificador** | Ninguno tradicional: combinación lineal `P = w₁P₁ + w₂P₂ + w₃P₃`, pesos normalizados por softmax y optimizados con `scipy.optimize.minimize(method='Nelder-Mead')` maximizando el AUC por fold |

**Resultado:** Δ AUC = **+0.0052**, IC [+0.0007, +0.0106], W/T/L 26/11/13, **p = 0.058**.

Pesos aprendidos (media ± desviación entre folds): `ctranspath` 17.6 % ± 11.2 %,
`virchow_v1` 10.4 % ± 7.8 %, **`conch_v1_5` 72.0 % ± 29.2 %** (rango 17.5 %–99.1 %).

**Veredicto: ❌ No concluyente.** Nominalmente p > 0.05. Y aunque su IC excluya el cero, **+0.0052
cae por debajo del suelo de reproducibilidad (≈0.004–0.01)**: es indistinguible de relanzar el
mismo trabajo con otra semilla de GPU.

**Por qué, en llano.** Nelder-Mead detecta algo real — que LogReg infrapondera a `conch_v1_5` — pero
el efecto es demasiado pequeño para separarlo del ruido. Y los pesos son **muy inestables fold a
fold** (una desviación del 29 % sobre una media del 72 %): en un fold asigna 0.95 a `ctranspath`, en
otro 0.99 a `conch_v1_5`. Eso es señal de que está ajustando ruido, no una estructura estable.

Es, aun así, **el mejor resultado que ha dado nunca tocar la regla de combinación** — lo cual dice
más sobre el techo de esa palanca que sobre Nelder-Mead.

---

### N6 · Escalera de características: desacuerdo, margen y entropía

| | |
|---|---|
| **Dataset** | `cptac_brca` / TP53_mutation, 50 folds |
| **Modelos base** | Trío histórico |
| **Metaclasificador** | LogReg, con bloque de columnas extra concatenado |
| **Extras** | Escalera aditiva: E0 (6 probabilidades) → E1 (+desacuerdo Jensen-gap) → E2 (+margen) → E3 (+**entropía** por modelo) |

| Escalón | Δ vs. E0 | IC 95 % | p | p Holm |
|---|---:|---|---:|---:|
| E1 (+desacuerdo) | −0.00038 | [−0.00109, +0.00031] | 0.294 | 0.294 |
| E2 (+margen) | −0.00057 | [−0.00155, +0.00026] | 0.214 | 0.642 |
| E3 (+entropía) | −0.00142 | [−0.00386, +0.00054] | 0.216 | 0.431 |

**Veredicto: ❌ Nulo.** Ningún escalón supera el baseline; todos los Δ son negativos y todos caen
dentro del MDE (≈±0.016).

**Por qué, en llano.** El argumento teórico era bueno: `H(p)` es simétrica (vale lo mismo en 0.1 que
en 0.9), así que ninguna combinación lineal de las probabilidades puede reproducirla, y en teoría
añade expresividad a un modelo lineal. Pero con **6 probabilidades ya informativas y 75 filas**,
LogReg ya extrae prácticamente toda la señal disponible a través de sus coeficientes. Las
transformaciones no lineales explícitas no aportan información nueva y sí consumen grados de
libertad. (Ver también la trampa de la columna degenerada en la sección 1.5.)

Una característica adicional, la divergencia KL/JS por pares entre modelos, se **descartó sin
implementar**: 3 columnas con fuerte colinealidad interna, información ya cubierta por el
Jensen-gap, y 25 filas por parámetro — zona de riesgo.

---

### N7 · Reponderación por importancia del modelo

| | |
|---|---|
| **Dataset** | `cptac_brca` / TP53_mutation, 50 folds |
| **Modelos base** | Trío histórico |
| **Metaclasificadores** | LogReg y SVM |
| **Extras** | Reescalar cada bloque de columnas por un escalar de importancia, con 3 métodos: `norm_ratio`, `softmax`, `signed` |

| Meta-modelo | Método | Δ AUC | W/T/L | p |
|---|---|---:|---|---:|
| logreg | `norm_ratio` | −0.0085 | 12/8/30 | 0.013 |
| logreg | `softmax` | −0.0123 | 11/5/34 | 0.002 |
| logreg | `signed` | −0.0105 | 17/7/26 | 0.337 |
| svm | `norm_ratio` | −0.0091 | 6/17/27 | 0.001 |
| svm | `softmax` | −0.0106 | 6/17/27 | 0.001 |
| svm | `signed` | −0.0064 | 20/3/27 | 0.657 |

**Veredicto: ❌ `norm_ratio` y `softmax` empeoran significativamente** (las 4 p sobreviven a Holm);
`signed` es indistinguible de no reponderar.

**Por qué, en llano.** Reescalar las características *antes* de un modelo que **ya va a aprender su
propia combinación lineal** es redundante: LogReg y SVM lineal ya optimizan exactamente esos pesos.
La reponderación no añade información, solo cambia el punto de partida de la regularización — y aquí
lo empeora.

---

### N8–N9 · Ampliar la capacidad de la atención

**N8 · Atención multi-cabeza.** `cptac_brca`, modelo base `ctranspath`, 4 cabezas con softmax
independiente y mean-pooling. **Δ AUC = −0.0240**, 5 folds, W/T/L 2/0/3.
**Veredicto: ❌ Empeora.** (Nota: 5 folds es exactamente el tamaño en el que un hallazgo de esta
campaña no replicó y cambió de signo; léase con esa cautela.)

**N9 · Atención con GNN espacial.** Ya detallada en P6: +0.0077 frente al baseline, no
significativo, aunque el control negativo con grafo barajado sí demuestra que hay señal espacial.

**Por qué, en llano — y es la explicación central de toda la campaña.** Hay que distinguir dos
presupuestos que se confunden constantemente:

- **Presupuesto de datos de la rama de atención:** ~7 000 parches por bolsa. Abundante.
- **Presupuesto estadístico del AUC final:** 50 folds sobre 103 pacientes, MDE ≈ ±0.016. Escaso.

Ampliar la atención mejora el primero y **no toca el segundo**. Con 75 slides de entrenamiento,
cualquier aumento de capacidad (multi-cabeza, GNN, transformers entre parches) se convierte en
memorización, no en señal nueva. Se probaron ocho variantes arquitectónicas a lo largo de la
campaña; ninguna mejoró, y la peor perdió 0.23 de AUC.

---

### N10 · Augmentación de bolsa (`bag_size`)

| | |
|---|---|
| **Dataset** | `cptac_brca` / TP53_mutation, 50 folds, baseline reentrenado en la misma pasada |
| **Modelos base** | Trío ganador |
| **Metaclasificador** | LogReg |
| **Extras** | Muestrear un subconjunto de parches por época durante el entrenamiento (256 / 512 / 1024), en vez de pasar la bolsa completa |

| Configuración | AUC | bacc | macro-F1 | kappa |
|---|---:|---:|---:|---:|
| Bolsa completa (baseline) | 0.8026 | **0.7448** | **0.7419** | **0.4886** |
| bag 256 | 0.8040 | 0.7428 | 0.7414 | 0.4872 |
| bag 512 | 0.8053 | 0.7442 | 0.7412 | 0.4870 |
| bag 1024 | **0.8103** | 0.7376 | 0.7349 | 0.4737 |

Δ frente a bolsa completa: bag 256 = +0.0014 (MDE 0.019), bag 512 = +0.0027 (MDE 0.017),
bag 1024 = +0.0077 (MDE 0.018). **Ninguna de las 12 comparaciones (3 tamaños × 4 métricas) es
significativa.**

**Veredicto: ❌ Nulo.** Y hay una razón adicional para no rescatar el +0.0077 de bag 1024: **sube el
AUC pero baja las otras tres métricas** (bacc −0.0072, macro-F1 −0.0070, kappa −0.0149). Un efecto
real se mueve en el mismo sentido en las cuatro, como hizo el cambio de modelo base. Éste no.

**Por qué, en llano.** La augmentación de bolsa **no crea pacientes nuevos** — siguen siendo 103. La
varianza que domina el error de generalización está *entre pacientes*, no entre submuestras de
parches del mismo paciente. Muestrear parches reduce una varianza que no era la que limitaba.

---

### N11 · Corregir la fuga de datos con predicciones out-of-fold

| | |
|---|---|
| **Dataset** | `cptac_brca` / TP53_mutation, 50 folds |
| **Modelos base** | Trío histórico (el único con OOF completo) |
| **Metaclasificador** | LogReg |
| **Extras** | `build_oof_features.py`: CV anidada agrupada por paciente |

**El problema es real y está medido.** El metaclasificador se entrena con predicciones que los
modelos base hicieron **sobre las slides que ellos mismos vieron en entrenamiento**. Es la fuga
clásica del *stacking* (Wolpert 1992, Breiman 1996):

| Superficie | Papel | AUC |
|---|---|---:|
| `_train_eval/val_outputs` | meta-entrenamiento (dentro de muestra) | **0.849–0.949** |
| `val_outputs` | parada temprana | 0.812–0.854 |
| `test_outputs` | reporte | 0.757–0.790 |
| `_train_eval_oof/val_outputs_oof` | meta-entrenamiento **sin fuga** | **0.722–0.730** |

El metaclasificador se entrena con señales que valen 0.95 y se aplica donde valen 0.77.

**Resultado de corregirlo:** in-sample 0.7616 vs. OOF 0.7571 → **Δ = −0.0045, IC [−0.0115, +0.0021],
W/T/L 18/11/21, no significativo.**

**Veredicto: ❌ Empate. Corregir la fuga no mejora nada** (para LogReg).

**Por qué, en llano.** Con solo 6 características y un combinador lineal, la fuga **infla las tres
columnas de forma aproximadamente pareja**, así que el peso *relativo* que aprende LogReg sobrevive
intacto. Y el OOF trae su propio coste: sus predicciones vienen de modelos entrenados con menos
datos, luego son más ruidosas. Las dos distorsiones se cancelan.

**Hipótesis que sigue abierta:** la fuga sí debería morder con meta-modelos **flexibles** (MLP,
LightGBM, TabPFN), que pueden explotar un espacio de 6 dimensiones casi separable. Eso explicaría
por qué el MLP colapsa y TabPFN infrarrinde. **Nunca se ha medido** (vía abierta 6).

**Nota sobre el coste muestral del OOF.** Se barrió también el número de pliegues internos de la CV
anidada (`n_inner` ∈ {3, 5, 10}). Más pliegues internos no compensan: el OOF entrena cada modelo
base con una fracción menor de los datos, y con 75 slides ese recorte es caro. Es la razón por la
que las predicciones OOF son más ruidosas y por la que las dos distorsiones acaban cancelándose.

---

### N12–N14 · Apagado dinámico del comité: el mecanismo funciona, el beneficio no existe

Cuatro pruebas independientes, todas negativas. El origen de la idea es PathBench
(arXiv 2505.20202): si el mejor modelo fundacional cambia según el órgano, ¿por qué fijar un comité
en vez de decidir sobre la marcha a quién escuchar?

**N12 · Ampliar el comité de 3 a 8 modelos.** `cptac_brca`, los **255 subconjuntos posibles**
(2⁸−1), LogReg.

| Configuración | AUC | Δ vs. trío ganador | IC 95 % | W/T/L | p |
|---|---:|---:|---|---|---:|
| Trío ganador (baseline) | 0.7986 | — | — | — | — |
| Los 8 completos | 0.7988 | +0.0002 | [−0.0160, +0.0163] | 23/2/25 | 0.99 |
| Mejor en test (6 modelos) | 0.8057 | +0.0071 | [−0.0074, +0.0215] | 29/5/16 | 0.30 |
| **Selección honesta (LOFO)** | 0.8011 | +0.0025 | [0.0000, +0.0075] | 1/49/0 | 0.32 |

Rendimiento por tamaño de comité (mejor / media de AUC en test): k=1 → 0.7902/0.7571 ·
k=2 → 0.8009/0.7759 · k=3 → 0.8040/0.7833 · k=4 → 0.8054/0.7883 · k=5 → 0.8055/0.7920 ·
**k=6 → 0.8057/0.7949** · k=7 → 0.8054/0.7970 · k=8 → 0.7988/0.7988.
**El óptimo está en k = 4–6, y con los 8 el rendimiento baja.**

**N13 · Enrutar el comité por órgano (el escenario ciego real).** 13 tareas binarias en 3 cohortes
(mama: `cptac_brca`, `bc_therapy`; colon: `cptac_coad`), 4 modelos comunes
(`hoptimus1`, `uni_v1`, `uni_v2`, `virchow2`), diseño *leave-one-task-out*.

| Política | Δ AUC vs. comité fijo | Gana / pierde | p |
|---|---:|---|---:|
| B) Comité global único | −0.0001 | 6/13 vs 4/13 | 0.85 |
| **C) Comité por órgano** (escenario realizable) | **+0.0015** | 6/13 vs 2/13 | **0.31** |
| Techo: selección conociendo la tarea | +0.0016 | 6/13 | 0.46 |
| Techo absoluto: oráculo (mira el test) | +0.0141 | — | — |

**N14 · Cambiar el criterio de selección.** Ocho criterios calculables sin etiquetas de test, sobre
los 255 subconjuntos:

| Criterio | Spearman con AUC test | AUC de la selección honesta |
|---|---:|---:|
| `auc_val_penaliz` (media − desviación entre folds) | **+0.814** | 0.8009 |
| `auc_val_medio` (el actual) | +0.750 | **0.8011** |
| `auc_val_agrupado` | +0.712 | 0.7992 |
| `brier_val` | +0.685 | 0.8006 |
| `logloss_val` | +0.682 | 0.8006 |
| `auc_metatrain` (control negativo) | +0.623 | 0.7960 |
| `avg_precision` | +0.559 | 0.7992 |
| `n_modelos` (control "usa más") | +0.449 | 0.7988 |

**Veredicto de las tres: ❌ No funciona.**

**Por qué, en llano — y es un resultado interesante en sí mismo.** El diagnóstico es que
**el AUC de validación no ordena los comités como lo hace el test**: la correlación de Spearman
entre ambos, sobre los 255 subconjuntos, es de solo **+0.750**. El mejor comité en validación
(el trío, 0.8681) queda **16.º en test**. El mejor en test (6 modelos, 0.8057) solo marca 0.8535 en
validación.

Y el mecanismo **sí discrimina de verdad**: en colon el enrutador selecciona
`hoptimus1+uni_v2+virchow2` y descarta `uni_v1`; en mama conserva los cuatro. Incluso reproduce con
datos propios el patrón de PathBench — `hoptimus1` aparece en 6 de los 8 comités óptimos de colon y
en **ninguno** de los 5 de mama. La información existe.

Lo que no existe es la forma de leerla sin las etiquetas de test que en producción no se tienen.
**Apagar modelos dinámicamente es implementable, pero no accionable.**

Dos resultados contraintuitivos que merecen mención: (a) agrupar los folds antes de calcular el
criterio (`auc_val_agrupado`) era el favorito teórico — más filas, mejor estimación — y salió
**peor**, porque mezcla folds con poblaciones distintas y esa heterogeneidad ensucia más de lo que
limpia el ruido; (b) penalizar la inestabilidad entre folds sí **ordena mejor** (+0.814 frente a
+0.750), pero ordenar mejor el conjunto no mueve el argmax, que es lo único que importa para
seleccionar.

---

### N15–N16 · Compartir o transferir la atención entre modelos

**N15 · Promediar las atenciones.** `cptac_brca` (par `ctranspath+conch_v1_5`) y `cptac_gbm` (trío):
Δ AUC = **−0.0009** (IC [−0.0083, +0.0061], MDE 0.010) y **−0.0061** (IC [−0.0157, +0.0029],
MDE 0.014), ambos p Holm = 1.00.
**Veredicto: ❌ Nulo estrecho** — no es ausencia de evidencia, es evidencia de ausencia dentro de
±0.014. Promediar aplana la atención selectiva sin ganar nada a cambio.

**N16 · Transferir la atención del mejor modelo.** En `cptac_brca` había una señal prometedora
(AP +0.0145, IC [+0.0034, +0.0258], W/T/L 31/4/15) al usar la atención de `conch_v1_5` para todos
los modelos. Réplica en `cptac_gbm`: Δ AUC +0.0079, Δ AP +0.0075, **W/T/L en AP = 25/0/25 (empate
exacto)**, p Holm = 1.00.
**Veredicto: ❌ No replica. Línea cerrada.** Es el desenlace que el propio protocolo del proyecto
anticipa para un hallazgo marginal en un solo dataset.

---

### N17 · FGE / SE en datasets pequeños

Sobre `cptac_brca` (50 folds, trío ganador, LogReg):

| Brazo | AUC | Δ AUC | IC 95 % | W/T/L | p Holm | **MDE** |
|---|---:|---:|---|---|---:|---:|
| `fge_trio:logreg` | 0.8006 | +0.0020 | [−0.0161, +0.0196] | **22/2/26** | 1.00 | **0.0261** |
| `se_trio:logreg` | 0.8000 | +0.0013 | [−0.0166, +0.0194] | 24/4/22 | 1.00 | 0.0259 |
| `fge_trio:mlp_snapshot` | 0.8007 | +0.0021 | [−0.0163, +0.0207] | 24/2/24 | 0.34 | 0.0266 |

Sobre `cervical_subtype` (5 folds): `fge_trio:logreg` Δ AUC = −0.0098, `se_trio:mlp_snapshot`
Δ AUC = −0.0258.

**Veredicto: ❌ Nulo en `cptac_brca`, negativo en `cervical_subtype`.** Obsérvese que
`fge_trio:logreg` tiene la media positiva pero **pierde en más folds de los que gana (22/26)**:
media a favor, mediana en contra.

**Por qué, en llano.** Los snapshots del mismo modelo correlacionan entre sí a **0.92–0.98**,
mientras que dos modelos fundacionales distintos correlacionan a **0.35–0.84**. **El presupuesto de
diversidad del ensemble ya está agotado por el eje multi-modelo**; la diversidad intra-modelo que
añaden los snapshots es esencialmente ruido correlacionado.

Además, el mecanismo no llega a desplegarse: la parada temprana entre ciclos corta sistemáticamente
en **2.3–3.8 snapshots** de los 6 configurados, en ambos datasets.

⚠️ **Contexto imprescindible:** este nulo tiene un **MDE de 0.026**, y en `cptac_gbm` el efecto real
resultó ser +0.022 (ficha P3). **No es que FGE no funcione: es que en `cptac_brca` no hay potencia
para verlo.** El nulo es correcto como nulo *acotado*; sería incorrecto citarlo como "los Snapshot
Ensembles no funcionan".

---

### N18 · Barrido de hiperparámetros del metaclasificador MLP

| | |
|---|---|
| **Dataset** | `cptac_brca` / TP53_mutation |
| **Modelos base** | Trío histórico |
| **Metaclasificadores** | `MLPMetaClassifier` (ciclo único) y `SnapshotMLPMetaClassifier` |
| **Extras** | Rejilla de 369 configuraciones (81 de ciclo único + 288 de snapshot) + estudio de ablación de 73 configuraciones para separar **capacidad** (red grande sin ciclos) de **mecanismo** (red pequeña con ciclos) |

Mejor resultado por métrica, frente a un baseline LogReg de AUC 0.744 / F1 0.672 / acc 0.701:

| Criterio | Configuración | AUC | F1 | acc |
|---|---|---:|---:|---:|
| Mejor AUC | ciclo único, `hidden=32`, `dropout=0.5`, `lr=5e-3` | **0.7672** | 0.4551 | 0.6097 |
| Mejor F1 | snapshot, `hidden=64`, 4 ciclos, `restart_lr=0.01` | 0.7464 | **0.6897** | 0.7099 |
| Mejor acc | snapshot, `hidden=128`, 8 ciclos | 0.7439 | 0.6832 | **0.7154** |

**Veredicto: ❌ Ninguna configuración gana en las tres métricas a la vez.** El AUC solo mejora con
una red pequeña y dropout alto, **a costa de hundir F1 y accuracy** (−0.217 de F1) — el colapso
hacia la clase mayoritaria típico de un meta-modelo flexible con pocos datos.

**Dos caveats importantes para citarlo:**

1. **Es un barrido pre-metodológico.** No reporta contraste pareado, ni IC, ni MDE, ni W/T/L — solo
   el error estándar por configuración. Es anterior al protocolo estadístico que se adoptó después,
   y por eso sus conclusiones quedan subsumidas por el barrido riguroso de N1.
2. **Su baseline (0.744) no coincide con ningún baseline posterior** (0.7918, luego 0.7616, luego
   0.7986). Corresponde a un estado anterior del pipeline y de los checkpoints. **Las cifras
   absolutas no son comparables con el resto del informe**; lo que sobrevive es la conclusión
   relativa: afinar hiperparámetros no rescata al MLP.

El estudio de ablación asociado se ejecutó solo en un **30 %** (73 de 240 configuraciones
planeadas), sin que quede documentado por qué, de modo que su conclusión sobre capacidad frente a
mecanismo es indicativa, no concluyente.

---

### N19 · Selección LOFO con bagging (stability selection)

| | |
|---|---|
| **Dataset** | `cptac_brca` / TP53_mutation, 50 folds |
| **Modelos base** | 8 modelos, 255 subconjuntos (mismo barrido que N12/N14) |
| **Método** | Selección honesta LOFO con bagging: bootstrap (2000 remuestreos) de los 49 folds != k antes del argmax, voto mayoritario en vez de un único punto — stability selection (Meinshausen & Bühlmann, 2010) / bagged ensemble selection (Caruana, Munson & Niculescu-Mizil, ICDM 2006) |

**Resultado:**

| | AUC | Subconjunto elegido |
|---|---:|---|
| Honesta simple (argmax único) | 0.8011 | trío ganador 49/50, otro (4 modelos) 1/50 |
| **Honesta con bagging** | **0.7986** | trío ganador **50/50** |
| Techo (mejor en test, optimista) | 0.8057 | — |

Δ bagging vs honesta simple: **−0.0025** [−0.0075, +0.0000], W/T/L 0/49/1, p=0.317. Δ bagging vs
baseline (trío ganador): +0.0000 exacto — elige el mismo subconjunto en los 50/50 folds. Estabilidad
media del voto ganador: **0.432** sobre 255 candidatos.

**Veredicto: ❌ No cierra la brecha — la invierte.**

**Por qué, en llano.** Motivado por el diagnóstico de N14 (Spearman(val,test)=0.75): si la selección
es ruidosa, remuestrearla debería estabilizarla. Ocurre lo contrario — el bagging vuelve la
selección **más conservadora, no más certera**: elige siempre el trío ganador (50/50), mientras que
la selección simple tenía un único fold donde ganaba un subconjunto de 4 modelos que, en ese fold
concreto, acertaba en test. Al promediar sobre remuestreos se pierde justo ese acierto puntual. La
estabilidad media de solo 0.432 confirma que hay ambigüedad genuina entre subconjuntos: el bagging
la resuelve por el lado "seguro" (el subconjunto más frecuente), no por el lado "acertado en ese
fold". **La varianza residual de la selección honesta no era ruido puro — tenía algo de señal
fold-específica real —, y suavizarla no ayuda.**

Reutiliza los `(V, T)` ya cacheados en `sweep_base_models.py` (flag `--bagged_boot`); no repite carga
de datos ni entrena de más. Código en `sweep_base_models_bagged.sbatch` (job 72431).

---

### N20 · CMA-ES para pesos del ensemble — frente a Nelder-Mead y LogReg

| | |
|---|---|
| **Dataset** | `cptac_brca` / TP53_mutation, 50 folds |
| **Modelos base** | Dos tríos, los mismos de P1 y N5: ganador (`ctranspath+uni_v2+conch_v1_5`) e histórico (`ctranspath+virchow_v1+conch_v1_5`) |
| **Método** | Combinación lineal `P = Σ wᵢPᵢ`, pesos softmax optimizados con CMA-ES (Hansen; paquete `cma==4.5.0`) maximizando AUC in-sample y evaluados en test — mismo objetivo y parametrización que N5, cambiando solo el optimizador. Es el optimizador que usa el post-hoc ensembling de Auto-Sklearn 2.0 para librerías de modelos (Purucker & Beel, arXiv:2307.00286), pensado para objetivos ruidosos y no convexos como este |

**Resultado:**

| Trío | LogReg | Nelder-Mead (N5) | CMA-ES |
|---|---:|---:|---:|
| ganador | 0.7986 | 0.7978 | 0.7990 |
| histórico | 0.7969 | **0.8020** | 0.7990 |

CMA-ES vs LogReg: Δ=+0.0004 [−0.0048,+0.0051] p=0.86 (ganador); Δ=+0.0021 [−0.0023,+0.0059] p=0.48
(histórico). CMA-ES vs Nelder-Mead: Δ=+0.0012 [−0.0021,+0.0046] p=0.38 (ganador); Δ=−0.0031
[−0.0077,+0.0009] p=0.20 (histórico).

**Veredicto: ❌ Ningún Δ es significativo, y todos caen bajo el suelo de reproducibilidad**
(≈0.004–0.01), igual que N5.

**Por qué, en llano.** Dato curioso: CMA-ES converge al **mismo 0.7990** en ambos tríos, poniendo
~77–78 % del peso en `conch_v1_5` y aplastando al tercer modelo (sea `uni_v2` o `virchow_v1`) a
<12 %, independientemente de cuál sea. Es la misma conclusión de P1 (`conch_v1_5` es la señal
fuerte, el resto casi no aporta) redescubierta por un optimizador continuo en vez de por búsqueda
discreta de subconjuntos — **confirma el diagnóstico, no añade AUC**. Con la elección de modelo base
ya fijando el techo, ni un optimizador más robusto a ruido que Nelder-Mead (CMA-ES es
poblacional, diseñado para objetivos no convexos y ruidosos) tiene margen para moverlo.

Código en `src/ensemble_cmaes.py` (job 72432); resultados en `results_opcion_b_cmaes.json`.

---

### I1–I2 · Los dos experimentos que quedaron incompletos

**I1 · Agregación a nivel de paciente.** En `cptac_brca` hay 9 pacientes con más de una slide. La
idea era agrupar las predicciones por `case_id` y promediarlas antes del metaclasificador, para
reducir la redundancia entre slides del mismo paciente. **Tres trabajos fallidos; nunca se
completó.** Sigue en la lista de prioridad baja.

**I2 · Atención con GAT y transformers entre parches.** Ocho trabajos lanzados; **ninguno llegó a
entrenar**, y el código se eliminó. No hay resultado que reportar. Dado el −0.024 de la atención
multi-cabeza (N8) y el +0.008 no significativo de la GNN espacial (N9), la expectativa era baja de
todas formas.

**Ninguno de los dos aporta evidencia en ninguna dirección.** Se listan para que el inventario de
la campaña esté completo y no parezca que se omitieron resultados desfavorables.

---

## 3.4 Las cuatro razones que explican casi todos los fracasos

Veinte resultados negativos, cuatro mecanismos:

### 1. El cuello de botella es muestral, no arquitectónico

103–166 pacientes, 75–164 slides de meta-entrenamiento. El ABMIL ya tiene ~0.5 M de parámetros.
**Todo lo que añade capacidad se convierte en memorización, no en señal**: MLP (−0.10), atención
multi-cabeza (−0.024), GNN espacial (+0.008 ns), fusión temprana de 4 096 D (−0.006), Deep MLP.

El control del grafo barajado (P6) lo demuestra por contraposición: la señal espacial **existe** y
es medible (+0.0119), pero el coste en parámetros de capturarla se come toda la ganancia.

**Corolario para la tesis:** con este tamaño de cohorte, la dirección productiva no es *añadir*
capacidad, sino *regularizar* o *elegir mejor la entrada*.

### 2. El presupuesto de diversidad ya está agotado

Correlación entre snapshots del mismo modelo: **0.92–0.98**. Entre modelos fundacionales distintos:
**0.35–0.84**. El eje multi-modelo ya captura casi toda la diversidad disponible; los Snapshot
Ensembles añaden variación correlacionada, que en un ensemble no es diversidad, es ruido.

Esto explica N17 y también por qué ampliar el comité de 3 a 8 (N12) no aporta: `ctranspath` y
`uni_v2` ya correlacionan a 0.84 entre sí.

### 3. Lo que ya es lineal no gana nada con no linealidad explícita

Con 6 probabilidades y 75 filas, **LogReg ya extrae toda la señal disponible**. Por eso fracasan la
escalera de características (N6, −0.0014), la reponderación por importancia (N7, −0.006 a −0.012) y
los metaclasificadores flexibles (N1). No es que las transformaciones sean malas: es que
**el modelo lineal ya las reconstruye implícitamente a través de sus coeficientes**, y hacerlas
explícitas solo consume grados de libertad.

### 4. El suelo de reproducibilidad vaporiza las mejoras pequeñas

Reentrenar la configuración idéntica mueve el AUC agregado **±0.004** y cada fold **±0.032**, solo
por no determinismo de GPU. Este único criterio descarta, sin necesidad de más análisis:
Nelder-Mead (+0.0052), GNN espacial (+0.0077), augmentación de bolsa (+0.0077), los snapshots en
`cptac_brca` (+0.0020) y CMA-ES (N20, ≤+0.0021).

**Regla operativa:** cualquier Δ por debajo de ≈0.01 de AUC es indistinguible de relanzar el mismo
trabajo. No es un resultado.

### Y una quinta razón, que es en realidad una advertencia sobre las cuatro anteriores

**Un nulo sin su MDE no es información.** Dos resultados de esta campaña se cerraron como "no
funciona" y luego resultaron ser ciertos al replicarlos con más datos: los Snapshot Ensembles y los
embeddings frente a las probabilidades. En ambos casos, el MDE de la medición original era **mayor
que el efecto real**. La medición no dijo "no hay efecto": dijo "no puedo ver un efecto de este
tamaño", y se leyó mal.

Todo resultado negativo de este informe debe leerse **junto a su MDE**, que por eso figura en todas
las tablas.

---

# 4. Vías abiertas de investigación

Ordenadas por relación entre efecto esperado y coste, según el criterio del informe más reciente.

## Prioridad alta

### 1. Entrenar los modelos fundacionales que faltan: UNI2, H-Optimus-0, Prov-GigaPath

**Por qué es la primera.** Es la única palanca del proyecto con un efecto grande medido
(+0.037 de AUC por cambiar un modelo). PathBench sitúa estos tres modelos en la parte alta del
ranking y ninguno está entrenado en este repositorio. Cada uno es una oportunidad del tamaño del
cambio `virchow_v1 → conch_v1_5`; ninguna variante de metaclasificador ha valido nunca más de
+0.008.

**Cómo:** `MODELS_OVERRIDE=".." sbatch train_base_models.sbatch` entrena un conjunto distinto sin
tocar la matriz de experimentos.

### 2. Fusión temprana a tres bandas en `cptac_gbm`, `bc_therapy` y `cervical_subtype`

**Por qué se reabre una línea cerrada.** El −0.0056 que cerró la fusión temprana se midió en
`cptac_brca`, con solo **2 de los 3** modelos (los únicos que compartían coordenadas allí) y con 75
slides de entrenamiento — el mismo régimen en el que los embeddings *también* parecían no funcionar
y luego resultaron funcionar. En los otros tres datasets el trío completo **sí** comparte
coordenadas de parche, y la fusión a tres bandas nunca se ha probado.

**Expectativa:** moderada. Pero es barata y el precedente de los embeddings aconseja no fiarse de un
nulo medido solo en el dataset más pequeño.

## Prioridad media

### 3. Repetir `cervical_subtype` con más de 5 folds

Con 5 folds no arbitra nada, y es el dataset con más slides de entrenamiento (407) — precisamente
el que debería resolver la pregunta de los embeddings. Se pueden generar particiones nuevas con
`src/create_val_splits.py`.

### 4. Graduar la selectividad de la atención

**Es el contraste decisivo pendiente.** La hipótesis de que el vuelco embeddings-vs-probabilidades
lo explica la selectividad de la atención (ficha P2) se apoya hoy en dos datasets y una única
manipulación llevada al extremo uniforme (`attn_mean`). Falta el experimento que la establecería o
la refutaría: **graduar** la selectividad — no solo aplanarla del todo — y comprobar si el Δ la
sigue de forma monótona.

Si se confirma, da un criterio *a priori* para decidir en qué datasets usar embeddings, calculable
sin etiquetas de test. Sería el primer criterio accionable de toda la campaña.

### 5. Reevaluar `virchow_v1` con una proyección mayor

`virchow_v1` retiene solo el **5 %** de su geometría entre parches tras la proyección 2560 → 512.
Es la única hipótesis mecánica que explica su AUC de 0.6823. Aumentar `proj_dim` es un cambio de un
parámetro, y comprobaría si el modelo es malo o si lo estamos estrangulando.

### 6. Predicciones out-of-fold + metaclasificador flexible

Para LogReg ya está medido: corregir la fuga es un empate (−0.0045, ns). Pero la hipótesis de que
**sí** importa para meta-modelos flexibles (MLP, LightGBM, TabPFN) — que pueden explotar un espacio
de 6 dimensiones casi separable — nunca se ha probado con los checkpoints vigentes. Explicaría por
qué el MLP colapsa y TabPFN infrarrinde.

Requisito previo: generar OOF para los 8 modelos (hoy solo existen para 3).

## Prioridad baja

### 7. Enrutado dinámico con más órganos

Es la única puerta que el estudio de apagado dinámico deja entreabierta. Solo se probaron **dos**
órganos (mama y colon); PathBench observa saltos de ranking mayores entre órganos más distantes
(pulmón, cerebro, estómago). Con `cptac_gbm` hay 3 tareas de cerebro, pero el núcleo común de
modelos cae a 3 y el experimento pierde potencia.

**No está descartado que el enrutado pague con más órganos y mayor separación entre ellos** — pero
el techo medido (oráculo, +0.0141) sigue siendo la mitad de lo que vale cambiar un modelo base.

### 8. Otras líneas menores

- **Gated attention**: nunca probada; es un cambio de una línea en la configuración del ABMIL.
- ~~Regularizar la atención con top-k de parches~~ — **probado el 6-sep-2026, ver P7 (§3.2):
  pooling duro top-10 en vez de blando. +0.0388 AUC significativo en `cptac_gbm`, nulo acotado en
  `cptac_brca`. Sigue pendiente la penalización de entropía (no probada) y el contraste *graduado*
  en K que pide la vía abierta 4.**
- **Calibración**: la log-loss se calcula pero nunca se ha usado para decidir. Si la salida va a
  informar una decisión clínica, la calibración importa tanto como el AUC.
- **Evaluación a nivel de paciente**: 9 pacientes de `cptac_brca` tienen más de una slide; la
  agregación por `case_id` antes del metaclasificador quedó incompleta (3 trabajos fallidos).
- **`logreg_l1` con `--meta_l1_c`**: implementado como *group-lasso* sobre modelos (coeficiente 0 =
  modelo apagado), **escrito pero nunca ejecutado** — perdió sentido al dar negativo el enrutado.
  Sigue disponible si se reabre la línea.

## Líneas cerradas — no reabrir sin motivo nuevo

| Línea | Resultado final |
|---|---|
| Fusión temprana en `cptac_brca` (concatenada y ponderada) | −0.0056 / −0.0182 |
| Atención con más capacidad (multi-cabeza, transformers entre parches, GAT) | −0.024 la mejor variante |
| Augmentación de bolsa | 12 comparaciones, 0 significativas |
| Transferencia de atención entre modelos | No replica (25/0/25) |
| Promediar atenciones | Nulo acotado a ±0.014 |
| Reponderación por importancia del modelo | −0.006 a −0.012 |
| Escalera de características (desacuerdo/margen/entropía) | −0.0014 |
| Apagado dinámico del comité | 4 pruebas, 4 negativos |

**Salvedad sobre la primera y la segunda:** el control negativo del grafo barajado demuestra que la
señal espacial **existe** (+0.0119, significativo). Lo que no funciona es pagarla con parámetros.
Reabrir si el cohorte crece sustancialmente por encima de las 112 slides de `cptac_brca`.

---

## Procedencia

Todas las cifras de este informe se han verificado contra los ficheros de resultados que las
produjeron, todos en la raíz del repositorio:

| Bloque | Fichero de resultados |
|---|---|
| Cambio de modelo base (P1) | `results_model_selection_sweep.json` |
| Barrido de 255 subconjuntos (N12) | `results_base_model_sweep_*.json`, `results_sweep_8models_*.json` |
| Campañas de 3 datasets (P2, P3, N1) | `results_campaign_{cptac_gbm,bc_therapy,cervical_subtype}_*.json` |
| Curva de tamaño de meta-train (P2) | `results_train_size_curve_cptac_gbm_TP53_mutation.json` |
| Diagnóstico de atención y entropía (§1.5) | `results_attention_diagnostics_*.json` |
| Atención compartida (P4, N15, N16) | `results_shared_attention_*.json` |
| Suite de metaclasificadores (N1, N2) | `results_meta_suite_cptac_brca_TP53_mutation.json` |
| Snapshots a nivel base (N17) | `results_snapshot_base_cptac_brca_TP53_mutation.json` |
| Escalera de características (N6) | `results_ladder_cptac_brca_TP53_mutation.json` |
| Nelder-Mead y fusión temprana (N3, N4, N5) | `results_opcion_{a,c,d}_*.json` |
| Augmentación de bolsa (N10) | `results_bag_size_sweep_cptac_brca_TP53_mutation.json` |
| Pooling duro top-K de atención (P7) | `logs/topk10_mlp_snapshot_71206.out` (job 71206); métricas por fold en `{dataset}/TP53_mutation/abmil/ensemble4_topk10/` y `ensemble4_mlp_snapshot_topk10/` bajo `PARADIS/datos/patches` — no hay `results_*.json` en el repo para este bloque, sólo el log y los directorios de resultados |
| Pooling duro top-K adaptativo por ESS (P8) | `logs/topk_ess_72435.out` (job 72435; job 72434 falló por un bug ya corregido, ver `src/abmil_engine.py::abmil_extract_adaptive_topk_embeddings`); métricas por fold en `{dataset}/TP53_mutation/abmil/ensemble4_topk_ess/` y `ensemble4_mlp_snapshot_topk_ess/` — tampoco hay `results_*.json` para este bloque |
| Selección LOFO con bagging (N19) | `sweep_base_models_bagged.sbatch` (job 72431) → `results_base_model_sweep_bagged_cptac_brca_TP53_mutation.json` |
| CMA-ES para pesos del ensemble (N20) | `run_opcion_b_cmaes.sbatch` (job 72432) → `results_opcion_b_cmaes.json` |
