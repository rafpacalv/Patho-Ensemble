# Selección de modelos base — cptac_brca / TP53_mutation

**Fecha**: 2026-08-21
**Origen**: pregunta sobre si se podía ensamblar los modelos fundación con
coordenadas compatibles al estilo de XGBoost (boosting por residuos).
**Datos**: `cptac_brca/TP53_mutation`, 112 slides, 103 pacientes, 50 folds
(75 train / 15 val / 22 test en fold_0).

---

## Resumen ejecutivo

La pregunta de partida llevó a auditar qué modelos base entran en el ensemble,
y ahí apareció el resultado principal: **cambiar `virchow_v1` por `conch_v1_5`
vale +0.037 AUC y +0.11 kappa**, aproximadamente **7 veces** la mejor mejora
que ha producido toda la campaña de reglas de combinación.

De paso se detectaron tres problemas de integridad de resultados que conviene
tener presentes (secciones 3 a 5).

---

## 1. La selección de modelos base domina a la regla de combinación

Medido con `ensemble4.py --meta_model logreg` (defaults: `feature_space=prob`,
sin `drop_redundant_class`), 50 folds, todo calculado en la misma pasada contra
las predicciones base vigentes.

| Configuración | AUC | bacc | macro-f1 | kappa |
|---|---:|---:|---:|---:|
| `ctranspath+uni_v2+virchow_v1` (trío histórico) | 0.7616 | 0.6702 | 0.6687 | 0.3502 |
| **`ctranspath+uni_v2+conch_v1_5`** | **0.7986** | **0.7302** | **0.7294** | **0.4629** |
| los 4 (añadir conch sin quitar virchow) | 0.7986 | 0.7291 | 0.7283 | 0.4606 |

Δ pareado del swap, con corrección de Holm sobre las cuatro métricas:

| Métrica | Δ | IC95% | W/T/L | p(Holm) | MDE |
|---|---:|---:|---:|---:|---:|
| AUC | +0.0370 | [+0.0230, +0.0514] | 34/4/12 | 1.4e-04 | 0.0209 |
| bacc | +0.0600 | [+0.0363, +0.0838] | 36/5/9 | 1.4e-04 | 0.0345 |
| macro-f1 | +0.0608 | [+0.0362, +0.0857] | 36/5/9 | 1.4e-04 | 0.0365 |
| kappa | +0.1127 | [+0.0657, +0.1601] | 36/5/9 | 1.4e-04 | 0.0680 |

Las cuatro significativas, con Δ entre 1.7× y 1.8× el MDE. No es un resultado
al límite como el `p = 0.058` de Nelder-Mead.

### Contexto: qué ha rendido cada línea de trabajo

| Intervención | Δ AUC | Estado |
|---|---:|---|
| **Cambiar un modelo base** | **+0.0370** | **p(Holm) = 1.4e-04** |
| Nelder-Mead sobre pesos | +0.0052 | p = 0.058, marginal |
| Escalera de features extras | −0.0014 | no significativo |
| Early fusion (3 modelos) | −0.0056 | no significativo |
| Weighted early fusion | −0.0182 | degrada |

### Dos lecturas

**`virchow_v1` es prescindible, no sólo peor.** Su AUC individual es 0.6823,
el más bajo de los ocho modelos disponibles. Añadir conch sin quitarlo (0.7986)
empata exactamente con el swap (0.7986): no aporta nada que conch no cubra. Y
quitarlo sin sustituirlo (`ctranspath+uni_v2` = 0.7650) **ya bate al trío**.

**La ganancia viene del modelo base, no del combinador.** `conch_v1_5` solo
puntúa 0.7902; el stacking con ctranspath+uni_v2 sube a 0.7986. El meta-learner
aporta +0.008 sobre el mejor modelo individual, mientras que cambiar el modelo
base aporta +0.037.

### Ranking completo (promedio de logits, sin entrenar)

```
0.7967  ctranspath+uni_v2+conch_v1_5
0.7955  ctranspath+conch_v1_5
0.7953  uni_v2+conch_v1_5
0.7936  ctranspath+virchow_v1+conch_v1_5
0.7934  ctranspath+uni_v2+virchow_v1+conch_v1_5
0.7907  uni_v2+virchow_v1+conch_v1_5
0.7902  conch_v1_5
0.7885  virchow_v1+conch_v1_5
0.7652  ctranspath+uni_v2
0.7634  uni_v2
0.7571  ctranspath
0.7568  ctranspath+uni_v2+virchow_v1   <-- trío actual
0.7498  uni_v2+virchow_v1
0.7382  ctranspath+virchow_v1
0.6823  virchow_v1
```

---

## 2. Grupos de coordenadas de patches

Verificado comparando el hash de `coords` en **las 112 slides** de los 12
directorios de features. Los modelos de un mismo grupo tienen coordenadas
byte-idénticas.

| Grupo | Patches/slide | Modelos |
|---|---:|---|
| 1 | 4068 | hoptimus1, phikon_v2, uni_v1, uni_v2, virchow2 |
| 2 | 2100 | conch_v1_5, ctranspath, virchow_v1 |
| 3 | 3148 | uni_v1_256px, uni_v2_256px |
| — | 857 / 734 | conch512, conch512_monai (cada uno solo) |

CLAUDE.md documentaba únicamente que ctranspath+virchow_v1 alinean y uni_v2 no.
En realidad hay **cinco** modelos alineados en el Grupo 1. El trío histórico
está partido entre los grupos 1 y 2, que es exactamente por lo que `ensemble5.py`
sólo podía fusionar dos de sus tres modelos.

Nota para la pregunta original: **el boosting por residuos no necesita
coordenadas compatibles**, porque el residuo se define sobre la etiqueta, que
es por slide. Las coordenadas sólo hacen falta para la variante que repondera
patches vía atención entre rondas.

---

## 3. ⚠️ Resultados `ensemble4_*` obsoletos

Las predicciones base de ctranspath, virchow_v1 y conch_v1_5 se regeneraron el
**2026-08-21 entre 19:26 y 19:28**. Todo directorio `ensemble4_*` anterior se
calculó contra predicciones que ya no existen en disco:

- `ensemble4_fea_e0..e3` (19-ago 21:15–21:17) — la escalera de features
- `ensemble4_idea3_e0..e2`, `ensemble4_idea3_attn_e0..e2` (20-ago 12:10–12:43)
- `ensemble4_{svm,knn,nb}_*` (19-ago 13:03–14:23)

**`E0 = 0.7918` no reproduce.** El valor vigente del mismo trío con el mismo
código es **0.7616**. Confirmado por reajuste: los coeficientes guardados en
`ensemble4_fea_e0/coefs.npy` no coinciden con un reajuste sobre los `preds.npy`
actuales en **ninguno** de los 50 folds.

Las comparaciones *dentro* de cada grupo siguen siendo válidas (mismas
predicciones base, mismo instante), así que la conclusión relativa de la
escalera —las features extras no aportan— se sostiene. Lo que no vale es citar
los absolutos ni comparar entre grupos.

**Regla**: recalcular el baseline en la misma pasada que cualquier contraste
nuevo. Para auditar un directorio, comparar `stat -c %y` de su `coefs.npy`
contra `{modelo}_{patching}_train_eval/val_outputs/fold_0/preds.npy`.

---

## 4. ⚠️ Dos campañas conviven en el mismo árbol

| Campaña | Modelos | Filas/fold | Motor |
|---|---|---|---|
| ago-2026 | ctranspath, uni_v2, virchow_v1, conch_v1_5 | 75/15/22 (112 slides) | `abmil_engine` |
| sept-2025 | hoptimus1, phikon_v2, uni_v1, virchow2 | 70/13/20 (103 = **pacientes**) | `patho_bench` |

La campaña de sept-2025 evaluó a **nivel paciente** (una slide por `case_id`).
Como `preds.npy` no guarda `slide_id`, las dos familias **no se pueden
realinear** y sus AUC no son comparables.

Esto estuvo a punto de producir una conclusión falsa: `phikon_v2` marcaba 0.8175
sobre 20 filas frente al ensemble sobre 22, lo que parecía indicar que un solo
modelo batía al trío. La comparación es inválida.

Sus checkpoints tampoco sirven: son
`patho_bench.TrainableSlideEncoder` (2.6M params, pre/post-attention 1024→1024,
atención **no** gated), incompatibles con `abmil_engine.ABMIL` (proyección
→512 + LayerNorm + GELU, gated attention). Hay que **reentrenar**, no
re-predecir.

Las salidas antiguas están archivadas en
`abmil/_legacy_patho_bench_sept2025/` con su README.

---

## 5. Estado: reentrenamiento en curso

Job SLURM **70270** (`gpu02`) reentrena `phikon_v2`, `uni_v1`, `hoptimus1` y
`virchow2` con el motor actual sobre las 112 slides, para poder barrer los ocho
modelos en igualdad de condiciones.

Lanzado con el override nuevo, que no toca la matriz de experimentos:

```bash
MODELS_OVERRIDE="phikon_v2 uni_v1 hoptimus1 virchow2" \
DATASET=cptac_brca TASK=TP53_mutation sbatch train_base_models.sbatch
```

---

## 6. Reproducibilidad

Réplica verificada: el barrido se calculó con una reimplementación del camino
de `ensemble4.py`, validada ejecutando el script oficial sobre el trío —
**ambos dan 0.7616 exacto**.

Directorios de resultados generados:

- `ensemble4_verif_trio` — baseline vigente del trío
- `ensemble4_swap_conch` — configuración ganadora
- `ensemble4_four` — los cuatro modelos
- `results_model_selection_sweep.json` — Δ, IC95%, W/T/L, p(Holm) y MDE por métrica

---

## 7. Próximos pasos

1. **Al terminar el job 70270**: barrer los 8 modelos y elegir el subconjunto
   por CV anidada, no por el máximo del ranking en test (el ranking de la
   sección 1 tiene selección sobre el propio test y sobreestima).
2. **Replicar en otros datasets/tareas** antes de cambiar `defaults.models` en
   `experiments.yaml`.
3. **Recalcular los contrastes de la campaña** (escalera, idea3, svm/knn/nb)
   contra el baseline vigente, o marcarlos explícitamente como históricos.
4. Sólo entonces, si procede, volver al boosting: exige residuos out-of-fold
   en cada ronda (los residuos in-sample son ~0 porque los modelos base ajustan
   su train a 0.94 AUC) y un learner base deliberadamente débil, con 75 bags de
   entrenamiento por fold.
