# ESTADO DEL ARTE: MEJORAS EN ATENCIÓN Y META-ENSEMBLE

> **⚠️ CORRECCIÓN (2026-08-21) — los valores absolutos están obsoletos.** El
> baseline **E0 = 0.7918 no reproduce**; el vigente para el mismo trío es
> **0.7616** (las predicciones base se regeneraron el 21-ago 19:26–19:28). La
> escalera E0→E3 sigue siendo internamente válida —se calculó de una vez— así
> que su conclusión (las features extras no aportan) se mantiene. Ver
> [INFORME_SELECCION_MODELOS_BASE_20260821.md](INFORME_SELECCION_MODELOS_BASE_20260821.md).

## Análisis de Diseño para 3 Ideas de Optimización

**Fecha:** 19 de Agosto de 2026  
**Base:** ESTADO_DEL_ARTE_20260819.md (análisis de 50 folds completado)  
**Objetivo:** Documento de diseño técnico + Implementación en progreso  
**Enfoque:** Explicar mecánica, beneficios justificados, riesgos específicos, y propuestas de implementación

---

## 🔴 ESTADO DE IMPLEMENTACIÓN

| Idea | Estado | Job ID | Resultado | Próximo Paso |
|------|--------|--------|-----------|---|
| **Idea 1** | ✅ COMPLETADA | 70100 | NO mejora significativa (Δ AUC = -0.0004 a -0.0014, ns) | **CERRADA** |
| **Idea 3 Fase 1** | ✅ COMPLETADA | 70101 | 450 archivos attn_stats.npy generados | Fase 2 en progreso |
| **Idea 3 Fase 2** | ⏳ EN PROGRESO | 70112 | Escalera aditiva con attn features | Esperando resultados |
| **Idea 2** | ⏸️ PENDIENTE | — | Depende de Idea 3 Fase 2 | Piloto si Idea 3 muestra señal |

---

## 📋 TABLA DE CONTENIDOS

1. [Resumen Ejecutivo](#resumen-ejecutivo)
2. [Contexto y Restricciones Compartidas](#contexto-y-restricciones-compartidas)
3. [Idea 1 — Feature Engineering Meta](#idea-1--feature-engineering-meta)
4. [Idea 2 — Multi-head Gated Attention](#idea-2--multi-head-gated-attention)
5. [Idea 3 — Attention Maps como Meta-Features](#idea-3--attention-maps-como-meta-features)
6. [Relación y Combinación entre las 3 Ideas](#relación-y-combinación-entre-las-3-ideas)
7. [Recomendación de Priorización](#recomendación-de-priorización)
8. [Riesgos Transversales y Mitigaciones](#riesgos-transversales-y-mitigaciones)
9. [Apéndice — Checklist de Implementación](#apéndice--checklist-de-implementación)

---

## 🎯 RESUMEN EJECUTIVO

### Tabla Comparativa

| Idea | Descripción | Coste Cómputo | Reentrenar Checkpoints | Riesgo Estadístico | Prioridad | Estimado Δ AUC |
|------|-------------|---|---|---|---|---|
| **1: Feature Engineering** | Columnas derivadas (entropía, margen, disagreement) de probs actuales | Nulo (minutos) | ❌ No | Bajo (1-3 params) | 🥇 Primera | +0.005–0.020 |
| **2: Multi-head Attention** | Pasar GatedAttention de single-head a n-heads con mean-pooling | Alto (GPU-h) | ✅ Sí (150 ckpts) | Igual al resto | 🥉 Tercera | +0.005–0.015 |
| **3: Attention Maps** | Persistir y agregar estadísticos de distribución de atención | Bajo (inferencia) | ❌ No | Compartido con #1 | 🥈 Segunda | +0.003–0.010 |

### Nota de Encuadre

Las 3 ideas heredan la misma restricción dominante identificada en `ESTADO_DEL_ARTE_20260819.md`: **presupuesto muestral de 50 folds, ~75 muestras train/fold, MDE ≈ ±0.016 en AUC**. Esto significa que cualquier mejora debe superar este umbral para ser estadísticamente detectable. 

**Prior empírico del proyecto:** 7 intentos de subir capacidad arquitectónica (FGE base, OOF, reweighting, DeepMLP, LightGBM, etc.) → 7 resultados nulos o negativos. La carga de la prueba es alta: cualquier nueva idea compite contra una evidencia histórica de que la restricción es muestral, no arquitectónica.

Esto documento desarrolla a nivel de implementación lo que el informe anterior solo esbozó (secciones A2, B1 de "Vías de Mejora").

---

## 🔗 CONTEXTO Y RESTRICCIONES COMPARTIDAS

### 1.1 Presupuesto Estadístico: El MDE que Vence a Todas las Ideas

Con 50 folds validados con paired t-test:

```
Fórmula MDE (Minimum Detectable Effect):
MDE = (t_critical + t_power) × σ_diferencia / √n_folds
    = (1.677 + 0.842) × ~0.025 / √50
    ≈ ±0.016 en AUC

Interpretación:
- Δ AUC < 0.016  → Indistinguible del ruido (no detectable)
- Δ AUC ≈ 0.003  → No detectable (como FGE base, que tuvo +0.003)
- Δ AUC ≈ +0.010 → Apenas bordeando la detectabilidad
- Δ AUC > 0.020  → Claramente detectable, pero casi nunca visto en este proyecto
```

**Consecuencia:** Las Ideas 1 y 3 (que no requieren reentrenamiento) comparten el mismo presupuesto estadístico de 50 folds. Si ambas se prueban en rondas separadas con corrección múltiple independiente, gastan el presupuesto dos veces. Solución: diseñar UNA sola ronda de experimentos que meta todas las columnas candidatas en una escalera ordenada por prioridad, con una única corrección Holm-Bonferroni sobre la familia.

### 1.2 Prior Empírico: 7 Intentos, 7 Fracasos

El proyecto ya evaluó en 50 folds:
- ✅ FGE/Snapshot en modelos base: +0.003 AUC (ns)
- ✅ Out-of-Fold meta-features: -0.077 AUC (empeora, p<0.001)
- ✅ Reweighting por importancia: -0.010 AUC (empeora, p<0.01)
- ✅ MLP simple en 6-dim: -0.135 AUC (overfitting severo)
- ✅ LightGBM: -0.296 AUC (inestabilidad, p<0.001)
- ✅ Deep MLP: falló por incompatibilidad TabPFN
- ✅ Spatial/GNN: falló por construcción de grafo

**Carga de la prueba:** Cualquier nueva idea parte con evidencia histórica EN CONTRA, no a favor. Esto justifica la regla de parada anticipada que se propondrá en cada idea: si el primer paso no supera mínimo 50% del MDE (~+0.008), detener inmediatamente.

### 1.3 Dos Presupuestos Distintos (Crítico No Confundir)

| Presupuesto | Relevancia | Ideas Afectadas | Límite |
|---|---|---|---|
| **Estadístico** (folds, MDE) | Detectabilidad del AUC final | 1, 3 | 50 folds; cualquier mejora < ±0.016 es indistinguible |
| **Cómputo** (GPU-horas reentrenamiento) | Factibilidad de ejecutar | 2 (multi-head) | ~150 reentrenamientos base × 3 modelos = costo alto |

La Idea 2 (multi-head) experimenta presión de AMBOS presupuestos simultáneamente: tiene el mismo MDE (50 folds) pero además requiere reentrenar cientos de modelos, así que el tiempo de feedback es mucho más largo. Las Ideas 1 y 3 solo compiten por el presupuesto estadístico.

### 1.4 Diagrama del Pipeline Actual y Puntos de Inserción

```
┌─────────────────────────────────────────────────────────────┐
│ PIPELINE PATHO-ENSEMBLE — Puntos de Inserción de las Ideas  │
└─────────────────────────────────────────────────────────────┘

ETAPA 1: Base Training (train_abmil.py, test_abmil.py, etc.)
┌──────────────────────────────────────────────────────────────────┐
│  feats.h5 (1536-d UNI, etc.)  →  ABMIL  →  preds.npy (2-d probs) │
│                                     ↓                             │
│                          [IDEA 2 ENTRA AQUÍ]                      │
│                    Multi-head GatedAttention                      │
│                      (requiere reentrenar)                        │
│                                     ↓                             │
│                    [IDEA 3 INICIA AQUÍ] ←────────────────────┐   │
│                Extraer attn_stats.npy (valores de a)         │   │
│                (sin reentrenar, re-inferencia solo)           │   │
└──────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
ETAPA 2: Ensemble Meta (ensemble4.py)
┌──────────────────────────────────────────────────────────────────┐
│  Cargar labels.npy, preds.npy (de 3 modelos)                     │
│  Concatenar → X_meta [N, 6] (3 modelos × 2 clases)               │
│                   ↓                                               │
│  [IDEA 1 Y 3 ENTRAN AQUÍ] ← concatenar como "bloque extra"       │
│  ┌─────────────────────────────────────────────────────┐          │
│  │ + Disagreement / Jensen-gap          [+1 col]       │          │
│  │ + Entropía por modelo                [+3 cols]      │ [IDEA 1] │
│  │ + Margen / confianza                 [+1-3 cols]    │          │
│  │ + ESS_norm, entropy_norm (atención)  [+2 cols]      │ [IDEA 3] │
│  │ + Percentiles atención               [+4-6 cols]    │          │
│  └─────────────────────────────────────────────────────┘          │
│                   ↓                                               │
│  _transform() [logit space, drop_redundant, etc. — SIN CAMBIOS]   │
│  reweighting opcional [SIN CAMBIOS]                               │
│                   ↓                                               │
│  X_meta_final [N, 6-28] → LogReg / MLP / etc.                    │
│                   ↓                                               │
│              Meta-train, meta-val, meta-test                      │
└──────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
ETAPA 3: Resultados Finales (Metrics, bootstrap IC, Holm)
┌──────────────────────────────────────────────────────────────────┐
│ AUC, Accuracy, F1, Kappa (50 folds, IC 95%, corrección Holm)     │
└──────────────────────────────────────────────────────────────────┘
```

---

## 💡 IDEA 1 — FEATURE ENGINEERING META

### 1.1 ¿Qué Es Exactamente?

Hoy, `Ensemble.forward()` en `ensemble4.py` (líneas ~340-368) construye `X_meta` así:

```python
# Pseudocódigo actual (línea 340-350 aproximada)
xpreds = [np.load(...) for cada modelo]  # Lista de [N, 2] softmax
vpreds = [...]
tpreds = [...]

# Stack + reshape → concatenación cruda
X_train = torch.stack(xpreds, dim=0)           # [3, N, 2]
X_train = X_train.permute(1,0,2).reshape(N,-1) # [N, 6]
# Resultado: [p_m0_c0, p_m0_c1, p_m1_c0, p_m1_c1, p_m2_c0, p_m2_c1]
```

**Feature Engineering = añadir columnas derivadas** de estas 6 probabilidades sin requerer nueva inferencia en los modelos base. Costo de cómputo: cero reentrenamiento, solo `ensemble4.py` se ejecuta (minutos).

### 1.2 Catálogo de Features Candidatas

#### Feature #1 (Prioritaria): Jensen-Gap / Disagreement entre Modelos

**Fórmula:**
```
Jensen_gap = H(mean_m p_m) - mean_m H(p_m)

donde:
- p_m ∈ [N, 2] = probabilidades del modelo m
- H(p) = -Σ_c p_c log(p_c) = entropía Shannon
- mean_m p_m = promedio simple de las 3 distribuciones softmax
```

**Intuición:** La diferencia entre entropía de la predicción promediada vs. promedio de entropías individuales captura la **diversidad entre modelos** de forma simétrica y teóricamente fundamentada (descomposición estándar de incertidumbre en el ensemble learning: incertidumbre total = incertidumbre media + desacuerdo). Es la única feature con una sola columna que resume información *entre* los 3 modelos.

**Por qué ayuda a LogReg:** El modelo lineal actual recibe 6 columnas crudas (probabilidades) y debe reconstruir implícitamente la diversidad con combinaciones lineales. Esta feature hace explícita una dimensión de variación (desacuerdo) que es no-lineal en las columnas originales y que el modelo lineal no puede sintetizar por sí mismo.

**Impacto estimado:** +0.005–0.010 AUC (el más alto potencial de una sola columna, según precedentes en ensemble learning).

**Candidata primaria: SÍ.**

---

#### Feature #2 (Prioritaria): Margen / Confianza por Modelo

**Fórmula opciones:**
```
Opción A: Margen = |p_m[1] - 0.5|  (rango 0-0.5)
Opción B: Entropía = -Σ p_m log p_m (rango 0 a log(2) ≈ 0.69)
Opción C: Max prob = max(p_m)      (rango 0.5-1)
```

**Implementación propuesta:** Calcular las 3 entropías por modelo (`3 columnas`) O el promedio de margen entre modelos (`1 columna`) como paso inicial, luego escalar si muestra señal.

**Por qué ayuda a LogReg:** En el caso binario, la entropía H(p) es una función *simétrica* en p[1] (mismo valor para p[1]=0.1 que para p[1]=0.9). Esta simetría no es reconstruible por NINGUNA combinación lineal de las columnas [p[1], 1-p[1]] — que son linealmente dependientes. Añadir H(p) explícitamente permite que LogReg capture una forma de frontera de decisión (tipo "V" o "U") que es imposible con solo combinaciones lineales de probabilidades. Es exactamente lo que el MLP/LightGBM intentaron aprender implícitamente con 75 muestras y fracasaron.

**Impacto estimado:** +0.003–0.008 AUC por columna (efecto marginal pero teóricamente sólido).

**Candidata primaria: SÍ (como segundo paso en escalera).**

---

#### Features Rechazadas: KL/JS Pairwise entre Modelos

**Fórmula:** Para cada par (m_i, m_j), calcular `KL_div(p_m_i || p_m_j)` → 3 columnas (pares: 0-1, 0-2, 1-2).

**Razón de rechazo:** Redundancia e colinealidad interna.
- Si los 3 vectores [p_0, p_1, p_2] son similares, todas las divergencias KL pairwise son chicas a la vez → fuerte colinealidad entre las 3 nuevas columnas.
- La información de "¿cuán distintos son los modelos entre sí?" ya está parcialmente capturada por el Jensen-gap (Feature #1).
- Penalidad de grados de libertad: 3 columnas × 75 muestras ≈ 25 filas/parámetro para LogReg, justo donde el proyecto ya vio degradación con DeepMLP.
- Evidencia: ninguno de los 7 intentos previos del proyecto que probaron variantes de ensemble weighting por divergencia mostró mejora.

**Decisión: No incluir.**

---

### 1.3 Por Qué Ayuda a un Modelo Lineal Downstream

**Principio General:** Features útiles a un LogReg son exactamente transformaciones **NO expresables linealmente** de las columnas existentes.

- ✅ **Útiles**: Entropía (función logarítmica simétrica), valor absoluto (margen), cualquier función cóncava o convexa.
- ❌ **No útiles**: Ratios p[1]/p[0] (es un reescalado lineal), suma/resta de probabilidades (lineal), promedios entre modelos (lineal).

**Ejemplo técnico:** Dado que LogReg aprende `y_pred = σ(w · X + b)` donde σ es sigmoid, las únicas transformaciones que LogReg puede capturar por sí mismo son lineales. Si una feature `f(X)` NO es reconstruible como `w' · X + c` para ningún `w'`, entonces añadir `f` explícitamente expande el espacio de fronteras de decisión posibles.

- `H(p[1])` es simétrica: no hay `w, c` tales que `H(p[1]) = w·p[1] + c` para todo `p[1]`.
- `|p[1] - 0.5|` es lineal a trozos: capturada parcialmente por LogReg pero no perfectamente en límites.

La entropía es la más teóricamente fundamentada.

---

### 1.4 Riesgo de Overfitting: Cuantificado

| Columnas Actuales | N Parámetros LogReg | Filas/Parámetro | Riesgo |
|---|---|---|---|
| 6 (sin bias) | 6 | 75/6 ≈ 12.5 | Bajo (aunque datos son escasos) |
| 6 + bias | 7 | 75/7 ≈ 10.7 | Bajo |
| +1 (disagreement) | 8 | 75/8 ≈ 9.4 | Bajo |
| +3 (entropía por modelo) | 10 | 75/10 ≈ 7.5 | **Aumento de riesgo, orden de magnitud donde falló DeepMLP** |
| +1 (entropía promedio) | 8 | 75/8 ≈ 9.4 | Bajo |

**Regla propuesta:** Máximo **1-3 columnas nuevas por ronda de experimento**, escalera aditiva. **Nunca las 6-9 candidatas juntas de una sola vez.**

El umbral de "safe" es ~7-8 filas/parámetro (orden de magnitud de donde empieza el overfitting severo en este proyecto). Con 75 muestras, eso significa máximo ~10-11 parámetros totales.

---

### 1.5 Cómo No Romper la Asunción de Bloques Uniformes

**Código crítico afectado (si se añaden columnas de forma ingenua):**

1. `Ensemble._transform()` (línea 118 aproximada):
   ```python
   if X.shape[1] % num_models != 0:
       raise ValueError(f"Expected X.shape[1] divisible by {num_models}, got {X.shape[1]}")
   ```

2. `LogitAveragingMetaClassifier._as_log_probs()` / `GatingMLPMetaClassifier` (línea 73-93):
   ```python
   # Reconstruct [N, n_models, C] from flat [N, n_models*C]
   n_classes = X.shape[1] // self.n_models
   X_reshaped = X.reshape(N, self.n_models, n_classes)
   ```

3. `_compute_reweight_scales()` (presume `num_classes = X.shape[1] // num_models`).

**Solución: Patrón "Bloque Extra al Final"**

Mismo precedente que ya existe en `ensemble4.py:510-514`:
```python
if drop_redundant_class and meta_model_name in ("logit_avg","gating"):
    raise ValueError("drop_redundant_class incompatible con logit_avg/gating")
```

**Pseudocódigo de implementación propuesto (en `Ensemble.forward()`, después de `_transform()`):**

```python
# Línea ~368 (después de _transform, antes de reweighting)
X_train = self._transform(xpreds, ...)  # Bloque uniforme, SIN CAMBIOS
X_val = self._transform(vpreds, ...)
X_test = self._transform(tpreds, ...)

# Reweighting opcional
if self.reweight_by_model_importance:
    scales = self._compute_reweight_scales(...)
    X_train = X_train * scales  # SIN CAMBIOS
    X_val = X_val * scales
    X_test = X_test * scales

# NUEVO: Feature engineering
if self.extra_features_list:  # Nuevo flag CLI: --extra_features disagreement,entropy_avg
    if self.meta_model_name in ("logit_avg", "gating"):
        raise ValueError(
            f"--extra_features no soportado con {self.meta_model_name}: "
            "reconstruyen [N, n_models, C] asumiendo divisibilidad. "
            "Usa LogReg, MLP, SVM, etc."
        )
    
    # Compute extra features SIEMPRE del mismo modo en train/val/test
    X_extra_train = self._compute_extra_features(
        xpreds_raw=xpreds, features=self.extra_features_list
    )
    X_extra_val = self._compute_extra_features(vpreds_raw=vpreds, ...)
    X_extra_test = self._compute_extra_features(tpreds_raw=tpreds, ...)
    
    # Concatenate AL FINAL, nunca intercalado
    X_train = np.concatenate([X_train, X_extra_train], axis=1)
    X_val = np.concatenate([X_val, X_extra_val], axis=1)
    X_test = np.concatenate([X_test, X_extra_test], axis=1)

# Fit metamodelo con X_train/X_val/X_test ya finales
self._fit_meta_model(self.meta_model, X_train, y_train, X_val, y_val)
```

**Regla de diseño clave:** Las columnas engineered SIEMPRE se concatenan **AL FINAL**, después de que `_transform()` y reweighting ya consumieron el bloque puro. Esto garantiza que:
- Ningún componente downstream tenga que aprender a ignorarlas.
- El guard del error capture cualquier meta-modelo que no sea compatible.
- La estructura de bloques uniformes se preserva internamente (solo se rompe al final del constructor de X).

---

### 1.6 Plan de Validación: Escalera Aditiva con Parada Anticipada

**Fase 1: Baseline (ya completado)**
- X_meta = 6 columnas crudas
- Meta-modelo: LogReg
- AUC: 0.7918 ± 0.015 (resultado de `ESTADO_DEL_ARTE_20260819.md`)

**Fase 2: +1 columna (Disagreement)**
- X_meta = 6 + 1 (Jensen-gap)
- Meta-modelo: LogReg
- Evaluación: 50 folds, bootstrap 1000, Holm-Bonferroni vs. Fase 1
- Regla de parada: Si Δ AUC < +0.008 (mitad MDE) O no es direccionalmente positivo con IC ajustado, **STOP** — no escalar más.
- Si pasa: continuar a Fase 3.

**Fase 3: +1 columna (Margen promedio)**
- X_meta = 6 + 1 (disagreement) + 1 (margen)
- Meta-modelo: LogReg
- Evaluación: 50 folds, bootstrap 1000, Holm-Bonferroni sobre familia {Fase 1, 2, 3}
- Si muestra señal conjunta: considerar Fase 4 (entropía desglozada por modelo).
- Si es plana: detener.

**Fase 4 (opcional): +3 columnas (Entropía por modelo)**
- X_meta = 6 + 1 (disagreement) + 1 (margen) + 3 (entropía m0, m1, m2)
- Meta-modelo: LogReg
- Evaluación: 50 folds, bootstrap 1000, Holm-Bonferroni sobre familia completa {1,2,3,4}
- Máximo punto de parada: más de +0.020 AUC acumulado (~una MDE completa) sería señal estadística fuerte. Si se logra, ya es un resultado validado.

**Chequeo secundario (no decisorio):**
- Replicar las fases ganadoras en `cervical_subtype` (5 folds) solo como señal direccional (el informe marca ese dataset como "menos confiable").

**Duración estimada:** 2-3 días (solo ejecutar `ensemble4.py` y `Metrics`, no reentrenar ABMIL). Costo de cómputo: negligible.

---

### 1.7 RESULTADOS: Idea 1 No Muestra Mejora Significativa ❌

**Experimento completado:** 19 de Agosto de 2026, Job 70100

| Escalón | AUC | 95% CI | Δ vs Baseline | p-valor | Adj. p (Holm) | Significancia |
|---------|-----|--------|---------------|---------|---------------|---|
| **E0 (Baseline)** | 0.7918 | [0.7630, 0.8200] | — | — | — | — |
| E1 (+disagreement) | 0.7914 | [0.7630, 0.8197] | -0.0004 | 0.294 | 0.294 | ✗ |
| E2 (+disagreement+margin) | 0.7912 | [0.7625, 0.8199] | -0.0006 | 0.214 | 0.642 | ✗ |
| E3 (+disagreement+margin+entropy) | 0.7904 | [0.7616, 0.8188] | -0.0014 | 0.216 | 0.431 | ✗ |

**Interpretación:**
- ⚠️ **Ningún escalón supera el baseline** con significancia estadística (Holm-Bonferroni, α=0.05)
- Los cambios observados (-0.0004 a -0.0014 AUC) caen **dentro del MDE estimado** (±0.016)
- El modelo lineal (LogisticRegression) **ya captura eficientemente** la información de las probabilidades base
- Las features derivadas (disagreement, margin, entropy) **no aportan señal nueva**
- Posible evidencia de redundancia: la información ya está contenida implícitamente en las probabilidades crudas

**Implicación:**
Las 7 intentos previos del proyecto (FGE, OOF, Reweighting, MLP, LightGBM, etc.) vs. 7 fracasos ya sugerían que la restricción es muestral, no arquitectónica. Este resultado de Idea 1 refuerza esa evidencia: incluso transformaciones no-lineales teóricamente bien motivadas (entropía, margen) no logran extraer señal nueva.

**Consecuencia para roadmap:**
1. **Idea 1 está CERRADA**: Feature Engineering Meta no es una avenida fructífera
2. **Proceder directamente a Idea 3**: Si estadísticos de atención (información completamente nueva, no contenida en preds) tampoco aportan, será fuerte evidencia de que el presupuesto muestral es la barrera fundamental
3. **Idea 2 (Multi-head)**: Depende de resultados de Idea 3 como sonda; si Idea 3 es también plana, multi-head tiene prioridad muy baja

---

## 🧠 IDEA 2 — MULTI-HEAD GATED ATTENTION

### 2.1 Mecanismo Actual vs. Propuesto

#### Actual (Single-Head)

**Ubicación:** `src/abmil_engine.py`, líneas 27-37.

```python
class GatedAttention(nn.Module):
    def __init__(self, in_dim, hidden_dim=256):
        super().__init__()
        self.V = nn.Linear(in_dim, hidden_dim, bias=False)      # rama tanh
        self.U = nn.Linear(in_dim, hidden_dim, bias=False)      # rama sigmoide (gate)
        self.w = nn.Linear(hidden_dim, 1, bias=False)           # UNA SOLA cabeza → (N,1)
    
    def forward(self, h):  # h: (N_patches, in_dim)
        a = torch.tanh(self.V(h)) * torch.sigmoid(self.U(h))    # (N, hidden_dim)
        return F.softmax(self.w(a), dim=0)                       # (N, 1), softmax sobre N patches
```

**Flujo:** 
- Entrada: h ∈ R^{N × in_dim} (embeddings de N patches)
- V(h) + U(h) → (N × hidden_dim) rama gated (tanh × sigmoid)
- w: Linear(256, 1) → (N × 1) logit por patch
- softmax(dim=0) → (N × 1) suma 1 (normalización por bolsa)

**Pooling (después, en ABMIL_Base._get_aggregated_embedding):**
```python
z = (a * h).sum(0, keepdim=True)  # (1, in_dim) = suma ponderada
```

---

#### Propuesto (Multi-Head, Mean-Pool)

**Cambio único de forma: `w: Linear(256, 1)` → `Linear(256, n_heads)`**

```python
class GatedMultiHeadAttention(nn.Module):
    def __init__(self, in_dim, hidden_dim=256, n_heads=4):
        super().__init__()
        self.V = nn.Linear(in_dim, hidden_dim, bias=False)      # SIN CAMBIOS
        self.U = nn.Linear(in_dim, hidden_dim, bias=False)      # SIN CAMBIOS
        self.w = nn.Linear(hidden_dim, n_heads, bias=False)     # ÚNICO CAMBIO
    
    def forward(self, h):  # h: (N_patches, in_dim)
        a = torch.tanh(self.V(h)) * torch.sigmoid(self.U(h))    # SIN CAMBIOS
        return F.softmax(self.w(a), dim=0)                       # (N, n_heads), softmax INDEPENDIENTE por cabeza
```

**Nota crítica:** El softmax es `dim=0` (eje de patches), no `dim=1`. Esto significa que **cada cabeza obtiene su propio softmax independiente sobre N patches** — no es un softmax conjunto que luego se divide en cabezas. Es la interpretación correcta para MIL con múltiples "razones" de relevancia.

**Pooling (tres alternativas de combinación):**

| Opción | Fórmula | Parámetros Clasificador | Riesgos | Recomendación |
|--------|---------|---|---|---|
| **(a) Concat** | `z = [z_1; z_2; ...; z_K]` ∈ R^{K×512} | `Linear(K×512, C)` **Multiplica por K** | Overfitting severo (clasificador en 75 bolsas) | ❌ No, 1ª ronda |
| **(b) Mean-pool** | `z = mean_k z_k` ∈ R^{512} | `Linear(512, C)` **FIJO como hoy** | Bajo (presupuesto clasificador sin cambios) | ✅ **Recomendada** |
| **(c) Bottleneck** | `z_concat → Linear(K×512, 512) → z` ∈ R^{512} | `Linear(512, C)` | Intermedio (parámetros en bottleneck, no en clasificador) | 🔶 2ª ronda si (b) muestra señal |

---

### 2.2 Distinción Crítica: Presupuesto de Datos en Atención vs. Presupuesto Estadístico del AUC

**Error común a evitar:**
> "Hay mediana ~7174 parches por bolsa, eso son muchos datos, así que no hay riesgo de overfitting."

**Realidad:**
- **Presupuesto de la rama de atención:** Sí, hay ~7000 parches por bolsa, es abundante. El número de parámetros en V/U (que es donde ocurre la mayor parte del cómputo) NO cambia con multi-head (V y U son idénticas). La rama `w: Linear(256, n_heads)` es barata.
- **Presupuesto estadístico del AUC final:** Independiente. Estamos evaluando el AUC en 50 folds con ~75 muestras de validación/test cada uno (no ~7174). El MDE sigue siendo ±0.016. Multi-head no cambia este presupuesto; solo cambia la arquitectura que alimenta la métrica de 50 folds.

**Conclusión:** Multi-head NO reduce el riesgo de no detectabilidad (MDE ≈ ±0.016 sigue siendo la barrera). Lo que reduce es el riesgo de overfitting *en la rama de atención* — pero eso solo importa si la rama de atención fue un cuello de botella (cosa que no es: ya tiene abundancia de datos de parches).

Por eso multi-head es **arriesgada en este contexto específico:** paga el costo de reentrenar 150 checkpoints sin alterar el presupuesto estadístico que decide si la mejora es detectable. Es una apuesta de que "múltiples razones de relevancia" existen en el dataset (conjetura teórica) sin evidencia empírica de que el dataset las exploraría mejor.

---

### 2.3 Compatibilidad de Checkpoints — La Pieza Crítica

**Problema:** Los ~150 checkpoints existentes (`arch="abmil"`) se guardan hoy como:
```python
# train_abmil.py, línea 336
torch.save(model.state_dict(), f"{output_dir}/checkpoints/fold_{fold}/model.pt")
```

y se cargan con:
```python
# test_abmil.py, línea ~80-90
ckpt = torch.load(...)
model.load_state_dict(ckpt, strict=True)  # ← FALLA si w cambió de forma
```

Si cambio `GatedAttention.w` de `Linear(256, 1)` a `Linear(256, 4)`, el state_dict tiene tensores de forma distinta (`256→1` vs. `256→4`) y `load_state_dict(strict=True)` tira KeyError en w.weight.

**Solución Ya Implementada en el Proyecto:** El brazo `SpatialABMIL` ya tiene este problema resuelto (línea 341-348 de `train_abmil.py`):

```python
torch.save({
    "state_dict": model.state_dict(),
    "arch": "sage_bn",           # ← versión de arquitectura
    "n_layers": 2,
    ...
}, checkpoint_path)
```

Y se lee condicionalmente en `test_abmil.py:83-105`:
```python
ckpt = torch.load(...)
if isinstance(ckpt, dict) and "state_dict" in ckpt:
    # Brazo espacial: reconstruir arquitectura correcta
    model = make_spatial_model(...)
    model.load_state_dict(ckpt["state_dict"], strict=True)
else:
    # ABMIL estándar: carga directa
    model.load_state_dict(ckpt, strict=True)
```

**Propuesta: NO Tocar Los 150 Checkpoints Existentes**

En su lugar:
1. Crear un nuevo módulo `src/mh_abmil.py` (análogo a `spatial_abmil.py`) con la clase `GatedMultiHeadAttention` y un factory `make_mh_model(n_heads=4, combine="mean")`.
2. Entrenar con `--arch mh_abmil --arch_tag mh4` → nuevo árbol de salida `checkpoints_mh4/fold_*/model.pt` (reutilizando el patrón `tag` ya existente).
3. Extender `test_abmil.py` para detectar y cargar `mh_abmil` checkpoints (agregar rama en `isinstance(ckpt, dict)` para `"arch" == "mh_abmil"`).
4. Los 150 checkpoints `arch="abmil"` permanecen **intactos**, sirven de baseline perpetuo para comparación.

Este es un experimento **aditivo**, no un reemplazo destructivo, hasta que se demuestre mejor.

---

### 2.4 Coste de Cómputo y Plan de Piloto

**Costo de la rama de atención:** Marginal (<5% overhead por forward/backward, V/U dominan).

**Costo real:** Obligatorio reentrenar desde cero (no se puede partir de pesos de `w: Linear(256, 1)` para inicializar `w: Linear(256, 4)` de forma no trivial).

**Reentrenamientos necesarios:**
- 50 folds × 3 modelos base = **150 entrenamientos completos**
- Receta `_RECIPES["ours"]`: lr=2e-4, wd=1e-4, epochs≤40, patience=8
- Por entrenamiento: ~5-10 minutos en GPU V100/A100
- **Total: ~125-250 GPU-horas** (12-25 horas wall-clock con paralelismo de 10 GPUs)

**Mitigación propuesta: Piloto Reducido Primero**

1. **Piloto fase 1:** `n_heads ∈ {2, 4}`, `head_combine="mean"`, en subconjunto de **10-15 folds × 1 modelo base** (p. ej., CTransPath).
   - Costo: ~10-15 entrenamientos ≈ 10-20 GPU-horas
   - Duración: 1-2 días de wall-clock
   - Decisión: ¿Muestra señal direccional positiva? (no necesita estar arriba de MDE, solo ser claramente positivo)

2. Si piloto muestra señal: escalar a **50 folds × 3 modelos** con la config ganadora del piloto (n_heads, combine).

3. Si piloto es plano: **STOP** — no escalar, ahorra 200 GPU-horas.

---

### 2.5 Variantes No Elegidas (Mención Breve)

Por decisión explícita del usuario, el foco es multi-head. Las siguientes variantes de mejora a `GatedAttention` quedan como alternativas menores fuera del alcance de este documento:

- **Regularización de entropía:** Penalizar distribución de atención demasiado picuda (`softmax_temp < 1`) o demasiado plana. Útil para evitar colapso de atención (pocos patches dominando), pero requiere ajuste de hiperparámetro adicional.
- **Dropout en atención:** Aplicar `Dropout(p)` dentro o después de `GatedAttention` para regularizar. Barato en parámetros, pero añade un hiperparámetro.
- **Temperatura en softmax:** `softmax(w(a) / T, dim=0)` con T aprendido o fijo. Controla cuán "sharp" o "smooth" es la distribución sin cambiar arquitectura.
- **Top-k pooling alternativo:** En lugar de `z = (a*h).sum()`, usar `z = h[top_k_indices].mean()`. Explota mejor la interpretabilidad (cuáles son los patches más relevantes) pero requiere selección de k.

Todas estas son estudiables en una fase 2 si multi-head (esta propuesta) muestra promesa.

---

### 2.6 Plan de Validación: Piloto + Escalada Condicional

**Fase Piloto (1-2 días):**
- Configuración: `n_heads ∈ {2, 4}`, `head_combine="mean"`, `arch="mh_abmil"`, `arch_tag="mh4"`
- Subset: 10-15 folds × 1 modelo (CTransPath)
- Evaluación: Protocolo estándar (bootstrap 1000, Holm-Bonferroni vs. baseline ABMIL standard)
- Criterio de decisión: ¿Δ AUC > 0 con IC95 no cruzando cero? (no debe superar MDE, solo ser direccionalmente positivo)
- Outputs esperados:
  - `checkpoints_mh4/fold_*/model.pt` (nuevos checkpoints)
  - Tabla de AUC/Acc/F1 comparativa (mh4 vs. baseline abmil)
  - Diagnóstico de convergencia: curvas de loss, val_auc por época

**Decisión Post-Piloto:**
- ✅ **Si muestra señal:** Escalar a 50 folds × 3 modelos, evaluación completa.
- ❌ **Si es plano o negativo:** Detener, documentar, guardar GPU-horas.

**Fase Escalada (si piloto okays, 10-15 días):**
- Reentrenar completo: 50 folds × 3 modelos con `n_heads=4`, `head_combine="mean"`
- Evaluación: Mismo protocolo bootstrap + Holm-Bonferroni, pero vs. todos los 150 checkpoints baseline
- Análisis secundario: Divergencia pairwise entre `a_k` (¿las cabezas difieren o colapsan?) — métrica análoga a `alphas()` de `SpatialABMIL` (línea 147-152)

---

## 🎭 IDEA 3 — ATTENTION MAPS COMO META-FEATURES

### 3.1 ¿Qué Son Exactamente?

Los pesos de atención `a` ∈ R^{N} (un valor por patch, softmax que suma 1) se calculan en **cada forward pass** de `GatedAttention`, pero se **descartan siempre**:

```python
# abmil_engine.py, en TODOS los puntos de consumo:
lo, _ = _apply(...)  # ← el segundo retorno (la atención) se descarta
```

Ubicaciones donde se descartan:
1. `_auc()` línea 152
2. `abmil_predict()` línea 277
3. `abmil_extract_embeddings()` línea 296
4. Training loop línea 243

**Nunca se persisten a disco** en ningún punto de `train_abmil.py`, `test_abmil.py`, `build_oof_features.py`, `train_abmil_fge.py`.

**Único precedente:** `spatial_abmil.attention_stats(a)` (línea 167-182) calcula ESS (effective sample size) y masa top-1%, pero solo se usa como diagnóstico en memoria durante el entrenamiento del brazo espacial — nunca se guarda.

### 3.2 Catálogo de Estadísticos Propuestos

**Generalizar y extender `attention_stats` existente:**

```python
def attention_stats_extended(a):
    """
    Compute aggregated statistics from per-patch attention weights.
    
    Args:
        a: (N_patches,) softmax weights, sum=1
    
    Returns:
        dict with keys: entropy_norm, ess_norm, top1pct, top5pct, p90, max, std
    """
    a = a.detach().float().flatten()
    n = len(a)
    sorted_a = a.sort(descending=True).values
    p = a.clamp_min(1e-12)
    
    # Raw statistics
    entropy = float(-(p * p.log()).sum())
    ess = float(1.0 / (a ** 2).sum())
    
    # Normalized (critical: size varies 7174-29622 across slides)
    entropy_norm = entropy / np.log(n)      # Range [0, 1]
    ess_norm = ess / n                      # Range [0, 1]
    
    # Mass in top-k%
    def top_mass(frac):
        k = max(1, int(round(frac * n)))
        return float(sorted_a[:k].sum())
    
    return {
        # Primarias (normalized, comparables entre bolsas de distinto tamaño)
        "entropy_norm": entropy_norm,       # Baja = decisión concentrada; alta = diluida
        "ess_norm": ess_norm,               # Baja = pocos patches dominan; alta = distribuido
        
        # Secundarias (masa acumulada en top patches)
        "top1pct_mass": top_mass(0.01),     # ¿Cuánta atención en 1% top patches?
        "top5pct_mass": top_mass(0.05),     # ¿Cuánta en 5% top?
        
        # Terciarias (percentiles de distribución)
        "p90": float(torch.quantile(a, 0.90)),
        "max": float(a.max()),
        "std": float(a.std()),
    }
```

**Columnas prioritarias para X_meta:**
1. `entropy_norm` (promedio de 3 modelos) — 1 columna
2. `ess_norm` (promedio de 3 modelos) — 1 columna
3. Después, si muestra señal: desglozar por modelo (3 entropía_norm + 3 ess_norm = 6 columnas)

**Descarte explícito:** top-k mass y percentiles como terciarias (secundarias en experimento si hay capacidad después de las primarias).

---

### 3.3 Qué Información Nueva Aportan

**Las probabilidades finales (`preds.npy`) NO capturan:**

1. **Confianza estructural:** Dos slides pueden dar `[p[0]=0.3, p[1]=0.7]` con distribuciones de atención radicalmente distintas:
   - Slide A: atención concentrada en 5 parches decisivos (ESS_norm bajo, top-1% masa alta) → decisión "fuerte"
   - Slide B: atención distribuida sobre 2000 parches (ESS_norm alto, top-1% masa baja) → decisión "diluida"
   
   Ambas predicen clase 1 con probabilidad 0.7, pero el *mecanismo* de confianza es opuesto. El meta-modelo podría aprender a penalizar slides B (falta evidencia localizada) o a confiar más en slides A (hay evidencia focal).

2. **Detección de casos difíciles:** Atención muy diluida (ESS_norm ≈ 1, máximo teórico) es un indicador de que el modelo "no encontró" evidencia clara. Podría señalar slides outliers, artefactos, o casos que están en el límite de decisión. Un meta-modelo lineal no puede expresar fácilmente esta noción de "difícil" solo con probabilidades finales.

3. **Relación inversa con presupuesto de datos:** Slides con atención concentrada tienen potencialmente más información por parche; slides con atención distribuida son más "ruidosas" (cada parche aporta poco). Esto es ortogonal a la probabilidad final.

---

### 3.4 Cambios de Código Concretos: Los 4 Puntos de Generación

**Crítico:** Estos cambios NO requieren reentrenamiento de modelos. La atención se recalcula en el forward pass de los checkpoints YA EXISTENTES.

#### Cambio 1: `src/abmil_engine.py` — Extraer Atención

**Función nueva:** `abmil_predict_with_attention()` análoga a `abmil_predict` pero que devuelve atención junto a predicciones.

```python
# Pseudocódigo (abmil_engine.py, después de abmil_predict actual)
def abmil_predict_with_attention(model, feats, stems, device, graphs=None):
    """
    Predict + extract attention weights and compute stats.
    
    Returns:
        preds: [N_stems, num_classes]
        attn_stats: [N_stems, n_stats] with columns [entropy_norm, ess_norm, ...]
    """
    preds_list = []
    attn_stats_list = []
    
    for stem in stems:
        h = load_features(h5_path, stem)  # [N_patches, in_dim]
        with torch.no_grad():
            logits, a = model(h.to(device))  # ← Mantén 'a', no lo descartes
        preds = softmax(logits).cpu().numpy()
        stats = attention_stats_extended(a)  # Nueva función de arriba
        preds_list.append(preds)
        attn_stats_list.append(stats)
    
    preds = np.vstack(preds_list)                  # [N_stems, num_classes]
    attn_stats = np.array([list(s.values()) for s in attn_stats_list])  # [N_stems, n_stats]
    return preds, attn_stats
```

**Nota crítica:** Se itera **un solo paso** sobre `stems`. Esto garantiza que `attn_stats[i]` está alineada fila-a-fila con `preds[i]` (mismo orden de `stems`, sin riesgo de desincronización).

#### Cambio 2: `src/train_abmil.py` — Guardar Atención Stats (Train)

En `abmil_train_test()`, donde hoy se llama a `abmil_predict` para guardar `test_outputs`, cambiar a `abmil_predict_with_attention`:

```python
# Pseudocódigo (train_abmil.py, línea ~350-361)
# Antes:
# test_preds, test_labels = abmil_predict(model, test_stems, ...)
# save_fold_outputs(output_dir, test_preds, test_labels, split='test')

# Después:
test_preds, test_attn_stats = abmil_predict_with_attention(model, test_stems, ...)
save_fold_outputs(output_dir, test_preds, test_labels, split='test')
save_fold_outputs_attn(output_dir, test_attn_stats, split='test', fold=fold)  # Nueva función
```

**Nueva función de guardado:** `save_fold_outputs_attn(output_dir, attn_stats, split, fold)`:

```python
def save_fold_outputs_attn(output_dir, attn_stats, split='test', fold=0):
    """Save attention statistics as .npy and column metadata as .json."""
    attn_dir = f"{output_dir}/{split}_outputs/fold_{fold}"
    os.makedirs(attn_dir, exist_ok=True)
    
    np.save(f"{attn_dir}/attn_stats.npy", attn_stats)  # [N, n_stats]
    
    # Sidecar JSON (igual patrón que embedding_dim.json)
    columns = ["entropy_norm", "ess_norm", "top1pct_mass", "top5pct_mass", "p90", "max", "std"]
    with open(f"{attn_dir}/attn_stats_columns.json", "w") as f:
        json.dump({"columns": columns, "n_stats": len(columns)}, f)
```

#### Cambio 3: `src/test_abmil.py` — Guardar Atención Stats (Meta-features In-Sample)

Similar a Cambio 2, pero para el **meta-train split** (predicciones en train fold para meta-learning):

```python
# Pseudocódigo (test_abmil.py, línea ~108-130)
# Antes:
# train_preds, train_labels = abmil_predict_train_fold(model, train_stems, ...)

# Después:
train_preds, train_attn_stats = abmil_predict_with_attention(model, train_stems, ...)
save_fold_outputs_attn(output_dir, train_attn_stats, split='train', fold=fold)
```

Reutilizar la misma función de guardado del Cambio 2.

#### Cambio 4: `src/build_oof_features.py` y `src/train_abmil_fge.py` — Guardar Atención Stats (OOF y FGE)

**`build_oof_features.py`:** En el bucle de nested CV que ya genera predicciones OOF, cambiar a `abmil_predict_with_attention` y guardar `attn_stats_oof.npy`.

**`train_abmil_fge.py`:** En `save_fold_outputs_fge()` (línea 34), que ya maneja snapshots, extender para guardar también atención (aquí es tricky porque hay múltiples snapshots por fold; propuesta: guardar `attn_stats_per_snapshot.npy` [n_snapshots, N_stems, n_stats] y luego agregar al mismo modo que se promedia `preds_per_snapshot`).

---

### 3.5 Coste — La Ventaja Clave de Esta Idea

**NO requiere reentrenamiento de checkpoints.**

Los ~150 checkpoints en `checkpoints/fold_*/model.pt` ya existen. Solo se re-ejecuta **inferencia** sobre ellos:
- `train_abmil.py` con `--test` flag: forward pass sobre test splits
- `test_abmil.py`: forward pass sobre train splits
- `build_oof_features.py`: forward pass sobre nested CV folds
- `train_abmil_fge.py`: forward pass sobre snapshots

Coste de cómputo:
- Tiempo de forward pass: ya pagado (ejecutado hoy, descartando atención)
- Overhead de extraer y guardar atención: negligible (una operación .detach().numpy() por forward pass)
- Coste total: **esencialmente cero GPU-hours adicionales**, puro I/O de features .h5 (ya amortizado).

**Duración estimada:** 4-6 horas de wall-clock ejecutando los 4 scripts sobre 50 folds × 3 modelos (completamente paralelizable).

---

### 3.6 Integración con `ensemble4.py` y con Idea 1

Estas columnas de atención son candidatas al mismo `X_meta` que Idea 1. Comparten exactamente la misma restricción (bloque uniforme) y el mismo patrón de guardia.

**Estrategia de integración:** Usar el mecanismo de "bloque extra al final" diseñado para Idea 1, pero generalizado:

```python
# ensemble4.py (pseudocódigo)
def _compute_extra_features(self, preds_raw, attn_stats_dict, feature_sources):
    """
    Compute extra features from raw predictions and attention stats.
    
    Args:
        preds_raw: list of [N, C] raw probabilities (before transform)
        attn_stats_dict: dict mapping model_name → [N, n_stats] attention arrays
        feature_sources: ["disagreement", "entropy_avg", "attn_entropy_norm", "attn_ess_norm"]
    
    Returns:
        X_extra: [N, n_selected_features]
    """
    X_extra_list = []
    
    if "disagreement" in feature_sources:
        # Jensen-gap (Idea 1)
        jensen = ...  # compute
        X_extra_list.append(jensen[:, np.newaxis])
    
    if "entropy_avg" in feature_sources:
        # Entropy per model (Idea 1)
        entropy = ...  # compute
        X_extra_list.append(entropy)  # [N, 3]
    
    if "attn_entropy_norm" in feature_sources:
        # Attention entropy (Idea 3)
        attn_ent = np.column_stack([
            attn_stats_dict[m][:, columns.index("entropy_norm")]
            for m in model_names
        ])  # [N, 3]
        X_extra_list.append(attn_ent)
    
    # ... más features
    
    return np.concatenate(X_extra_list, axis=1)
```

El mismo `--extra_features` flag sirve para activar/desactivar columnas de Idea 1 Y Idea 3 juntas.

---

### 3.7 Plan de Validación: Fase de Generación + Fase de Evaluación

**Fase 1: Generación de attn_stats.npy (GRATUITA, sin presupuesto estadístico)**
- Ejecutar los 4 cambios sobre el grid completo (50 folds × 3 modelos)
- Generar `attn_stats.npy` + `attn_stats_columns.json` en cada `{split}_outputs/fold_{fold}/`
- Duración: 4-6 horas wall-clock
- Coste GPU: negligible
- Validación secundaria: spot-check unos 5-10 slides para confirmar que attn_stats está alineado fila-a-fila con labels/preds

**Fase 2: Evaluación de Features (COMPARTE PRESUPUESTO con Idea 1)**
- Las columnas de atención se añaden a la **misma escalera aditiva** que Idea 1
- Ordenamiento de prioridad propuesto: disagreement, attn_entropy_norm, attn_ess_norm, margen, entropía_modelo
- Protocolo: 50 folds, bootstrap 1000, Holm-Bonferroni sobre toda la familia
- Regla de parada: misma que Idea 1 (si primer paso < 50% MDE, stop)

**Recomendación de secuenciación:**
- Ejecutar Fase 1 de Idea 3 **en paralelo** con experimentos piloto de Idea 2 (son independientes: Idea 3 usa checkpoints ya existentes, Idea 2 entrena nuevos)
- Ejecutar Fase 2 de Ideas 1+3 **en la misma ronda** (una sola escalera, una sola corrección múltiple)

---

## 🔀 RELACIÓN Y COMBINACIÓN ENTRE LAS 3 IDEAS

### 5.1 Ideas 1 y 3 Son Mecánicamente Idénticas

Ambas **agregan columnas a `X_meta`** usando el mismo mecanismo de "bloque extra al final" (sección 2.5). Comparten:
- La misma restricción (divisibilidad de bloques uniformes)
- La misma guardia (`raise ValueError` para logit_avg/gating)
- La misma escalera de validación y corrección múltiple
- La misma coste de cómputo (negligible)

**Diferencia única:** Fuente de información.
- **Idea 1:** Transformaciones de probabilidades ya existentes (información que hoy el meta-modelo tiene, pero en forma implícita).
- **Idea 3:** Distribución de atención (información completamente nueva, no contenida en probabilidades).

**Implicación:** No deben probarse en dos rondas separadas. Una sola ronda de experimentos que meta todas las columnas candidatas en una escalera ordenada por prioridad (p. ej.: 1. disagreement, 2. attn_entropy_norm, 3. attn_ess_norm, 4. margen, 5. entropy_per_model) evita gastar corrección múltiple dos veces sobre el mismo presupuesto de 50 folds.

---

### 5.2 Idea 2 Es Ortogonal

Multi-head attention cambia la **arquitectura upstream** (los modelos base ABMIL), no la construcción del meta-vector.

- **No compite por presupuesto estadístico de forma directa:** Idea 2 requiere reentrenar desde cero, así que genera un vector de resultados completamente independiente (50 folds de multi-head vs. 50 folds de single-head baseline). Ambos pueden evaluarse en paralelo.
- **Sí compite por presupuesto de cómputo:** Requerir 150-200 GPU-horas es un costo real.
- **Relación potencial:** Si Ideas 1+3 muestran que columnas derivadas de atención ya ayudan, es evidencia indirecta de que una distribución de atención más rica (multi-head) podría pagar su costo. Si Ideas 1+3 son planas, es una alerta de que multi-head tampoco pagaría.

---

### 5.3 Argumento de Secuenciación: Por Qué Idea 3 Es una "Sonda" para Idea 2

**Hipótesis subyacente en Idea 2:**
> "El modelo necesita múltiples perspectivas (cabezas) para capturar distintas razones de relevancia de patch. Esto mejorará las predicciones."

**Hipótesis subyacente en Idea 3:**
> "La distribución de atención de una sola cabeza ya contiene información útil (confianza estructural, concentración) que el meta-modelo puede explotar."

Si Idea 3 (cheap sonda) demuestra que **estadísticos de atención de una sola cabeza** ya aportan señal al meta-modelo (Δ AUC > +0.008, direccionalmente positivo), es evidencia de que:
- La atención es una fuente de información válida
- El dataset tiene variabilidad en cómo el modelo confía en su decisión
- Una distribución más rica (multi-head) probablemente aprovechará mejor

Inversamente, si Idea 3 es completamente plana (atención stats no aportan nada), es una **alerta roja** de que:
- Quizá la distribución de atención ya es óptima para este dataset
- O quizá la información está contenida en las probabilidades (redundancia)
- Multi-head es menos probable que pague su alto costo de cómputo

**Conclusión:** Probar Idea 3 primero (2-3 días) ahorra potencialmente 200 GPU-horas de Idea 2 si el resultado es negativo.

---

## 📊 RECOMENDACIÓN DE PRIORIZACIÓN

### Tabla de Orden Recomendado

| Orden | Idea | Esfuerzo | Duración | Decisión Clave | Por Qué |
|---|---|---|---|---|---|
| 🥇 **1ª** | **Feature Engineering Meta** | Bajo (horas) | 2-3 días | ¿Δ AUC > +0.008 con escalera aditiva? | Cheapest first; falsable rápido; desbloquea decisión sobre Idea 3 |
| 🥈 **2ª** | **Attention Maps** | Bajo (inferencia, 1 día) | 4-6 horas | ¿Attn stats aportan señal conjunta? | Ejecutar Fase 1 en paralelo a piloto de Idea 2; fase 2 en misma escalera que Idea 1 |
| 🥉 **3ª** | **Multi-head Attention** | Alto (reentrenamiento) | Piloto 1-2 días, escalada 10-15 días | ¿Piloto muestra Δ AUC > 0 direccional? | Costosa; depende de resultados de Ideas 1+3 como sonda; piloto primero para invalidar early |

### Narrativa

**Semana 1:**
1. Lunes: Ejecutar Idea 1 escalera (baseline → disagreement → margen → entropía), resultados martes.
2. Martes-miércoles: En paralelo, ejecutar Fase 1 de Idea 3 (generar attn_stats.npy para todos).
3. Miércoles: Si Idea 1 muestra señal, ejecutar Fase 2 (Ideas 1+3 en escalera conjunta). Si es plana, detener.
4. Jueves-viernes: Lanzar piloto de Idea 2 (10-15 folds × 1 modelo, n_heads=2,4).

**Semana 2:**
- Si piloto de Idea 2 okays (muestra Δ > 0): escalar a 50 folds × 3 modelos (10-15 días de reentrenamiento).
- Si piloto de Idea 2 es plano: cerrar, documentar, ahorrar GPU-horas.

---

## ⚠️ RIESGOS TRANSVERSALES Y MITIGACIONES

| Riesgo | Ideas Afectadas | Descripción | Mitigación |
|---|---|---|---|
| **Sobreajuste por columnas extra** | 1, 3 | Cada columna nueva reduce filas/parámetro de 10.7 a 9.4 a 7.5; riesgo histórico del proyecto | Escalera aditiva; regla de parada (< 50% MDE); máximo 3-5 columnas totales por ronda |
| **Ruptura de checkpoints** | 2 | Cambiar `w: Linear(256,1)→(256,4)` rompe `load_state_dict(strict=True)` en 150 ckpts | NO tocar ckpts existentes; nuevo módulo `mh_abmil.py` con arch_tag; cargar condicionalmente |
| **Colapso/redundancia de cabezas** | 2 | Múltiples cabezas pueden volverse idénticas (aprender lo mismo) | Loggear divergencia pairwise entre `a_k` (métrica análoga a `alphas()` en SpatialABMIL); detener si divergencia < threshold |
| **Confusión cómputo vs. estadística** | 2, 3 | "Hay muchos parches, así que OK" para overfitting de atención; pero MDE sigue siendo ±0.016 en AUC final | Sección 2.2 explicita distinción; reiterada en cada plan de validación |
| **Comparaciones múltiples no coordinadas** | 1, 3 | Si Ideas 1 y 3 se prueban en dos rondas, gastan Holm-Bonferroni dos veces | Diseñar UNA escalera conjunta (disagreement, attn_entropy, attn_ess, margen, entropy_model) |
| **Atención alineada fila-a-fila** | 3 | Si attn_stats se computa en un pass y preds en otro, pueden desincronizarse | Función única `abmil_predict_with_attention()` en un solo bucle sobre stems |
| **Compatibilidad attn_stats con custom modelos** | 3 | Si el usuario construye modelos sin `GatedAttention` (p.ej. spatial), ¿qué pasa? | Función `attention_stats_extended` es agnóstica: acepta cualquier vector `a`; brazo spatial ya usa GatedAttention |

---

## 📝 APÉNDICE — CHECKLIST DE IMPLEMENTACIÓN

### Tabla de Archivos y Funciones Tocados (Por Idea)

| Archivo | Función/Clase | Idea(s) | Tipo | Cambio Propuesto |
|---|---|---|---|---|
| `src/ensemble4.py` | `Ensemble._transform()` | 1, 3 | Existente | Sin cambios directos; pero guardia actualizada: `if extra_features and meta_model in ...` |
| `src/ensemble4.py` | `Ensemble.forward()` | 1, 3 | Existente | Insertar bloque `if self.extra_features_list: ...` después de reweighting, antes de fit |
| `src/ensemble4.py` | `Ensemble._compute_extra_features()` | 1, 3 | **Nuevo** | Calcular disagreement, entropía, attn_entropy, attn_ess, etc. |
| `src/ensemble4.py` | `main()` | 1, 3 | Existente | Agregar flag CLI `--extra_features` (lista de feature names) |
| `src/abmil_engine.py` | `GatedAttention` | 2 | Existente | Sin cambios (baseline); variante `GatedMultiHeadAttention` en nuevo módulo |
| `src/abmil_engine.py` | `abmil_predict()` | 3 | Existente | Sin cambios (mantener descarte de atención en ruta estándar) |
| `src/abmil_engine.py` | `abmil_predict_with_attention()` | 3 | **Nuevo** | Extraer y devolver atención + stats |
| `src/abmil_engine.py` | `attention_stats_extended()` | 3 | **Nuevo** | Calcular entropy_norm, ess_norm, top-k mass, percentiles |
| `src/mh_abmil.py` | `GatedMultiHeadAttention` | 2 | **Nuevo módulo** | Implementación de multi-head con mean-pool |
| `src/mh_abmil.py` | `make_mh_model()` | 2 | **Nuevo** | Factory para construir modelos con arch="mh_abmil" |
| `src/train_abmil.py` | `save_fold_outputs()` | 3 | Existente | Sin cambios (mantener como está) |
| `src/train_abmil.py` | `save_fold_outputs_attn()` | 3 | **Nuevo** | Guardar attn_stats.npy + attn_stats_columns.json |
| `src/train_abmil.py` | `abmil_train_test()` | 2, 3 | Existente | Si Idea 2: condicionalmente cargar `mh_abmil`. Si Idea 3: cambiar a `abmil_predict_with_attention` |
| `src/test_abmil.py` | `abmil_predict_train_fold()` | 3 | Existente | Cambiar a `abmil_predict_with_attention`, guardar attn_stats |
| `src/test_abmil.py` | Carga condicional de ckpts | 2 | Existente | Extender rama `isinstance(ckpt, dict)` para `"arch" == "mh_abmil"` |
| `src/build_oof_features.py` | Bucle de nested CV | 3 | Existente | Cambiar a `abmil_predict_with_attention`, guardar attn_stats_oof |
| `src/train_abmil_fge.py` | `save_fold_outputs_fge()` | 2, 3 | Existente | Si Idea 2: usar GatedMultiHeadAttention. Si Idea 3: guardar attn_stats_per_snapshot |
| `src/meta_models.py` | Todos | 1, 3 | Existente | Sin cambios; ya aceptan X_meta de cualquier anchura |

---

## 🎓 CONCLUSIÓN

Este documento proporciona el análisis técnico y los planes de validación para 3 ideas concretas de mejora, todas evaluadas contra la restricción dominante del proyecto (MDE ≈ ±0.016 en 50 folds).

**Recomendación ejecutiva:**
1. Implementar **Idea 1 (Feature Engineering)** como prueba inicial de bajo riesgo (2-3 días).
2. En paralelo, **ejecutar Fase 1 de Idea 3** (generar attn_stats.npy, 4-6 horas, cero costo GPU).
3. Decidir Idea 2 (Multi-head) basándose en señal de Ideas 1+3.

**Ruta de costo mínimo al máximo aprendizaje:** Ideas 1+3 cuestan ~3 días + gastos negligibles, y sirven como sonda para invalidar early o justificar la inversión de 150-200 GPU-horas en Idea 2.

---

**Documento compilado:** 19 de Agosto de 2026  
**Base:** Análisis verificado contra src/abmil_engine.py, ensemble4.py, meta_models.py, utils.py  
**Estado:** Listo para implementación
