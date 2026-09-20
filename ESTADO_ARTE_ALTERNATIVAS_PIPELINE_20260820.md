# ESTADO DEL ARTE: Alternativas al Pipeline de Ensemble
## Fusión Temprana a Nivel de Parche

**Fecha**: 20 de agosto de 2026  
**Experimento**: IDEA 4 — Early Fusion Pilot (5 folds)  
**Dataset**: cptac_brca / TP53_mutation  
**Status**: En ejecución (Job 70184)

---

## 📋 Resumen Ejecutivo

| Alternativa | Factibilidad | Coste | Riesgo | Prioridad | Status |
|---|---|---|---|---|---|
| **Late Fusion (actual)** | Alta | Bajo | Bajo | — | ✅ Probado |
| **Early Fusion (ctranspath⊕virchow_v1)** | Alta | Muy bajo | Bajo | **1** | 🔄 En piloto |
| Feature Engineering Meta (Idea 1) | Alta | Cero | Bajo | 2 | ❌ Nulo (Δ=-0.0014) |
| Attention Maps (Idea 3) | Media | Medio | Medio | 3 | ❌ Nulo (Δ=+0.0013) |
| Multi-head Attention (Idea 2) | Alta | Bajo | Medio | 4 | ❌ Negativo (Δ=-0.024) |

---

## 🔍 Diagnóstico: Todo es Late Fusion

**Exploración completada**: 2 agentes Explore verificaron el codebase.

### Hallazgos

**Meta-learners (15 variantes en `meta_models.py`/`ensemble4.py`)**:
- LogisticRegression, MLP, MLP_Snapshot, MLP_FGE, MLP_DeepEns, LogitAvg, Gating, DeepMLP, DeepMLP_Snapshot, TabPFN, TabPFN_Snapshot, LightGBM, SVM, KNN, NB
- **Todas**: late fusion — combinan únicamente vectores ya colapsados por slide (prob/logit/embedding-512)

**Arquitecturas de pooling (3 variantes en `abmil_engine.py`/`spatial_abmil.py`)**:
- ABMIL (gated attention, single-head)
- SpatialABMIL (smooth/sage_bn/sage — graph message passing antes de atención)
- MultiHeadGatedAttention (multi-head, mean-pool — **fallido en piloto Idea 2**, Δ=-0.024)
- **Todas**: asumen un único `in_dim` por modelo — no hay concatenación a nivel de parche

**Consecuencia**: Cada modelo procesa independientemente, collapsa su opinión en el softmax final por slide, y solo entonces se combinan esas 3 opiniones ya firmes. Cuando el meta-learner ve un desacuerdo, la información de qué parches específicos lo causaron ya se perdió.

### Hallazgo Crítico: Alineación de Coordenadas

**Verificado leyendo `.h5` directamente** (no solo código):

| Modelo | # slides | Coordenadas |
|---|---|---|
| ctranspath | 387 | ✓ Idénticas row-by-row |
| virchow_v1 | 387 | ✓ Idénticas row-by-row |
| uni_v2 | 650 | ✗ Escala/rango distinto (~1.7× más parches) |

**En cptac_brca, ctranspath y virchow_v1 comparten exactamente las mismas 387 slides
con coordenadas idénticas fila-a-fila.** Esto significa que se pueden fusionar a
nivel de parche con **coste de alineación = CERO** (solo concatenar filas).

---

## 💡 Alternativa Propuesta: Fusión Temprana

### Mecánica

**En vez de 3 ABMIL independientes + stacking → 1 ABMIL que ve información de múltiples modelos por parche**

```
Actual (Late Fusion):
  ctranspath  ────→ [ABMIL₁] ────→ [p₁] ──┐
  uni_v2      ────→ [ABMIL₂] ────→ [p₂] ──┼→ [LogReg] → final
  virchow_v1  ────→ [ABMIL₃] ────→ [p₃] ──┘

Propuesto (Early Fusion):
  [ctranspath ⊕ virchow_v1] ──→ [ABMIL_fused] ──→ [p_fused]
         (768+2560=3328)              ↓
                          Atención resuelve desacuerdo LOCAL
  
  uni_v2 ──→ [ABMIL_uni] ──→ [p_uni]
  
  [p_fused, p_uni] ──→ [LogReg] → final
```

### Por qué es distinto

**Late fusion** ve: "modelo A confía en clase 0, modelo B confía en clase 1"  
→ No sabe qué parches causaron esa divergencia

**Early fusion** ve: "en el parche i, modelo A ve feature X, modelo B ve feature Y"  
→ Atención aprende a pesarlos localmente

### Coste de Parámetros

- Antes: `in_dim=2560` (virchow_v1 máximo) → 2560×512 = 1.3M parámetros en la 1ª capa
- Después: `in_dim=3328` → 3328×512 = 1.7M parámetros
- **Δ = +0.4M** (~30% aumento)

**Comparación**: MLP fallido tenía ~5M parámetros. Este cambio es menor, en un modelo
que ya soporta presupuesto de datos actual (50 folds, ~75 samples/fold).

### Limitación: uni_v2

**No fusionable directamente** — usa rejilla de parcheo distinta (650 slides, coordenadas
en escala diferente). **Solución**: mantener uni_v2 como modelo independiente en el
stacking final.

---

## 🛠️ Implementación (`src/ensemble5.py`)

### Verificación de Alineación (Hard Fail)

```python
def _verify_patch_alignment(work_dir, train_source, fuse_models, slide_ids):
    # Lee coords de cada modelo directamente de .h5
    # Assert fila-a-fila igualdad
    # Falla explícitamente si hay divergencia
```

**Estilo**: No silencioso, no warning — exacto como `FileNotFoundError` en `_load_attn_stats`.

### Carga Fusionada

```python
def load_fused_features_cached(work_dir, train_source, fuse_models, slide_ids, latent_dims):
    # load_features_cached(modelo₁) → {slide: tensor [N, d₁]}
    # load_features_cached(modelo₂) → {slide: tensor [N, d₂]}
    # torch.cat(..., dim=1) → {slide: tensor [N, d₁+d₂]}
```

### Entrenamiento

Reusa `abmil_engine.train_abmil` sin modificación:
```python
model = abmil_engine.train_abmil(
    feats=fused_feats,        # {slide: [N, 3328]}
    in_dim=3328,              # suma de dims
    ...
)
```

### Combinación con uni_v2

```python
from ensemble4 import Ensemble

# Instancia fold-by-fold para evitar assert len(folds)==1
ensemble = Ensemble(
    xdirs=[fused_train_eval, uni_v2_train_eval],
    tdirs=[fused_test, uni_v2_test],
    ...
)
```

---

## 📊 Protocolo de Validación

### Piloto (5 folds)

1. Entrenar ABMIL fusionado (ctranspath⊕virchow_v1) en folds 0-4
2. Generar meta-features (train_eval + test)
3. Combinar con uni_v2 vía LogisticRegression
4. **Comparación pareada**: Δ AUC (fused + uni_v2) − (baseline 3 modelos) en esos mismos 5 folds

**Sin pretensión de significancia** (n=5, mismo caveat que Idea 2).

### Si hay señal positiva o neutra

Escalar a 50 folds × Holm-Bonferroni (mismo rigor que Ideas 1-3).

### Si hay señal negativa

Cerrar esta vía, documentar como intento #8 fallido.

---

## ❌ Alternativas Descartadas

### TransMIL, CLAM, DSMIL, DTFD-MIL

**Por qué no**: Siguen siendo mono-modelo, siguen necesitando stacking. Aumentan
complejidad sin cambiar mecanismo de fusión.

### Cross-Attention Aprendida

Atención cruzada entre modelos (transformer cross-encoder).

**Problema**: Más parámetros que concatenación, mismo overfitting risk con ~75 samples/fold.
**Prior histórico**: "7 intentos, 7 nulos" — carga de la prueba EN CONTRA de aumentar capacidad.

---

## 🚀 Alternativas Secundarias (Futuras)

### Data Augmentation a Nivel de Bolsa

Subsampleo/bootstrap de parches en train → regularización sin parámetros nuevos.
**Status**: Pendiente (complementaria, no compete por cómputo con piloto actual).

### Registro Espacial de uni_v2

Proyectar coordenadas de uni_v2 a la rejilla de ctranspath/virchow_v1 vía nearest-neighbor.
**Status**: Follow-up si piloto da señal positiva.

---

**Job Status**: 🔄 70184 en ejecución  
**Resultado Esperado**: ~1-2 horas  
**Versión**: v1 — 20 de agosto de 2026
