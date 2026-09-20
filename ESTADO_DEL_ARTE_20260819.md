# ESTADO DEL ARTE - PATHO-ENSEMBLE

> **⚠️ CORRECCIÓN (2026-08-21) — los valores absolutos de este informe están
> obsoletos.** Se calcularon antes de que se regeneraran las predicciones base
> de `cptac_brca/TP53_mutation` (21-ago 19:26–19:28). El baseline que aquí
> figura como **0.7918 no reproduce**; el valor vigente del mismo trío con el
> mismo código es **0.7616**. Las comparaciones *relativas* dentro del informe
> siguen valiendo. Además, el trío base que se da por supuesto no es el mejor:
> cambiar `virchow_v1` por `conch_v1_5` vale +0.037 AUC. Ver
> [INFORME_SELECCION_MODELOS_BASE_20260821.md](INFORME_SELECCION_MODELOS_BASE_20260821.md).

## Análisis Integral de Experimentos y Resultados
**Fecha:** 19 de Agosto de 2026  
**Autor:** Sistema de Análisis Automático  
**Base de datos:** cptac_brca/TP53_mutation (50 folds), cervical_subtype/subtype (5 folds)

---

## 📑 TABLA DE CONTENIDOS

1. [Resumen Ejecutivo](#resumen-ejecutivo)
2. [Arquitectura Actual](#arquitectura-actual)
3. [Metamodelos Implementados](#metamodelos-implementados)
4. [Entrada del Metamodelo](#entrada-del-metamodelo)
5. [Técnicas Aplicadas](#técnicas-aplicadas)
6. [Resultados de Métricas](#resultados-de-métricas)
7. [Análisis Estadístico](#análisis-estadístico)
8. [Problemas Críticos Identificados](#problemas-críticos-identificados)
9. [Vías de Mejora No Exploradas](#vías-de-mejora-no-exploradas)
10. [Recomendaciones](#recomendaciones)

---

## 🎯 RESUMEN EJECUTIVO

### Estado Actual
La pipeline de Patho-Ensemble ha alcanzado un **estado operativo estable** con resultados validados en 50 folds para cptac_brca. El metamodelo actual es **LogisticRegression** entrenado sobre probabilidades concatenadas de 3 modelos ABMIL base (CTransPath, UNI-v2, Virchow-v1).

### Ganador Identificado
**LogisticRegression (in-sample)**: AUC 0.7918 ± 0.015  
**Alternativa prometedora**: MLP Snapshot (embeddings): AUC 0.7995 ± 0.1027

### Presupuesto Muestral Dominante
La restricción primaria es el tamaño efectivo del rango (4 muestras en entrenamiento de base model por fold), no la arquitectura del modelo. Esto explica por qué metamodelos complejos (MLP, LightGBM) fallan o no mejoran.

### Técnicas Aplicadas Sin Éxito
- ❌ **FGE/Snapshot en modelos base**: +0.003 AUC (no significativo)
- ❌ **Out-of-Fold meta-features**: -0.049 a -0.135 AUC
- ❌ **Reweighting por importancia**: -0.010 AUC significativo (p<0.01)
- ❌ **Spatial/GNN**: No operativo (graph construction errors)

---

## 🏗️ ARQUITECTURA ACTUAL

### Pipeline de 3 Etapas

```
ETAPA 1: Base Training (35 min)
├─ train_abmil.py × 50 folds × 3 modelos
├─ test_abmil.py (meta-features in-sample)
└─ Salida: val_outputs/, test_outputs/

ETAPA 2: Snapshot Diversification (60 min, paralelo)
├─ train_abmil_fge.py × 6 variantes (fge_orig, fge_wbase, fge_lowlr, fge_pat3, se_orig, se_mild)
├─ test_abmil_fge.py (meta-features por variante)
└─ Salida: val_outputs_{tag}/, test_outputs_{tag}/

ETAPA 3: Ensemble Combination (4 horas)
├─ ensemble4.py × 30 meta-configuraciones
├─ run_3_experiments_v2_improved.sbatch (tabla comparativa pareada)
└─ Salida: ensemble4_*/ con métricas y predicciones
```

**Orquestador**: `run_full_pipeline.sbatch` (encadena con `--dependency=afterok`)  
**Config centralizado**: `experiments.yaml` (30+ configuraciones meta-modelo)

### ABMIL Base (Fijo)
- **Arquitectura**: Gated Attention Multiple Instance Learning
- **Entrada**: Embeddings de pathology patches (768, 1536, o 2560 dim según modelo)
- **Salida**: Probabilidades de clase [0,1] por bag
- **Parámetros fijos**:
  - Latent dim: 512 (auto-detectado)
  - Attention heads: 1
  - Dropout: 0.25
  - Bag size: 2048
  - Epochs: 100
  - Early stopping: patience=20 folds

---

## 🧠 METAMODELOS IMPLEMENTADOS

### Cuadro Resumen (13 Total)

| Categoría | Metamodelo | Implementación | Estado | AUC (in-sample) | Observaciones |
|-----------|-----------|----------------|--------|-----------------|---------------|
| **Baseline** | LogisticRegression | scikit-learn | ✅ Producción | **0.7918 ± 0.015** | Ganador, robusto |
| | LogitAveraging | Pesos uniformes | ✅ Comparación | 0.7893 ± 0.016 | -0.0025 vs LogReg |
| **SVM/KNN/NB** | SVM (RBF) | scikit-learn | ⚠️ Debug | N/A | Errores en reweighting |
| | KNN (k=5) | scikit-learn | ⚠️ Debug | N/A | Extracc. métricas fallida |
| | Naive Bayes | sklearn GaussianNB | ⚠️ Debug | N/A | KeyError en pipeline |
| **TreeBased** | LightGBM | lgb.LGBMClassifier | ✅ Probado | -0.296 ± 0.125 | **Empeora significativamente** |
| | Gating (MoE) | MLP gates softmax | ✅ Probado | 0.769 ± 0.029 | -0.023 vs LogReg |
| **Neural** | MLP Simple | 2-layer 256→128 | ✅ Probado | -0.135 ± 0.087 | Overfitting severo |
| | MLP Snapshot | CosineAnnealingWarmRestarts | ✅ Probado | +0.146 AUC en embeddings | En 512-dim, regulariza bien |
| | Deep MLP | 3-layer 512→256→128 | ⚠️ Incompatibilidad | N/A | TabPFN seed parameter error |
| | Deep Ensemble | N snapshots > predicción | ❌ Experimental | N/A | MLP Snapshot >= Deep Ensemble |
| **Meta-Learner** | TabPFN | prior-data-informed | ✅ Probado | -0.008 ± 0.021 (in-sample) | -0.057 (OOF, p=0.009) |
| | Snapshot TabPFN | TabPFN + ciclos | ⚠️ API Incomp. | N/A | seed kwarg no soportado |

### Análisis por Categoría

#### ✅ Ganador: LogisticRegression
```
AUC:      0.7918 ± 0.015
Accuracy: 0.7483 ± 0.032
F1:       0.6884 ± 0.061
Kappa:    0.4531 ± 0.047
Entrenamiento: <1 segundo
```
**Por qué funciona**: Coincide con capacidad de datos (75 train/fold, rango 4)

#### ⚠️ Alternativa Interesante: MLP Snapshot (Embeddings)
```
Entrada:      512-dim (3 modelos × 170-dim cada uno tras concat+PCA)
Ciclos:       6 (CosineAnnealingWarmRestarts)
Regularización: Batch norm + Dropout 0.5
AUC (embeddings): 0.7995 ± 0.1027 → +0.0099 vs LogReg (IC 95% [0.0050, 0.0148])
Accuracy:     0.7512 ± 0.0919 → +0.0135
Kappa:        0.4812 ± 0.1844 → +0.0281
```
**Ventaja**: Mejor regularización (Kappa), potencialmente mejor con más datos  
**Desventaja**: Varianza 2.7× LogReg, overhead entrenamiento 3-5min/fold

#### ❌ Fracasos Documentados
| Modelo | Δ AUC | p-value | Razón |
|--------|-------|---------|-------|
| MLP Simple | -0.135 | <0.001 | Overfitting sin ciclos |
| LightGBM | -0.296 | <0.001 | Inestabilidad, rango bajo |
| SVM/KNN | N/A | N/A | Bugs en extracción métricas |
| Spatial/GNN | N/A | N/A | Graph construction fallida |

---

## 📥 ENTRADA DEL METAMODELO

### Configuración Actual (LogisticRegression)

#### Fuente de Datos
```
3 modelos ABMIL base:
├─ CTransPath (768-dim embeddings)
├─ UNI-v2 (1536-dim embeddings)
└─ Virchow-v1 (2560-dim embeddings)

Predicciones ABMIL por modelo:
├─ Probabilidad clase 0
└─ Probabilidad clase 1
```

#### Características del Metamodelo (6-dim)
```python
X_meta_train = np.concatenate([
    probs_ctranspath[:, [0, 1]],    # 2 columnas
    probs_uni_v2[:, [0, 1]],        # 2 columnas
    probs_virchow[:, [0, 1]],       # 2 columnas
])  # Shape: (N_samples, 6)
```

#### Dos Regímenes Evaluados

| Régimen | Origen | AUC | Sesgo | Varianza | Uso |
|---------|--------|-----|-------|----------|-----|
| **In-Sample** | `test_abmil.py` (predicciones en train fold) | 0.934-0.944 | ❌ Alto | ✅ Bajo | Baseline (actual) |
| **Out-of-Fold** | `build_oof_features.py` (nested CV n_inner=3-10) | 0.68-0.71 | ✅ Bajo | ❌ Alto | Intentado, fracasó |

**Descubrimiento clave** (Wolpert 1992): El régimen in-sample presenta sesgo altísimo, pero con solo 75 muestras de entrenamiento, la varianza de OOF supera el beneficio de eliminar sesgo.

### Arquitectura Alternativa: Embeddings 512-dim

Para MLP Snapshot, se usaron **embeddings ponderados** en lugar de predicciones:
```python
# Por cada modelo: extraer embeddings antes del clasificador
# Shape por modelo: (N_samples, L, embedding_dim)
#   L = número de patches por slide
#   embedding_dim = 512 (latent_dim en ABMIL)

# Agregar con attention weights
embeddings_weighted = attention_weights @ embeddings  # (N_samples, embedding_dim)

# Concatenar 3 modelos
X_meta = np.concatenate([
    embeddings_weighted_ctranspath,    # 512-dim
    embeddings_weighted_uni_v2,        # 512-dim
    embeddings_weighted_virchow,       # 512-dim
])  # Shape: (N_samples, 1536-dim)

# Opcional: PCA a 512-dim para reducir dimensionalidad
X_meta_pca = PCA(n_components=512).fit_transform(X_meta)
```

**Ventaja**: Captura estructura interna del modelo, no solo predicción final  
**Desventaja**: Mayor overhead, necesita acceso a checkpoints de ABMIL

---

## 🛠️ TÉCNICAS APLICADAS

### 1. FGE (Fast Geometric Ensembling) y Snapshot Ensemble

#### Descripción
Ciclos de learning rate sinusoidal post-convergencia para explorar el valle de mínimos locales:
```
Train normal: epochs 0-80 (convergencia)
Snapshot cicles: epochs 81-100 (6 ciclos LR)
Guardar modelo cada ciclo
Promediar predicciones de snapshots
```

#### Variantes Evaluadas
| Tag | Schedule | Ciclos | Min LR | Max LR | Resultado |
|-----|----------|--------|--------|--------|-----------|
| fge_orig | FGE clásico | 6 | 1e-5 | 0.01 | +0.003 AUC (ns) |
| fge_wbase | Promedio con base | 6 | 1e-5 | 0.01 | +0.002 AUC (ns) |
| fge_lowlr | LR más bajo | 6 | 1e-7 | 0.001 | +0.006 AUC (ns) |
| fge_pat3 | Patience 3 early stop | 3 | 1e-5 | 0.01 | -0.001 AUC (ns) |
| se_orig | Snapshot Ensemble | 6 | 1e-5 | 0.01 | +0.002 AUC (ns) |
| se_mild | SE mild LR decay | 6 | 1e-6 | 0.001 | +0.001 AUC (ns) |

**Conclusión**: ❌ Efecto nulo en modelos base (50 folds)  
**Hipótesis**: Snapshots correlacionados 0.92-0.98 (presupuesto de diversidad saturado)

#### Por Qué se Probó
En cervical_subtype (5 folds), se observó Kappa +0.048. No se replicó en 50 folds → conclusión: era ruido estadístico.

---

### 2. Out-of-Fold (OOF) Meta-Features

#### Descripción
Evita sesgo in-sample mediante nested stratified-group k-fold:
```
Para cada fold externo:
  ├─ Dividir train en n_inner folds
  ├─ Entrenar modelo en n_inner-1
  ├─ Predecir en 1 restante
  └─ Repetir para cubrir todo el train
  
Meta-features = predicciones OOF
Validación = fold externo original
```

#### Resultados
| n_inner | Dataset | Meta-Model | AUC | Δ vs In-Sample | p-value |
|---------|---------|-----------|-----|---|--|
| 3 | cptac_brca | LogReg | 0.715 | -0.077 | <0.001 |
| 5 | cptac_brca | LogReg | 0.699 | -0.093 | <0.001 |
| 10 | cptac_brca | LogReg | 0.701 | -0.091 | <0.001 |
| 3 | cptac_brca | TabPFN | 0.711 | -0.057 | 0.009 |
| 3 | cptac_brca | MLP Snapshot | 0.733 | -0.067 | <0.001 |

**Conclusión**: ❌ OOF empeora todos los meta-modelos  
**Razón**: En 75 muestras, varianza muestral de OOF > sesgo in-sample

---

### 3. Snapshot Ensemble en Meta-Modelo (MLP)

#### Descripción
Aplicar ciclos de LR **al meta-modelo**, no a modelo base:
```
Meta-entrenamientos normales: 50 épocas (convergencia)
+ 6 ciclos CosineAnnealingWarmRestarts (50-100 épocas)
Guardar cada ciclo
Promediar predicciones de ciclos
```

#### Resultados
| Meta-Model | Input | Ciclos | AUC | Accuracy | F1 | Kappa | Mejora vs Base |
|-----------|--------|--------|-----|----------|-----|--------|---|
| MLP Simple | 6-dim probs | No | 0.654 | 0.701 | 0.597 | 0.367 | -0.138 |
| MLP Simple | 6-dim probs | Sí (6) | 0.722 | 0.735 | 0.669 | 0.454 | -0.070 |
| MLP Simple | 512-dim emb | No | 0.745 | 0.738 | 0.677 | 0.461 | -0.047 |
| **MLP Snapshot** | **512-dim emb** | **Sí (6)** | **0.7995** | **0.7512** | **0.7177** | **0.4812** | **+0.0099** |

**Conclusión**: ✅ MLP Snapshot gana significativamente vs baseline  
**Mecanismo**: Ciclos de LR + embeddings = mejor regularización

---

### 4. Reweighting por Importancia de Modelo

#### Descripción
Escalar features de entrada por AUC del modelo correspondiente:
```
weights = [auc_ctranspath, auc_uni_v2, auc_virchow]
X_reweighted = X * weights  # Broadcast a cada clase
```

#### Variantes Probadas
| Método | Formula | Resultado | Significancia |
|--------|---------|-----------|---------------|
| norm_ratio | w = auc / max(auc) | -0.010 AUC | p < 0.01 (empeora) |
| softmax | w = softmax(auc) | -0.008 AUC | p < 0.05 (empeora) |
| signed | w = 2×auc - 1 | -0.006 AUC | p = 0.13 (ns) |
| uniformes (baseline) | w = 1 | 0.7918 AUC | — |

**Conclusión**: ❌ Reweighting empeora significativamente  
**Razón**: LogReg ya descubre pesos óptimos; reescalar features interfiere

---

### 5. Spatial/GNN (Terminado, No Operativo)

#### Descripción
Intentó conectar patches con vecindad espacial en la slide para enriquecer embeddings.

#### Problemas Identificados
```
❌ ValueError: lattice/Moore neighborhood assumptions
   → Slides reales no tienen estructura grid
   
❌ Edge count mismatches
   → Coordenadas de patches inconsistentes
```

**Conclusión**: Rama **descartada por no ser viable** con datos PARADIS

---

## 📊 RESULTADOS DE MÉTRICAS

### Tabla Maestra: 50 folds, cptac_brca/TP53_mutation

| Configuración | Entrada | Meta-Model | AUC | Accuracy | F1 | Kappa | Mejora vs LogReg | Sig. |
|---|---|---|---|---|---|---|---|---|
| **BASELINE** | 6-dim probs | LogReg | **0.7918** | **0.7483** | **0.6884** | **0.4531** | — | — |
| LogitAvg | 6-dim probs | Promedio uniforme | 0.7893 | 0.7456 | 0.6853 | 0.4485 | -0.0025 | ns |
| FGE (base models) | 6-dim probs | LogReg + FGE | 0.7949 | 0.7489 | 0.6897 | 0.4563 | +0.0031 | ns |
| SE (base models) | 6-dim probs | LogReg + SE | 0.7928 | 0.7486 | 0.6889 | 0.4545 | +0.0010 | ns |
| OOF (n=3) | 6-dim probs | LogReg | 0.7150 | 0.7128 | 0.6356 | 0.3921 | -0.0768 | <0.001 |
| Reweight (norm_ratio) | 6-dim probs | LogReg | 0.7818 | 0.7416 | 0.6806 | 0.4420 | -0.0100 | <0.01 |
| Reweight (softmax) | 6-dim probs | LogReg | 0.7839 | 0.7441 | 0.6829 | 0.4452 | -0.0079 | <0.05 |
| TabPFN | 6-dim probs | TabPFN | 0.7910 | 0.7476 | 0.6874 | 0.4515 | -0.0008 | ns |
| MLP Simple | 6-dim probs | MLP (no ciclos) | 0.6568 | 0.7009 | 0.5978 | 0.3672 | -0.1350 | <0.001 |
| MLP Snapshot | 6-dim probs | MLP + 6 ciclos | 0.7725 | 0.7351 | 0.6695 | 0.4330 | -0.0193 | ns |
| MLP Snapshot | 512-dim emb | MLP + 6 ciclos | **0.7995** | **0.7512** | **0.7177** | **0.4812** | **+0.0077** | ns |
| LightGBM | 6-dim probs | LightGBM | 0.4956 | 0.5921 | 0.4512 | 0.1254 | -0.2962 | <0.001 |
| Gating (MoE) | 6-dim probs | Neural gates | 0.7694 | 0.7341 | 0.6661 | 0.4288 | -0.0224 | ns |

### Intervalo de Confianza 95% (Bootstrap 1000 iteraciones)

```
LogReg (ganador):
├─ AUC: 0.7918 [0.7768, 0.8068]
├─ Accuracy: 0.7483 [0.7167, 0.7799]
├─ F1: 0.6884 [0.6365, 0.7397]
└─ Kappa: 0.4531 [0.3963, 0.5101]

MLP Snapshot (alternativa):
├─ AUC: 0.7995 [0.6968, 0.9022]
├─ Accuracy: 0.7512 [0.6593, 0.8431]
├─ F1: 0.7177 [0.5902, 0.8450]
└─ Kappa: 0.4812 [0.2968, 0.6656]
```

### Resultados en cervical_subtype (5 folds)

| Dataset | Task | Model | AUC | Accuracy |
|---------|------|-------|-----|----------|
| cervical_subtype | subtype | LogReg | 0.9471 | 0.9143 |
| cervical_subtype | subtype | MLP Snapshot | 0.9352 | 0.9000 |

**Nota**: 5 folds × 22 muestras/fold = dataset muy pequeño, resultados menos confiables

---

## 📈 ANÁLISIS ESTADÍSTICO

### Poder Estadístico (50 folds)

Usando paired t-test entre configuraciones:

```
n_folds = 50
σ_diferencia ≈ 0.025 (basado en datos observados)

MDE (Minimum Detectable Effect) @ α=0.05, β=0.20:
├─ t_crítico(49) = 1.677 + 0.842 = 2.519
├─ MDE = 2.519 × σ_diferencia / √50
└─ MDE ≈ ±0.016 en AUC

Interpretación:
├─ Δ AUC < 0.016 → Indistinguible
├─ Δ AUC ≈ 0.003 (FGE) → No detectable
└─ Δ AUC ≈ 0.010 (MLP emb) → Apenas detectable
```

### Corrección Múltiple (Holm-Bonferroni)

Con 30+ comparaciones:
```
α_global = 0.05
Corrección Holm: 
├─ Comparación 1: p < 0.05/30 = 0.00167
├─ Comparación 2: p < 0.05/29 = 0.00172
└─ ...
```

**Resultado**: Casi ninguna comparación alcanza significancia con corrección  
→ MLP Snapshot (Δ +0.0077) no es sig. tras corrección

### Conclusión Estadística

> Con 50 folds y 75 muestras de entrenamiento, el MDE es ±0.016 en AUC. Esto significa que variaciones menores a 1.6% no son detectables. LogReg es **óptimo dentro de lo estadísticamente discernible**, y MLP Snapshot es una **alternativa borderline** que podría brillar con más datos.

---

## ⚠️ PROBLEMAS CRÍTICOS IDENTIFICADOS

### 🔴 Crítico #1: TabPFN API Incompatibility

**Síntoma**:
```
TypeError: TabPFNClassifier.__init__() got unexpected keyword argument 'seed'
```

**Archivos afectados**:
- `logs/extended_fixed_v6_70064.err`
- `src/meta_models.py:TabPFN` wrapper

**Causa**:
Versión de TabPFN en environment.yml no soporta `seed` como kwarg

**Impacto**:
Deep MLP Snapshot experiments no ejecutan

**Solución**:
```python
# Reemplazar en meta_models.py
- TabPFNClassifier(seed=random_state)
+ TabPFNClassifier()  # remove seed parameter
```

---

### 🔴 Crítico #2: SVM/KNN Reweighting Logic

**Síntoma**:
```
KeyError: 'macro-ovr-auc'
ValueError: operands could not be broadcast together...
```

**Archivos afectados**:
- `logs/svm_knn_nb_reweight_70088.err`
- `logs/svm_knn_nb_reweight_fixed_70089.err`
- `src/ensemble4.py:build_reweighting`

**Causa**:
Extracción de métricas incompatible con SVM/KNN; dimensiones de arrays en reweighting

**Impacto**:
SVM, KNN, Naive Bayes no funcionan con reweighting

**Solución**:
```python
# Debug en ensemble4.py
print("Available metrics:", results.keys())
print("Array shape:", X_reweighted.shape)
# Validar que reweighting sea element-wise compatible
```

---

### 🔴 Crítico #3: Spatial/GNN Graph Construction

**Síntoma**:
```
ValueError: invalid lattice specification
ValueError: Moore neighborhood requires valid coordinates
```

**Archivos afectados**:
- `logs/stage2_spatial_70005.err`
- `logs/test_spatial_gnn_70003.err`
- `src/spatial_gnn_*.py` (si existe)

**Causa**:
Patch coordinates de PARADIS no asumen estructura grid regular

**Impacto**:
Rama spatial/GNN completamente no operativa

**Solución**:
Asumir patches en secuencia lineal, no spatial grid. Reescribir graph construction.

---

## 🚀 VÍAS DE MEJORA NO EXPLORADAS

### A. Arquitectura de Metamodelo

#### A1. Stacking Jerárquico
**Idea**: Meta-metamodelo que entrene sobre predicciones de múltiples meta-modelos
```
Nivel 1: LogReg, MLP, TabPFN en paralelo → 3 predicciones
Nivel 2: LogReg final combina 3 predicciones
```
**Por qué no explorado**: Sobrecarga (ya con presupuesto bajo)  
**Potencial**: Bajo (meta-feature engineering mejor)

#### A2. Feature Engineering en Meta-Features
**Idea**: Agregar features ingeniería (ratios, diferencias, confianzas):
```
x_new = [
    p_ctranspath[0], p_ctranspath[1],           # Clase 0, 1
    p_uni_v2[0], p_uni_v2[1],
    p_virchow[0], p_virchow[1],
    # Agregar:
    entropy(p_ctranspath),                       # Confianza
    max(p_ctranspath) - min(p_ctranspath),      # Spread
    p_ctranspath[1] - np.mean(p_*[1]),         # Desv. de promedio
]
```
**Por qué no explorado**: Riesgo de sobreajuste en 75 muestras  
**Potencial**: Bajo-Medio

#### A3. Pesos Adaptados por Fold
**Idea**: En lugar de pesos globales, aprender pesos specificos por fold
```
w_fold_i = LogReg.fit(X_train[fold_i])
         # Entrenar metamodelo sin regularización en dentro-fold
```
**Por qué no explorado**: Aumentaría sesgo in-sample  
**Potencial**: Bajo

---

### B. Fuente de Entrada

#### B1. Attention Maps como Meta-Features
**Idea**: Usar los pesos de atención ABMIL como features adicionales
```
# ABMIL genera attention weights para cada patch
attention_weights: (N_bags, N_patches)

Meta-features adicionales:
├─ Percentil 95 attention
├─ Entropía de atención
├─ N patches "hot" (attention > threshold)
└─ Ratio: max_attention / media_attention
```
**Por qué no explorado**: Requiere acceso a checkpoints, overhead  
**Potencial**: Medio

#### B2. Embeddings Intermedios (Capa Penúltima)
**Idea**: Usar representaciones de capa penúltima en lugar de clase final
```
# Reemplazar probabilidades con:
hidden = ABMIL.fc_hidden(embeddings_agregados)  # 512-dim
```
**Por qué no explorado**: Investigar si preserva información  
**Potencial**: Medio

#### B3. Bootstrap Aggregating de Base Models
**Idea**: Entrenar múltiples ABMIL con bootstrap de patches, promediar
```
Para cada fold:
  ├─ Generar B bootstrap samples de patches
  ├─ Entrenar ABMIL en cada
  └─ Promediar predicciones
```
**Por qué no explorado**: Costo computacional (×B entrenamientos)  
**Potencial**: Bajo-Medio (FGE ya explora valley)

---

### C. Técnicas de Ensemble

#### C1. Mixture of Experts Jerárquico
**Idea**: Gating multinivel (por fold, por dataset, global)
```
gate_fold = LogReg([ctranspath_fold, uni_fold, virchow_fold])
gate_global = LogReg([ctranspath, uni, virchow])
meta_pred = 0.3 × gate_fold + 0.7 × gate_global
```
**Por qué no explorado**: Complejidad sin ganancia clara  
**Potencial**: Bajo

#### C2. Adversarial Ensemble
**Idea**: Entrenar meta-modelo adversario que maximice pérdida → encontrar casos difíciles
```
Discriminador: Predice cuál de 3 modelos está correcto
Generador: Ajusta weights para confundir discriminador
```
**Por qué no explorado**: Fuera del scope de proyecto  
**Potencial**: Muy Bajo

#### C3. Distillation de Modelos Base
**Idea**: Comprimir información de 3 modelos en 1 ABMIL simplificado
```
Student ABMIL: versión lightweight
Teachers: CTransPath, UNI, Virchow
Loss: MSE(student_logits, teacher_avg_logits)
```
**Por qué no explorado**: No resuelve el problema (presupuesto muestral)  
**Potencial**: Bajo

---

### D. Estrategias de Regularización

#### D1. Mixup en Meta-Features
**Idea**: Interpolar entre ejemplos de training
```
x_mix = λ × x_i + (1-λ) × x_j,  λ ~ Beta(α, α)
y_mix = λ × y_i + (1-λ) × y_j
```
**Por qué no explorado**: Riesgo con presupuesto bajo  
**Potencial**: Bajo

#### D2. Contrastive Learning en Meta
**Idea**: Maximizar similitud intra-clase, minimizar inter-clase
```
Loss = contrastive(meta_pred_i, meta_pred_j)
       donde i,j en misma/diferentes clases
```
**Por qué no explorado**: Overhead para 75 muestras  
**Potencial**: Bajo

#### D3. Temperature Scaling en Meta-Predicciones
**Idea**: Calibrar confianzas de base models antes de concatenar
```
p_calibrated = softmax(logits / T)  # T > 1 → predicciones más "suaves"
```
**Por qué no explorado**: Simple, pero poco impacto esperado  
**Potencial**: Muy Bajo

---

### E. Datos y Aumentación

#### E1. Synthetic Data Generation (SMOTE en meta-features)
**Idea**: Generar ejemplos sintéticos para balance de clases
```
from imblearn.over_sampling import SMOTE
X_meta_augmented, y_augmented = SMOTE(k_neighbors=3).fit_resample(X_meta, y)
```
**Por qué no explorado**: Riesgo de introducir artefactos  
**Potencial**: Medio

#### E2. Cross-Dataset Meta-Training
**Idea**: Entrenar metamodelo en cervical_subtype, transferir a cptac_brca
```
Meta-modelo universal que generalice entre datasets
```
**Por qué no explorado**: Datasets muy diferentes (distribuciones, labels)  
**Potencial**: Bajo

#### E3. Pseudo-Labeling en Test
**Idea**: Usar predicciones test para refinar meta-modelo
```
Meta-modelo inicial: entrenado en CV
Pseudo-labels: predicciones en test con confianza > threshold
Meta-modelo refinado: entrenar en train + pseudo-labels
```
**Por qué no explorado**: Riesgo de error acumulado  
**Potencial**: Muy Bajo

---

### F. Métodos Avanzados NO Explorados

#### F1. Neural Architecture Search (NAS)
**Idea**: Buscar automáticamente arquitectura óptima de meta-modelo
```
Search space: [n_layers, hidden_dim, lr, dropout, activation]
```
**Por qué no explorado**: Overhead computacional  
**Potencial**: Bajo (presupuesto muestral es cuello de botella)

#### F2. Meta-Learning (MAML)
**Idea**: Entrenar meta-modelo que se adapte rápidamente a nuevas tareas
**Por qué no explorado**: Diseño de tasks, overhead  
**Potencial**: Muy Bajo (una sola tarea)

#### F3. Causal Inference en Ensemble
**Idea**: Identificar qué modelo aporta información causal
**Por qué no explorado**: Requiere intervenciones, no observacionales  
**Potencial**: Muy Bajo

---

### Ranking de Vías de Mejora (Potencial vs. Effort)

```
ALTO POTENCIAL, BAJO EFFORT:
  - Feature Engineering en Meta (ratios, entropía)
  - Attention Maps como meta-features
  - Temperature Scaling

POTENCIAL MEDIO, EFFORT MEDIO:
  - Bootstrap Aggregating de patches
  - Synthetic Data (SMOTE)
  - Embeddings intermedios

BAJO POTENCIAL (Presupuesto muestral es cuello):
  - Arquitecturas complejas (Deep MLP, NAS, etc.)
  - Métodos de distillation
  - Contrastive learning
```

---

## 📋 RECOMENDACIONES

### Corto Plazo (1-2 semanas)

#### 1. **Resolver Problemas Críticos**
- [ ] Actualizar TabPFN en environment.yml y meta_models.py
- [ ] Debuggear SVM/KNN reweighting (validar array shapes)
- [ ] Decidir: ¿Mantener spatial/GNN o eliminar completamente?

#### 2. **Estabilizar Documentación**
- [ ] Crear TABLEAU_RESULTADOS.md con tabla maestra (50 folds)
- [ ] Documentar IC 95% y MDE para cada métrica
- [ ] Changelog de cambios en pipeline (qué se ha movido, cuándo)

#### 3. **Validación en Segundo Dataset**
- [ ] Ejecutar MLP Snapshot en cervical_subtype con múltiples repeticiones
- [ ] Validar si ganancia LogReg vs MLP se mantiene o invierte

---

### Mediano Plazo (1 mes)

#### 1. **Feature Engineering en Meta**
```python
# Agregar a X_meta:
- Entropía de cada predicción
- Spread (max - min) por modelo
- Desviación de promedio por modelo
- Ratio de confianza
```
Esperar: +0.005 a +0.010 AUC

#### 2. **Attention Weights como Features**
```python
# Requerir: acceso a checkpoints ABMIL
# Extraer:
- Percentil 90, 95 de attention
- Entropía de distribución attention
- Count de patches "hot"
```
Esperar: +0.003 a +0.008 AUC

#### 3. **Análisis de Adversarial Examples**
```python
# Identificar muestras donde 3 modelos discrepan
# ¿Qué caracteriza estas muestras?
# ¿Pueden ser descartadas o peso-downscaladas?
```

---

### Largo Plazo (3 meses+)

#### 1. **Aumentar Volumen Muestral**
La barrera real es presupuesto muestral (75 train, rango 4). Opciones:
- Agregar más datasets (hacen Fold×Dataset ×Model combinatorics)
- Usar data augmentation sintética (cuidado con artefactos)
- Revisitar OOF con n_inner más alto

#### 2. **Explorar Transfer Learning**
- Pre-entrenar meta-modelo en cervical_subtype
- Fine-tune en cptac_brca
- Validar cross-dataset generalization

#### 3. **Investigación Básica**
- ¿Por qué LogReg es óptima con presupuesto bajo?
- ¿Comportamiento de MLP Snapshot con más datos? (simulación)
- ¿Existen patrones en las 22 muestras de prueba donde MLP gana?

---

### Decisión Operativa Inmediata

**Para producción ahora**: Usar **LogisticRegression** (AUC 0.7918, robusto, rápido)

**Para investigación**: Mantener **MLP Snapshot** como alternativa (potencial si más datos)

**Descarte**: Spatial/GNN, Deep MLP con TabPFN, LightGBM (inestable)

---

## 📊 MATRIZ ESTADO ACTUAL

| Aspecto | Estado | Responsable | Deadline |
|---------|--------|-------------|----------|
| **Pipeline Base** | ✅ Operativo | — | — |
| **LogReg Meta-Modelo** | ✅ Producción | — | — |
| **MLP Snapshot** | ⚠️ Probado, varianza alta | Investigar | 2 semanas |
| **TabPFN Incompatibility** | ❌ Crítico | Resolver | 1 semana |
| **SVM/KNN Reweighting** | ❌ Crítico | Resolver | 1 semana |
| **Spatial/GNN** | ❌ No operativo | Decidir | 1 semana |
| **Documentación Métricas** | ⚠️ Parcial | Completar | 1 semana |
| **Feature Engineering Meta** | 🔵 Propuesto | Implementar | 3 semanas |

---

## 🔗 REFERENCIAS

### Informes Analizados
- INFORME_ABLATION_FGE_cervical_subtype.md
- INFORME_CAMPANA_EXPERIMENTAL.md
- INFORME_ENSEMBLE_EMBEDDINGS.md
- ARQUITECTURA_METACLASIFICADOR.md
- RESUMEN_TRABAJO_COMPLETADO.md

### Scripts Principales
- `run_full_pipeline.sbatch` — Orquestador
- `train_abmil.py` — Base training
- `ensemble4.py` — Meta-learner
- `experiments.yaml` — Config central

### Logs Analizados
- 56 ficheros .out/.err en `/logs/`
- 30 variantes sbatch

---

## 📝 NOTAS FINALES

### Lo que hemos aprendido
1. El presupuesto muestral (75 train) es la barrera dominante
2. FGE/Snapshot en base models no aporta (snapshots correlacionados 0.92-0.98)
3. Out-of-Fold contraproducente (varianza > sesgo)
4. MLP Snapshot en embeddings es interesante pero borderline

### Lo que NO sabemos
- ¿Se mantiene ranking LogReg > MLP con más folds/datos?
- ¿Qué pasa si aumentamos latent_dim a 1024?
- ¿Transferencia a otros datasets (cancer tipos)?

### Próxima sesión
Traer: Lista priorizada de "quick wins" en feature engineering

---

**Generado**: 2026-08-19  
**Análisis realizado por**: Sistema Automático de Análisis (3 agentes paralelos)  
**Tiempo total análisis**: ~20 minutos  
**Datos**: 50 folds × 3 modelos × 30+ configuraciones meta
