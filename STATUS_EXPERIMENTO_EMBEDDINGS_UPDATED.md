# Status: Experimento de Ensemble con Embeddings Ponderados (ACTUALIZADO)

**Fecha actualización:** Agosto 18, 2026  
**Status general:** ⚙️ Fase 1 completada, Fase 2 parcial + reparación

---

## Fase 1: Experimento Inicial ✅ COMPLETADO

### Modelos Entrenados (4)
1. ✅ **LogisticRegression** — Baseline interpretable
2. ✅ **MLPMetaClassifier** — Red simple con CosineAnnealingLR
3. ✅ **SnapshotMLPMetaClassifier** ⭐ **GANADOR** — Con warm-restart cycles
4. ✅ **FGEMLPMetaClassifier** — Con ciclos FGE

### Resultados Fase 1
| Modelo | AUC | Accuracy | F1 | Kappa |
|---|---|---|---|---|
| **MLP Snapshot** | **0.7995 ± 0.1027** | **0.7512 ± 0.0919** | **0.7498 ± 0.0908** | **0.4812 ± 0.1844** |
| MLP FGE | 0.7947 ± 0.1011 | 0.7319 ± 0.0900 | 0.7142 ± 0.1052 | 0.4095 ± 0.2057 |
| LogReg | 0.7896 ± 0.0907 | 0.7377 ± 0.0824 | 0.7366 ± 0.0812 | 0.4531 ± 0.1665 |
| MLP | 0.7901 ± 0.1019 | 0.7052 ± 0.0930 | 0.6589 ± 0.1354 | 0.3223 ± 0.2420 |

**Job ID:** 70054  
**Tiempo total:** ~2 horas  
**Status:** ✅ Completado exitosamente

---

## Fase 2a: Deep MLP ✅ COMPLETADO

### Modelo Entrenado (1)
1. ✅ **DeepMLPMetaClassifier** — 3 capas (256→128→64) con BatchNorm

### Resultados Fase 2a
| Modelo | AUC | Accuracy | F1 | Kappa |
|---|---|---|---|---|
| Deep MLP | 0.7882 ± 0.1020 | 0.6717 ± 0.1127 | 0.6326 ± 0.1491 | 0.3071 ± 0.2276 |

**Conclusión:** Arquitectura más compleja causa overfitting. Underperforms vs MLP Snapshot en todos los métricas. No recomendado.

**Incluido en:** Job 70055 (completó exitosamente antes de fallar en Fase 2b)

---

## Fase 2b: Deep MLP Snapshot + TabPFN ⚠️ EN REPARACIÓN

### Modelos Pendientes (3)
1. ❌ **SnapshotDeepMLPMetaClassifier** — Deep MLP + warm-restart (ERROR REPARADO)
2. ❌ **TabPFNMetaClassifier** — Tabular Prior Foundation Network
3. ❌ **SnapshotTabPFNMetaClassifier** — TabPFN con seed variation

### Error Encontrado (Job 70055)
```
File "/shared/home/JKP6679/Patho-Ensemble/src/meta_models.py", line 1018, in _auc
    return roc_auc_score(y_val, probs[:, 1])
IndexError: invalid index to scalar variable.
```

**Root Cause:** Cuando `X_val` contiene un solo sample, `predict_proba()` retorna array 1D en lugar de 2D.

### Fix Aplicado (meta_models.py)
```python
def _auc(self, X_val, y_val):
    probs = self.predict_proba(X_val)
    probs = np.atleast_2d(probs)  # ← NUEVO: Garantizar siempre 2D
    y_val = np.asarray(y_val)
    ...
```

**Commit:** Cambio aplicado a `/shared/home/JKP6679/Patho-Ensemble/src/meta_models.py`

### Próxima Ejecución
```bash
sbatch run_extended_models_fixed.sbatch
```

**Job estimado:** ~2 horas  
**Modelos a entrenar:** 3 (Deep MLP Snapshot, TabPFN, TabPFN Snapshot)  
**Logs:** `/shared/home/JKP6679/Patho-Ensemble/logs/extended_models_fixed_*.out`

---

## Resumen Completo (5/8 Modelos)

### Modelos Completados
1. ✅ LogisticRegression — AUC 0.7896
2. ✅ MLPMetaClassifier — AUC 0.7901
3. ✅ **SnapshotMLPMetaClassifier** ⭐ **GANADOR** — **AUC 0.7995**
4. ✅ FGEMLPMetaClassifier — AUC 0.7947
5. ✅ DeepMLPMetaClassifier — AUC 0.7882

### Modelos Pendientes (Reparados)
6. ⏳ SnapshotDeepMLPMetaClassifier — (fix applied)
7. ⏳ TabPFNMetaClassifier — (ready)
8. ⏳ SnapshotTabPFNMetaClassifier — (ready)

---

## Infraestructura

### Dataset
- **Nombre:** cptac_brca/TP53_mutation (binary classification)
- **Modelos Base:** ctranspath, uni_v2, virchow_v1
- **Meta-features:** Embeddings concatenados (512×3 = 1536-dim)
- **Folds:** 50

### Directorios de Resultados
```
/home/JKP6679/Patho-Ensemble/PARADIS/datos/patches/cptac_brca/TP53_mutation/abmil/
├── ensemble4_embeddings/                           ✅ 50 folds
├── ensemble4_mlp_embeddings/                       ✅ 50 folds
├── ensemble4_mlp_snapshot_embeddings/              ✅ 50 folds (GANADOR)
├── ensemble4_mlp_fge_embeddings/                   ✅ 50 folds
├── ensemble4_deep_mlp_embeddings/                  ✅ 50 folds
├── ensemble4_deep_mlp_snapshot_embeddings/         ⏳ (en reparación)
├── ensemble4_tabpfn_embeddings/                    ⏳ (en reparación)
└── ensemble4_tabpfn_snapshot_embeddings/           ⏳ (en reparación)
```

---

## Timeline

### ✅ Completado
- **Fase 1:** 4 modelos iniciales
  - Duración: ~2 horas
  - Job: 70054
  - Status: EXITOSO

- **Fase 2a:** 1 modelo (Deep MLP)
  - Duración: ~30 min (incluido en Job 70055)
  - Status: EXITOSO

### 🔄 En Curso
- **Fase 2b:** 3 modelos (reparados)
  - Job: (próximo, sbatch run_extended_models_fixed.sbatch)
  - Duración estimada: ~2 horas
  - Status: LISTO PARA EJECUTAR

### 📅 Próximos Pasos
1. Ejecutar `sbatch run_extended_models_fixed.sbatch`
2. Esperar completación (~2 horas)
3. Generar informe final con 8 modelos
4. Comparación estadística: Fase 1 vs Fase 2
5. Actualizar documentación

---

## Archivos Generados

### Documentación
- ✅ INFORME_ENSEMBLE_EMBEDDINGS.html (Fase 1)
- ✅ INFORME_ENSEMBLE_EMBEDDINGS.md (Fase 1)
- ✅ INFORME_ENSEMBLE_EMBEDDINGS_EXTENDED.html (5 modelos)
- ✅ INFORME_ENSEMBLE_EMBEDDINGS_EXTENDED.md (5 modelos)
- ✅ QUICK_START_EMBEDDINGS.md
- ✅ RESUMEN_TRABAJO_COMPLETADO.md
- ✅ README_EMBEDDINGS_ENSEMBLE.md

### Scripts SLURM
- ✅ run_embeddings_4models.sbatch (Fase 1)
- ⚠️ run_extended_models.sbatch (Fase 2, job 70055 parcialmente fallido)
- ✅ run_extended_models_fixed.sbatch (Fase 2b reparado)

### Código
- ✅ src/ensemble4.py (flexible 8 meta-modelos)
- ✅ src/meta_models.py (todos los meta-modelos + FIX aplicado)
- ✅ src/test_abmil.py (embeddings in-sample)
- ✅ src/build_oof_features.py (embeddings out-of-fold)

---

## 🎯 Recomendación

### Usar: MLP Snapshot ⭐

**Por qué:**
- Mejor AUC: 0.7995 ± 0.1027
- Mejor generalización: Kappa = 0.4812
- Menos complejo que Deep MLP
- Ciclos de warm-restart actúan como regularización automática

**No usar:**
- **Deep MLP:** Overfitting (AUC 0.7882, acc 0.6717)
- **MLP puro:** Oscilación alta sin regularización (σ_kappa = 0.2420)

---

## 📊 Cambios de Código

### meta_models.py
- Fix: `np.atleast_2d(probs)` en `_auc()` (ambas implementaciones)
- Afecta: SnapshotDeepMLPMetaClassifier, TabPFNMetaClassifier, SnapshotTabPFNMetaClassifier

### ensemble4.py
- Soporta 8 meta-modelos (sin cambios en esta versión)
- Auto-detección de dimensiones de embeddings
- Compatible con in-sample y out-of-fold meta-features

---

**Última actualización:** Agosto 18, 2026, 19:45 UTC  
**Usuario:** rafael.pachon.alvarez@gmail.com  
**Branch:** pruebas_rafa  
**Próxima acción:** Ejecutar `sbatch run_extended_models_fixed.sbatch`
