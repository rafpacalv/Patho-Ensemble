# Status: Experimento de Ensemble con Embeddings Ponderados

## Fase 1: Experimento Inicial ✅ COMPLETADO

### Modelos Entrenados (4)
1. **LogisticRegression** - Baseline interpretable
2. **MLPMetaClassifier** - Red simple con CosineAnnealingLR
3. **SnapshotMLPMetaClassifier** ⭐ GANADOR - Con warm-restart cycles
4. **FGEMLPMetaClassifier** - Con ciclos FGE

### Resultados Iniciales
| Modelo | AUC | Accuracy | F1 | Kappa |
|---|---|---|---|---|
| LogReg | 0.7896 ± 0.0907 | 0.7377 ± 0.0824 | 0.7366 ± 0.0812 | 0.4531 ± 0.1665 |
| MLP | 0.7901 ± 0.1019 | 0.7052 ± 0.0930 | 0.6589 ± 0.1354 | 0.3223 ± 0.2420 |
| **MLP Snapshot** | **0.7995 ± 0.1027** | **0.7512 ± 0.0919** | **0.7498 ± 0.0908** | **0.4812 ± 0.1844** |
| MLP FGE | 0.7947 ± 0.1011 | 0.7319 ± 0.0900 | 0.7142 ± 0.1052 | 0.4095 ± 0.2057 |

### Análisis de Features
**Top modelo informativo: ctranspath (65% de top 20 features)**
- ctranspath es el modelo fundacional más importante para TP53_mutation
- uni_v2 contribuye con 35%
- virchow_v1 no aparece en top 20

### Archivos Generados (Fase 1)
- ✅ `/home/JKP6679/Patho-Ensemble/INFORME_ENSEMBLE_EMBEDDINGS.html` (387 KB)
- ✅ `/home/JKP6679/Patho-Ensemble/INFORME_ENSEMBLE_EMBEDDINGS.md` (3.9 KB)

---

## Fase 2: Experimento Extendido 🔄 EN CURSO

### Modelos en Entrenamiento (4 Nuevos)
1. **DeepMLPMetaClassifier** - 3 capas (256→128→64) con BatchNorm
2. **SnapshotDeepMLPMetaClassifier** - Deep MLP con warm-restart cycles
3. **TabPFNMetaClassifier** - Tabular Prior Foundation Network
4. **SnapshotTabPFNMetaClassifier** - TabPFN con múltiples inicializaciones

### Job Status
- **Job ID:** 70055
- **Status:** EN EJECUCIÓN
- **Expected Time:** ~2-4 horas
- **Logs:** `/shared/home/JKP6679/Patho-Ensemble/logs/extended_models_70055.out`

### Arquitecturas Nuevas
```
DeepMLP:
  Input(1536) → Linear(256) → BN → ReLU → Dropout
             → Linear(128) → BN → ReLU → Dropout
             → Linear(64)  → BN → ReLU → Dropout
             → Linear(2)

TabPFN:
  Tabular Prior Foundation Network (arquitectura propietaria)
  - n_ensemble=32 (standard), seed-based variation
```

### Modificaciones al Código
1. **meta_models.py**: +150 líneas
   - Clase `_DeepMLP` (arquitectura)
   - Clase `DeepMLPMetaClassifier`
   - Clase `SnapshotDeepMLPMetaClassifier`
   - Clase `TabPFNMetaClassifier`
   - Clase `SnapshotTabPFNMetaClassifier`

2. **ensemble4.py**: +30 líneas
   - Actualizado imports
   - Actualizado results_subdir_map
   - Actualizado instantiation logic
   - Actualizado argument parser

3. **run_extended_models.sbatch**: Script SLURM para Fase 2

---

## Infraestructura

### Dataset
- **Nombre:** cptac_brca/TP53_mutation (binary classification)
- **Modelos Base:** ctranspath, uni_v2, virchow_v1
- **Meta-features:** Embeddings concatenados (512×3 = 1536-dim)
- **Folds:** 50

### Configuración Entrenamiento
- **Optimizador:** AdamW (lr=1e-3, wd=1e-4)
- **Early Stopping:** patience=8, basado en AUC de validación
- **Ciclos:** CosineAnnealingWarmRestarts (T_0=20, T_mult=1)
- **Device:** GPU CUDA

### Directorios de Resultados
```
/home/JKP6679/Patho-Ensemble/PARADIS/datos/patches/cptac_brca/TP53_mutation/abmil/
├── ensemble4_embeddings/              # LogReg
├── ensemble4_mlp_embeddings/          # MLP
├── ensemble4_mlp_snapshot_embeddings/ # MLP Snapshot ⭐
├── ensemble4_mlp_fge_embeddings/      # MLP FGE
├── ensemble4_deep_mlp_embeddings/     # [EN ENTRENAMIENTO]
├── ensemble4_deep_mlp_snapshot_embeddings/
├── ensemble4_tabpfn_embeddings/
└── ensemble4_tabpfn_snapshot_embeddings/
```

---

## Timeline Esperado

### ✅ Completado
- Fase 1: Experimento inicial (4 modelos)
  - Tiempo: ~2 horas
  - Fecha: Agosto 18, 2026

### 🔄 En Curso
- Fase 2: Experimento extendido (4 modelos nuevos)
  - Job: 70055
  - Tiempo estimado: 2-4 horas más
  - ETA: Agosto 18 (~21:00-23:00)

### ⏳ Pendiente
- Generación de informe extendido
- Actualización de HTML/MD con 8 modelos
- Análisis comparativo final

---

## Cambios Técnicos Resumidos

### 1. Instalación de Dependencias
```bash
pip install tabpfn==8.3.0
```

### 2. Nuevas Clases Python
- **DeepMLPMetaClassifier**: MLP con 3 capas + BatchNorm
- **SnapshotDeepMLPMetaClassifier**: Deep MLP + CosineAnnealingWarmRestarts
- **TabPFNMetaClassifier**: Wrapper para TabPFN
- **SnapshotTabPFNMetaClassifier**: Ensemble de TabPFN

### 3. Actualización de ensemble4.py
- Soporta los 8 meta-modelos (4 originales + 4 nuevos)
- Mantiene backward compatibility
- Argumentos de línea de comando actualizados

---

## Próximos Pasos

1. ✅ Esperar a que complete Job 70055
2. ⏳ Generar reporte extendido con todos los 8 modelos
3. ⏳ Crear tabla comparativa
4. ⏳ Actualizar informe HTML/Markdown
5. ⏳ Copiar informe al proyecto local

---

**Última actualización:** Agosto 18, 2026  
**Usuario:** rafael.pachon.alvarez@gmail.com  
**Branch:** pruebas_rafa
