# 📊 Proyecto: Ensemble con Embeddings Ponderados

## 📑 Documentación Generada

### 1. **INFORME_ENSEMBLE_EMBEDDINGS.html** 🌟
- **Tipo:** Informe interactivo con gráficos
- **Contenido:** 
  - Resumen ejecutivo
  - Análisis estadístico (paired t-tests)
  - 4 gráficos comparativos
  - Feature importance analysis
  - Recomendaciones

### 2. **INFORME_ENSEMBLE_EMBEDDINGS.md**
- **Tipo:** Markdown legible
- **Contenido:** Versión texto del informe HTML

### 3. **RESUMEN_TRABAJO_COMPLETADO.md** 📋
- **Tipo:** Documentación técnica detallada
- **Contenido:**
  - Objetivo del proyecto
  - Arquitectura de embeddings
  - Cambios técnicos realizados
  - Resultados Fase 1
  - Status Fase 2

### 4. **STATUS_EXPERIMENTO_EMBEDDINGS.md** ⚡
- **Tipo:** Estado actual del proyecto
- **Contenido:**
  - Fase 1: ✅ COMPLETADA (4 modelos)
  - Fase 2: 🔄 EN CURSO (4 modelos, Job 70055)
  - Timeline y próximos pasos

### 5. **QUICK_START_EMBEDDINGS.md** 🚀
- **Tipo:** Guía de inicio rápido
- **Contenido:**
  - Comandos para generar embeddings
  - Opciones de meta-modelos
  - Troubleshooting

### 6. **README_EMBEDDINGS_ENSEMBLE.md** (este archivo)
- **Tipo:** Índice y guía general

---

## 📊 Resultados Fase 1

### Tabla Resumen (50 folds)
| Meta-Modelo | AUC | Accuracy | F1 | Kappa |
|---|---|---|---|---|
| LogReg | 0.7896 ± 0.0907 | 0.7377 ± 0.0824 | 0.7366 ± 0.0812 | 0.4531 ± 0.1665 |
| MLP | 0.7901 ± 0.1019 | 0.7052 ± 0.0930 | 0.6589 ± 0.1354 | 0.3223 ± 0.2420 |
| **MLP Snapshot** ⭐ | **0.7995 ± 0.1027** | **0.7512 ± 0.0919** | **0.7498 ± 0.0908** | **0.4812 ± 0.1844** |
| MLP FGE | 0.7947 ± 0.1011 | 0.7319 ± 0.0900 | 0.7142 ± 0.1052 | 0.4095 ± 0.2057 |

### Feature Importance
```
Top Modelos (% en top 20 features):
- ctranspath:  65% (dominante)
- uni_v2:      35%
- virchow_v1:   0%
```

---

## 🔬 Cambios Técnicos Principales

### 1. Nuevas Clases (abmil_engine.py)
- `ABMIL_Base`: Clase base con lógica compartida
- `ABMIL_EMBEDDING`: Extrae embeddings (512-dim) sin clasificar

### 2. Actualización test_abmil.py
- Flag `--use_embeddings` para generar embeddings
- Genera 3 tipos: train, val, test

### 3. Modificación ensemble4.py
- Detección automática de dimensiones
- Flexible concatenation vs fixed stacking

### 4. 4 Nuevos Meta-Modelos (meta_models.py)
- `DeepMLPMetaClassifier` (3 capas)
- `SnapshotDeepMLPMetaClassifier` (Deep MLP + cycles)
- `TabPFNMetaClassifier` (Pre-trained foundation model)
- `SnapshotTabPFNMetaClassifier` (TabPFN + ensemble)

---

## 🏃 Cómo Usar

### Opción 1: Usar Scripts SLURM Prebuild

```bash
# Fase 1 (ya ejecutado)
sbatch run_embeddings_4models.sbatch

# Fase 2 (en curso)
sbatch run_extended_models.sbatch
```

### Opción 2: Manual Step-by-Step

```bash
# 1. Generar embeddings
python src/test_abmil.py --foundational_model ctranspath --use_embeddings ...

# 2. Entrenar meta-modelo
python src/ensemble4.py --meta_model mlp_snapshot --meta_features embeddings ...
```

Ver `QUICK_START_EMBEDDINGS.md` para comandos completos.

---

## 📈 Fase 2: En Curso (Job 70055)

### 4 Nuevos Meta-Modelos
1. **DeepMLP** - Red de 3 capas con BatchNorm
2. **DeepMLP Snapshot** - Deep MLP + warm-restart cycles
3. **TabPFN** - Tabular Prior Foundation Network
4. **TabPFN Snapshot** - TabPFN ensemble

### Timeline
- **Inicio:** Agosto 18, 2026 (~19:30)
- **Duración estimada:** 2-4 horas
- **GPU:** gpu04
- **Logs:** `/shared/home/JKP6679/Patho-Ensemble/logs/extended_models_70055.out`

### Próximos Pasos (después de Job 70055)
1. ✅ Generar reporte extendido (8 modelos)
2. ✅ Comparar resultados
3. ✅ Copiar informe actualizado al proyecto

---

## 🎯 Key Insights

### ✅ Lo que Funciona
1. **Embeddings > Predicciones**: 512-dim vs 2-dim proporciona mucha más información
2. **Warm-Restart Cycles**: MLP Snapshot regula mejor que FGE en este problema
3. **LogReg Estable**: Sin overfitting (Kappa=0.453), buen baseline
4. **ctranspath Dominante**: Más informativo que otros modelos para TP53_mutation

### ❌ Lo que NO Funciona
1. **MLP Simple**: Colapsa por dimensionalidad (1536→16→2)
2. **FGE Cycles**: Menos efectivo que warm-restart aquí
3. **Complejidad Extra**: Deep MLP puede overfit con pocos datos

### 🎯 Recomendación Final
**Usar MLP Snapshot** como meta-modelo:
- Mejor rendimiento (AUC 0.800, Acc 0.751)
- Regularización natural mediante averaging
- Robusto a hyperparámetros
- Tiempo de entrenamiento razonable

---

## 📂 Estructura de Directorios

```
/home/JKP6679/Patho-Ensemble/
├── src/
│   ├── abmil_engine.py          (modificado: +80 líneas)
│   ├── test_abmil.py            (modificado: +50 líneas)
│   ├── ensemble4.py             (modificado: +70 líneas)
│   ├── meta_models.py           (modificado: +250 líneas)
│   └── build_oof_features.py    (modificado: +30 líneas)
│
├── run_embeddings_4models.sbatch    (nuevo)
├── run_extended_models.sbatch       (nuevo)
│
├── INFORME_ENSEMBLE_EMBEDDINGS.html (nuevo, 387 KB)
├── INFORME_ENSEMBLE_EMBEDDINGS.md   (nuevo, 3.9 KB)
├── RESUMEN_TRABAJO_COMPLETADO.md    (nuevo)
├── STATUS_EXPERIMENTO_EMBEDDINGS.md (nuevo)
├── QUICK_START_EMBEDDINGS.md        (nuevo)
└── README_EMBEDDINGS_ENSEMBLE.md    (este archivo)

/shared/home/JKP6679/Patho-Ensemble/PARADIS/datos/patches/
└── cptac_brca/TP53_mutation/abmil/
    ├── ensemble4_embeddings/                    (LogReg)
    ├── ensemble4_mlp_embeddings/                (MLP)
    ├── ensemble4_mlp_snapshot_embeddings/       (MLP Snapshot) ⭐
    ├── ensemble4_mlp_fge_embeddings/            (MLP FGE)
    ├── ensemble4_deep_mlp_embeddings/           ([en training])
    ├── ensemble4_deep_mlp_snapshot_embeddings/  ([en training])
    ├── ensemble4_tabpfn_embeddings/             ([en training])
    └── ensemble4_tabpfn_snapshot_embeddings/    ([en training])
```

---

## 🔗 Referencias Rápidas

| Documento | Para | Acceso |
|---|---|---|
| INFORME_ENSEMBLE_EMBEDDINGS.html | Ejecutivo + gráficos | `open INFORME_ENSEMBLE_EMBEDDINGS.html` |
| INFORME_ENSEMBLE_EMBEDDINGS.md | Lectura simple | Abrir con editor |
| RESUMEN_TRABAJO_COMPLETADO.md | Técnico detallado | Para developers |
| STATUS_EXPERIMENTO_EMBEDDINGS.md | Estado actual | Chequear Fase 2 |
| QUICK_START_EMBEDDINGS.md | Comenzar a usar | Copiar comandos |

---

## 📞 Soporte

### Errores Comunes

**Q: "No module named 'tabpfn'"**  
A: Instalar con `pip install tabpfn`

**Q: "Directory not found: val_outputs_embeddings"**  
A: Ejecutar `test_abmil.py --use_embeddings` primero

**Q: ¿Cuánto tiempo toma?**  
A: Fase 1 (4 modelos): ~3 horas | Fase 2 (4 modelos): ~4 horas

---

**Proyecto:** Ensemble with Weighted Embeddings  
**Dataset:** cptac_brca / TP53_mutation  
**Autor:** Claude Code  
**Fecha:** Agosto 18, 2026  
**Status:** ✅ Fase 1 completa, 🔄 Fase 2 en curso  
**Email:** rafael.pachon.alvarez@gmail.com  
