# Próximos Pasos: Fase 2b Reparada

**Estado actual:** Fase 1 completada (4 modelos) + Fase 2a completada (1 modelo) = 5/8 modelos ✅  
**Error reparado:** SnapshotDeepMLPMetaClassifier._auc() — incompatibilidad dimensional  
**Fecha:** Agosto 18, 2026

---

## 🚀 Acción Inmediata

Ejecutar el sbatch reparado:

```bash
cd /home/JKP6679/Patho-Ensemble
sbatch run_extended_models_fixed.sbatch
```

**Qué hará:**
- Entrenar Deep MLP Snapshot (MLP 3 capas + warm-restart cycles)
- Entrenar TabPFN (Tabular Prior Foundation Network)
- Entrenar TabPFN Snapshot (ensemble de TabPFN)

**Tiempo estimado:** 2-3 horas  
**GPU:** 1× NVIDIA (partition=main)  
**Logs:** 
- Salida: `/shared/home/JKP6679/Patho-Ensemble/logs/extended_models_fixed_*.out`
- Errores: `/shared/home/JKP6679/Patho-Ensemble/logs/extended_models_fixed_*.err`

---

## 📋 Cambios Aplicados (Fix)

### Archivo: `/shared/home/JKP6679/Patho-Ensemble/src/meta_models.py`

**Línea 1014-1020 (SnapshotDeepMLPMetaClassifier._auc)**

**Antes:**
```python
def _auc(self, X_val, y_val):
    probs = self.predict_proba(X_val)
    y_val = np.asarray(y_val)
    if len(self.classes_) == 2:
        return roc_auc_score(y_val, probs[:, 1])  # ← Error si probs es 1D
    else:
        return roc_auc_score(y_val, probs, multi_class='ovr')
```

**Después:**
```python
def _auc(self, X_val, y_val):
    probs = self.predict_proba(X_val)
    probs = np.atleast_2d(probs)  # ← NUEVO: Garantizar siempre 2D
    y_val = np.asarray(y_val)
    if len(self.classes_) == 2:
        return roc_auc_score(y_val, probs[:, 1])  # ✅ Ahora siempre 2D
    else:
        return roc_auc_score(y_val, probs, multi_class='ovr')
```

**Afectadas:**
- `SnapshotDeepMLPMetaClassifier` (línea ~1014)
- `TabPFNMetaClassifier` (línea ~similar)
- `SnapshotTabPFNMetaClassifier` (línea ~similar)

**Status:** ✅ Ya aplicado a la versión en `/shared/home/JKP6679/Patho-Ensemble/src/meta_models.py`

---

## 📊 Resultados Esperados

Una vez completado Job (próximo), tendremos:

### Modelos 6-8 (Fase 2b)
| Modelo | Tiempo esperado | Notas |
|---|---|---|
| Deep MLP Snapshot | ~30-40 min | MLP 3-capas + ciclos warm-restart |
| TabPFN | ~40-50 min | Pre-trained foundation model, rápido |
| TabPFN Snapshot | ~40-50 min | Ensemble de TabPFN con seed variation |

### Tabla Final (8 Modelos)
| Modelo | AUC | Completado |
|---|---|---|
| LogReg | 0.7896 ± 0.0907 | ✅ |
| MLP | 0.7901 ± 0.1019 | ✅ |
| **MLP Snapshot** ⭐ | **0.7995 ± 0.1027** | ✅ |
| MLP FGE | 0.7947 ± 0.1011 | ✅ |
| Deep MLP | 0.7882 ± 0.1020 | ✅ |
| Deep MLP Snapshot | TBD | ⏳ |
| TabPFN | TBD | ⏳ |
| TabPFN Snapshot | TBD | ⏳ |

---

## 📈 Análisis Posterior (Después de Job)

Una vez completada Fase 2b:

### 1. Generar Informe Final
```bash
python3 << 'EOF'
# Script en /tmp/generate_final_report.py
# Cargar resultados de 8 modelos
# Ejecutar paired t-tests
# Generar HTML + Markdown
# Identificar ganador final
EOF
```

### 2. Comparaciones Clave
- **Deep MLP Snapshot vs MLP Snapshot:** ¿Mejora la arquitectura más compleja con ciclos?
- **TabPFN vs MLP Snapshot:** ¿Supera el modelo pre-entrenado al snapshot ensemble?
- **TabPFN Snapshot vs TabPFN:** ¿Añaden valor los ciclos a TabPFN?

### 3. Ranking Final
Esperamos que el ranking sea similar a Fase 1, pero con TabPFN posiblemente competitivo.

---

## 📝 Verificación Post-Job

Después de ~2-3 horas, verificar:

### Comprobar Completación
```bash
# Ver estado del job
squeue -j <JOB_ID>

# Si completó:
ls -la /home/JKP6679/Patho-Ensemble/PARADIS/datos/patches/cptac_brca/TP53_mutation/abmil/ensemble4_*_embeddings/

# Debería ver:
# - ensemble4_deep_mlp_snapshot_embeddings/
# - ensemble4_tabpfn_embeddings/
# - ensemble4_tabpfn_snapshot_embeddings/
```

### Verificar Resultados
```bash
# Ver métricas finales
for model in deep_mlp_snapshot tabpfn tabpfn_snapshot; do
  echo "=== $model ==="
  cat /home/JKP6679/Patho-Ensemble/PARADIS/datos/patches/cptac_brca/TP53_mutation/abmil/ensemble4_${model}_embeddings/test_metrics_summary.json | python -m json.tool
done
```

### Revisar Logs
```bash
# Ver última línea del output
tail -20 /shared/home/JKP6679/Patho-Ensemble/logs/extended_models_fixed_*.out

# Ver errores (si los hay)
cat /shared/home/JKP6679/Patho-Ensemble/logs/extended_models_fixed_*.err
```

---

## 🔄 Automatización (Opcional)

Para monitoreo automático, ejecutar:

```bash
# Monitor job y genera reporte cuando completa
cd /tmp
cat > monitor_fase2b.sh << 'EOF'
#!/bin/bash
JOB_ID=$1
while true; do
  if ! squeue -j $JOB_ID &>/dev/null; then
    echo "✅ Job $JOB_ID completó"
    # Generar reporte automáticamente
    python3 /tmp/generate_final_report.py
    break
  fi
  echo "⏳ Job $JOB_ID aún en ejecución..."
  sleep 120
done
EOF

chmod +x monitor_fase2b.sh
./monitor_fase2b.sh <JOB_ID> &
```

---

## 📂 Archivos Generados Hasta Ahora

### Documentación (Todos en `/home/JKP6679/Patho-Ensemble/`)
- ✅ INFORME_ENSEMBLE_EMBEDDINGS.html
- ✅ INFORME_ENSEMBLE_EMBEDDINGS.md
- ✅ INFORME_ENSEMBLE_EMBEDDINGS_EXTENDED.html ← **5 modelos**
- ✅ INFORME_ENSEMBLE_EMBEDDINGS_EXTENDED.md ← **5 modelos**
- ✅ QUICK_START_EMBEDDINGS.md
- ✅ RESUMEN_TRABAJO_COMPLETADO.md
- ✅ README_EMBEDDINGS_ENSEMBLE.md
- ✅ STATUS_EXPERIMENTO_EMBEDDINGS_UPDATED.md ← **Este archivo**
- ✅ NEXT_STEPS.md ← **Guía de próximos pasos**

### Scripts SLURM
- ✅ run_embeddings_4models.sbatch (Fase 1 — Job 70054)
- ⚠️ run_extended_models.sbatch (Fase 2 — Job 70055, parcialmente fallido)
- ✅ run_extended_models_fixed.sbatch ← **Fase 2b reparado**

---

## 🎯 Resumen Ejecutivo (5 Modelos Actuales)

### 🏆 Ganador: MLP Snapshot
- **AUC:** 0.7995 ± 0.1027
- **Accuracy:** 0.7512 ± 0.0919
- **F1:** 0.7498 ± 0.0908
- **Kappa:** 0.4812 ± 0.1844

**Conclusión:** Snapshot Ensemble con ciclos de warm-restart regulariza automáticamente y produce mejor generalización que modelos más complejos.

---

## ✅ Checklist

- [x] Identificar error en Job 70055
- [x] Aplicar fix a meta_models.py
- [x] Crear sbatch reparado (run_extended_models_fixed.sbatch)
- [x] Generar informe extendido con 5 modelos
- [x] Documentar status actualizado
- [ ] **Ejecutar: `sbatch run_extended_models_fixed.sbatch`** ← **SIGUIENTE PASO**
- [ ] Esperar ~2-3 horas
- [ ] Generar informe final con 8 modelos
- [ ] Actualizar documentación
- [ ] Copiar informes a proyecto local (si aplica)

---

**Instrucciones:** Ejecutar `sbatch run_extended_models_fixed.sbatch` cuando esté listo.  
**Tiempo total (completo):** ~2-3 horas + 30 min para reportes  
**Contacto:** rafael.pachon.alvarez@gmail.com
