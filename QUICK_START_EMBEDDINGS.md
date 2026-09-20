# Quick Start: Embeddings Ponderados para Ensemble

## 1️⃣ Generar Embeddings (uno por modelo)

```bash
python src/test_abmil.py \
    --foundational_model ctranspath \
    --use_embeddings \
    --work_dir /home/JKP6679/Patho-Ensemble/PARADIS/datos/patches \
    --train_source cptac_brca \
    --tissue_patching 20x_224px_0px_overlap \
    --task_name TP53_mutation

# Repite para: uni_v2, virchow_v1
```

**Salida:** `{model}_train_eval_embeddings/val_outputs_embeddings/fold_k/`

## 2️⃣ Entrenar Meta-Modelos

### Opción A: Un modelo a la vez
```bash
python src/ensemble4.py \
    --foundational_models ctranspath uni_v2 virchow_v1 \
    --meta_model mlp_snapshot \
    --meta_features embeddings \
    --work_dir /home/JKP6679/Patho-Ensemble/PARADIS/datos/patches \
    --train_source cptac_brca \
    --tissue_patching 20x_224px_0px_overlap \
    --task_name TP53_mutation
```

### Opción B: Usar sbatch (batch)
```bash
sbatch run_embeddings_4models.sbatch       # 4 modelos simples
sbatch run_extended_models.sbatch          # 4 modelos complejos
```

## 3️⃣ Ver Resultados

```bash
# Resumen agregado
cat /home/JKP6679/Patho-Ensemble/PARADIS/datos/patches/cptac_brca/TP53_mutation/abmil/ensemble4_mlp_snapshot_embeddings/test_metrics_summary.json | python -m json.tool

# Por fold
ls /home/JKP6679/Patho-Ensemble/PARADIS/datos/patches/cptac_brca/TP53_mutation/abmil/ensemble4_mlp_snapshot_embeddings/test_metrics/fold_*/metrics.json

# Visualizar informe
open INFORME_ENSEMBLE_EMBEDDINGS.html
```

## 4️⃣ Opciones de Meta-Modelos

| Opción | Tiempo | Complejidad | Recomendación |
|---|---|---|---|
| `logreg` | Rápido | Mínima | Baseline, interpretable |
| `mlp` | Medio | Baja | Rápido, simple |
| `mlp_snapshot` ⭐ | Medio | Media | **MEJOR en embeddings** |
| `mlp_fge` | Medio | Media | Alternativa a Snapshot |
| `deep_mlp` | Lento | Alta | Más expresivo |
| `deep_mlp_snapshot` | Lento | Alta | Deep + regularización |
| `tabpfn` | Rápido | Alta | Modelo pre-entrenado |
| `tabpfn_snapshot` | Rápido | Alta | TabPFN + ensemble |

## 5️⃣ Opciones Clave de ensemble4.py

```bash
--meta_model {logreg,mlp,mlp_snapshot,mlp_fge,
              deep_mlp,deep_mlp_snapshot,
              tabpfn,tabpfn_snapshot}
--meta_features embeddings        # Usar embeddings en lugar de predictions
--meta_epochs 100                 # Épocas de training
--meta_lr 1e-3                   # Learning rate
--meta_wd 1e-4                   # Weight decay
--meta_patience 8                # Early stopping patience
--n_cycles 6                     # Ciclos de Snapshot/FGE
--restart_lr 5e-3               # Learning rate en restart
```

## 6️⃣ Monitoreo de Jobs

```bash
# Ver estado
squeue -j 70055

# Ver logs en tiempo real
tail -f /shared/home/JKP6679/Patho-Ensemble/logs/extended_models_70055.out

# Ver errores
cat /shared/home/JKP6679/Patho-Ensemble/logs/extended_models_70055.err

# Cancelar si falla
scancel 70055
```

## 7️⃣ Troubleshooting

### ❌ "No module named tabpfn"
```bash
pip install tabpfn
```

### ❌ "Directory not found: val_outputs_embeddings"
Asegúrate de que `test_abmil.py --use_embeddings` se ejecutó primero.

### ❌ "CUDA out of memory"
```bash
# Reducir batch size (no disponible)
# O usar CPU: --meta_model logreg (no necesita GPU)
```

### ❌ Shape mismatch en ensemble4.py
El código detecta dimensiones automáticamente. Si falla, chequea que todos los modelos tengan embeddings generados.

## 📊 Benchmarks

### Fase 1 (4 modelos) - ~2-3 horas
| Modelo | AUC | Acc | F1 |
|---|---|---|---|
| MLP Snapshot | 0.800 | 0.751 | 0.750 |
| LogReg | 0.790 | 0.738 | 0.737 |

### Fase 2 (8 modelos) - ~4 horas adicionales
Resultados pendientes (Job 70055 en ejecución)

## 🔗 Referencias

- Informe completo: `INFORME_ENSEMBLE_EMBEDDINGS.html`
- Resumen técnico: `RESUMEN_TRABAJO_COMPLETADO.md`
- Status: `STATUS_EXPERIMENTO_EMBEDDINGS.md`

---

**Última actualización:** Agosto 18, 2026  
**Status:** ⚙️ Job 70055 en ejecución
