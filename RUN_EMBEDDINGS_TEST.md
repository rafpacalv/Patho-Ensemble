# 🚀 Ejecutar Experimento: Embeddings + 4 Meta-Modelos

## Inicio Rápido

```bash
sbatch /shared/home/JKP6679/Patho-Ensemble/run_embeddings_4models.sbatch
```

## ¿Qué hace?

1. **Genera embeddings ponderados** (512-dim) para 3 modelos base
   - ctranspath, uni_v2, virchow_v1
   - Se guardan en `*_train_eval_embeddings/`

2. **Entrena 4 meta-modelos** con los embeddings concatenados:
   - **LogisticRegression** (baseline)
   - **MLP** (CosineAnnealingLR + early stopping)
   - **Snapshot MLP** (CosineAnnealingWarmRestarts + snapshots)
   - **FGE MLP** (warm-up + ciclos FGE)

3. **Computa métricas** por fold y agregadas

## Tiempo estimado

- **Total**: ~2-4 horas (GPU)
- Embeddings: ~20 min
- Meta-modelos: ~1-3 horas (según convergencia)

## Logs

Mientras se ejecuta:

```bash
# Ver el estado en tiempo real
tail -f embeddings_4models_<JOB_ID>.out
```

Al terminar:

```bash
# Ver el resultado final
cat embeddings_4models_<JOB_ID>.out | tail -20
```

## Resultados

Se guardan en:
```
/home/JKP6679/Patho-Ensemble/PARADIS/datos/patches/cptac_brca/TP53_mutation/abmil/
```

Por meta-modelo:
- `ensemble4_embeddings/` → LogisticRegression
- `ensemble4_mlp_embeddings/` → MLP
- `ensemble4_mlp_snapshot_embeddings/` → Snapshot MLP
- `ensemble4_mlp_fge_embeddings/` → FGE MLP

Cada uno contiene:
- `test_metrics_summary.json` - AUC, ACC, F1 agregados
- `test_metrics/fold_*/metrics.json` - Métricas por fold
- `coefs.npy` (solo LogReg/MLP) - Coeficientes del modelo

## Comparar Resultados

Una vez completado, los resultados están **alineados por fold**, listos para:

```bash
# Estadística apareada (t-test por fold)
python scripts/compare_metrics.py \
    --model1 ensemble4_embeddings \
    --model2 ensemble4_mlp_embeddings \
    --dataset cptac_brca \
    --task TP53_mutation
```

## Verificar antes de ejecutar

```bash
# Comprobar que todo compila
python -m py_compile \
    src/abmil_engine.py \
    src/test_abmil.py \
    src/build_oof_features.py \
    src/ensemble4.py

# Comprobar que los datos existen
ls -d /home/JKP6679/Patho-Ensemble/PARADIS/datos/patches/cptac_brca/TP53_mutation/
ls -d /home/JKP6679/Patho-Ensemble/PARADIS/datos/features/cptac_brca/features_*/

# Comprobar que las clases están disponibles
python -c "from src import abmil_engine; print('✓ ABMIL_EMBEDDING:', hasattr(abmil_engine, 'ABMIL_EMBEDDING'))"
```

---

**¡Listo! Ejecuta el sbatch cuando quieras.** 🚀
