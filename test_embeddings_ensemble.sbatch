#!/bin/bash
#SBATCH --job-name=test_embeddings_4x
#SBATCH --output=test_embeddings_4x_%j.out
#SBATCH --error=test_embeddings_4x_%j.err
#SBATCH --time=04:00:00
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --partition=gpu

set -euo pipefail

# ============================================================================
# Test embeddings in-sample with 4 meta-models on cptac_brca/TP53_mutation
# ============================================================================
#
# Pipeline:
# 1. Generar embeddings ponderados in-sample (512-dim cada modelo)
# 2. Entrenar 4 meta-modelos diferentes (todos usan AdamW + early stopping):
#    - LogisticRegression (baseline)
#    - MLPMetaClassifier (CosineAnnealingLR + early stopping)
#    - SnapshotMLPMetaClassifier (CosineAnnealingWarmRestarts + snapshots)
#    - FGEMLPMetaClassifier (warm-up + FGE cycles)
# 3. Comparar resultados
#
# Nota: Los embeddings ponderados tienen dimensión 512 por modelo base.
#       Con 3 modelos: 512*3 = 1536 dimensiones totales (flexible).
#       El metaclasificador detecta dimensiones y se adapta automáticamente.

REPO_DIR="/shared/home/JKP6679/Patho-Ensemble"
cd "$REPO_DIR"

# Configuración del dataset
WORK_DIR="/home/JKP6679/Patho-Ensemble/PARADIS/datos/patches"
TRAIN_SOURCE="cptac_brca"
TISSUE_PATCHING="20x_224px_0px_overlap"
TASK_NAME="TP53_mutation"
EPOCHS=100

# Modelos base a usar
MODELS=("ctranspath" "uni_v2" "virchow_v1")

echo "================================================================================"
echo "TEST: Embeddings In-Sample Ensemble - 4 Meta-Models"
echo "================================================================================"
echo "Dataset: $TRAIN_SOURCE/$TASK_NAME"
echo "Modelos base: ${MODELS[*]}"
echo "Tissue patching: $TISSUE_PATCHING"
echo "Base epochs: $EPOCHS"
echo ""
echo "Meta-models (all with AdamW + early stopping):"
echo "  1. LogisticRegression (baseline)"
echo "  2. MLP (CosineAnnealingLR)"
echo "  3. Snapshot MLP (CosineAnnealingWarmRestarts)"
echo "  4. FGE MLP (warm-up + FGE cycles)"
echo "================================================================================"
echo ""

# ============================================================================
# STEP 1: Generate embeddings (in-sample, 512-dim per model)
# ============================================================================
echo "[STEP 1] Generating in-sample embeddings (512-dim ponderados)..."
echo "----------------------------------------------------------------------"

for model in "${MODELS[@]}"; do
    echo "  → Generando embeddings ponderados para $model..."
    python src/test_abmil.py \
        --foundational_model "$model" \
        --work_dir "$WORK_DIR" \
        --train_source "$TRAIN_SOURCE" \
        --tissue_patching "$TISSUE_PATCHING" \
        --task_name "$TASK_NAME" \
        --epochs "$EPOCHS" \
        --use_embeddings
    echo "    ✓ Embeddings ponderados de $model completados"
done

echo ""
echo "✅ Todos los embeddings ponderados generados (se guardan en *_train_eval_embeddings/)"
echo ""

# ============================================================================
# STEP 2: Train 4 different meta-models
# ============================================================================
echo "[STEP 2] Training 4 meta-models with embeddings..."
echo "----------------------------------------------------------------------"
echo ""

# 2.1 LogisticRegression (baseline)
echo "  [2.1] Logistic Regression (baseline)..."
python src/ensemble4.py \
    --foundational_models ctranspath uni_v2 virchow_v1 \
    --work_dir "$WORK_DIR" \
    --train_source "$TRAIN_SOURCE" \
    --tissue_patching "$TISSUE_PATCHING" \
    --task_name "$TASK_NAME" \
    --meta_model logreg \
    --meta_features embeddings
echo "  ✓ Logistic Regression completada"
echo ""

# 2.2 MLPMetaClassifier (CosineAnnealingLR + early stopping)
echo "  [2.2] MLP (CosineAnnealingLR + early stopping)..."
python src/ensemble4.py \
    --foundational_models ctranspath uni_v2 virchow_v1 \
    --work_dir "$WORK_DIR" \
    --train_source "$TRAIN_SOURCE" \
    --tissue_patching "$TISSUE_PATCHING" \
    --task_name "$TASK_NAME" \
    --meta_model mlp \
    --meta_features embeddings \
    --meta_epochs 100 \
    --meta_lr 1e-3 \
    --meta_wd 1e-4 \
    --meta_patience 8
echo "  ✓ MLP completada"
echo ""

# 2.3 SnapshotMLPMetaClassifier (CosineAnnealingWarmRestarts + snapshots)
echo "  [2.3] Snapshot MLP (CosineAnnealingWarmRestarts + snapshots)..."
python src/ensemble4.py \
    --foundational_models ctranspath uni_v2 virchow_v1 \
    --work_dir "$WORK_DIR" \
    --train_source "$TRAIN_SOURCE" \
    --tissue_patching "$TISSUE_PATCHING" \
    --task_name "$TASK_NAME" \
    --meta_model mlp_snapshot \
    --meta_features embeddings \
    --meta_epochs 120 \
    --meta_lr 1e-3 \
    --meta_wd 1e-4 \
    --n_cycles 6 \
    --restart_lr 5e-3 \
    --n_snapshots_ensemble 6
echo "  ✓ Snapshot MLP completada"
echo ""

# 2.4 FGEMLPMetaClassifier (warm-up + FGE cycles)
echo "  [2.4] FGE MLP (warm-up + FGE cycles)..."
python src/ensemble4.py \
    --foundational_models ctranspath uni_v2 virchow_v1 \
    --work_dir "$WORK_DIR" \
    --train_source "$TRAIN_SOURCE" \
    --tissue_patching "$TISSUE_PATCHING" \
    --task_name "$TASK_NAME" \
    --meta_model mlp_fge \
    --meta_features embeddings \
    --meta_epochs 100 \
    --meta_lr 1e-3 \
    --meta_wd 1e-4 \
    --fge_n_cycles 6 \
    --fge_cycle_length 3 \
    --fge_lr_1 1e-3 \
    --fge_lr_2 1e-5 \
    --fge_cycle_patience 2 \
    --n_snapshots_ensemble 6
echo "  ✓ FGE MLP completada"
echo ""

# ============================================================================
# STEP 3: Comparison summary
# ============================================================================
echo "================================================================================"
echo "[STEP 3] RESULTS SUMMARY"
echo "================================================================================"
echo ""

RESULTS_BASE="$WORK_DIR/$TRAIN_SOURCE/$TASK_NAME/abmil"

echo "Buscando resultados en: $RESULTS_BASE"
echo ""
echo "Modelos meta-aprendices (embeddings):"
echo "┌─────────────────────────────┬──────────────────────────────────────┐"
echo "│ Meta-Model                  │ Directorio de resultados             │"
echo "├─────────────────────────────┼──────────────────────────────────────┤"

# Check each result
for model_name in "ensemble4_embeddings" "ensemble4_mlp_embeddings" "ensemble4_mlp_snapshot_embeddings" "ensemble4_mlp_fge_embeddings"; do
    if [ -d "$RESULTS_BASE/$model_name" ]; then
        metrics_count=$(ls "$RESULTS_BASE/$model_name"/test_metrics/fold_*/metrics.json 2>/dev/null | wc -l)
        printf "│ %-27s │ %d folds completados                 │\n" "${model_name%_embeddings}" "$metrics_count"
    fi
done

echo "└─────────────────────────────┴──────────────────────────────────────┘"
echo ""

# ============================================================================
# Final summary
# ============================================================================
echo "================================================================================"
echo "✅ PIPELINE DE PRUEBA COMPLETADO"
echo "================================================================================"
echo ""
echo "Archivos de resultados por meta-modelo:"
echo "  • Logistic Regression:  $RESULTS_BASE/ensemble4_embeddings/"
echo "  • MLP:                  $RESULTS_BASE/ensemble4_mlp_embeddings/"
echo "  • Snapshot MLP:         $RESULTS_BASE/ensemble4_mlp_snapshot_embeddings/"
echo "  • FGE MLP:              $RESULTS_BASE/ensemble4_mlp_fge_embeddings/"
echo ""
echo "Cada directorio contiene:"
echo "  • test_metrics/fold_*/metrics.json - Métricas por fold"
echo "  • test_metrics_summary.json - Métricas agregadas"
echo "  • (coefs.npy, odds_ratios.npy si aplica)"
echo ""
echo "Arquitectura neural (todos excepto LogReg):"
echo "  • Input: embeddings concatenados (512*3 = 1536-dim flexible)"
echo "  • Hidden: 16-dim + ReLU + Dropout(0.1)"
echo "  • Output: num_classes"
echo "  • Optimizador: AdamW (lr=1e-3, wd=1e-4)"
echo "  • Early stopping: basado en validación AUC"
echo ""
echo "✏️  Nota: Para comparar estadísticamente los resultados,"
echo "   los datos están alineados (mismo fold, mismo split)."
echo ""
