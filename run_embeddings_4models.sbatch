#!/bin/bash
#SBATCH --job-name=embeddings_4models
#SBATCH --output=logs/embeddings_4models_%j.out
#SBATCH --error=logs/embeddings_4models_%j.err
#SBATCH --time=12:00:00
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --partition=main

set -euo pipefail

# Activate conda environment
source /home/JKP6679/miniconda3/etc/profile.d/conda.sh
conda activate ensemble

cd /home/JKP6679/Patho-Ensemble

echo "================================================================================"
echo "🚀 EXPERIMENTO: Embeddings Ponderados + 4 Meta-Modelos"
echo "================================================================================"
echo ""
echo "Configuración:"
echo "  Dataset: cptac_brca / TP53_mutation"
echo "  Modelos base: ctranspath, uni_v2, virchow_v1"
echo "  Meta-modelos: LogReg, MLP, Snapshot, FGE (todos con AdamW)"
echo "  Embeddings: 512-dim ponderados × 3 modelos = 1536-dim flexible"
echo ""

WORK_DIR="/home/JKP6679/Patho-Ensemble/PARADIS/datos/patches"
TRAIN_SOURCE="cptac_brca"
TISSUE_PATCHING="20x_224px_0px_overlap"
TASK_NAME="TP53_mutation"

# ============================================================================
# PASO 1: Generar embeddings in-sample
# ============================================================================
echo "📝 PASO 1: Generando embeddings ponderados in-sample..."
echo "----------------------------------------------------------------------"

for model in ctranspath uni_v2 virchow_v1; do
    echo "  • $model..."
    python src/test_abmil.py \
        --foundational_model "$model" \
        --work_dir "$WORK_DIR" \
        --train_source "$TRAIN_SOURCE" \
        --tissue_patching "$TISSUE_PATCHING" \
        --task_name "$TASK_NAME" \
        --epochs 100 \
        --use_embeddings
done

echo "✅ Paso 1 completado"
echo ""

# ============================================================================
# PASO 2: Entrenar 4 meta-modelos
# ============================================================================
echo "🧠 PASO 2: Entrenando 4 meta-modelos..."
echo "----------------------------------------------------------------------"
echo ""

echo "  [1/4] Logistic Regression (baseline)..."
python src/ensemble4.py \
    --foundational_models ctranspath uni_v2 virchow_v1 \
    --work_dir "$WORK_DIR" \
    --train_source "$TRAIN_SOURCE" \
    --tissue_patching "$TISSUE_PATCHING" \
    --task_name "$TASK_NAME" \
    --meta_model logreg \
    --meta_features embeddings
echo "      ✓ Completado"
echo ""

echo "  [2/4] MLP (CosineAnnealingLR + early stopping)..."
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
echo "      ✓ Completado"
echo ""

echo "  [3/4] Snapshot MLP (CosineAnnealingWarmRestarts)..."
python src/ensemble4.py \
    --foundational_models ctranspath uni_v2 virchow_v1 \
    --work_dir "$WORK_DIR" \
    --train_source "$TRAIN_SOURCE" \
    --tissue_patching "$TISSUE_PATCHING" \
    --task_name "$TASK_NAME" \
    --meta_model mlp_snapshot \
    --meta_features embeddings \
    --meta_epochs 120 \
    --n_cycles 6 \
    --restart_lr 5e-3
echo "      ✓ Completado"
echo ""

echo "  [4/4] FGE MLP (warm-up + FGE cycles)..."
python src/ensemble4.py \
    --foundational_models ctranspath uni_v2 virchow_v1 \
    --work_dir "$WORK_DIR" \
    --train_source "$TRAIN_SOURCE" \
    --tissue_patching "$TISSUE_PATCHING" \
    --task_name "$TASK_NAME" \
    --meta_model mlp_fge \
    --meta_features embeddings \
    --meta_epochs 100 \
    --fge_n_cycles 6 \
    --fge_cycle_length 3
echo "      ✓ Completado"
echo ""

# ============================================================================
# RESUMEN FINAL
# ============================================================================
echo "================================================================================"
echo "✅ EXPERIMENTO COMPLETADO"
echo "================================================================================"
echo ""
echo "Resultados en:"
echo "  $WORK_DIR/$TRAIN_SOURCE/$TASK_NAME/abmil/"
echo ""
echo "Por meta-modelo:"
echo "  • ensemble4_embeddings/                (LogReg)"
echo "  • ensemble4_mlp_embeddings/            (MLP)"
echo "  • ensemble4_mlp_snapshot_embeddings/   (Snapshot)"
echo "  • ensemble4_mlp_fge_embeddings/        (FGE)"
echo ""
echo "Cada uno contiene:"
echo "  • test_metrics_summary.json            (métricas agregadas)"
echo "  • test_metrics/fold_*/metrics.json     (métricas por fold)"
echo ""
echo "⏱️  Tiempo total: Ver logs en embeddings_4models_${SLURM_JOB_ID}.out"
echo ""
