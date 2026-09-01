#!/bin/bash
#SBATCH --job-name=extended_models
#SBATCH --output=logs/extended_models_%j.out
#SBATCH --error=logs/extended_models_%j.err
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
echo "🚀 EXPERIMENTO EXTENDIDO: Deep MLP + TabPFN (con/sin Snapshot)"
echo "================================================================================"
echo ""
echo "Configuración:"
echo "  Dataset: cptac_brca / TP53_mutation"
echo "  Modelos base: ctranspath, uni_v2, virchow_v1"
echo "  Meta-modelos NUEVOS:"
echo "    1. Deep MLP (3 capas, 256-128-64)"
echo "    2. Deep MLP Snapshot (con CosineAnnealingWarmRestarts)"
echo "    3. TabPFN (N-Point Tabular Prior Foundation Network)"
echo "    4. TabPFN Snapshot"
echo "  Meta-features: Embeddings 1536-dim (ya generados)"
echo ""

WORK_DIR="/home/JKP6679/Patho-Ensemble/PARADIS/datos/patches"
TRAIN_SOURCE="cptac_brca"
TISSUE_PATCHING="20x_224px_0px_overlap"
TASK_NAME="TP53_mutation"

echo "🧠 Entrenando 4 meta-modelos NUEVOS..."
echo "========================================================================"
echo ""

echo "  [1/4] Deep MLP (3 capas, 256→128→64)..."
python src/ensemble4.py \
    --foundational_models ctranspath uni_v2 virchow_v1 \
    --work_dir "$WORK_DIR" \
    --train_source "$TRAIN_SOURCE" \
    --tissue_patching "$TISSUE_PATCHING" \
    --task_name "$TASK_NAME" \
    --meta_model deep_mlp \
    --meta_features embeddings \
    --meta_epochs 100 \
    --meta_lr 1e-3 \
    --meta_wd 1e-4 \
    --meta_patience 8
echo "      ✓ Completado"
echo ""

echo "  [2/4] Deep MLP Snapshot (CosineAnnealingWarmRestarts)..."
python src/ensemble4.py \
    --foundational_models ctranspath uni_v2 virchow_v1 \
    --work_dir "$WORK_DIR" \
    --train_source "$TRAIN_SOURCE" \
    --tissue_patching "$TISSUE_PATCHING" \
    --task_name "$TASK_NAME" \
    --meta_model deep_mlp_snapshot \
    --meta_features embeddings \
    --meta_epochs 120 \
    --n_cycles 6 \
    --restart_lr 5e-3
echo "      ✓ Completado"
echo ""

echo "  [3/4] TabPFN (sin ensemble de ciclos)..."
python src/ensemble4.py \
    --foundational_models ctranspath uni_v2 virchow_v1 \
    --work_dir "$WORK_DIR" \
    --train_source "$TRAIN_SOURCE" \
    --tissue_patching "$TISSUE_PATCHING" \
    --task_name "$TASK_NAME" \
    --meta_model tabpfn \
    --meta_features embeddings
echo "      ✓ Completado"
echo ""

echo "  [4/4] TabPFN Snapshot (con CosineAnnealingWarmRestarts)..."
python src/ensemble4.py \
    --foundational_models ctranspath uni_v2 virchow_v1 \
    --work_dir "$WORK_DIR" \
    --train_source "$TRAIN_SOURCE" \
    --tissue_patching "$TISSUE_PATCHING" \
    --task_name "$TASK_NAME" \
    --meta_model tabpfn_snapshot \
    --meta_features embeddings \
    --meta_epochs 120 \
    --n_cycles 6 \
    --restart_lr 5e-3
echo "      ✓ Completado"
echo ""

echo "================================================================================"
echo "✅ EXPERIMENTO EXTENDIDO COMPLETADO"
echo "================================================================================"
echo ""
echo "Resultados en:"
echo "  $WORK_DIR/$TRAIN_SOURCE/$TASK_NAME/abmil/"
echo ""
echo "Nuevos directorios (4 modelos):"
echo "  • ensemble4_deep_mlp_embeddings/            (Deep MLP)"
echo "  • ensemble4_deep_mlp_snapshot_embeddings/   (Deep MLP + Snapshot)"
echo "  • ensemble4_tabpfn_embeddings/              (TabPFN)"
echo "  • ensemble4_tabpfn_snapshot_embeddings/     (TabPFN + Snapshot)"
echo ""
echo "⏱️  Tiempo total: Ver logs en extended_models_${SLURM_JOB_ID}.out"
echo ""
