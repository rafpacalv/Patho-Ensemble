#!/bin/bash
#SBATCH --job-name=extended_fixed_v7
#SBATCH --partition=main
#SBATCH --gpus=1
#SBATCH --time=08:00:00
#SBATCH --output=/shared/home/JKP6679/Patho-Ensemble/logs/extended_fixed_v7_%J.out
#SBATCH --error=/shared/home/JKP6679/Patho-Ensemble/logs/extended_fixed_v7_%J.err

set -euo pipefail

# Activate conda
source /home/JKP6679/miniconda3/etc/profile.d/conda.sh
conda activate ensemble

cd /shared/home/JKP6679/Patho-Ensemble

echo "================================================================================"
echo "🚀 EXPERIMENTO EXTENDIDO (FIXED v7): TabPFNClassifier minimal initialization"
echo "================================================================================"
echo ""

WORK_DIR="/home/JKP6679/Patho-Ensemble/PARADIS/datos/patches"
DATASET="cptac_brca"
TASK="TP53_mutation"
PATCHING="20x_224px_0px_overlap"

# [1/3] Deep MLP Snapshot
echo "=========================================================================="
echo "  [1/3] Deep MLP Snapshot (CosineAnnealingWarmRestarts)"
echo "=========================================================================="
echo ""

python src/ensemble4.py \
    --foundational_models ctranspath uni_v2 virchow_v1 \
    --meta_model deep_mlp_snapshot \
    --meta_features embeddings \
    --work_dir "$WORK_DIR" \
    --train_source "$DATASET" \
    --tissue_patching "$PATCHING" \
    --task_name "$TASK" \
    --meta_epochs 120 \
    --meta_lr 1e-3 \
    --meta_wd 1e-4 \
    --meta_patience 8 \
    --n_cycles 6 \
    --restart_lr 5e-3

echo ""
echo "✅ Deep MLP Snapshot completado"
echo ""

# [2/3] TabPFN
echo "=========================================================================="
echo "  [2/3] TabPFN (Tabular Prior Foundation Network)"
echo "=========================================================================="
echo ""

python src/ensemble4.py \
    --foundational_models ctranspath uni_v2 virchow_v1 \
    --meta_model tabpfn \
    --meta_features embeddings \
    --work_dir "$WORK_DIR" \
    --train_source "$DATASET" \
    --tissue_patching "$PATCHING" \
    --task_name "$TASK"

echo ""
echo "✅ TabPFN completado"
echo ""

# [3/3] TabPFN Snapshot
echo "=========================================================================="
echo "  [3/3] TabPFN Snapshot"
echo "=========================================================================="
echo ""

python src/ensemble4.py \
    --foundational_models ctranspath uni_v2 virchow_v1 \
    --meta_model tabpfn_snapshot \
    --meta_features embeddings \
    --work_dir "$WORK_DIR" \
    --train_source "$DATASET" \
    --tissue_patching "$PATCHING" \
    --task_name "$TASK"

echo ""
echo "✅ TabPFN Snapshot completado"
echo ""

echo "================================================================================"
echo "✅ EXPERIMENTO EXTENDIDO (FIXED v7) COMPLETADO EXITOSAMENTE"
echo "================================================================================"
echo ""
echo "Resultados guardados en:"
echo "  - ensemble4_deep_mlp_snapshot_embeddings/"
echo "  - ensemble4_tabpfn_embeddings/"
echo "  - ensemble4_tabpfn_snapshot_embeddings/"
echo ""
