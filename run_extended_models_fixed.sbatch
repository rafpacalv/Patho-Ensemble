#!/bin/bash
#SBATCH --job-name=extended_fixed
#SBATCH --partition=main
#SBATCH --gpus=1
#SBATCH --time=06:00:00
#SBATCH --output=/shared/home/JKP6679/Patho-Ensemble/logs/extended_models_fixed_%J.out
#SBATCH --error=/shared/home/JKP6679/Patho-Ensemble/logs/extended_models_fixed_%J.err

set -euo pipefail

# Activate conda
source /home/JKP6679/miniconda3/etc/profile.d/conda.sh
conda activate ensemble

cd /shared/home/JKP6679/Patho-Ensemble

echo "================================================================================"
echo "🚀 EXPERIMENTO EXTENDIDO (FIXED): Deep MLP Snapshot + TabPFN (con/sin Snapshot)"
echo "================================================================================"
echo ""
echo "Configuración:"
echo "  Dataset: cptac_brca / TP53_mutation"
echo "  Modelos base: ctranspath, uni_v2, virchow_v1"
echo "  Meta-modelos (reparados):"
echo "    1. Deep MLP Snapshot (con CosineAnnealingWarmRestarts)"
echo "    2. TabPFN (N-Point Tabular Prior Foundation Network)"
echo "    3. TabPFN Snapshot"
echo "  Meta-features: Embeddings 1536-dim (ya generados)"
echo ""
echo "🧠 Entrenando 3 meta-modelos (REPARADOS)..."
echo "========================================================================"
echo ""

WORK_DIR="/home/JKP6679/Patho-Ensemble/PARADIS/datos/patches"
DATASET="cptac_brca"
TASK="TP53_mutation"
PATCHING="20x_224px_0px_overlap"

# [1/3] Deep MLP Snapshot (CosineAnnealingWarmRestarts)
echo "  [1/3] Deep MLP Snapshot (CosineAnnealingWarmRestarts)..."
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
    --restart_lr 5e-3 2>&1 | tee -a "${WORK_DIR}/${DATASET}/${TASK}/abmil/ensemble4_deep_mlp_snapshot_embeddings/log.txt"

echo ""
echo "      ✓ Completado"
echo ""

# [2/3] TabPFN
echo "  [2/3] TabPFN (Tabular Prior Foundation Network)..."
echo ""

python src/ensemble4.py \
    --foundational_models ctranspath uni_v2 virchow_v1 \
    --meta_model tabpfn \
    --meta_features embeddings \
    --work_dir "$WORK_DIR" \
    --train_source "$DATASET" \
    --tissue_patching "$PATCHING" \
    --task_name "$TASK" 2>&1 | tee -a "${WORK_DIR}/${DATASET}/${TASK}/abmil/ensemble4_tabpfn_embeddings/log.txt"

echo ""
echo "      ✓ Completado"
echo ""

# [3/3] TabPFN Snapshot
echo "  [3/3] TabPFN Snapshot..."
echo ""

python src/ensemble4.py \
    --foundational_models ctranspath uni_v2 virchow_v1 \
    --meta_model tabpfn_snapshot \
    --meta_features embeddings \
    --work_dir "$WORK_DIR" \
    --train_source "$DATASET" \
    --tissue_patching "$PATCHING" \
    --task_name "$TASK" 2>&1 | tee -a "${WORK_DIR}/${DATASET}/${TASK}/abmil/ensemble4_tabpfn_snapshot_embeddings/log.txt"

echo ""
echo "      ✓ Completado"
echo ""

echo "========================================================================"
echo "✅ EXPERIMENTO EXTENDIDO (FIXED) COMPLETADO"
echo "========================================================================"
echo ""
echo "Resultados guardados en:"
echo "  - ensemble4_deep_mlp_snapshot_embeddings/"
echo "  - ensemble4_tabpfn_embeddings/"
echo "  - ensemble4_tabpfn_snapshot_embeddings/"
echo ""
