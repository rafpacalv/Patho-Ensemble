#!/bin/bash
#SBATCH --job-name=idea4_early_fusion
#SBATCH --partition=main
#SBATCH --constraint=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=logs/idea4_early_fusion_%j.out
#SBATCH --error=logs/idea4_early_fusion_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=rafael.pachon.alvarez@gmail.com

set -uo pipefail

source $(conda info --base)/etc/profile.d/conda.sh
conda activate ensemble

PROJECT_DIR="/home/JKP6679/Patho-Ensemble"
cd "$PROJECT_DIR"

DATASET="cptac_brca"
TASK="TP53_mutation"
WORK_DIR="/home/JKP6679/Patho-Ensemble/PARADIS/datos/patches"
TISSUE_PATCHING="20x_224px_0px_overlap"
EPOCHS=100

echo "============================================================================"
echo "IDEA 4 — ESCALA COMPLETA: EARLY FUSION A NIVEL DE PARCHE (50 FOLDS)"
echo "============================================================================"
echo "Configuración:"
echo "  Dataset: $DATASET / Task: $TASK"
echo "  Modelos fusionados: ctranspath + virchow_v1"
echo "  Modelo complementario: uni_v2"
echo "  Folds: 0-49 (50 folds completos — validación definitiva)"
echo "  Concatenación: por slide, ctranspath[768] ⊕ virchow_v1[2560] → [3328]"
echo "  Piloto 5-fold: Δ AUC = +0.0329 (4/5 folds mejores)"
echo "============================================================================"
echo "Fecha: $(date)"
echo ""

python src/ensemble5.py \
    --work_dir "$WORK_DIR" \
    --train_source "$DATASET" \
    --tissue_patching "$TISSUE_PATCHING" \
    --task_name "$TASK" \
    --fuse_models ctranspath virchow_v1 \
    --other_model uni_v2 \
    --fold_start 0 \
    --fold_end 49 \
    --epochs "$EPOCHS" \
    --meta_model logreg

echo ""
echo "============================================================================"
echo "Piloto completado: $(date)"
echo "============================================================================"
echo ""
echo "Resultados disponibles en:"
echo "  Fused ABMIL: $WORK_DIR/$DATASET/$TASK/abmil/ctranspath_virchow_v1_fused_${TISSUE_PATCHING}/"
echo "  Stacking: $WORK_DIR/$DATASET/$TASK/abmil/ctranspath_virchow_v1_fused_${TISSUE_PATCHING}_ctranspath_virchow_v1_fused_uni_v2_stacked/"
echo ""
