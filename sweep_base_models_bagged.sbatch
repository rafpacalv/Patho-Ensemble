#!/bin/bash
#SBATCH --job-name=sweep_base_models_bagged
#SBATCH --partition=main
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --output=logs/sweep_base_models_bagged_%j.out
#SBATCH --error=logs/sweep_base_models_bagged_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=rafael.pachon.alvarez@gmail.com

# ============================================================================
# Barrido de subconjuntos de modelos base + selección LOFO con bagging
# ============================================================================
# Misma barrida que sweep_base_models.sbatch (255 subconjuntos, stacking
# logreg), pero además compara la selección honesta actual (argmax único
# sobre la media de los folds != k) contra una variante con bagging
# (bootstrap de esos folds, voto mayoritario — stability selection,
# Meinshausen & Bühlmann 2010 / bagged ensemble selection, Caruana et al.
# ICDM 2006). Ataca directamente la brecha de N14: Spearman(val, test) = 0.75
# sobre los 255 subconjuntos, sin cambiar el criterio de selección, sólo su
# varianza. Sin GPU: sólo reajusta LogisticRegression + numpy sobre
# predicciones ya guardadas.
#
#   sbatch sweep_base_models_bagged.sbatch
#   SWEEP_MODELS="a b c" BAGGED_BOOT=5000 sbatch sweep_base_models_bagged.sbatch
# ============================================================================

set -euo pipefail

source $(conda info --base)/etc/profile.d/conda.sh
conda activate ensemble

PROJECT_DIR="/home/JKP6679/Patho-Ensemble"
cd "$PROJECT_DIR"

DATASET="${DATASET:-cptac_brca}"
TASK="${TASK:-TP53_mutation}"
PATCHING="${PATCHING:-20x_224px_0px_overlap}"
WORK_DIR="${WORK_DIR:-$PROJECT_DIR/PARADIS/datos/patches}"
BAGGED_BOOT="${BAGGED_BOOT:-2000}"

# Mismos 8 modelos y mismo baseline que sweep_base_models.sbatch, para que la
# comparación honesta-vs-bagging sea directa.
SWEEP_MODELS="${SWEEP_MODELS:-ctranspath uni_v2 virchow_v1 conch_v1_5 phikon_v2 uni_v1 hoptimus1 virchow2}"
BASELINE="${BASELINE:-ctranspath uni_v2 conch_v1_5}"

OUT="results_base_model_sweep_bagged_${DATASET}_${TASK}.json"

echo "============================================================================"
echo "Barrido de modelos base con selección LOFO bagged — $DATASET / $TASK"
echo "============================================================================"
echo "Inicio:      $(date)"
echo "modelos:     $SWEEP_MODELS"
echo "baseline:    $BASELINE"
echo "bootstrap:   $BAGGED_BOOT"
echo "salida:      $OUT"
echo "============================================================================"
echo ""

python src/sweep_base_models.py \
    --work_dir "$WORK_DIR" \
    --train_source "$DATASET" \
    --task_name "$TASK" \
    --tissue_patching "$PATCHING" \
    --models $SWEEP_MODELS \
    --baseline $BASELINE \
    --bagged_boot "$BAGGED_BOOT" \
    --out "$OUT"

echo ""
echo "Fin: $(date)"
echo "============================================================================"
