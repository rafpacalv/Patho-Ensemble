#!/bin/bash
#SBATCH --job-name=sweep_base_models
#SBATCH --partition=main
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --output=logs/sweep_base_models_%j.out
#SBATCH --error=logs/sweep_base_models_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=rafael.pachon.alvarez@gmail.com

# ============================================================================
# Barrido de subconjuntos de modelos base (stacking logreg)
# ============================================================================
# Responde a qué modelos fundación deben entrar en el ensemble. Sin GPU: sólo
# reajusta LogisticRegression sobre predicciones ya guardadas.
#
#   sbatch sweep_base_models.sbatch
#   SWEEP_MODELS="a b c" sbatch sweep_base_models.sbatch
#
# Encadenado tras un entrenamiento:
#   sbatch --dependency=afterok:<jobid> sweep_base_models.sbatch
#
# El script aborta si dos modelos no coinciden fila a fila — es la comprobación
# que evita comparar campañas evaluadas sobre poblaciones distintas.
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

# Los 8 modelos base de cptac_brca. Los 4 primeros son la campaña ago-2026;
# los 4 últimos se reentrenaron el 21-ago para poder compararlos (antes eran
# runs de patho_bench a nivel paciente, con 103 filas en vez de 112).
SWEEP_MODELS="${SWEEP_MODELS:-ctranspath uni_v2 virchow_v1 conch_v1_5 phikon_v2 uni_v1 hoptimus1 virchow2}"
BASELINE="${BASELINE:-ctranspath uni_v2 virchow_v1}"

OUT="results_base_model_sweep_${DATASET}_${TASK}.json"

echo "============================================================================"
echo "Barrido de modelos base — $DATASET / $TASK"
echo "============================================================================"
echo "Inicio:   $(date)"
echo "modelos:  $SWEEP_MODELS"
echo "baseline: $BASELINE"
echo "salida:   $OUT"
echo "============================================================================"
echo ""

python src/sweep_base_models.py \
    --work_dir "$WORK_DIR" \
    --train_source "$DATASET" \
    --task_name "$TASK" \
    --tissue_patching "$PATCHING" \
    --models $SWEEP_MODELS \
    --baseline $BASELINE \
    --out "$OUT"

echo ""
echo "Fin: $(date)"
echo "============================================================================"
