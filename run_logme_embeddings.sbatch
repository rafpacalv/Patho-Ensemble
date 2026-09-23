#!/bin/bash
#SBATCH --job-name=logme_embeddings
#SBATCH --partition=main
#SBATCH --constraint=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=06:00:00
#SBATCH --output=logs/logme_embeddings_%j.out
#SBATCH --error=logs/logme_embeddings_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=rafael.pachon.alvarez@gmail.com

# ============================================================================
# LogME como criterio de selección de modelos base — sin entrenar el ABMIL
# ============================================================================
# Paso 1 (GPU): extrae embeddings de atención blanda por slide para los 8
# modelos de cptac_brca/TP53_mutation (test_abmil.py --use_embeddings, no
# existían para este dataset — solo se habían generado para cptac_gbm en P2).
# Paso 2 (CPU): calcula LogME sobre esos embeddings (src/logme_criterion.py)
# y lo contrasta contra el AUC real, sin mirar test en la selección.
#
#   sbatch run_logme_embeddings.sbatch
# ============================================================================

set -euo pipefail

count_files() {
    local n
    n=$(ls -1d "$@" 2>/dev/null | wc -l) || n=0
    echo "$n"
}

source $(conda info --base)/etc/profile.d/conda.sh
conda activate ensemble

PROJECT_DIR="/home/JKP6679/Patho-Ensemble"
cd "$PROJECT_DIR"

DATASET="cptac_brca"
TASK="TP53_mutation"
PATCHING="20x_224px_0px_overlap"
WORK_DIR="$PROJECT_DIR/PARADIS/datos/patches"
MODELS="ctranspath uni_v2 virchow_v1 conch_v1_5 phikon_v2 uni_v1 hoptimus1 virchow2"
BASELINE="ctranspath uni_v2 conch_v1_5"
TOPK="${TOPK:-3}"
ABMIL_DIR="$WORK_DIR/$DATASET/$TASK/abmil"
EXPECTED_FOLDS=50

echo "============================================================================"
echo "Paso 1: embeddings de atención blanda (GPU) — $DATASET / $TASK"
echo "============================================================================"
echo "Inicio:  $(date)"
echo "modelos: $MODELS"
echo "============================================================================"
echo ""

FAILED=()
for MODEL in $MODELS; do
    EMB_DIR="${ABMIL_DIR}/${MODEL}_${PATCHING}_train_eval_embeddings/val_outputs_embeddings"
    N=$(count_files "$EMB_DIR"/fold_*)
    if [ "$N" -ge "$EXPECTED_FOLDS" ]; then
        echo "✓ $MODEL: embeddings ya existen ($N/$EXPECTED_FOLDS). Se salta."
        continue
    fi

    echo "[$(date)] Extrayendo embeddings para $MODEL..."
    if python src/test_abmil.py \
        --foundational_model "$MODEL" \
        --work_dir "$WORK_DIR" \
        --train_source "$DATASET" \
        --tissue_patching "$PATCHING" \
        --task_name "$TASK" \
        --use_embeddings \
        2>&1 | tee -a "logs/logme_embeddings_${MODEL}.log"; then

        N=$(count_files "$EMB_DIR"/fold_*)
        echo "[$(date)] ✓ $MODEL — $N/$EXPECTED_FOLDS folds"
        [ "$N" -ge "$EXPECTED_FOLDS" ] || FAILED+=("$MODEL (embeddings: $N/$EXPECTED_FOLDS)")
    else
        echo "[$(date)] ❌ $MODEL: fallo extrayendo embeddings"
        FAILED+=("$MODEL (embeddings)")
    fi
done

if [ ${#FAILED[@]} -ne 0 ]; then
    echo ""
    echo "⚠ Incidencias en la extracción de embeddings:"
    printf '  ❌ %s\n' "${FAILED[@]}"
    echo "Abortando antes del cálculo de LogME."
    exit 1
fi

echo ""
echo "============================================================================"
echo "Paso 2: criterio LogME (CPU) — top-$TOPK, selección honesta LOFO"
echo "============================================================================"

OUT="results_logme_criterion_${DATASET}_${TASK}.json"

python src/logme_criterion.py \
    --work_dir "$WORK_DIR" \
    --train_source "$DATASET" \
    --task_name "$TASK" \
    --tissue_patching "$PATCHING" \
    --models $MODELS \
    --baseline $BASELINE \
    --topk "$TOPK" \
    --out "$OUT"

echo ""
echo "Fin: $(date)"
echo "============================================================================"
