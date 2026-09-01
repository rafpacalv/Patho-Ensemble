#!/bin/bash
#SBATCH --job-name=meta_suite_trio
#SBATCH --partition=main
#SBATCH --constraint=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=logs/meta_suite_trio_%j.out
#SBATCH --error=logs/meta_suite_trio_%j.err

set -euo pipefail

source $(conda info --base)/etc/profile.d/conda.sh
conda activate ensemble

PROJECT_DIR="/home/JKP6679/Patho-Ensemble"
cd "$PROJECT_DIR"

DATASET="cptac_brca"
TASK="TP53_mutation"
WORK_DIR="/home/JKP6679/Patho-Ensemble/PARADIS/datos/patches"
TISSUE_PATCHING="20x_224px_0px_overlap"
MODELS="ctranspath uni_v2 conch_v1_5"
N_FOLDS=50

echo "================================================================================"
echo "SUITE DE METACLASIFICADORES — trío ganador (ctranspath + uni_v2 + conch_v1_5)"
echo "Job $SLURM_JOB_ID — $(date)"
echo "================================================================================"

# ============================================================================
# FASE 1 — Regenerar embeddings de slide desde los checkpoints VIGENTES
# ----------------------------------------------------------------------------
# Los embeddings que había en disco eran del 18-ago, anteriores al checkpoint de
# ctranspath (20-ago 12:52), y conch_v1_5 no tenía ninguno. La regla operativa
# del proyecto (CLAUDE.md, "Stored ensemble4_* results go stale") exige que las
# features salgan de los checkpoints actuales, así que se regeneran los tres.
# Es re-predicción determinista: no reentrena nada.
# ============================================================================
echo ""
echo "### FASE 1 — extracción de embeddings (512-D por modelo)"
for MODEL in $MODELS; do
    echo "--- $MODEL ---"
    python src/test_abmil.py \
        --foundational_model "$MODEL" \
        --work_dir "$WORK_DIR" \
        --train_source "$DATASET" \
        --tissue_patching "$TISSUE_PATCHING" \
        --task_name "$TASK" \
        --epochs 100 \
        --use_embeddings
done

echo ""
echo "### FASE 1b — verificación de que los embeddings son posteriores al checkpoint"
ABMIL_DIR="$WORK_DIR/$DATASET/$TASK/abmil"
for MODEL in $MODELS; do
    D="$ABMIL_DIR/${MODEL}_${TISSUE_PATCHING}"
    CKPT=$(stat -c %Y "$D/checkpoints/fold_0/model.pt")
    EMB=$(stat -c %Y "${D}_train_eval_embeddings/val_outputs_embeddings/fold_0/preds.npy")
    if [ "$EMB" -le "$CKPT" ]; then
        echo "❌ $MODEL: embeddings ($EMB) NO son posteriores al checkpoint ($CKPT)"
        exit 1
    fi
    echo "✓ $MODEL: embeddings posteriores al checkpoint"
done

# ============================================================================
# FASE 2 — Los 7 brazos, en UNA sola pasada
# ----------------------------------------------------------------------------
# Brazos 1-6 (espacio de probabilidades, 2 col/modelo = 6 features):
#   logreg · mlp · mlp_snapshot · svm · knn · nb
# Brazo 7 (§8.1 del estado del arte — la comparación abierta pendiente):
#   los mismos metaclasificadores sobre el embedding de slide de 512-D
#   (1536 features), con los checkpoints actuales.
# Todo comparte proceso, splits y predicciones base, así que el baseline queda
# recalculado en la misma pasada y los contrastes son pareados.
# ============================================================================
echo ""
echo "### FASE 2 — suite de metaclasificadores"
python src/run_meta_suite.py \
    --work_dir "$WORK_DIR" \
    --train_source "$DATASET" \
    --task_name "$TASK" \
    --tissue_patching "$TISSUE_PATCHING" \
    --foundational_models $MODELS \
    --n_folds $N_FOLDS \
    --baseline prob:logreg \
    --configs \
        prob:logreg prob:mlp prob:mlp_snapshot prob:svm prob:knn prob:nb \
        emb:logreg emb:mlp emb:mlp_snapshot emb:mlp_fge \
    --out "$PROJECT_DIR/results_meta_suite_${DATASET}_${TASK}.json"

echo ""
echo "================================================================================"
echo "✅ COMPLETADO — $(date)"
echo "================================================================================"
