#!/bin/bash
#SBATCH --job-name=corr_prune_emb
#SBATCH --partition=main
#SBATCH --constraint=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=logs/corr_prune_emb_%j.out
#SBATCH --error=logs/corr_prune_emb_%j.err
#SBATCH --mail-user=rafael.pachon.alvarez@gmail.com
#SBATCH --mail-type=END,FAIL

set -euo pipefail

source $(conda info --base)/etc/profile.d/conda.sh
conda activate ensemble

PROJECT_DIR="/home/JKP6679/Patho-Ensemble"
cd "$PROJECT_DIR"

DATASET="cptac_brca"
TASK="TP53_mutation"
WORK_DIR="/home/JKP6679/Patho-Ensemble/PARADIS/datos/patches"
TISSUE_PATCHING="20x_224px_0px_overlap"
N_FOLDS=50
ABMIL_DIR="$WORK_DIR/$DATASET/$TASK/abmil"

echo "================================================================================"
echo "PODA POR CORRELACION SOBRE EMBEDDINGS DE SLIDE (arXiv 2512.11104)"
echo "Job $SLURM_JOB_ID — $(date)"
echo "================================================================================"
echo ""
echo "Idea del paper 'Information-Driven Fusion of Pathology Foundation Models':"
echo "rankear dimensiones del embedding por separacion de clase, podar cualquier"
echo "dimension cuya correlacion de Pearson con una ya seleccionada de mayor rango"
echo "supere un umbral theta, ANTES de concatenar los embeddings de los modelos base."
echo "ensemble5.py ya probo la fusion temprana NAIVE a nivel de PARCHE (sin poda,"
echo "documentada como fallo en CLAUDE.md, Delta AUC = -0.0056). Esto es distinto:"
echo "poda por correlacion a nivel de EMBEDDING DE SLIDE (512-D/modelo, ya colapsado"
echo "por ABMIL), naive vs podada, nunca antes ejecutada en este repo."
echo ""

# ============================================================================
# FASE 0 — refrescar embeddings que esten STALE respecto a su checkpoint
# ----------------------------------------------------------------------------
# stat confirmo antes de lanzar este job: ctranspath, uni_v2 y conch_v1_5 ya
# tienen embeddings posteriores a sus checkpoints (regenerados 22-ago para
# run_meta_suite_winning_trio.sbatch). virchow_v1 NO: sus embeddings son del
# 18-ago 19:02, ANTERIORES a su checkpoint actual (19-ago 21:48) -> estan
# calculados sobre un modelo que ya no existe en disco. Se regenera aqui.
# Es re-prediccion determinista (test_abmil.py --use_embeddings): no reentrena.
# ============================================================================
echo "### FASE 0 — refrescar embeddings de virchow_v1 (stale: 18-ago, checkpoint 19-ago)"
python src/test_abmil.py \
    --foundational_model virchow_v1 \
    --work_dir "$WORK_DIR" \
    --train_source "$DATASET" \
    --tissue_patching "$TISSUE_PATCHING" \
    --task_name "$TASK" \
    --epochs 100 \
    --use_embeddings

echo ""
echo "### FASE 0b — verificacion de frescura (embeddings posteriores al checkpoint)"
for MODEL in ctranspath uni_v2 virchow_v1 conch_v1_5; do
    CKPT="$ABMIL_DIR/${MODEL}_${TISSUE_PATCHING}/checkpoints/fold_0/model.pt"
    EMB="$ABMIL_DIR/${MODEL}_${TISSUE_PATCHING}/test_outputs_embeddings/fold_0/preds.npy"
    if [ -f "$CKPT" ] && [ -f "$EMB" ]; then
        CKPT_T=$(stat -c %Y "$CKPT")
        EMB_T=$(stat -c %Y "$EMB")
        if [ "$EMB_T" -ge "$CKPT_T" ]; then
            echo "  OK    $MODEL: embeddings ($(date -d @$EMB_T)) >= checkpoint ($(date -d @$CKPT_T))"
        else
            echo "  STALE $MODEL: embeddings ($(date -d @$EMB_T)) < checkpoint ($(date -d @$CKPT_T)) — ABORTANDO"
            exit 1
        fi
    else
        echo "  FALTA $MODEL: checkpoint o embeddings no encontrados — ABORTANDO"
        exit 1
    fi
done

# ============================================================================
# FASE 1 — trio HISTORICO (ctranspath + uni_v2 + virchow_v1)
# ----------------------------------------------------------------------------
# Una sola pasada: baseline naive (emb:logreg, TODAS las 1536-D sin podar) y
# los tres brazos podados (theta=0.3/0.5/0.7), pareados contra ese mismo
# baseline recalculado ahora. prob:logreg se incluye solo como referencia de
# contexto (el 0.7616 conocido, espacio de probabilidad in-sample).
# ============================================================================
echo ""
echo "### FASE 1 — trio HISTORICO (ctranspath + uni_v2 + virchow_v1)"
python src/run_meta_suite.py \
    --work_dir "$WORK_DIR" \
    --train_source "$DATASET" \
    --task_name "$TASK" \
    --tissue_patching "$TISSUE_PATCHING" \
    --foundational_models ctranspath uni_v2 virchow_v1 \
    --n_folds $N_FOLDS \
    --configs prob:logreg emb:logreg emb_corr0.3:logreg emb_corr0.5:logreg emb_corr0.7:logreg \
    --baseline emb:logreg \
    --out results_corr_prune_hist_cptac_brca_TP53_mutation.json

# ============================================================================
# FASE 2 — trio GANADOR (ctranspath + uni_v2 + conch_v1_5)
# ============================================================================
echo ""
echo "### FASE 2 — trio GANADOR (ctranspath + uni_v2 + conch_v1_5)"
python src/run_meta_suite.py \
    --work_dir "$WORK_DIR" \
    --train_source "$DATASET" \
    --task_name "$TASK" \
    --tissue_patching "$TISSUE_PATCHING" \
    --foundational_models ctranspath uni_v2 conch_v1_5 \
    --n_folds $N_FOLDS \
    --configs prob:logreg emb:logreg emb_corr0.3:logreg emb_corr0.5:logreg emb_corr0.7:logreg \
    --baseline emb:logreg \
    --out results_corr_prune_win_cptac_brca_TP53_mutation.json

echo ""
echo "================================================================================"
echo "COMPLETADO — $(date)"
echo "Resultados: results_corr_prune_hist_cptac_brca_TP53_mutation.json"
echo "            results_corr_prune_win_cptac_brca_TP53_mutation.json"
echo "================================================================================"
