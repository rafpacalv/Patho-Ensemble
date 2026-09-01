#!/bin/bash
#SBATCH --job-name=snapshot_base_trio
#SBATCH --partition=main
#SBATCH --constraint=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=logs/snapshot_base_trio_%j.out
#SBATCH --error=logs/snapshot_base_trio_%j.err

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
EPOCHS=100

# ============================================================================
# Snapshot ensembles A NIVEL BASE sobre el trío ganador.
#
# Distinto del `mlp_snapshot` de la suite de metaclasificadores: aquí los
# snapshots son de los ABMIL, no del MLP del stacking.
#
# Se generan tags NUEVOS (fge_trio / se_trio) en vez de reutilizar fge_orig /
# se_orig: aquéllos son del 14-15 ago y anteriores a los checkpoints base
# actuales (ctranspath 20-ago, conch_v1_5 21-ago), así que no son citables ni
# comparables. Además conch_v1_5 nunca tuvo snapshots.
#
# Hiperparámetros: los de fge_orig y se_orig en experiments.yaml, que son los
# validados en la ablación previa.
#   fge (Garipov et al., ICLR 2018): lr_1=2e-4  lr_2=2e-5  cycle_length=4
#   se  (Huang   et al., ICLR 2017): lr_1=1e-3  lr_2=1e-6  cycle_length=15
# ============================================================================

echo "================================================================================"
echo "SNAPSHOT ENSEMBLE A NIVEL BASE — trío ganador"
echo "Job $SLURM_JOB_ID — $(date)"
echo "================================================================================"

run_variant () {
    local TAG=$1 SCHED=$2 LR1=$3 LR2=$4 CLEN=$5

    echo ""
    echo "################################################################################"
    echo "# VARIANTE $TAG  (schedule=$SCHED lr_1=$LR1 lr_2=$LR2 cycle_length=$CLEN)"
    echo "################################################################################"

    for MODEL in $MODELS; do
        echo ""
        echo "--- [$TAG] entrenando ciclos: $MODEL --- $(date)"
        python src/train_abmil_fge.py \
            --foundational_model "$MODEL" \
            --work_dir "$WORK_DIR" \
            --train_source "$DATASET" \
            --tissue_patching "$TISSUE_PATCHING" \
            --task_name "$TASK" \
            --epochs $EPOCHS \
            --fold all \
            --n_cycles 6 \
            --cycle_length "$CLEN" \
            --lr_1 "$LR1" \
            --lr_2 "$LR2" \
            --cycle_patience 2 \
            --fge_tag "$TAG" \
            --schedule "$SCHED"

        echo "--- [$TAG] meta-features: $MODEL --- $(date)"
        python src/test_abmil_fge.py \
            --foundational_model "$MODEL" \
            --work_dir "$WORK_DIR" \
            --train_source "$DATASET" \
            --tissue_patching "$TISSUE_PATCHING" \
            --task_name "$TASK" \
            --epochs $EPOCHS \
            --fge_tag "$TAG"
    done
}

run_variant fge_trio fge 2e-4 2e-5 4
run_variant se_trio  se  1e-3 1e-6 15

# ============================================================================
# Comparación pareada, con el baseline recalculado en esta misma pasada.
# ============================================================================
echo ""
echo "### Comparación pareada — snapshots base vs. ABMIL base"
python src/run_meta_suite.py \
    --work_dir "$WORK_DIR" \
    --train_source "$DATASET" \
    --task_name "$TASK" \
    --tissue_patching "$TISSUE_PATCHING" \
    --foundational_models $MODELS \
    --n_folds 50 \
    --baseline prob:logreg \
    --configs \
        prob:logreg prob:mlp_snapshot \
        fge_trio:logreg fge_trio:mlp_snapshot \
        se_trio:logreg  se_trio:mlp_snapshot \
    --out "$PROJECT_DIR/results_snapshot_base_${DATASET}_${TASK}.json"

echo ""
echo "================================================================================"
echo "✅ COMPLETADO — $(date)"
echo "================================================================================"
