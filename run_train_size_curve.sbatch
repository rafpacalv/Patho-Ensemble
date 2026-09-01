#!/bin/bash
#SBATCH --job-name=size_curve
#SBATCH --partition=main
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=logs/size_curve_%j.out
#SBATCH --error=logs/size_curve_%j.err

# ============================================================================
# ¿Es el tamaño del train lo que decide embeddings vs. probabilidades?
#
# Entre datasets, `emb:logreg` pierde con 75 slides de meta-train (cptac_brca,
# −0.0160) y gana con 164 (cptac_gbm, +0.0230 ✱). Pero los datasets difieren en
# mucho más que en n. Este job convierte esa correlación en una relación causal
# DENTRO de cptac_gbm: recorta las filas de entrenamiento del metaclasificador y
# mide el Δ a cada tamaño, con todo lo demás fijo.
#
# Sólo inferencia + ajustes de logreg sobre predicciones ya en disco: no
# reentrena ningún ABMIL, así que no pide GPU y no compite con nada.
#
# NUNCA ejecutar esto en ctl01: es el nodo de login.
# ============================================================================

set -euo pipefail
source $(conda info --base)/etc/profile.d/conda.sh
conda activate ensemble
cd /home/JKP6679/Patho-Ensemble

DATASET="${DATASET:-cptac_gbm}"
TASK="${TASK:-TP53_mutation}"
REPEATS="${REPEATS:-5}"

echo "Job $SLURM_JOB_ID en $(hostname) — $(date)"
echo "================================================================================"

python src/train_size_curve.py \
    --train_source "$DATASET" \
    --task_name "$TASK" \
    --foundational_models ctranspath uni_v2 conch_v1_5 \
    --n_folds 50 \
    --sizes 40 55 75 95 120 145 0 \
    --repeats "$REPEATS" \
    --meta_model logreg \
    --spaces prob emb \
    --out "results_train_size_curve_${DATASET}_${TASK}.json" \
    --device cpu

echo ""
echo "✅ COMPLETADO — $(date)"
