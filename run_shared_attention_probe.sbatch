#!/bin/bash
#SBATCH --job-name=attn_probe
#SBATCH --partition=main
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=logs/attn_probe_%j.out
#SBATCH --error=logs/attn_probe_%j.err

# ============================================================================
# ¿Merece la pena compartir el vector de atención entre modelos base?
#
# Sólo se puede entre modelos que compartan rejilla de parches. Verificado
# sobre las 112 slides de cptac_brca: ctranspath y conch_v1_5 tienen `coords`
# byte-idénticos, y uni_v2 NO (5487 parches donde los otros tienen 4101), así
# que la sonda es del PAR, no del trío.
#
# Es sólo inferencia sobre los checkpoints ya entrenados: no reentrena nada,
# por eso no pide GPU y no compite con las campañas de los otros datasets.
#
# Cuatro regímenes:
#   own    cada modelo con su atención        (statu quo)
#   avg    media de las atenciones            (la propuesta)
#   conch  la atención del mejor modelo       (¿es promediar o es conch?)
#   mean   atención uniforme = mean pooling   (CONTROL NEGATIVO)
#
# `mean` es el que más informa y el que falta en todos los informes: si empata
# con `own`, la atención no aporta nada aquí y la pregunta de cómo combinarla
# es discutible.
# ============================================================================

set -euo pipefail

source $(conda info --base)/etc/profile.d/conda.sh
conda activate ensemble

PROJECT_DIR="/home/JKP6679/Patho-Ensemble"
cd "$PROJECT_DIR"

DATASET="${DATASET:-cptac_brca}"
TASK="${TASK:-TP53_mutation}"
PAIR="${PAIR:-ctranspath conch_v1_5}"
ATTN_MODEL="${ATTN_MODEL:-conch_v1_5}"
N_FOLDS="${N_FOLDS:-50}"

echo "================================================================================"
echo "SONDA DE ATENCIÓN COMPARTIDA — $DATASET / $TASK"
echo "Job $SLURM_JOB_ID en $(hostname) — $(date)"
echo "================================================================================"

# ---- Fase 1: generar los cuatro regímenes de embeddings --------------------
python src/shared_attention_probe.py \
    --work_dir "$PROJECT_DIR/PARADIS/datos/patches" \
    --train_source "$DATASET" \
    --task_name "$TASK" \
    --models $PAIR \
    --attn_model "$ATTN_MODEL" \
    --n_folds "$N_FOLDS" \
    --device cpu

# ---- Fase 2: análisis pareado, baseline recalculado en la misma pasada -----
# `prob:logreg` va incluido como referencia absoluta del par: sin él no se
# sabría si un régimen "mejor" sigue estando por debajo de las probabilidades.
echo ""
echo "### Análisis"
python src/run_meta_suite.py \
    --work_dir "$PROJECT_DIR/PARADIS/datos/patches" \
    --train_source "$DATASET" \
    --task_name "$TASK" \
    --foundational_models $PAIR \
    --n_folds "$N_FOLDS" \
    --baseline attn_own:logreg \
    --configs prob:logreg attn_own:logreg attn_avg:logreg attn_conch:logreg attn_mean:logreg \
    --out "$PROJECT_DIR/results_shared_attention_${DATASET}_${TASK}.json" \
    --device cpu

echo ""
echo "================================================================================"
echo "✅ COMPLETADO — $(date)"
echo "================================================================================"
