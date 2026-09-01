#!/bin/bash
#SBATCH --job-name=idea3_phase2_eval
#SBATCH --partition=main
#SBATCH --constraint=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=logs/idea3_phase2_eval_%j.out
#SBATCH --error=logs/idea3_phase2_eval_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=rafael.pachon.alvarez@gmail.com

# ============================================================================
# IDEA 3 — FASE 2: EVALUACIÓN CON ESCALERA ADITIVA
# ============================================================================
# Ejecuta escalera de feature engineering que incluye estadísticos de atención.
#
# Escalera:
#   E0: Baseline (6 features crudas)
#   E1: + disagreement (1 columna extra)
#   E2: + attn_entropy_norm + attn_ess_norm (3+3 = 6 columnas extra)
#
# Comparación pareada (50 folds): Δ AUC, IC95%, Holm-Bonferroni
# ============================================================================

set -uo pipefail

source $(conda info --base)/etc/profile.d/conda.sh
conda activate ensemble

PROJECT_DIR="/home/JKP6679/Patho-Ensemble"
cd "$PROJECT_DIR"

# Configuración: cptac_brca / TP53_mutation
DATASET="cptac_brca"
TASK="TP53_mutation"

eval "$(python3 src/experiment_config.py --emit-shell --dataset "$DATASET" --task "$TASK")"

ABMIL_DIR="$WORK_DIR/$DATASET/$TASK/abmil"
LOG_DIR="$PROJECT_DIR/logs/${DATASET}_${TASK}"
mkdir -p "$LOG_DIR"

echo "============================================================================"
echo "IDEA 3 — FASE 2: EVALUACIÓN CON ESCALERA ADITIVA (FEATURES DE ATENCIÓN)"
echo "============================================================================"
echo "Fecha: $(date)"
echo "Dataset: $DATASET, Task: $TASK"
echo "Modelos base: $MODELS"
echo "Patching: $TISSUE_PATCHING"
echo "============================================================================"
echo ""

# ---------------------------------------------------------------------------
# Función auxiliar: ejecutar un escalón
# ---------------------------------------------------------------------------
run_escalon() {
    local STEP_NAME=$1
    local EXTRA_FEATURES=$2
    local RESULTS_SUFFIX=$3

    echo "============================================================================"
    echo "ESCALÓN: $STEP_NAME"
    if [ -z "$EXTRA_FEATURES" ]; then
        echo "  Extra features: (none — baseline)"
    else
        echo "  Extra features: $EXTRA_FEATURES"
    fi
    echo "  Results subdir suffix: $RESULTS_SUFFIX"
    echo "  Inicio: $(date)"
    echo "============================================================================"

    # Construir el comando ensemble4.py
    CMD="python src/ensemble4.py"
    CMD="$CMD --foundational_models $MODELS"
    CMD="$CMD --work_dir $WORK_DIR"
    CMD="$CMD --train_source $DATASET"
    CMD="$CMD --tissue_patching $TISSUE_PATCHING"
    CMD="$CMD --task_name $TASK"
    CMD="$CMD --results_subdir_suffix $RESULTS_SUFFIX"
    CMD="$CMD --meta_model logreg"

    if [ -n "$EXTRA_FEATURES" ]; then
        CMD="$CMD --extra_features $EXTRA_FEATURES"
    fi

    # Ejecutar
    if eval "$CMD"; then
        echo "✓ $STEP_NAME completado exitosamente"
        echo ""
    else
        echo "❌ $STEP_NAME falló con código $?"
        echo ""
        return 1
    fi
}

# ---------------------------------------------------------------------------
# Ejecutar los 3 escalones
# ---------------------------------------------------------------------------

run_escalon "E0 (Baseline — sin features extras)" \
            "" \
            "_idea3_e0"

run_escalon "E1 (+ disagreement)" \
            "disagreement" \
            "_idea3_e1"

run_escalon "E2 (+ disagreement + margin)" \
            "disagreement margin_avg" \
            "_idea3_e2"

# ---------------------------------------------------------------------------
# Análisis pareado: comparación con Holm-Bonferroni
# ---------------------------------------------------------------------------
echo ""
echo "============================================================================"
echo "ANÁLISIS PAREADO CON HOLM-BONFERRONI"
echo "============================================================================"
python src/analyze_feature_engineering_ladder_v2.py \
    --dataset "$DATASET" \
    --task "$TASK" \
    --work_dir "$WORK_DIR" \
    --subdirs "E0 (Baseline)=ensemble4_idea3_e0" \
              "E1 (+disagreement)=ensemble4_idea3_e1" \
              "E2 (+disagreement+margin)=ensemble4_idea3_e2" \
    --output "results_ladder_idea3_${DATASET}_${TASK}.json"

# ---------------------------------------------------------------------------
# Resumen final
# ---------------------------------------------------------------------------
echo ""
echo "============================================================================"
echo "FASE 2 COMPLETADA"
echo "Fin: $(date)"
echo "============================================================================"
echo ""
echo "Resultados disponibles en:"
echo "  E0: $ABMIL_DIR/ensemble4_idea3_e0/"
echo "  E1: $ABMIL_DIR/ensemble4_idea3_e1/"
echo "  E2: $ABMIL_DIR/ensemble4_idea3_e2/"
echo ""
echo "Comparación pareada guardada en: results_ladder_${DATASET}_${TASK}.json"
echo ""
