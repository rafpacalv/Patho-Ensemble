#!/bin/bash
#SBATCH --job-name=feature_engineering_ladder
#SBATCH --partition=main
#SBATCH --constraint=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=logs/feature_engineering_ladder_%j.out
#SBATCH --error=logs/feature_engineering_ladder_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=rafael.pachon.alvarez@gmail.com

# ============================================================================
# ESCALERA DE VALIDACIÓN: IDEA 1 (FEATURE ENGINEERING)
# ============================================================================
# Ejecuta 4 escalones progresivos de agregación de features extras:
#
#   E0: baseline (sin features extras)
#   E1: + disagreement (Jensen-gap)
#   E2: + disagreement + margin_avg
#   E3: + disagreement + margin_avg + entropy_per_model
#
# Cada escalón corre con 50 folds, LogisticRegression, bootstrap 1000.
# Las métricas se comparan pareadas (Holm-Bonferroni corrected).
#
# Uso:
#   sbatch run_feature_engineering_ladder.sbatch
#
# Nota: DATASET y TASK están hardcodeados abajo (línea ~43). Modificar si necesario.
#
# Requiere: train_base_models.sbatch completado (bases estándar en _train_eval/val_outputs)
# Duración estimada: 4-5 h (4 × 50 folds × ~55 min cada uno en GPU)
# ============================================================================

set -uo pipefail   # sin -e: un escalón que falle no debe abortar el resto

source $(conda info --base)/etc/profile.d/conda.sh
conda activate ensemble

PROJECT_DIR="/home/JKP6679/Patho-Ensemble"
cd "$PROJECT_DIR"

# Configuración: incluir dataset y task aquí
DATASET="cptac_brca"
TASK="TP53_mutation"

eval "$(python3 src/experiment_config.py --emit-shell --dataset "$DATASET" --task "$TASK")"

LOG_DIR="$PROJECT_DIR/logs/${DATASET}_${TASK}"
mkdir -p "$LOG_DIR"

ABMIL_DIR="$WORK_DIR/$DATASET/$TASK/abmil"

# ---------------------------------------------------------------------------
# Comprobaciones previas
# ---------------------------------------------------------------------------
echo "============================================================================"
echo "ESCALERA DE FEATURE ENGINEERING — $DATASET / $TASK"
echo "============================================================================"
echo "Fecha: $(date)"
echo "Modelos base: $MODELS"
echo "Patching: $TISSUE_PATCHING"
echo "============================================================================"
echo ""

BASES_OK=1
for MODEL in $MODELS; do
    B="$ABMIL_DIR/${MODEL}_${TISSUE_PATCHING}"
    if [ ! -d "${B}_train_eval/val_outputs/fold_0" ]; then
        echo "❌ falta ${MODEL}: _train_eval/val_outputs/fold_0"
        BASES_OK=0
    fi
done

if [ "$BASES_OK" -eq 0 ]; then
    echo ""
    echo "❌ Faltan las bases estándar (train_eval/val_outputs). Ejecuta primero:"
    echo "   DATASET=$DATASET TASK=$TASK sbatch train_base_models.sbatch"
    exit 1
fi

echo "✓ Bases estándar disponibles"
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
# Ejecutar los 4 escalones
# ---------------------------------------------------------------------------

run_escalon "E0 (Baseline — sin features extras)" \
            "" \
            "_fea_e0"

run_escalon "E1 (+ disagreement)" \
            "disagreement" \
            "_fea_e1"

run_escalon "E2 (+ disagreement + margin_avg)" \
            "disagreement margin_avg" \
            "_fea_e2"

run_escalon "E3 (+ disagreement + margin_avg + entropy_per_model)" \
            "disagreement margin_avg entropy_per_model" \
            "_fea_e3"

# ---------------------------------------------------------------------------
# Resumen final
# ---------------------------------------------------------------------------
echo "============================================================================"
echo "ESCALERA COMPLETADA"
echo "Fin: $(date)"
echo "============================================================================"
echo ""
echo "Resultados disponibles en:"
echo "  E0: $ABMIL_DIR/ensemble4_fea_e0/"
echo "  E1: $ABMIL_DIR/ensemble4_fea_e1/"
echo "  E2: $ABMIL_DIR/ensemble4_fea_e2/"
echo "  E3: $ABMIL_DIR/ensemble4_fea_e3/"
echo ""
echo "Para analizar métricas pareadas, ejecuta:"
echo "  python src/analyze_feature_engineering_ladder_v2.py \\"
echo "    --dataset $DATASET \\"
echo "    --task $TASK \\"
echo "    --work_dir $WORK_DIR \\"
echo "    --subdirs 'E0 (Baseline)=ensemble4_fea_e0' \\"
echo "              'E1 (+disagreement)=ensemble4_fea_e1' \\"
echo "              'E2 (+disagreement+margin)=ensemble4_fea_e2' \\"
echo "              'E3 (+disagreement+margin+entropy)=ensemble4_fea_e3'"
echo ""
