#!/bin/bash
# ============================================================================
# LAUNCHER: Feature Engineering Ladder (Idea 1)
# ============================================================================
# Script interactivo para ejecutar la escalera de validación de features extras.
#
# Uso:
#   bash launch_feature_engineering_ladder.sh
#
# O directamente con parámetros:
#   DATASET=cptac_brca TASK=TP53_mutation bash launch_feature_engineering_ladder.sh
# ============================================================================

set -e

PROJECT_DIR="/home/JKP6679/Patho-Ensemble"
cd "$PROJECT_DIR"

source $(conda info --base)/etc/profile.d/conda.sh
conda activate ensemble

echo "============================================================================"
echo "LANZADOR: Feature Engineering Ladder (Idea 1)"
echo "============================================================================"
echo ""

# Obtener dataset y task activos
if [ -z "${DATASET:-}" ] || [ -z "${TASK:-}" ]; then
    echo "📋 Datasets y tareas disponibles:"
    python3 src/experiment_config.py --list-active
    echo ""
    read -rp "🔹 Ingresa DATASET (e.g., cptac_brca): " DATASET
    read -rp "🔹 Ingresa TASK (e.g., TP53_mutation): " TASK
fi

echo ""
echo "Configuración:"
echo "  DATASET: $DATASET"
echo "  TASK: $TASK"
echo ""

# Validar que la configuración existe
if ! python3 src/experiment_config.py --dataset "$DATASET" --task "$TASK" --emit-shell &>/dev/null; then
    echo "❌ Configuración no encontrada: $DATASET / $TASK"
    exit 1
fi

echo "✓ Configuración validada"
echo ""

# Confirmación
read -rp "¿Lanzar escalera (sbatch)? [y/N]: " CONFIRM
if [ "$CONFIRM" != "y" ]; then
    echo "Cancelado."
    exit 0
fi

echo ""
echo "🚀 Lanzando escalera de Feature Engineering..."
echo ""

# Lanzar sbatch
JOB_ID=$(DATASET="$DATASET" TASK="$TASK" sbatch run_feature_engineering_ladder.sbatch | grep -oP '\d+$')

echo "✓ Job lanzado: $JOB_ID"
echo ""
echo "Monitorear progreso:"
echo "  squeue -j $JOB_ID"
echo "  tail -f logs/feature_engineering_ladder_${JOB_ID}.out"
echo ""
echo "Analizar resultados (después de completarse):"
echo "  python src/analyze_feature_engineering_ladder.py \\"
echo "    --dataset $DATASET \\"
echo "    --task $TASK \\"
echo "    --work_dir $PROJECT_DIR/PARADIS/datos/patches"
echo ""
