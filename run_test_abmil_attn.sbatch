#!/bin/bash
#SBATCH --job-name=test_abmil_attn
#SBATCH --partition=main
#SBATCH --constraint=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=06:00:00
#SBATCH --output=logs/test_abmil_attn_%j.out
#SBATCH --error=logs/test_abmil_attn_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=rafael.pachon.alvarez@gmail.com

set -uo pipefail

source $(conda info --base)/etc/profile.d/conda.sh
conda activate ensemble

PROJECT_DIR="/home/JKP6679/Patho-Ensemble"
cd "$PROJECT_DIR"

# Configuración
DATASET="cptac_brca"
TASK="TP53_mutation"
WORK_DIR="/home/JKP6679/Patho-Ensemble/PARADIS/datos/patches"
TISSUE_PATCHING="20x_224px_0px_overlap"
MODELS="ctranspath uni_v2 virchow_v1"
EPOCHS=100

echo "============================================================================"
echo "Generando attn_stats correctos para train_eval (meta-train split)"
echo "============================================================================"
echo "Fecha: $(date)"
echo ""

for model in $MODELS; do
    echo "Procesando modelo: $model"
    python src/test_abmil.py \
        --foundational_model "$model" \
        --work_dir "$WORK_DIR" \
        --train_source "$DATASET" \
        --tissue_patching "$TISSUE_PATCHING" \
        --task_name "$TASK" \
        --epochs "$EPOCHS"
    echo ""
done

echo "============================================================================"
echo "Verificación de archivos generados:"
echo "============================================================================"
ls -lh "$WORK_DIR/$DATASET/$TASK/abmil/ctranspath_${TISSUE_PATCHING}_train_eval/val_outputs/fold_0/"

echo ""
echo "✓ Completado: $(date)"
