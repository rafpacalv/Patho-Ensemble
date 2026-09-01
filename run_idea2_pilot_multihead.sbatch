#!/bin/bash
#SBATCH --job-name=idea2_pilot_mh
#SBATCH --partition=main
#SBATCH --constraint=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=logs/idea2_pilot_mh_%j.out
#SBATCH --error=logs/idea2_pilot_mh_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=rafael.pachon.alvarez@gmail.com

set -uo pipefail

source $(conda info --base)/etc/profile.d/conda.sh
conda activate ensemble

PROJECT_DIR="/home/JKP6679/Patho-Ensemble"
cd "$PROJECT_DIR"

DATASET="cptac_brca"
TASK="TP53_mutation"
WORK_DIR="/home/JKP6679/Patho-Ensemble/PARADIS/datos/patches"
TISSUE_PATCHING="20x_224px_0px_overlap"
MODEL="ctranspath"
EPOCHS=100

echo "============================================================================"
echo "IDEA 2 — PILOTO: MULTI-HEAD GATED ATTENTION EN ABMIL"
echo "============================================================================"
echo "Configuración:"
echo "  Dataset: $DATASET / Task: $TASK"
echo "  Modelo: $MODEL"
echo "  Folds: 0-4 (piloto de 5 folds para validar)"
echo "  n_attention_heads: 4"
echo "  Combinación: mean-pool"
echo "============================================================================"
echo ""

# Entrenar modelo base (single-head) para comparación
echo "=== BASELINE: Single-head Gated Attention ==="
for fold in 0 1 2 3 4; do
    python src/train_abmil.py \
        --foundational_model "$MODEL" \
        --work_dir "$WORK_DIR" \
        --train_source "$DATASET" \
        --tissue_patching "$TISSUE_PATCHING" \
        --task_name "$TASK" \
        --epochs "$EPOCHS" \
        --fold "$fold"
done
echo "✓ Baseline completado"

echo ""
echo "=== MULTI-HEAD: 4-head Gated Attention ==="
for fold in 0 1 2 3 4; do
    python src/train_abmil.py \
        --foundational_model "$MODEL" \
        --work_dir "$WORK_DIR" \
        --train_source "$DATASET" \
        --tissue_patching "$TISSUE_PATCHING" \
        --task_name "$TASK" \
        --epochs "$EPOCHS" \
        --fold "$fold" \
        --multi_head_attention \
        --n_attention_heads 4 \
        --arch_tag "multihead_4"
done
echo "✓ MultiHead completado"

echo ""
echo "============================================================================"
echo "Piloto completado: $(date)"
echo "============================================================================"
echo ""
echo "Comparar resultados:"
echo "  Baseline: $WORK_DIR/$DATASET/$TASK/abmil/${MODEL}_${TISSUE_PATCHING}/test_metrics/"
echo "  MultiHead: $WORK_DIR/$DATASET/$TASK/abmil/${MODEL}_multihead_4_${TISSUE_PATCHING}/test_metrics/"
echo ""
