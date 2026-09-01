#!/bin/bash
#SBATCH --job-name=train_base_models_fge
#SBATCH --partition=main
#SBATCH --constraint=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=16:00:00
#SBATCH --output=logs/train_base_models_fge_%j.out
#SBATCH --error=logs/train_base_models_fge_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=rafael.pachon.alvarez@gmail.com

# ============================================================================
# Training 3 Base ABMIL Models with Fast Geometric Ensembles (FGE)
# ============================================================================
# This script trains all 3 foundational models with FGE cycles for all 50 folds
# (based on: train_abmil.py + abmil_fge.py, train_abmil_fge.py, test_abmil_fge.py)
# Required for: Experiments 2 & 3 (base_source=fge)
# Estimated runtime: 14-16 hours (50% more than standard ABMIL due to FGE cycles)
# ============================================================================

set -e

source $(conda info --base)/etc/profile.d/conda.sh
conda activate ensemble

PROJECT_DIR="/shared/home/JKP6679/Patho-Ensemble"
WORK_DIR="/shared/home/PARADIS/datos"
TRAIN_SOURCE="cptac_brca"
TISSUE_PATCHING="20x_224px_0px_overlap"
TASK_NAME="TP53_mutation"
EPOCHS=100

# FGE hyperparameters
N_CYCLES=6
CYCLE_LENGTH=4
LR_1=2e-4
LR_2=2e-5
CYCLE_PATIENCE=2

MODELS=("ctranspath" "uni_v2" "virchow_v1")
declare -A LATENT_DIMS
LATENT_DIMS[ctranspath]=768
LATENT_DIMS[uni_v2]=768
LATENT_DIMS[virchow_v1]=2560

mkdir -p "$PROJECT_DIR/logs"

echo "============================================================================"
echo "Training Base ABMIL Models with FGE (3 models × 50 folds each)"
echo "============================================================================"
echo "Start time: $(date)"
echo "Project dir: $PROJECT_DIR"
echo "Work dir: $WORK_DIR"
echo "Dataset: $TRAIN_SOURCE / $TASK_NAME"
echo "ABMIL Epochs (convergence): $EPOCHS"
echo "FGE: n_cycles=$N_CYCLES, cycle_length=$CYCLE_LENGTH"
echo "     lr_1=$LR_1, lr_2=$LR_2, cycle_patience=$CYCLE_PATIENCE"
echo "============================================================================"
echo ""

TOTAL_START=$(date +%s)

for MODEL in "${MODELS[@]}"; do
    echo ""
    echo "╔════════════════════════════════════════════════════════════════════════╗"
    echo "║ Model: $MODEL (FGE)"
    echo "╚════════════════════════════════════════════════════════════════════════╝"
    echo ""

    LATENT_DIM=${LATENT_DIMS[$MODEL]}

    # Check if already trained with FGE (checkpoints_fge exists with 50 folds)
    MODEL_DIR="$WORK_DIR/$TRAIN_SOURCE/$TASK_NAME/abmil/${MODEL}_${TISSUE_PATCHING}"
    if [ -d "$MODEL_DIR/checkpoints_fge" ]; then
        FOLD_COUNT=$(ls -d "$MODEL_DIR/checkpoints_fge/fold_"* 2>/dev/null | wc -l)
        if [ "$FOLD_COUNT" -eq 50 ]; then
            echo "✓ $MODEL FGE already trained (50 folds found). Skipping..."
            continue
        fi
    fi

    MODEL_START=$(date +%s)
    echo "[$(date)] Starting FGE training for $MODEL (latent_dim=$LATENT_DIM)..."

    python "$PROJECT_DIR/src/train_abmil_fge.py" \
        --foundational_model "$MODEL" \
        --latent_dim "$LATENT_DIM" \
        --work_dir "$WORK_DIR" \
        --train_source "$TRAIN_SOURCE" \
        --tissue_patching "$TISSUE_PATCHING" \
        --task_name "$TASK_NAME" \
        --epochs "$EPOCHS" \
        --fold all \
        --n_cycles "$N_CYCLES" \
        --cycle_length "$CYCLE_LENGTH" \
        --lr_1 "$LR_1" \
        --lr_2 "$LR_2" \
        --cycle_patience "$CYCLE_PATIENCE" \
        2>&1 | tee -a "logs/train_fge_${MODEL}.log"

    MODEL_END=$(date +%s)
    MODEL_DURATION=$((MODEL_END - MODEL_START))

    echo "[$(date)] ✓ $MODEL FGE training COMPLETED"
    echo "Duration: $((MODEL_DURATION / 60)) minutes $((MODEL_DURATION % 60)) seconds"
    echo ""
done

# ============================================================================
# Step 2: Generate validation predictions for meta-learner (FGE snapshots)
# ============================================================================
echo ""
echo "╔════════════════════════════════════════════════════════════════════════╗"
echo "║ Step 2: Generating Validation Predictions (FGE Snapshots) for Meta-learner"
echo "╚════════════════════════════════════════════════════════════════════════╝"
echo ""

for MODEL in "${MODELS[@]}"; do
    LATENT_DIM=${LATENT_DIMS[$MODEL]}

    echo "[$(date)] Generating FGE validation predictions for $MODEL..."

    python "$PROJECT_DIR/src/test_abmil_fge.py" \
        --foundational_model "$MODEL" \
        --latent_dim "$LATENT_DIM" \
        --work_dir "$WORK_DIR" \
        --train_source "$TRAIN_SOURCE" \
        --tissue_patching "$TISSUE_PATCHING" \
        --task_name "$TASK_NAME" \
        --epochs "$EPOCHS" \
        2>&1 | tee -a "logs/test_fge_${MODEL}.log"

    echo "[$(date)] ✓ FGE validation predictions generated for $MODEL"
    echo ""
done

# ============================================================================
# Summary
# ============================================================================
TOTAL_END=$(date +%s)
TOTAL_DURATION=$((TOTAL_END - TOTAL_START))

echo "============================================================================"
echo "EXECUTION SUMMARY (FGE Training)"
echo "============================================================================"
echo ""
echo "Total FGE Training + Validation Predictions: $((TOTAL_DURATION / 60)) minutes $((TOTAL_DURATION % 60)) seconds"
echo ""
echo "Next steps:"
echo "  1. Verify all output directories exist (see logs above)"
echo "  2. If OK, run: sbatch run_3_experiments_v2.sbatch"
echo ""
echo "Log files:"
echo "  - logs/train_base_models_fge_${SLURM_JOB_ID}.out (this file)"
echo "  - logs/train_fge_ctranspath.log"
echo "  - logs/train_fge_uni_v2.log"
echo "  - logs/train_fge_virchow_v1.log"
echo "  - logs/test_fge_ctranspath.log"
echo "  - logs/test_fge_uni_v2.log"
echo "  - logs/test_fge_virchow_v1.log"
echo ""
echo "============================================================================"
echo "End time: $(date)"
echo "✓ FGE base model training pipeline completed!"
echo "============================================================================"
