#!/bin/bash
# Local grid search (without SLURM)
# Usage: ./run_grid_search_local.sh
# This runs the MLP hyperparameter optimization on local machine

set -e

# Activate conda environment
source $(conda info --base)/etc/profile.d/conda.sh
conda activate ensemble

PROJECT_DIR="/home/JKP6679/Patho-Ensemble"
mkdir -p "$PROJECT_DIR/logs"

echo "============================================================================"
echo "MLP Hyperparameter Grid Search (Local)"
echo "============================================================================"
echo "Start time: $(date)"
echo "Project dir: $PROJECT_DIR"
echo ""
echo "Grid configuration:"
echo "  Single-Cycle MLP: 81 combinations"
echo "  Snapshot Ensemble: 144 combinations"
echo "  Total: 225 trials (estimated 4-6 hours)"
echo ""
echo "Baseline LogReg: AUC=0.744, F1=0.672, Acc=0.701"
echo "============================================================================"
echo ""

cd "$PROJECT_DIR"
python src/grid_search_mlp.py 2>&1 | tee -a logs/grid_search_local.log

echo ""
echo "============================================================================"
echo "✓ Grid Search Completed: $(date)"
echo "============================================================================"
echo ""
echo "Results:"
echo "  JSON: $PROJECT_DIR/grid_search_results.json"
echo "  Log:  $PROJECT_DIR/logs/grid_search_local.log"
echo ""
echo "To view results:"
echo "  jq '.best_trial | {type, params, auc: .auc}' $PROJECT_DIR/grid_search_results.json"
echo "============================================================================"
