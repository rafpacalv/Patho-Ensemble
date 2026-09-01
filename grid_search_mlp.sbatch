#!/bin/bash
#SBATCH --job-name=mlp_gridsearch
#SBATCH --partition=main
#SBATCH --constraint=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --output=logs/grid_search_%j.out
#SBATCH --error=logs/grid_search_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=rafael.pachon.alvarez@gmail.com

# ============================================================================
# MLP Hyperparameter Grid Search
# ============================================================================
# Optimiza hiperparámetros de MLP single-cycle y snapshot ensemble
# para superar el baseline de Logistic Regression (AUC=0.744)
# ============================================================================

set -e

# Activate conda environment
source $(conda info --base)/etc/profile.d/conda.sh
conda activate ensemble

PROJECT_DIR="/home/JKP6679/Patho-Ensemble"
mkdir -p "$PROJECT_DIR/logs"

echo "============================================================================"
echo "MLP Hyperparameter Grid Search Optimization"
echo "============================================================================"
echo "Start time: $(date)"
echo "Project dir: $PROJECT_DIR"
echo "============================================================================"
echo ""

cd "$PROJECT_DIR"
python src/grid_search_mlp.py

echo ""
echo "============================================================================"
echo "Grid Search Completed: $(date)"
echo "Results saved to: $PROJECT_DIR/grid_search_results.json"
echo "============================================================================"
