#!/bin/bash
#SBATCH --job-name=ablation_mlp
#SBATCH --output=logs/ablation_study_%J.out
#SBATCH --error=logs/ablation_study_%J.err
#SBATCH --time=20:00:00
#SBATCH --mem=64GB
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --partition=main
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=rafael.pachon.alvarez@gmail.com

# Ablation Study: Desacoplar arquitectura vs snapshot ensemble
# ============================================================
#
# Job allocation:
#   - 64 GB RAM
#   - 8 CPUs
#   - 1 GPU
#   - 20 hours max
#
# Expected runtime:
#   Phase 1: Single-Cycle LARGE (24 trials) ≈ 0.5 horas
#   Phase 2: Snapshot SMALL (216 trials)    ≈ 5-6 horas
#   ──────────────────────────────────────────────────
#   Total:                                   ≈ 6-7 horas
#
# Outputs:
#   - logs/ablation_study_<jobid>.out  (stdout)
#   - logs/ablation_study_<jobid>.err  (stderr)
#   - ablation_study_results.json       (metrics)
#   - abmil/ensemble4_mlp_ablation_*/   (individual trial results)

set -e

echo "======================================================================"
echo "Ablation Study: Arquitectura vs Snapshot Ensemble"
echo "======================================================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Start time: $(date)"
echo "Node: $(hostname)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo ""

# Change to project directory
cd /shared/home/JKP6679/Patho-Ensemble

# Activate conda environment
source /home/JKP6679/miniconda3/etc/profile.d/conda.sh
conda activate ensemble

echo "Environment activated: $(which python)"
echo "PyTorch version: $(python -c 'import torch; print(torch.__version__)')"
echo ""

# Create logs directory if needed
mkdir -p logs

# Run ablation study
echo "Running ablation study..."
echo ""

python src/ablation_study_mlp.py

ABLATION_EXIT=$?

echo ""
echo "======================================================================"
echo "Ablation Study Complete"
echo "======================================================================"
echo "Exit code: $ABLATION_EXIT"
echo "End time: $(date)"
echo ""

if [ $ABLATION_EXIT -eq 0 ]; then
    echo "✅ SUCCESS: Ablation study completed"
    echo ""
    echo "Results saved to: ablation_study_results.json"
    echo ""
    echo "View results with:"
    echo "  jq '.' ablation_study_results.json | head -100"
    echo ""
else
    echo "❌ FAILED: Ablation study exited with code $ABLATION_EXIT"
fi

exit $ABLATION_EXIT
