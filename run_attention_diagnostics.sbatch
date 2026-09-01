#!/bin/bash
#SBATCH --job-name=attn_diag
#SBATCH --partition=main
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=logs/attn_diag_%j.out
#SBATCH --error=logs/attn_diag_%j.err

# ¿Por qué uni_v2 no aprende a atender (entropía 0.9939 ≈ uniforme)?
# Discrimina entre: (A) la dimensión de entrada / compresión de la proyección,
# (B) el número de parches, (C) la geometría de las features.
# Sólo inferencia sobre checkpoints existentes: no pide GPU.

set -euo pipefail
source $(conda info --base)/etc/profile.d/conda.sh
conda activate ensemble
cd /home/JKP6679/Patho-Ensemble

echo "Job $SLURM_JOB_ID en $(hostname) — $(date)"
python src/attention_diagnostics.py \
    --train_source "${DATASET:-cptac_brca}" \
    --task_name "${TASK:-TP53_mutation}" \
    --n_slides "${N_SLIDES:-25}" \
    --out "results_attention_diagnostics_${DATASET:-cptac_brca}_${TASK:-TP53_mutation}.json"
echo "✅ $(date)"
