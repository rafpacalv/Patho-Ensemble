#!/bin/bash
#SBATCH --job-name=reanalyze_ladder
#SBATCH --partition=main
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=00:15:00
#SBATCH --output=logs/reanalyze_ladder_%j.out
#SBATCH --error=logs/reanalyze_ladder_%j.err

# ============================================================================
# Re-corre solo el análisis (sin GPU, no reentrena ni re-predice nada) de la
# escalera de feature engineering v2 (job 72573), usando
# src/analyze_feature_engineering_ladder_v2.py después de arreglar el bug de
# interpretación: antes, cualquier escalón con marca "*" o "**" (incluido el
# umbral marginal 0.05<=p<0.10) se imprimía como "mejora significativa (α=0.05)"
# sin mirar el signo de Δ, y el campo 'significant' guardado en el JSON tenía
# el mismo problema. Esto solo regenera los dos results_ladder_*.json con la
# interpretación correcta; los ensemble4_fea_e*/ensemble4_fea_conch_e* en
# disco no cambian.
# ============================================================================

set -euo pipefail

source $(conda info --base)/etc/profile.d/conda.sh
conda activate ensemble

PROJECT_DIR="/home/JKP6679/Patho-Ensemble"
cd "$PROJECT_DIR"

DATASET="cptac_brca"
TASK="TP53_mutation"
WORK_DIR="/home/JKP6679/Patho-Ensemble/PARADIS/datos/patches"

python src/analyze_feature_engineering_ladder_v2.py \
    --dataset "$DATASET" \
    --task "$TASK" \
    --work_dir "$WORK_DIR" \
    --subdirs 'E0 (Baseline)=ensemble4_fea_e0' \
              'E1 (+disagreement)=ensemble4_fea_e1' \
              'E2 (+disagreement+margin)=ensemble4_fea_e2' \
              'E3 (+disagreement+margin+entropy)=ensemble4_fea_e3' \
    --output "$PROJECT_DIR/results_ladder_cptac_brca_TP53_mutation.json"

python src/analyze_feature_engineering_ladder_v2.py \
    --dataset "$DATASET" \
    --task "$TASK" \
    --work_dir "$WORK_DIR" \
    --subdirs 'E0 (Baseline)=ensemble4_fea_conch_e0' \
              'E1 (+disagreement)=ensemble4_fea_conch_e1' \
              'E2 (+disagreement+margin)=ensemble4_fea_conch_e2' \
              'E3 (+disagreement+margin+entropy)=ensemble4_fea_conch_e3' \
    --output "$PROJECT_DIR/results_ladder_conch_cptac_brca_TP53_mutation.json"
