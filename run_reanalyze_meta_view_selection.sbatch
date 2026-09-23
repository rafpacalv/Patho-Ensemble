#!/bin/bash
#SBATCH --job-name=reanalyze_view_sel
#SBATCH --partition=main
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=00:15:00
#SBATCH --output=logs/reanalyze_view_sel_%j.out
#SBATCH --error=logs/reanalyze_view_sel_%j.err

# Re-corre solo el análisis del sweep de view selection (job 72576, cuyos 16
# ensemble4.py sí completaron bien en disco) tras arreglar dos bugs en
# analyze_feature_engineering_ladder_v2.py: etiquetas con '=' interno se
# cortaban mal (split en el primer '=') y colisionaban en silencio,
# perdiendo 3 de 4 escalones L1 sin error. Ahora usa 'C:0.01' en vez de
# 'C=0.01' y el script aborta si detecta una etiqueta repetida.

set -euo pipefail

source $(conda info --base)/etc/profile.d/conda.sh
conda activate ensemble

PROJECT_DIR="/home/JKP6679/Patho-Ensemble"
cd "$PROJECT_DIR"

DATASET="cptac_brca"
TASK="TP53_mutation"
WORK_DIR="/home/JKP6679/Patho-Ensemble/PARADIS/datos/patches"

for TRIO_TAG in hist win; do
    python src/analyze_feature_engineering_ladder_v2.py \
        --dataset "$DATASET" \
        --task "$TASK" \
        --work_dir "$WORK_DIR" \
        --subdirs "Baseline logreg=ensemble4_logit_dropc_${TRIO_TAG}" \
                  "L1 C:0.01=ensemble4_logreg_l1_logit_dropc_c001_${TRIO_TAG}" \
                  "L1 C:0.1=ensemble4_logreg_l1_logit_dropc_c01_${TRIO_TAG}" \
                  "L1 C:1.0=ensemble4_logreg_l1_logit_dropc_c1_${TRIO_TAG}" \
                  "L1 C:10.0=ensemble4_logreg_l1_logit_dropc_c10_${TRIO_TAG}" \
                  "ElasticNet l1_ratio:0.2=ensemble4_logreg_en_logit_dropc_l02_c1_${TRIO_TAG}" \
                  "ElasticNet l1_ratio:0.5=ensemble4_logreg_en_logit_dropc_l05_c1_${TRIO_TAG}" \
                  "ElasticNet l1_ratio:0.8=ensemble4_logreg_en_logit_dropc_l08_c1_${TRIO_TAG}" \
        --output "$PROJECT_DIR/results_meta_view_selection_${TRIO_TAG}_${DATASET}_${TASK}.json"
done
