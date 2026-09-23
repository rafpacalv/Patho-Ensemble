#!/bin/bash
#SBATCH --job-name=meta_view_sel
#SBATCH --partition=main
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --output=logs/meta_view_sel_%j.out
#SBATCH --error=logs/meta_view_sel_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=rafael.pachon.alvarez@gmail.com

# ============================================================================
# VIEW SELECTION PARA STACKING (L1 / elastic net) — nuevo, 2026-09-22
# ============================================================================
# Bibliografía: Sylvain et al., "View selection in multi-view stacking:
# choosing the meta-learner" — cada modelo base es una "vista"; lasso /
# elastic net no-negativos sobre el meta-learner seleccionan vistas. Aquí se
# prueba la parte no-restringida (sin la restricción de signo ≥0, que
# requeriría un optimizador propio — sklearn LogisticRegression no la
# soporta): logreg_l1 (ya existía en ensemble4.py, nunca evaluado limpio) y
# logreg_en (elastic net, NUEVO — penalty='elasticnet', solver='saga').
#
# Todo corre con --feature_space logit --drop_redundant_class: en tarea
# binaria eso deja EXACTAMENTE 1 columna por modelo base (log-odds), así que
# la L1/elastic-net sobre esas columnas es selección de MODELO, no de columna
# suelta (ver comentario en ensemble4.py junto a logreg_l1). El baseline
# 'logreg' de este experimento corre bajo el mismo feature_space —no es
# directamente el 0.7616/0.7986 de la escalera (esa usaba --feature_space
# prob) — para que la comparación sea pareada dentro del mismo espacio de
# features. CPU-only: ningún meta-modelo aquí usa GPU.
#
# Barrido, por trío (histórico y ganador, en la misma tanda):
#   logreg                                (baseline, este espacio de features)
#   logreg_l1   C ∈ {0.01, 0.1, 1.0, 10.0}
#   logreg_en   l1_ratio ∈ {0.2, 0.5, 0.8}, C=1.0
# = 8 configs × 2 tríos = 16 runs de 50 folds cada uno.
#
# Análisis pareado (Holm-Bonferroni) reutilizando
# analyze_feature_engineering_ladder_v2.py (es agnóstico al significado del
# meta-modelo: solo lee test_metrics/fold_*/metrics.json por subdirectorio).
# ============================================================================

set -uo pipefail   # sin -e: una config que falle no debe abortar el resto

source $(conda info --base)/etc/profile.d/conda.sh
conda activate ensemble

PROJECT_DIR="/home/JKP6679/Patho-Ensemble"
cd "$PROJECT_DIR"

DATASET="cptac_brca"
TASK="TP53_mutation"
WORK_DIR="/home/JKP6679/Patho-Ensemble/PARADIS/datos/patches"
TISSUE_PATCHING="20x_224px_0px_overlap"

MODELS_HIST="ctranspath uni_v2 virchow_v1"
MODELS_WIN="ctranspath uni_v2 conch_v1_5"

echo "============================================================================"
echo "VIEW SELECTION (L1 / elastic net) PARA STACKING — $DATASET / $TASK"
echo "Fecha: $(date)"
echo "============================================================================"

run_config() {
    local LABEL=$1
    local MODELS=$2
    local META_MODEL=$3
    local EXTRA_ARGS=$4
    local SUFFIX=$5

    echo "============================================================================"
    echo "CONFIG: $LABEL"
    echo "  Modelos: $MODELS | meta_model: $META_MODEL | extra: $EXTRA_ARGS | suffix: $SUFFIX"
    echo "  Inicio: $(date)"
    echo "============================================================================"

    CMD="python src/ensemble4.py"
    CMD="$CMD --foundational_models $MODELS"
    CMD="$CMD --work_dir $WORK_DIR"
    CMD="$CMD --train_source $DATASET"
    CMD="$CMD --tissue_patching $TISSUE_PATCHING"
    CMD="$CMD --task_name $TASK"
    CMD="$CMD --feature_space logit --drop_redundant_class"
    CMD="$CMD --meta_model $META_MODEL"
    CMD="$CMD --results_subdir_suffix $SUFFIX"
    CMD="$CMD $EXTRA_ARGS"

    if eval "$CMD"; then
        echo "✓ $LABEL completado"
        echo ""
    else
        echo "❌ $LABEL falló con código $?"
        echo ""
        return 1
    fi
}

for TRIO_TAG in hist win; do
    if [ "$TRIO_TAG" = "hist" ]; then
        MODELS="$MODELS_HIST"
    else
        MODELS="$MODELS_WIN"
    fi

    run_config "logreg ($TRIO_TAG)"              "$MODELS" "logreg"    ""                        "_$TRIO_TAG"
    run_config "logreg_l1 C=0.01 ($TRIO_TAG)"     "$MODELS" "logreg_l1" "--meta_l1_c 0.01"        "_c001_$TRIO_TAG"
    run_config "logreg_l1 C=0.1 ($TRIO_TAG)"      "$MODELS" "logreg_l1" "--meta_l1_c 0.1"         "_c01_$TRIO_TAG"
    run_config "logreg_l1 C=1.0 ($TRIO_TAG)"      "$MODELS" "logreg_l1" "--meta_l1_c 1.0"         "_c1_$TRIO_TAG"
    run_config "logreg_l1 C=10.0 ($TRIO_TAG)"     "$MODELS" "logreg_l1" "--meta_l1_c 10.0"        "_c10_$TRIO_TAG"
    run_config "logreg_en l1_ratio=0.2 ($TRIO_TAG)" "$MODELS" "logreg_en" "--meta_l1_ratio 0.2 --meta_l1_c 1.0" "_l02_c1_$TRIO_TAG"
    run_config "logreg_en l1_ratio=0.5 ($TRIO_TAG)" "$MODELS" "logreg_en" "--meta_l1_ratio 0.5 --meta_l1_c 1.0" "_l05_c1_$TRIO_TAG"
    run_config "logreg_en l1_ratio=0.8 ($TRIO_TAG)" "$MODELS" "logreg_en" "--meta_l1_ratio 0.8 --meta_l1_c 1.0" "_l08_c1_$TRIO_TAG"
done

echo "============================================================================"
echo "ANÁLISIS PAREADO"
echo "============================================================================"

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

echo "============================================================================"
echo "COMPLETADO — Fin: $(date)"
echo "============================================================================"
echo ""
echo "Resultados:"
echo "  $PROJECT_DIR/results_meta_view_selection_hist_${DATASET}_${TASK}.json"
echo "  $PROJECT_DIR/results_meta_view_selection_win_${DATASET}_${TASK}.json"
