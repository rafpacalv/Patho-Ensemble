#!/bin/bash
#SBATCH --job-name=fea_ladder_v2
#SBATCH --partition=main
#SBATCH --constraint=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=logs/fea_ladder_v2_%j.out
#SBATCH --error=logs/fea_ladder_v2_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=rafael.pachon.alvarez@gmail.com

# ============================================================================
# ESCALERA DE FEATURE ENGINEERING — v2 (repesca + extensión)
# ============================================================================
# `run_feature_engineering_ladder.sbatch` (19-ago) corrió E0..E3 sobre el
# trío histórico (ctranspath+uni_v2+virchow_v1). Sus resultados quedaron
# obsoletos: las predicciones base de ctranspath, virchow_v1 y conch_v1_5 se
# regeneraron el 21-ago 19:26-19:28 (ver CLAUDE.md, "Stored ensemble4_*
# results go stale"), y E0=0.7918 en results_ladder_cptac_brca_TP53_mutation.json
# no reproduce (el valor vigente para ese mismo trío y código es 0.7616).
#
# Este script hace dos cosas en una sola tanda (ambas comparten proceso y
# predicciones base, así que cada una queda pareada internamente):
#
#   (A) Repite la escalera original sobre el trío HISTÓRICO con las
#       predicciones base vigentes → corrige la cifra que no reproducía.
#         E0..E3 → ensemble4_fea_e{0..3}/  (mismos subdirs que la v1: se
#         sobrescriben con datos correctos)
#
#   (B) Corre la misma escalera sobre el trío GANADOR (ctranspath+uni_v2+
#       conch_v1_5, +0.037 AUC sobre el histórico — CLAUDE.md "Base model
#       selection dominates"). Pregunta nueva: ¿el resultado nulo de
#       feature engineering ("las features extra no ayudan") se sostiene
#       también cuando el trío de partida ya es fuerte, o solo se veía
#       porque el trío de partida era mediocre?
#         E0..E3 → ensemble4_fea_conch_e{0..3}/  (subdirs nuevos, no colisionan)
#
# Verificado antes de escribir este script: los ficheros
# {model}_..._train_eval/val_outputs/fold_0/preds.npy de los 4 modelos
# candidatos (ctranspath, uni_v2, virchow_v1, conch_v1_5) son todos
# posteriores a los checkpoints y a los coefs.npy de la escalera v1 (19-ago
# 21:15-21:17) → no hace falta re-predecir, solo re-ajustar el meta-learner.
#
# Al final corre también el análisis pareado (Holm-Bonferroni) de cada
# escalera por separado, con --output explícito para no pisarse.
#
# Uso:
#   sbatch run_feature_engineering_ladder_v2.sbatch
# ============================================================================

set -uo pipefail   # sin -e: un escalón que falle no debe abortar el resto

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

ABMIL_DIR="$WORK_DIR/$DATASET/$TASK/abmil"

echo "============================================================================"
echo "ESCALERA DE FEATURE ENGINEERING v2 — $DATASET / $TASK"
echo "Fecha: $(date)"
echo "Trío histórico: $MODELS_HIST"
echo "Trío ganador:   $MODELS_WIN"
echo "============================================================================"
echo ""

# ---------------------------------------------------------------------------
# Comprobaciones previas: bases estándar para AMBOS tríos
# ---------------------------------------------------------------------------
check_bases() {
    local LABEL=$1
    local MODELS=$2
    local OK=1
    for MODEL in $MODELS; do
        B="$ABMIL_DIR/${MODEL}_${TISSUE_PATCHING}"
        if [ ! -d "${B}_train_eval/val_outputs/fold_0" ]; then
            echo "❌ [$LABEL] falta ${MODEL}: _train_eval/val_outputs/fold_0"
            OK=0
        fi
    done
    if [ "$OK" -eq 0 ]; then
        echo "❌ [$LABEL] faltan bases estándar. Ejecuta primero:"
        echo "   MODELS_OVERRIDE=\"$MODELS\" DATASET=$DATASET TASK=$TASK sbatch train_base_models.sbatch"
        return 1
    fi
    echo "✓ [$LABEL] bases estándar disponibles ($MODELS)"
    return 0
}

BASES_OK=1
check_bases "histórico" "$MODELS_HIST" || BASES_OK=0
check_bases "ganador"   "$MODELS_WIN"  || BASES_OK=0

if [ "$BASES_OK" -eq 0 ]; then
    exit 1
fi
echo ""

# ---------------------------------------------------------------------------
# Función auxiliar: ejecutar un escalón para un trío dado
# ---------------------------------------------------------------------------
run_escalon() {
    local STEP_NAME=$1
    local MODELS=$2
    local EXTRA_FEATURES=$3
    local RESULTS_SUFFIX=$4

    echo "============================================================================"
    echo "ESCALÓN: $STEP_NAME"
    echo "  Modelos: $MODELS"
    if [ -z "$EXTRA_FEATURES" ]; then
        echo "  Extra features: (none — baseline)"
    else
        echo "  Extra features: $EXTRA_FEATURES"
    fi
    echo "  Results subdir suffix: $RESULTS_SUFFIX"
    echo "  Inicio: $(date)"
    echo "============================================================================"

    CMD="python src/ensemble4.py"
    CMD="$CMD --foundational_models $MODELS"
    CMD="$CMD --work_dir $WORK_DIR"
    CMD="$CMD --train_source $DATASET"
    CMD="$CMD --tissue_patching $TISSUE_PATCHING"
    CMD="$CMD --task_name $TASK"
    CMD="$CMD --results_subdir_suffix $RESULTS_SUFFIX"
    CMD="$CMD --meta_model logreg"

    if [ -n "$EXTRA_FEATURES" ]; then
        CMD="$CMD --extra_features $EXTRA_FEATURES"
    fi

    if eval "$CMD"; then
        echo "✓ $STEP_NAME completado exitosamente"
        echo ""
    else
        echo "❌ $STEP_NAME falló con código $?"
        echo ""
        return 1
    fi
}

# ---------------------------------------------------------------------------
# (A) Trío histórico — corrige la escalera v1
# ---------------------------------------------------------------------------
run_escalon "A-E0 histórico (Baseline)"                              "$MODELS_HIST" ""                                             "_fea_e0"
run_escalon "A-E1 histórico (+ disagreement)"                        "$MODELS_HIST" "disagreement"                                 "_fea_e1"
run_escalon "A-E2 histórico (+ disagreement + margin_avg)"           "$MODELS_HIST" "disagreement margin_avg"                      "_fea_e2"
run_escalon "A-E3 histórico (+ disagreement + margin_avg + entropy)" "$MODELS_HIST" "disagreement margin_avg entropy_per_model"    "_fea_e3"

# ---------------------------------------------------------------------------
# (B) Trío ganador — extensión nueva
# ---------------------------------------------------------------------------
run_escalon "B-E0 ganador (Baseline)"                              "$MODELS_WIN" ""                                             "_fea_conch_e0"
run_escalon "B-E1 ganador (+ disagreement)"                        "$MODELS_WIN" "disagreement"                                 "_fea_conch_e1"
run_escalon "B-E2 ganador (+ disagreement + margin_avg)"           "$MODELS_WIN" "disagreement margin_avg"                      "_fea_conch_e2"
run_escalon "B-E3 ganador (+ disagreement + margin_avg + entropy)" "$MODELS_WIN" "disagreement margin_avg entropy_per_model"    "_fea_conch_e3"

# ---------------------------------------------------------------------------
# Análisis pareado (Holm-Bonferroni) — uno por trío, con --output explícito
# ---------------------------------------------------------------------------
echo "============================================================================"
echo "ANÁLISIS PAREADO"
echo "============================================================================"

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

echo "============================================================================"
echo "ESCALERA v2 COMPLETADA — Fin: $(date)"
echo "============================================================================"
echo ""
echo "Resultados:"
echo "  Histórico (A): $ABMIL_DIR/ensemble4_fea_e{0..3}/"
echo "                 $PROJECT_DIR/results_ladder_cptac_brca_TP53_mutation.json"
echo "  Ganador   (B): $ABMIL_DIR/ensemble4_fea_conch_e{0..3}/"
echo "                 $PROJECT_DIR/results_ladder_conch_cptac_brca_TP53_mutation.json"
