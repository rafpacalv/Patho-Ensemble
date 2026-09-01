#!/bin/bash
#SBATCH --job-name=idea3_phase1_attn
#SBATCH --partition=main
#SBATCH --constraint=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=logs/idea3_phase1_attn_%j.out
#SBATCH --error=logs/idea3_phase1_attn_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=rafael.pachon.alvarez@gmail.com

# ============================================================================
# IDEA 3 — FASE 1: Extracción de Atención Statistics
# ============================================================================
# Re-inferencia sobre checkpoints existentes para extraer attn_stats.npy
# NO requiere reentrenamiento — solo forward pass sobre ya entrenados.
#
# Ejecuta:
#   1. train_abmil.py --test (test splits)
#   2. test_abmil.py (meta-train splits, in-sample)
#   3. build_oof_features.py (OOF splits, si existe)
#
# Duración estimada: 4-6 horas (50 folds × 3 modelos, paralelizable)
# Coste GPU: Negligible (forward pass solamente, sin backward)
# ============================================================================

set -uo pipefail

source $(conda info --base)/etc/profile.d/conda.sh
conda activate ensemble

PROJECT_DIR="/home/JKP6679/Patho-Ensemble"
cd "$PROJECT_DIR"

# Configuración: cptac_brca / TP53_mutation
DATASET="cptac_brca"
TASK="TP53_mutation"

eval "$(python3 src/experiment_config.py --emit-shell --dataset "$DATASET" --task "$TASK")"

LOG_DIR="$PROJECT_DIR/logs/${DATASET}_${TASK}"
mkdir -p "$LOG_DIR"

echo "============================================================================"
echo "IDEA 3 — FASE 1: EXTRACCIÓN DE ATENCIÓN STATISTICS"
echo "============================================================================"
echo "Fecha: $(date)"
echo "Dataset: $DATASET, Task: $TASK"
echo "Modelos base: $MODELS"
echo "Patching: $TISSUE_PATCHING"
echo "============================================================================"
echo ""

# ---------------------------------------------------------------------------
# Paso 1: train_abmil.py --test (extrae attn_stats de test splits)
# ---------------------------------------------------------------------------
echo "📊 Paso 1: Extrayendo attn_stats de test splits (train_abmil.py)"
echo "============================================================================"

for MODEL in $MODELS; do
    echo ""
    echo "Modelo: $MODEL"
    CMD="python src/train_abmil.py"
    CMD="$CMD --foundational_model $MODEL"
    CMD="$CMD --work_dir $WORK_DIR"
    CMD="$CMD --train_source $DATASET"
    CMD="$CMD --tissue_patching $TISSUE_PATCHING"
    CMD="$CMD --task_name $TASK"
    CMD="$CMD --fold all"
    CMD="$CMD --epochs 1"  # dummy epochs, solo inferencia

    if eval "$CMD"; then
        echo "✓ $MODEL: test splits completados"
    else
        echo "❌ $MODEL: test splits fallaron"
    fi
done

# ---------------------------------------------------------------------------
# Paso 2: test_abmil.py (extrae attn_stats de meta-train splits, in-sample)
# ---------------------------------------------------------------------------
echo ""
echo "📊 Paso 2: Extrayendo attn_stats de meta-train splits (test_abmil.py)"
echo "============================================================================"

for MODEL in $MODELS; do
    echo ""
    echo "Modelo: $MODEL"
    CMD="python src/test_abmil.py"
    CMD="$CMD --foundational_model $MODEL"
    CMD="$CMD --work_dir $WORK_DIR"
    CMD="$CMD --train_source $DATASET"
    CMD="$CMD --tissue_patching $TISSUE_PATCHING"
    CMD="$CMD --task_name $TASK"
    CMD="$CMD --fold all"

    if eval "$CMD"; then
        echo "✓ $MODEL: meta-train splits completados"
    else
        echo "❌ $MODEL: meta-train splits fallaron"
    fi
done

# ---------------------------------------------------------------------------
# Paso 3: build_oof_features.py (extrae attn_stats de OOF splits, si aplica)
# ---------------------------------------------------------------------------
echo ""
echo "📊 Paso 3: Extrayendo attn_stats de OOF splits (build_oof_features.py)"
echo "============================================================================"

for MODEL in $MODELS; do
    echo ""
    echo "Modelo: $MODEL"
    CMD="python src/build_oof_features.py"
    CMD="$CMD --foundational_model $MODEL"
    CMD="$CMD --work_dir $WORK_DIR"
    CMD="$CMD --train_source $DATASET"
    CMD="$CMD --tissue_patching $TISSUE_PATCHING"
    CMD="$CMD --task_name $TASK"
    CMD="$CMD --n_inner 5"  # número de folds internos

    if eval "$CMD"; then
        echo "✓ $MODEL: OOF splits completados"
    else
        echo "❌ $MODEL: OOF splits fallaron"
    fi
done

# ---------------------------------------------------------------------------
# Resumen final
# ---------------------------------------------------------------------------
echo ""
echo "============================================================================"
echo "FASE 1 COMPLETADA"
echo "Fin: $(date)"
echo "============================================================================"
echo ""
echo "✅ attn_stats.npy generados en:"
echo "   - {model}_{patching}/test_outputs/fold_*/attn_stats.npy"
echo "   - {model}_{patching}_train_eval/val_outputs/fold_*/attn_stats.npy"
echo "   - {model}_{patching}_train_eval_oof/val_outputs_oof/fold_*/attn_stats.npy"
echo ""
echo "📄 Metadatos de columnas guardados en:"
echo "   - .../fold_*/attn_stats_columns.json"
echo ""
echo "Próximo paso: Ejecutar src/analyze_feature_engineering_ladder_v2.py"
echo "con --extra_features incluyendo attn_entropy_norm, attn_ess_norm"
echo ""
