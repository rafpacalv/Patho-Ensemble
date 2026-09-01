#!/bin/bash
#SBATCH --job-name=auc_fixed
#SBATCH --partition=main
#SBATCH --constraint=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=logs/auc_fixed_%j.out
#SBATCH --error=logs/auc_fixed_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=rafael.pachon.alvarez@gmail.com

# ============================================================================
# AUC-BASED REWEIGHTING (FIXED) — Compare norm_ratio vs auc_based con MetricDistance corregido
# ============================================================================

set -uo pipefail

source $(conda info --base)/etc/profile.d/conda.sh
conda activate ensemble

PROJECT_DIR="/home/JKP6679/Patho-Ensemble"
cd "$PROJECT_DIR"

DATASET="${DATASET:-cptac_brca}"
TASK="${TASK:-TP53_mutation}"

eval "$(python3 src/experiment_config.py --emit-shell --dataset "$DATASET" --task "$TASK")"

LOG_DIR="$PROJECT_DIR/logs/${DATASET}_${TASK}"
mkdir -p "$LOG_DIR"

ABMIL_DIR="$WORK_DIR/$DATASET/$TASK/abmil"

echo "============================================================================"
echo "PRE-FLIGHT CHECKS — $DATASET / $TASK"
echo "============================================================================"

STANDARD_OK=1
for MODEL in $MODELS; do
    B="$ABMIL_DIR/${MODEL}_${TISSUE_PATCHING}"
    if [ ! -d "${B}_train_eval/val_outputs/fold_0" ]; then
        echo "❌ Missing base model: ${MODEL}"
        STANDARD_OK=0
    fi
done

if [ "$STANDARD_OK" -eq 1 ]; then
    echo "✓ All base models available"
else
    echo "❌ Missing base models. Run:"
    echo "   DATASET=$DATASET TASK=$TASK sbatch train_base_models.sbatch"
    exit 1
fi

echo ""
echo "============================================================================"
echo "TESTING: SVM with norm_ratio (baseline) vs auc_based (FIXED) — $DATASET / $TASK"
echo "============================================================================"
echo "Start time: $(date)"
echo "Base models: $MODELS"
echo "============================================================================"

TOTAL_START=$(date +%s)
FAILED=()
RAN=()

function run_config() {
    local META=$1 METHOD=$2 SUFFIX=$3
    local LABEL="${META}_${METHOD}"

    echo ""
    echo "╔════════════════════════════════════════════════════════════════════════╗"
    echo "║ $LABEL (reweight_method=$METHOD)"
    echo "╚════════════════════════════════════════════════════════════════════════╝"

    START=$(date +%s)
    if python src/ensemble4.py \
        --foundational_models $MODELS \
        --work_dir "$WORK_DIR" \
        --train_source "$DATASET" \
        --tissue_patching "$TISSUE_PATCHING" \
        --task_name "$TASK" \
        --meta_model "$META" \
        --meta_features "insample" \
        --reweight_by_model_importance \
        --reweight_method "$METHOD" \
        --results_subdir_suffix "$SUFFIX" \
        2>&1 | tee -a "$LOG_DIR/${LABEL}.log"; then

        DUR=$(( $(date +%s) - START ))
        echo "[$(date)] ✓ $LABEL completed ($((DUR/60)) min $((DUR%60)) s)"
        RAN+=("$LABEL")
    else
        echo "[$(date)] ❌ $LABEL FAILED"
        FAILED+=("$LABEL")
    fi
}

# Test only SVM (fastest) to verify the fix works
run_config "svm" "norm_ratio" "_nr_fixed"
run_config "svm" "auc_based" "_auc_fixed"

# Comparison table
echo ""
echo "╔════════════════════════════════════════════════════════════════════════╗"
echo "║ RESULTS — norm_ratio (baseline) vs. auc_based (FIXED)"
echo "╚════════════════════════════════════════════════════════════════════════╝"

ABMIL_DIR="$ABMIL_DIR" DATASET="$DATASET" TASK="$TASK" python3 << 'PYEOF'
import json
import os
from pathlib import Path
import numpy as np
from scipy.stats import wilcoxon

ABMIL_DIR = Path(os.environ["ABMIL_DIR"])
DATASET = os.environ["DATASET"]
TASK = os.environ["TASK"]

METRICS = [("macro-ovr-auc", "AUC-ROC"), ("weighted_kappa", "Kappa"),
           ("macro-f1", "Macro-F1"), ("bacc", "Balanced acc"), ("acc", "Accuracy")]

# For SVM only
baseline_dir = ABMIL_DIR / "ensemble4_svm_nr_fixed"
auc_dir = ABMIL_DIR / "ensemble4_svm_auc_fixed"

baseline_data = {}
auc_data = {}

for sub, data_dict in [(baseline_dir, baseline_data), (auc_dir, auc_data)]:
    folds = sorted(sub.glob("test_metrics/fold_*"),
                  key=lambda p: int(p.name.split("_")[1]))
    for fd in folds:
        try:
            m = json.load(open(fd / "metrics.json"))["overall"]
            for k, _ in METRICS:
                data_dict.setdefault(k, []).append(m[k])
        except:
            pass

for k in baseline_data:
    baseline_data[k] = np.array(baseline_data[k])
    auc_data[k] = np.array(auc_data[k])

print(f"\n{'='*100}")
print(f"{DATASET} / {TASK}   —   {len(baseline_data.get('macro-ovr-auc', []))} folds")
print(f"{'='*100}\n")

for key, label in METRICS:
    if key not in baseline_data or key not in auc_data:
        continue

    b_vals = baseline_data[key]
    a_vals = auc_data[key]

    d = a_vals - b_vals
    mean_d = d.mean()
    sd_d = d.std(ddof=1)
    ci = 1.96 * sd_d / np.sqrt(len(d)) if sd_d > 0 else 0.0
    p = wilcoxon(a_vals, b_vals, alternative='two-sided').pvalue if sd_d > 0 else 1.0
    wins = int((d > 0).sum())
    ties = int((d == 0).sum())

    print(f"{label:12s}  baseline={b_vals.mean():.4f}  auc_based={a_vals.mean():.4f}  "
          f"Δ={mean_d:+.4f}  CI={ci:+.4f}s  wins={wins:2d}  ties={ties:2d}  p={p:.4f}")

PYEOF

# Summary
TOTAL_DUR=$(( $(date +%s) - TOTAL_START ))
echo ""
echo "╔════════════════════════════════════════════════════════════════════════╗"
echo "║ SUMMARY"
echo "╚════════════════════════════════════════════════════════════════════════╝"
echo "Completed: ${#RAN[@]}"
echo "Failed:    ${#FAILED[@]}"
if [ ${#FAILED[@]} -gt 0 ]; then
    echo "           ${FAILED[*]}"
fi
echo "Total duration: $((TOTAL_DUR/60)) min $((TOTAL_DUR%60)) s"
echo "End time: $(date)"
