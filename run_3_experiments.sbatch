#!/bin/bash
#SBATCH --job-name=ensemble_3exp
#SBATCH --partition=main
#SBATCH --constraint=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=logs/ensemble_3exp_%j.out
#SBATCH --error=logs/ensemble_3exp_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=rafael.pachon.alvarez@gmail.com

# ============================================================================
# Snapshot Ensembles Experiments: Logistic Regression vs MLP vs Snapshot MLP
# ============================================================================
# This script runs 3 meta-learner experiments on PARADIS data and compares results
# Expected total runtime: ~40-60 minutes (50 folds × 3 experiments)
# ============================================================================

set -e  # Exit on error

# Activate conda environment
source $(conda info --base)/etc/profile.d/conda.sh
conda activate ensemble

PROJECT_DIR="/home/JKP6679/Patho-Ensemble"
WORK_DIR="/home/JKP6679/Patho-Ensemble/PARADIS/datos"
TRAIN_SOURCE="cptac_brca"
TISSUE_PATCHING="20x_224px_0px_overlap"
TASK_NAME="TP53_mutation"

# Create logs directory if it doesn't exist
mkdir -p "$PROJECT_DIR/logs"

echo "============================================================================"
echo "Snapshot Ensembles: 3 Meta-learner Experiments"
echo "============================================================================"
echo "Start time: $(date)"
echo "Project dir: $PROJECT_DIR"
echo "Work dir: $WORK_DIR"
echo "Dataset: $TRAIN_SOURCE / $TASK_NAME"
echo "============================================================================"
echo ""

# ============================================================================
# EXPERIMENT 1: Logistic Regression (Baseline)
# ============================================================================
echo "[$(date)] Starting EXPERIMENT 1: Logistic Regression (baseline)..."
echo "---"
cd "$PROJECT_DIR"

START_EXP1=$(date +%s)
python src/ensemble4.py \
    --foundational_models ctranspath uni_v2 virchow_v1 \
    --work_dir "$WORK_DIR" \
    --train_source "$TRAIN_SOURCE" \
    --tissue_patching "$TISSUE_PATCHING" \
    --task_name "$TASK_NAME" \
    --meta_model logreg \
    2>&1 | tee -a logs/exp1_logreg.log

END_EXP1=$(date +%s)
DURATION_EXP1=$((END_EXP1 - START_EXP1))
echo "[$(date)] ✓ Experiment 1 COMPLETED (duration: ${DURATION_EXP1}s)"
echo ""

# ============================================================================
# EXPERIMENT 2: MLP + Single-Cycle Cosine + Early Stopping
# ============================================================================
echo "[$(date)] Starting EXPERIMENT 2: MLP + Single-Cycle Cosine + Early Stopping..."
echo "---"

START_EXP2=$(date +%s)
python src/ensemble4.py \
    --foundational_models ctranspath uni_v2 virchow_v1 \
    --work_dir "$WORK_DIR" \
    --train_source "$TRAIN_SOURCE" \
    --tissue_patching "$TISSUE_PATCHING" \
    --task_name "$TASK_NAME" \
    --meta_model mlp \
    --meta_hidden_dim 16 \
    --meta_dropout 0.1 \
    --meta_epochs 100 \
    --meta_lr 1e-3 \
    --meta_wd 1e-4 \
    --meta_patience 8 \
    2>&1 | tee -a logs/exp2_mlp.log

END_EXP2=$(date +%s)
DURATION_EXP2=$((END_EXP2 - START_EXP2))
echo "[$(date)] ✓ Experiment 2 COMPLETED (duration: ${DURATION_EXP2}s)"
echo ""

# ============================================================================
# EXPERIMENT 3: MLP + Snapshot Ensembles (Cyclic Cosine + Snapshots)
# ============================================================================
echo "[$(date)] Starting EXPERIMENT 3: MLP + Snapshot Ensembles (Cyclic Cosine + Snapshots)..."
echo "---"

START_EXP3=$(date +%s)
python src/ensemble4.py \
    --foundational_models ctranspath uni_v2 virchow_v1 \
    --work_dir "$WORK_DIR" \
    --train_source "$TRAIN_SOURCE" \
    --tissue_patching "$TISSUE_PATCHING" \
    --task_name "$TASK_NAME" \
    --meta_model mlp_snapshot \
    --meta_hidden_dim 16 \
    --meta_dropout 0.1 \
    --meta_epochs 120 \
    --n_cycles 6 \
    --restart_lr 5e-3 \
    --meta_wd 1e-4 \
    --n_snapshots_ensemble 6 \
    2>&1 | tee -a logs/exp3_snapshot.log

END_EXP3=$(date +%s)
DURATION_EXP3=$((END_EXP3 - START_EXP3))
echo "[$(date)] ✓ Experiment 3 COMPLETED (duration: ${DURATION_EXP3}s)"
echo ""

# ============================================================================
# COMPARISON: Extract and display metrics
# ============================================================================
echo "============================================================================"
echo "RESULTS COMPARISON"
echo "============================================================================"
echo ""

METRICS_DIR="$WORK_DIR/$TRAIN_SOURCE/$TASK_NAME/abmil"

# Define color codes for nicer output
RED='\033[0;31m'
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m'  # No Color

echo -e "${BLUE}Extracting metrics from all 3 experiments...${NC}"
echo ""

# Create a Python script to format and compare results nicely
python3 << 'PYEOF'
import json
from pathlib import Path

work_dir = Path("/home/JKP6679/Patho-Ensemble/PARADIS/datos/cptac_brca/TP53_mutation/abmil")
experiments = [
    ("ensemble4", "Logistic Regression (Baseline)"),
    ("ensemble4_mlp", "MLP + Single-Cycle Cosine"),
    ("ensemble4_mlp_snapshot", "MLP + Snapshot Ensembles"),
]

results = []
print("=" * 110)
print(f"{'Experiment':<38} {'AUC-ROC':<28} {'F1':<20} {'Accuracy':<18}")
print("-" * 110)

for exp_dir, exp_name in experiments:
    metrics_file = work_dir / exp_dir / "test_metrics_summary.json"

    try:
        with open(metrics_file) as f:
            data = json.load(f)

        # Data is at root level, not under "metrics" key
        auc_data = data.get("macro-ovr-auc", {})
        f1_data = data.get("macro-f1", {})
        acc_data = data.get("acc", {})

        auc_formatted = auc_data.get("formatted", "N/A")
        f1_formatted = f1_data.get("formatted", "N/A")
        acc_formatted = acc_data.get("formatted", "N/A")

        # Extract mean values for comparison
        auc_mean = auc_data.get("mean", None)
        f1_mean = f1_data.get("mean", None)
        acc_mean = acc_data.get("mean", None)

        print(f"{exp_name:<38} {auc_formatted:<28} {f1_formatted:<20} {acc_formatted:<18}")
        results.append({
            "experiment": exp_name,
            "auc": auc_mean,
            "f1": f1_mean,
            "accuracy": acc_mean
        })

    except FileNotFoundError:
        print(f"{exp_name:<38} {'FILE NOT FOUND':<28} {'FILE NOT FOUND':<20} {'FILE NOT FOUND':<18}")
    except Exception as e:
        print(f"{exp_name:<38} {'ERROR':<28} {'ERROR':<20} {'ERROR':<18}")
        print(f"  └─ {type(e).__name__}: {e}")

print("=" * 110)
print()

# Calculate and display improvements
print("=" * 110)
print("IMPROVEMENT ANALYSIS (vs Logistic Regression baseline)")
print("-" * 110)

if len(results) >= 3:
    baseline_auc = results[0]["auc"]
    baseline_f1 = results[0]["f1"]
    baseline_acc = results[0]["accuracy"]

    for i in range(1, len(results)):
        exp = results[i]
        exp_name = exp["experiment"]

        auc_str = "N/A"
        f1_str = "N/A"
        acc_str = "N/A"

        if baseline_auc is not None and exp["auc"] is not None:
            auc_diff = exp["auc"] - baseline_auc
            auc_pct = (auc_diff / baseline_auc * 100) if baseline_auc != 0 else 0
            auc_str = f"{auc_diff:+.4f} ({auc_pct:+.2f}%)"

        if baseline_f1 is not None and exp["f1"] is not None:
            f1_diff = exp["f1"] - baseline_f1
            f1_pct = (f1_diff / baseline_f1 * 100) if baseline_f1 != 0 else 0
            f1_str = f"{f1_diff:+.4f} ({f1_pct:+.2f}%)"

        if baseline_acc is not None and exp["accuracy"] is not None:
            acc_diff = exp["accuracy"] - baseline_acc
            acc_pct = (acc_diff / baseline_acc * 100) if baseline_acc != 0 else 0
            acc_str = f"{acc_diff:+.4f} ({acc_pct:+.2f}%)"

        print(f"{exp_name:<38} ΔAuc={auc_str:<22} ΔF1={f1_str:<16} ΔAcc={acc_str:<16}")

print("=" * 110)

PYEOF

echo ""

# ============================================================================
# Per-fold statistics (optional, for detailed analysis)
# ============================================================================
echo ""
echo "============================================================================"
echo "PER-FOLD STATISTICS (first 5 folds)"
echo "============================================================================"
echo ""

python3 << 'PYEOF'
import json
from pathlib import Path

work_dir = Path("/home/JKP6679/Patho-Ensemble/PARADIS/datos/cptac_brca/TP53_mutation/abmil")
experiments = ["ensemble4", "ensemble4_mlp", "ensemble4_mlp_snapshot"]

for exp_dir in experiments:
    metrics_file = work_dir / exp_dir / "test_metrics_summary.json"

    try:
        with open(metrics_file) as f:
            data = json.load(f)

        print(f"\n{exp_dir}:")
        print(f"  {'Fold':<8} {'AUC-ROC':<15} {'F1':<15} {'Accuracy':<15}")
        print(f"  {'-'*55}")

        # Data is at root level for aggregate metrics
        # Per-fold data would be in fold subdirectories, but we just show the summary here
        auc_data = data.get("macro-ovr-auc", {})
        f1_data = data.get("macro-f1", {})
        acc_data = data.get("acc", {})

        auc_formatted = auc_data.get("formatted", "N/A")
        f1_formatted = f1_data.get("formatted", "N/A")
        acc_formatted = acc_data.get("formatted", "N/A")

        print(f"  {'ALL':<8} {auc_formatted:<15} {f1_formatted:<15} {acc_formatted:<15}")

    except Exception as e:
        print(f"\n{exp_dir}: ERROR - {type(e).__name__}: {e}")

PYEOF

echo ""

# ============================================================================
# Summary
# ============================================================================
echo "============================================================================"
echo "EXECUTION SUMMARY"
echo "============================================================================"
echo ""
echo "Experiment 1 (LogReg):           ${DURATION_EXP1}s"
echo "Experiment 2 (MLP):              ${DURATION_EXP2}s"
echo "Experiment 3 (SnapshotMLP):      ${DURATION_EXP3}s"
echo ""
TOTAL_DURATION=$((DURATION_EXP1 + DURATION_EXP2 + DURATION_EXP3))
echo "Total duration:                  ${TOTAL_DURATION}s ($(printf '%d:%02d' $((TOTAL_DURATION/60)) $((TOTAL_DURATION%60))))"
echo ""
echo "Results directory: $METRICS_DIR"
echo "Log files:"
echo "  - $PROJECT_DIR/logs/exp1_logreg.log"
echo "  - $PROJECT_DIR/logs/exp2_mlp.log"
echo "  - $PROJECT_DIR/logs/exp3_snapshot.log"
echo "  - $PROJECT_DIR/logs/ensemble_3exp_${SLURM_JOB_ID}.out (this file)"
echo ""
echo "To view detailed metrics from a specific experiment:"
echo "  jq '.metrics' $METRICS_DIR/ensemble4/test_metrics_summary.json"
echo "  jq '.metrics' $METRICS_DIR/ensemble4_mlp/test_metrics_summary.json"
echo "  jq '.metrics' $METRICS_DIR/ensemble4_mlp_snapshot/test_metrics_summary.json"
echo ""
echo "============================================================================"
echo "End time: $(date)"
echo "✓ All experiments completed successfully!"
echo "============================================================================"
