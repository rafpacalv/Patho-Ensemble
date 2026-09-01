#!/bin/bash
#
# Local execution script for 3 meta-learner experiments (without Slurm)
# Use this if you want to run directly on your machine or in a tmux session
#
# Usage:
#   bash run_3_experiments_local.sh
# Or in tmux:
#   tmux new-session -d -s ensemble_exp 'bash run_3_experiments_local.sh'
#

set -e  # Exit on error

# Activate conda environment
source $(conda info --base)/etc/profile.d/conda.sh
conda activate patho-ensemble

PROJECT_DIR="/home/JKP6679/Patho-Ensemble"
WORK_DIR="/home/JKP6679/Patho-Ensemble/PARADIS/datos/patches"
TRAIN_SOURCE="cptac_brca"
TISSUE_PATCHING="20x_224px_0px_overlap"
TASK_NAME="TP53_mutation"

# Create logs directory if it doesn't exist
mkdir -p "$PROJECT_DIR/logs"

echo "============================================================================"
echo "Snapshot Ensembles: 3 Meta-learner Experiments (Local Execution)"
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

echo "Extracting metrics from all 3 experiments..."
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
print("=" * 100)
print(f"{'Experiment':<38} {'AUC-ROC':<25} {'F1':<18} {'Accuracy':<15}")
print("-" * 100)

for exp_dir, exp_name in experiments:
    metrics_file = work_dir / exp_dir / "test_metrics_summary.json"

    try:
        with open(metrics_file) as f:
            data = json.load(f)
            metrics = data.get("metrics", {})

        auc = metrics.get("test_auc_roc", None)
        auc_ci = metrics.get("test_auc_roc_ci", None)
        f1 = metrics.get("test_f1", None)
        f1_ci = metrics.get("test_f1_ci", None)
        acc = metrics.get("test_accuracy", None)
        acc_ci = metrics.get("test_accuracy_ci", None)

        # Format with confidence intervals
        if auc is not None and auc_ci is not None:
            auc_str = f"{auc:.4f} ± {auc_ci:.4f}"
        elif auc is not None:
            auc_str = f"{auc:.4f}"
        else:
            auc_str = "N/A"

        if f1 is not None and f1_ci is not None:
            f1_str = f"{f1:.4f} ± {f1_ci:.4f}"
        elif f1 is not None:
            f1_str = f"{f1:.4f}"
        else:
            f1_str = "N/A"

        if acc is not None and acc_ci is not None:
            acc_str = f"{acc:.4f} ± {acc_ci:.4f}"
        elif acc is not None:
            acc_str = f"{acc:.4f}"
        else:
            acc_str = "N/A"

        print(f"{exp_name:<38} {auc_str:<25} {f1_str:<18} {acc_str:<15}")
        results.append({
            "experiment": exp_name,
            "auc": auc,
            "f1": f1,
            "accuracy": acc
        })

    except FileNotFoundError:
        print(f"{exp_name:<38} {'MISSING':<25} {'MISSING':<18} {'MISSING':<15}")
    except Exception as e:
        print(f"{exp_name:<38} {'ERROR':<25} {'ERROR':<18} {'ERROR':<15}")
        print(f"  └─ {e}")

print("=" * 100)
print()

# Calculate and display improvements
print("=" * 95)
print("IMPROVEMENT ANALYSIS (vs Logistic Regression baseline)")
print("-" * 100)

if len(results) >= 3:
    baseline_auc = results[0]["auc"]
    baseline_f1 = results[0]["f1"]
    baseline_acc = results[0]["accuracy"]

    for i in range(1, len(results)):
        exp = results[i]
        exp_name = exp["experiment"]

        auc_diff = None
        f1_diff = None
        acc_diff = None

        if baseline_auc is not None and exp["auc"] is not None:
            auc_diff = exp["auc"] - baseline_auc
            auc_pct = (auc_diff / baseline_auc * 100) if baseline_auc != 0 else 0
            auc_str = f"{auc_diff:+.4f} ({auc_pct:+.2f}%)"
        else:
            auc_str = "N/A"

        if baseline_f1 is not None and exp["f1"] is not None:
            f1_diff = exp["f1"] - baseline_f1
            f1_pct = (f1_diff / baseline_f1 * 100) if baseline_f1 != 0 else 0
            f1_str = f"{f1_diff:+.4f} ({f1_pct:+.2f}%)"
        else:
            f1_str = "N/A"

        if baseline_acc is not None and exp["accuracy"] is not None:
            acc_diff = exp["accuracy"] - baseline_acc
            acc_pct = (acc_diff / baseline_acc * 100) if baseline_acc != 0 else 0
            acc_str = f"{acc_diff:+.4f} ({acc_pct:+.2f}%)"
        else:
            acc_str = "N/A"

        print(f"{exp_name:<38} ΔAuc={auc_str:<20} ΔF1={f1_str:<15} ΔAcc={acc_str:<15}")

print("=" * 95)

PYEOF

echo ""

# ============================================================================
# Per-fold statistics (optional, for detailed analysis)
# ============================================================================
echo ""
echo "============================================================================"
echo "PER-FOLD STATISTICS (first 10 folds)"
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
            fold_metrics = data.get("fold_metrics", {})

        print(f"\n{exp_dir}:")
        print(f"  {'Fold':<8} {'AUC-ROC':<12} {'F1':<12} {'Accuracy':<12}")
        print(f"  {'-'*50}")

        for i, fold_id in enumerate(sorted(fold_metrics.keys())[:10]):
            metrics = fold_metrics[fold_id]
            auc = metrics.get("test_auc_roc", "N/A")
            f1 = metrics.get("test_f1", "N/A")
            acc = metrics.get("test_accuracy", "N/A")

            if isinstance(auc, (int, float)):
                auc = f"{auc:.4f}"
            if isinstance(f1, (int, float)):
                f1 = f"{f1:.4f}"
            if isinstance(acc, (int, float)):
                acc = f"{acc:.4f}"

            print(f"  {fold_id:<8} {auc:<12} {f1:<12} {acc:<12}")

    except Exception as e:
        print(f"\n{exp_dir}: ERROR - {e}")

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
