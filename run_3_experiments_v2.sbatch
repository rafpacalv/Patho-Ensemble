#!/bin/bash
#SBATCH --job-name=run_3_experiments_v2
#SBATCH --partition=main
#SBATCH --constraint=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=logs/run_3_experiments_v2_%j.out
#SBATCH --error=logs/run_3_experiments_v2_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=rafael.pachon.alvarez@gmail.com

# ============================================================================
# Run 4 Meta-learner Configurations: Baseline + 3 Experiments
# ============================================================================
# Exp 0 (Baseline RE-RUN with leakage fix):
#   - bases: standard (checkpoint único)
#   - meta: LogReg
#   - output: ensemble4/
#
# Exp 1: TabPFN on standard bases
#   - bases: standard
#   - meta: TabPFN
#   - output: ensemble4_tabpfn/
#
# Exp 2: TabPFN on FGE-diversified bases
#   - bases: fge (snapshots promediados)
#   - meta: TabPFN
#   - output: ensemble4_tabpfn_fge/
#
# Exp 3: LogReg on FGE-diversified bases
#   - bases: fge
#   - meta: LogReg
#   - output: ensemble4_fge/
#
# Prerequisites:
#   - ABMIL base training (run_3_experiments.sbatch OR train_base_models_fge.sbatch)
#   - test_abmil.py outputs in {model}_train_eval/val_outputs/fold_k/ (standard)
#   - test_abmil_fge.py outputs in {model}_train_eval_fge/val_outputs/fold_k/ (FGE)
#
# Estimated runtime: 3-4 hours
# ============================================================================

set -e

source $(conda info --base)/etc/profile.d/conda.sh
conda activate ensemble

PROJECT_DIR="/shared/home/JKP6679/Patho-Ensemble"
WORK_DIR="/shared/home/PARADIS/datos"
TRAIN_SOURCE="cptac_brca"
TISSUE_PATCHING="20x_224px_0px_overlap"
TASK_NAME="TP53_mutation"

MODELS="ctranspath uni_v2 virchow_v1"

mkdir -p "$PROJECT_DIR/logs"

echo "============================================================================"
echo "Running Meta-learner Experiments (4 configurations)"
echo "============================================================================"
echo "Start time: $(date)"
echo "Project dir: $PROJECT_DIR"
echo "Work dir: $WORK_DIR"
echo "Dataset: $TRAIN_SOURCE / $TASK_NAME"
echo "Models: $MODELS"
echo "============================================================================"
echo ""

cd "$PROJECT_DIR"

# ============================================================================
# Experiment 0: Baseline (LogReg, standard bases) — RE-RUN WITH LEAKAGE FIX
# ============================================================================
echo ""
echo "╔════════════════════════════════════════════════════════════════════════╗"
echo "║ Experiment 0: Baseline (LogReg + Standard Bases) — Leakage Fix"
echo "╚════════════════════════════════════════════════════════════════════════╝"
echo ""
echo "[$(date)] Running baseline ensemble (LogReg on standard bases)..."

python src/ensemble4.py \
    --foundational_models $MODELS \
    --work_dir "$WORK_DIR" \
    --train_source "$TRAIN_SOURCE" \
    --tissue_patching "$TISSUE_PATCHING" \
    --task_name "$TASK_NAME" \
    --meta_model logreg \
    --base_source standard \
    2>&1 | tee -a "logs/exp0_baseline.log"

echo "[$(date)] ✓ Baseline completed"
echo ""

# ============================================================================
# Experiment 1: TabPFN on Standard Bases
# ============================================================================
echo ""
echo "╔════════════════════════════════════════════════════════════════════════╗"
echo "║ Experiment 1: TabPFN + Standard Bases"
echo "╚════════════════════════════════════════════════════════════════════════╝"
echo ""
echo "[$(date)] Running TabPFN on standard bases..."

python src/ensemble4.py \
    --foundational_models $MODELS \
    --work_dir "$WORK_DIR" \
    --train_source "$TRAIN_SOURCE" \
    --tissue_patching "$TISSUE_PATCHING" \
    --task_name "$TASK_NAME" \
    --meta_model tabpfn \
    --base_source standard \
    2>&1 | tee -a "logs/exp1_tabpfn_standard.log"

echo "[$(date)] ✓ Experiment 1 completed"
echo ""

# ============================================================================
# Experiment 2: TabPFN on FGE-Diversified Bases
# ============================================================================
echo ""
echo "╔════════════════════════════════════════════════════════════════════════╗"
echo "║ Experiment 2: TabPFN + FGE-Diversified Bases"
echo "╚════════════════════════════════════════════════════════════════════════╝"
echo ""
echo "[$(date)] Running TabPFN on FGE-diversified bases..."

python src/ensemble4.py \
    --foundational_models $MODELS \
    --work_dir "$WORK_DIR" \
    --train_source "$TRAIN_SOURCE" \
    --tissue_patching "$TISSUE_PATCHING" \
    --task_name "$TASK_NAME" \
    --meta_model tabpfn \
    --base_source fge \
    2>&1 | tee -a "logs/exp2_tabpfn_fge.log"

echo "[$(date)] ✓ Experiment 2 completed"
echo ""

# ============================================================================
# Experiment 3: LogReg on FGE-Diversified Bases
# ============================================================================
echo ""
echo "╔════════════════════════════════════════════════════════════════════════╗"
echo "║ Experiment 3: LogReg + FGE-Diversified Bases"
echo "╚════════════════════════════════════════════════════════════════════════╝"
echo ""
echo "[$(date)] Running LogReg on FGE-diversified bases..."

python src/ensemble4.py \
    --foundational_models $MODELS \
    --work_dir "$WORK_DIR" \
    --train_source "$TRAIN_SOURCE" \
    --tissue_patching "$TISSUE_PATCHING" \
    --task_name "$TASK_NAME" \
    --meta_model logreg \
    --base_source fge \
    2>&1 | tee -a "logs/exp3_logreg_fge.log"

echo "[$(date)] ✓ Experiment 3 completed"
echo ""

# ============================================================================
# Comparison Table
# ============================================================================
echo ""
echo "╔════════════════════════════════════════════════════════════════════════╗"
echo "║ Results Summary: Comparing 4 Configurations"
echo "╚════════════════════════════════════════════════════════════════════════╝"
echo ""

python3 << 'PYEOF'
import json
import os
from pathlib import Path

WORK_DIR = "/shared/home/PARADIS/datos"
TRAIN_SOURCE = "cptac_brca"
TASK_NAME = "TP53_mutation"
ABMIL_DIR = Path(WORK_DIR) / TRAIN_SOURCE / TASK_NAME / "abmil"

configs = [
    ("Exp 0 (Baseline)", "ensemble4"),
    ("Exp 1 (TabPFN Std)", "ensemble4_tabpfn"),
    ("Exp 2 (TabPFN FGE)", "ensemble4_tabpfn_fge"),
    ("Exp 3 (LogReg FGE)", "ensemble4_fge"),
]

print("\n{:<25} {:<12} {:<12} {:<12}".format("Configuration", "AUC", "F1", "Accuracy"))
print("-" * 65)

for name, subdir in configs:
    metrics_file = ABMIL_DIR / subdir / "test_metrics_summary.json"
    if metrics_file.exists():
        try:
            with open(metrics_file) as f:
                metrics = json.load(f)

            auc = metrics.get("macro-ovr-auc", {}).get("mean", "N/A")
            f1 = metrics.get("macro-f1", {}).get("mean", "N/A")
            acc = metrics.get("acc", {}).get("mean", "N/A")

            if isinstance(auc, (int, float)):
                auc = f"{auc:.4f}"
            if isinstance(f1, (int, float)):
                f1 = f"{f1:.4f}"
            if isinstance(acc, (int, float)):
                acc = f"{acc:.4f}"

            print("{:<25} {:<12} {:<12} {:<12}".format(name, auc, f1, acc))
        except Exception as e:
            print("{:<25} {:<12}".format(name, f"Error: {e}"))
    else:
        print("{:<25} {:<12}".format(name, "Pending"))

print("\nNote: Baseline AUC before fix was 0.744; re-run validates the fix.")
print("TabPFN and FGE are new — these are first-run results.")

PYEOF

echo ""
echo "============================================================================"
echo "EXECUTION SUMMARY"
echo "============================================================================"
echo ""
echo "Completed configurations:"
echo "  - Exp 0: Baseline LogReg + standard bases (leakage fix validation)"
echo "  - Exp 1: TabPFN + standard bases"
echo "  - Exp 2: TabPFN + FGE-diversified bases"
echo "  - Exp 3: LogReg + FGE-diversified bases"
echo ""
echo "Results saved in:"
echo "  - ABMIL_DIR/ensemble4/"
echo "  - ABMIL_DIR/ensemble4_tabpfn/"
echo "  - ABMIL_DIR/ensemble4_tabpfn_fge/"
echo "  - ABMIL_DIR/ensemble4_fge/"
echo ""
echo "Log files:"
echo "  - logs/run_3_experiments_v2_${SLURM_JOB_ID}.out (this file)"
echo "  - logs/exp0_baseline.log"
echo "  - logs/exp1_tabpfn_standard.log"
echo "  - logs/exp2_tabpfn_fge.log"
echo "  - logs/exp3_logreg_fge.log"
echo ""
echo "============================================================================"
echo "End time: $(date)"
echo "✓ Meta-learner experiments completed!"
echo "============================================================================"
