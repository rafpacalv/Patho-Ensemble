#!/bin/bash
#SBATCH --job-name=opcion_a_nelder_mead
#SBATCH --partition=main
#SBATCH --constraint=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --output=logs/opcion_a_nelder_mead_%j.out
#SBATCH --error=logs/opcion_a_nelder_mead_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=rafael.pachon.alvarez@gmail.com

set -euo pipefail

source $(conda info --base)/etc/profile.d/conda.sh
conda activate ensemble

PROJECT_DIR="/home/JKP6679/Patho-Ensemble"
cd "$PROJECT_DIR"

DATASET="cptac_brca"
TASK="TP53_mutation"
WORK_DIR="/home/JKP6679/Patho-Ensemble/PARADIS/datos/patches"
TISSUE_PATCHING="20x_224px_0px_overlap"

echo "============================================================================"
echo "OPCIÓN A: Late Fusion + Nelder-Mead Weight Optimization"
echo "============================================================================"
echo "Dataset: $DATASET / Task: $TASK"
echo "Models: ctranspath, virchow_v1, conch_v1_5"
echo "Fecha: $(date)"
echo "============================================================================"
echo ""

# ============================================================================
# PILOTO: 5 folds
# ============================================================================

echo "============================================================================"
echo "PILOTO: Nelder-Mead optimization on 5 folds"
echo "============================================================================"

python src/ensemble_nelder_mead.py \
    --work_dir "$WORK_DIR" \
    --train_source "$DATASET" \
    --tissue_patching "$TISSUE_PATCHING" \
    --task_name "$TASK" \
    --foundational_models ctranspath virchow_v1 conch_v1_5 \
    --fold_start 0 \
    --fold_end 4

echo ""
echo "✓ Piloto completado"
echo ""

# ============================================================================
# ESCALA COMPLETA: 50 folds
# ============================================================================

echo "============================================================================"
echo "ESCALA COMPLETA: Nelder-Mead optimization on 50 folds"
echo "============================================================================"

python src/ensemble_nelder_mead.py \
    --work_dir "$WORK_DIR" \
    --train_source "$DATASET" \
    --tissue_patching "$TISSUE_PATCHING" \
    --task_name "$TASK" \
    --foundational_models ctranspath virchow_v1 conch_v1_5 \
    --fold_start 0 \
    --fold_end 49

echo ""
echo "✓ Escala completa completada"
echo ""

# ============================================================================
# COMPARACIÓN PAREADA: Nelder-Mead vs LogReg Baseline
# ============================================================================

echo "============================================================================"
echo "Comparación pareada: Nelder-Mead vs LogReg Baseline"
echo "============================================================================"

cat > /tmp/compare_opcion_a.py << 'PYTHON_EOF'
import json
import numpy as np
from scipy import stats as sp_stats
import os

WORK_DIR = "/home/JKP6679/Patho-Ensemble/PARADIS/datos/patches"
DATASET = "cptac_brca"
TASK = "TP53_mutation"
ABMIL_DIR = f"{WORK_DIR}/{DATASET}/{TASK}/abmil"

def load_fold_metrics(metrics_dir):
    """Load AUC from metrics.json files."""
    aucs = {}
    for fold_k in range(50):
        # Try test_metrics/fold_k first
        metrics_file = f"{metrics_dir}/test_metrics/fold_{fold_k}/metrics.json"
        if not os.path.exists(metrics_file):
            metrics_file = f"{metrics_dir}/fold_{fold_k}/metrics.json"

        if os.path.exists(metrics_file):
            with open(metrics_file) as f:
                metrics = json.load(f)
                auc = metrics["overall"]["macro-ovr-auc"]
                aucs[fold_k] = auc
    return aucs

# Load Nelder-Mead
nelder_mead_dir = f"{ABMIL_DIR}/ensemble4_nelder_mead_ctranspath_virchow_v1_conch_v1_5"
nelder_mead_aucs = load_fold_metrics(nelder_mead_dir)
print(f"✓ Loaded Nelder-Mead AUCs: {len(nelder_mead_aucs)} folds")

# Load LogReg Baseline (Opción C)
logreg_dir = f"{ABMIL_DIR}/ensemble4_opcion_c_late_fusion_baseline"
logreg_aucs = load_fold_metrics(logreg_dir)
print(f"✓ Loaded LogReg Baseline AUCs: {len(logreg_aucs)} folds")

if len(nelder_mead_aucs) == 0 or len(logreg_aucs) == 0:
    print("ERROR: No metrics found")
    exit(1)

print(f"\n{'='*80}")
print("COMPARACIÓN PAREADA: Nelder-Mead vs LogReg Baseline")
print(f"{'='*80}\n")

# Compute paired differences
paired_deltas = []
wins, ties, losses = 0, 0, 0

for fold_k in sorted(logreg_aucs.keys()):
    if fold_k in nelder_mead_aucs:
        logreg_auc = logreg_aucs[fold_k]
        nm_auc = nelder_mead_aucs[fold_k]
        delta = nm_auc - logreg_auc

        paired_deltas.append(delta)

        if delta > 0.001:
            wins += 1
            sign = "✓"
        elif delta < -0.001:
            losses += 1
            sign = "✗"
        else:
            ties += 1
            sign = "="

        print(f"Fold {fold_k:2d}: LogReg={logreg_auc:.4f}  Nelder-Mead={nm_auc:.4f}  Δ={delta:+.4f} {sign}")

paired_deltas = np.array(paired_deltas)
logreg_arr = np.array([logreg_aucs[k] for k in sorted(logreg_aucs.keys()) if k in nelder_mead_aucs])
nm_arr = np.array([nelder_mead_aucs[k] for k in sorted(logreg_aucs.keys()) if k in nelder_mead_aucs])

print(f"\n{'='*80}")
print("ESTADÍSTICA")
print(f"{'='*80}")
print(f"N folds: {len(paired_deltas)}")
print(f"Wins/Ties/Losses: {wins}/{ties}/{losses}")
print(f"Mean Δ AUC: {paired_deltas.mean():+.4f}")
print(f"Median Δ AUC: {np.median(paired_deltas):+.4f}")
print(f"Std Δ AUC: {paired_deltas.std():.4f}")

# Bootstrap CI
np.random.seed(42)
boot_means = []
for _ in range(1000):
    boot = np.random.choice(paired_deltas, len(paired_deltas), replace=True)
    boot_means.append(boot.mean())

boot_means = np.array(boot_means)
ci_lower = np.percentile(boot_means, 2.5)
ci_upper = np.percentile(boot_means, 97.5)

print(f"Bootstrap 95% CI: [{ci_lower:+.4f}, {ci_upper:+.4f}]")

# Paired t-test
t_stat, p_value = sp_stats.ttest_rel(nm_arr, logreg_arr)
print(f"Paired t-test: t={t_stat:.4f}, p={p_value:.4f}")

print(f"\n{'='*80}")
print("CONCLUSIÓN")
print(f"{'='*80}")
if ci_lower > 0:
    print(f"✅ MEJORA SIGNIFICATIVA: Nelder-Mead supera LogReg")
elif ci_upper < 0:
    print(f"❌ EMPEORAMIENTO SIGNIFICATIVO: Nelder-Mead peor que LogReg")
else:
    print(f"⚠️  SIN DIFERENCIA SIGNIFICATIVA")

print(f"Δ AUC: {paired_deltas.mean():+.4f} [{ci_lower:+.4f}, {ci_upper:+.4f}]")
print(f"Wins: {wins}/50")

# Save
result = {
    "method": "nelder_mead_vs_logreg",
    "mean_delta_auc": float(paired_deltas.mean()),
    "ci_lower": float(ci_lower),
    "ci_upper": float(ci_upper),
    "wins_ties_losses": [wins, ties, losses],
    "p_value": float(p_value),
    "n_folds": len(paired_deltas),
}

with open("results_opcion_a_nelder_mead.json", "w") as f:
    json.dump(result, f, indent=2)

print(f"\n✓ Resultados guardados en results_opcion_a_nelder_mead.json")
PYTHON_EOF

python /tmp/compare_opcion_a.py

echo ""
echo "============================================================================"
echo "TODO COMPLETADO"
echo "Fecha: $(date)"
echo "============================================================================"
echo ""
echo "Resultados disponibles:"
echo "  Nelder-Mead: ${WORK_DIR}/${DATASET}/${TASK}/abmil/ensemble4_nelder_mead_ctranspath_virchow_v1_conch_v1_5/"
echo "  Comparación: results_opcion_a_nelder_mead.json"
echo ""
