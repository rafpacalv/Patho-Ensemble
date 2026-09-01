#!/bin/bash
#SBATCH --job-name=opcion_d_weighted
#SBATCH --partition=main
#SBATCH --constraint=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=logs/opcion_d_weighted_early_fusion_%j.out
#SBATCH --error=logs/opcion_d_weighted_early_fusion_%j.err
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
EPOCHS=100

echo "============================================================================"
echo "OPCIÓN D: Weighted Early Fusion + Nelder-Mead (Pesos en Embeddings)"
echo "============================================================================"
echo "Dataset: $DATASET / Task: $TASK"
echo "Models: ctranspath, virchow_v1, conch_v1_5"
echo "Strategy: w_ct*emb_ct ⊕ w_vir*emb_vir ⊕ w_conch*emb_conch → ABMIL"
echo "Fecha: $(date)"
echo "============================================================================"
echo ""

# ============================================================================
# PILOTO: 5 folds
# ============================================================================

echo "============================================================================"
echo "PILOTO: Weighted Early Fusion on 5 folds"
echo "============================================================================"

python src/ensemble_weighted_early_fusion.py \
    --work_dir "$WORK_DIR" \
    --train_source "$DATASET" \
    --tissue_patching "$TISSUE_PATCHING" \
    --task_name "$TASK" \
    --foundational_models ctranspath virchow_v1 conch_v1_5 \
    --fold_start 0 \
    --fold_end 4 \
    --epochs "$EPOCHS"

echo ""
echo "✓ Piloto completado"
echo ""

# ============================================================================
# ESCALA COMPLETA: 50 folds
# ============================================================================

echo "============================================================================"
echo "ESCALA COMPLETA: Weighted Early Fusion on 50 folds"
echo "============================================================================"

python src/ensemble_weighted_early_fusion.py \
    --work_dir "$WORK_DIR" \
    --train_source "$DATASET" \
    --tissue_patching "$TISSUE_PATCHING" \
    --task_name "$TASK" \
    --foundational_models ctranspath virchow_v1 conch_v1_5 \
    --fold_start 0 \
    --fold_end 49 \
    --epochs "$EPOCHS"

echo ""
echo "✓ Escala completa completada"
echo ""

# ============================================================================
# COMPARACIÓN PAREADA: Opción D vs Opción A (Nelder-Mead baseline)
# ============================================================================

echo "============================================================================"
echo "Comparación pareada: Opción D (Weighted Early) vs Opción A (Nelder-Mead)"
echo "============================================================================"

cat > /tmp/compare_opcion_d.py << 'PYTHON_EOF'
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
        metrics_file = f"{metrics_dir}/test_metrics/fold_{fold_k}/metrics.json"
        if not os.path.exists(metrics_file):
            metrics_file = f"{metrics_dir}/fold_{fold_k}/metrics.json"

        if os.path.exists(metrics_file):
            with open(metrics_file) as f:
                metrics = json.load(f)
                auc = metrics["overall"]["macro-ovr-auc"]
                aucs[fold_k] = auc
    return aucs

# Load Opción D (Weighted Early Fusion)
opcion_d_dir = f"{ABMIL_DIR}/weighted_early_fusion_ctranspath_virchow_v1_conch_v1_5"
opcion_d_aucs = load_fold_metrics(opcion_d_dir)
print(f"✓ Loaded Opción D AUCs: {len(opcion_d_aucs)} folds")

# Load Opción A Baseline (Nelder-Mead Late Fusion)
opcion_a_dir = f"{ABMIL_DIR}/ensemble4_nelder_mead_ctranspath_virchow_v1_conch_v1_5"
opcion_a_aucs = load_fold_metrics(opcion_a_dir)
print(f"✓ Loaded Opción A AUCs: {len(opcion_a_aucs)} folds")

if len(opcion_d_aucs) == 0 or len(opcion_a_aucs) == 0:
    print("ERROR: No metrics found")
    exit(1)

print(f"\n{'='*80}")
print("COMPARACIÓN PAREADA: Opción D (Weighted Early) vs Opción A (Nelder-Mead)")
print(f"{'='*80}\n")

# Compute paired differences
paired_deltas = []
wins, ties, losses = 0, 0, 0

for fold_k in sorted(opcion_a_aucs.keys()):
    if fold_k in opcion_d_aucs:
        opcion_a_auc = opcion_a_aucs[fold_k]
        opcion_d_auc = opcion_d_aucs[fold_k]
        delta = opcion_d_auc - opcion_a_auc

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

        print(f"Fold {fold_k:2d}: OpA={opcion_a_auc:.4f}  OpD={opcion_d_auc:.4f}  Δ={delta:+.4f} {sign}")

paired_deltas = np.array(paired_deltas)
opcion_a_arr = np.array([opcion_a_aucs[k] for k in sorted(opcion_a_aucs.keys()) if k in opcion_d_aucs])
opcion_d_arr = np.array([opcion_d_aucs[k] for k in sorted(opcion_a_aucs.keys()) if k in opcion_d_aucs])

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
t_stat, p_value = sp_stats.ttest_rel(opcion_d_arr, opcion_a_arr)
print(f"Paired t-test: t={t_stat:.4f}, p={p_value:.4f}")

print(f"\n{'='*80}")
print("CONCLUSIÓN")
print(f"{'='*80}")
if ci_lower > 0:
    print(f"✅ MEJORA SIGNIFICATIVA: Opción D supera Opción A")
elif ci_upper < 0:
    print(f"❌ EMPEORAMIENTO SIGNIFICATIVO: Opción D peor que Opción A")
else:
    print(f"⚠️  SIN DIFERENCIA SIGNIFICATIVA")

print(f"Δ AUC: {paired_deltas.mean():+.4f} [{ci_lower:+.4f}, {ci_upper:+.4f}]")
print(f"Wins: {wins}/50")

# Save
result = {
    "method": "weighted_early_fusion_vs_nelder_mead",
    "mean_delta_auc": float(paired_deltas.mean()),
    "ci_lower": float(ci_lower),
    "ci_upper": float(ci_upper),
    "wins_ties_losses": [wins, ties, losses],
    "p_value": float(p_value),
    "n_folds": len(paired_deltas),
}

with open("results_opcion_d_weighted_early_fusion.json", "w") as f:
    json.dump(result, f, indent=2)

print(f"\n✓ Resultados guardados en results_opcion_d_weighted_early_fusion.json")
PYTHON_EOF

python /tmp/compare_opcion_d.py

echo ""
echo "============================================================================"
echo "TODO COMPLETADO"
echo "Fecha: $(date)"
echo "============================================================================"
echo ""
echo "Resultados disponibles:"
echo "  Opción D: ${WORK_DIR}/${DATASET}/${TASK}/abmil/weighted_early_fusion_ctranspath_virchow_v1_conch_v1_5/"
echo "  Comparación: results_opcion_d_weighted_early_fusion.json"
echo ""
