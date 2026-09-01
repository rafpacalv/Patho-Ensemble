#!/bin/bash
#SBATCH --job-name=opcion_b_patient
#SBATCH --partition=main
#SBATCH --constraint=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=logs/opcion_b_per_patient_%j.out
#SBATCH --error=logs/opcion_b_per_patient_%j.err
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
echo "OPCIÓN B: Per-Patient Aggregation"
echo "============================================================================"
echo "Dataset: $DATASET / Task: $TASK"
echo "Models: ctranspath, uni_v2, virchow_v1"
echo "Strategy: Aggregate per-slide predictions to per-patient level"
echo "Fecha: $(date)"
echo "============================================================================"
echo ""

# ============================================================================
# PASO 1: PILOTO - 5 folds
# ============================================================================

echo "============================================================================"
echo "PASO 1: PILOTO - Per-Patient Aggregation (5 folds)"
echo "============================================================================"

python src/aggregate_to_patient.py \
    --work_dir "$WORK_DIR" \
    --train_source "$DATASET" \
    --tissue_patching "$TISSUE_PATCHING" \
    --task_name "$TASK" \
    --foundational_models ctranspath uni_v2 virchow_v1 \
    --fold_start 0 \
    --fold_end 4

echo ""
echo "✓ Piloto completado"
echo ""

# ============================================================================
# PASO 2: ESCALA COMPLETA - 50 folds
# ============================================================================

echo "============================================================================"
echo "PASO 2: ESCALA COMPLETA - Per-Patient Aggregation (50 folds)"
echo "============================================================================"

python src/aggregate_to_patient.py \
    --work_dir "$WORK_DIR" \
    --train_source "$DATASET" \
    --tissue_patching "$TISSUE_PATCHING" \
    --task_name "$TASK" \
    --foundational_models ctranspath uni_v2 virchow_v1 \
    --fold_start 0 \
    --fold_end 49

echo ""
echo "✓ Escala completa completada"
echo ""

# ============================================================================
# PASO 3: META-LEARNER - Patient Level (LogReg)
# ============================================================================

echo "============================================================================"
echo "PASO 3: Meta-Learner Training (Patient-Level)"
echo "============================================================================"

python src/ensemble4.py \
    --foundational_models ctranspath uni_v2 virchow_v1 \
    --work_dir "$WORK_DIR" \
    --train_source "$DATASET" \
    --tissue_patching "$TISSUE_PATCHING" \
    --task_name "$TASK" \
    --meta_model logreg \
    --patient_level

echo ""
echo "✓ Meta-learner training complete (patient-level)"
echo ""

# ============================================================================
# PASO 4: COMPARACIÓN PAREADA - Patient vs Slide Level
# ============================================================================

echo "============================================================================"
echo "PASO 4: Comparación Pareada - Patient Level vs Slide Level"
echo "============================================================================"

cat > /tmp/compare_patient_level.py << 'PYTHON_EOF'
import json
import numpy as np
from scipy import stats as sp_stats
import os

WORK_DIR = "/home/JKP6679/Patho-Ensemble/PARADIS/datos/patches"
DATASET = "cptac_brca"
TASK = "TP53_mutation"
ABMIL_DIR = f"{WORK_DIR}/{DATASET}/{TASK}/abmil"

def load_fold_metrics(metrics_dir, level="slide"):
    """Load AUC from metrics.json files."""
    aucs = {}
    for fold_k in range(50):
        if level == "patient":
            metrics_file = f"{metrics_dir}/test_metrics/fold_{fold_k}/metrics.json"
        else:
            metrics_file = f"{metrics_dir}/test_metrics/fold_{fold_k}/metrics.json"

        if os.path.exists(metrics_file):
            with open(metrics_file) as f:
                metrics = json.load(f)
                auc = metrics["overall"]["macro-ovr-auc"]
                aucs[fold_k] = auc
    return aucs

# Load Slide-Level (Baseline - LogReg on slides)
slide_dir = f"{ABMIL_DIR}/ensemble4"
slide_aucs = load_fold_metrics(slide_dir, level="slide")
print(f"✓ Loaded Slide-Level AUCs: {len(slide_aucs)} folds")

# Load Patient-Level (Opción B - LogReg on patients)
patient_dir = f"{ABMIL_DIR}/ensemble4_patient"
patient_aucs = load_fold_metrics(patient_dir, level="patient")
print(f"✓ Loaded Patient-Level AUCs: {len(patient_aucs)} folds")

if len(slide_aucs) == 0 or len(patient_aucs) == 0:
    print("ERROR: No metrics found")
    print(f"  Slide-level dir: {slide_dir}")
    print(f"  Patient-level dir: {patient_dir}")

    # Try alternative naming
    alt_patient_dir = f"{ABMIL_DIR}/ensemble4_patient_logreg"
    if os.path.exists(alt_patient_dir):
        print(f"  Found alternative: {alt_patient_dir}")
        patient_aucs = load_fold_metrics(alt_patient_dir, level="patient")
        print(f"  Loaded: {len(patient_aucs)} folds")

    if len(slide_aucs) == 0 or len(patient_aucs) == 0:
        exit(1)

print(f"\n{'='*80}")
print("COMPARACIÓN PAREADA: Patient-Level vs Slide-Level (LogReg)")
print(f"{'='*80}\n")

# Compute paired differences
paired_deltas = []
wins, ties, losses = 0, 0, 0

for fold_k in sorted(slide_aucs.keys()):
    if fold_k in patient_aucs:
        slide_auc = slide_aucs[fold_k]
        patient_auc = patient_aucs[fold_k]
        delta = patient_auc - slide_auc

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

        print(f"Fold {fold_k:2d}: Slide={slide_auc:.4f}  Patient={patient_auc:.4f}  Δ={delta:+.4f} {sign}")

paired_deltas = np.array(paired_deltas)
slide_arr = np.array([slide_aucs[k] for k in sorted(slide_aucs.keys()) if k in patient_aucs])
patient_arr = np.array([patient_aucs[k] for k in sorted(slide_aucs.keys()) if k in patient_aucs])

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
t_stat, p_value = sp_stats.ttest_rel(patient_arr, slide_arr)
print(f"Paired t-test: t={t_stat:.4f}, p={p_value:.4f}")

print(f"\n{'='*80}")
print("CONCLUSIÓN")
print(f"{'='*80}")
if ci_lower > 0:
    print(f"✅ MEJORA SIGNIFICATIVA: Patient-level supera Slide-level")
elif ci_upper < 0:
    print(f"❌ EMPEORAMIENTO SIGNIFICATIVO: Patient-level peor que Slide-level")
else:
    print(f"⚠️  SIN DIFERENCIA SIGNIFICATIVA")

print(f"Δ AUC: {paired_deltas.mean():+.4f} [{ci_lower:+.4f}, {ci_upper:+.4f}]")
print(f"Wins: {wins}/50")

# Save
result = {
    "method": "patient_level_vs_slide_level",
    "mean_delta_auc": float(paired_deltas.mean()),
    "ci_lower": float(ci_lower),
    "ci_upper": float(ci_upper),
    "wins_ties_losses": [wins, ties, losses],
    "p_value": float(p_value),
    "n_folds": len(paired_deltas),
}

with open("results_opcion_b_patient_level.json", "w") as f:
    json.dump(result, f, indent=2)

print(f"\n✓ Resultados guardados en results_opcion_b_patient_level.json")
PYTHON_EOF

python /tmp/compare_patient_level.py

echo ""
echo "============================================================================"
echo "TODO COMPLETADO"
echo "Fecha: $(date)"
echo "============================================================================"
echo ""
echo "Resultados disponibles:"
echo "  Patient-Level: ${WORK_DIR}/${DATASET}/${TASK}/abmil/ensemble4_patient/"
echo "  Comparación: results_opcion_b_patient_level.json"
echo ""
