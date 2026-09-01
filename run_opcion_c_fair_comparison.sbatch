#!/bin/bash
#SBATCH --job-name=opcion_c_fair_comparison
#SBATCH --partition=main
#SBATCH --constraint=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=logs/opcion_c_fair_comparison_%j.out
#SBATCH --error=logs/opcion_c_fair_comparison_%j.err
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

# ============================================================================
# OPCIÓN C: FAIR COMPARISON — Late Fusion + Meta-learner vs Early Fusion + Meta-learner
# ============================================================================
# Modelos: ctranspath, virchow_v1, conch_v1_5 (todos con 387 slides, coordenadas alineadas)
# ============================================================================

echo "============================================================================"
echo "OPCIÓN C — FAIR COMPARISON: Late vs Early Fusion + Meta-learner"
echo "============================================================================"
echo "Fecha: $(date)"
echo "Dataset: $DATASET / Task: $TASK"
echo "Modelos: ctranspath, virchow_v1, conch_v1_5"
echo "============================================================================"
echo ""

# ============================================================================
# FASE 0: ENTRENAR CONCH_V1_5 (si no existe)
# ============================================================================

CONCH_DIR="${WORK_DIR}/${DATASET}/${TASK}/abmil/conch_v1_5_${TISSUE_PATCHING}"

if [ ! -d "$CONCH_DIR/checkpoints/fold_0" ]; then
    echo "============================================================================"
    echo "FASE 0: Entrenando conch_v1_5 (50 folds)"
    echo "============================================================================"
    
    python src/train_abmil.py \
        --foundational_model conch_v1_5 \
        --work_dir "$WORK_DIR" \
        --train_source "$DATASET" \
        --tissue_patching "$TISSUE_PATCHING" \
        --task_name "$TASK" \
        --epochs "$EPOCHS" \
        --fold all
    
    echo "✓ conch_v1_5 training complete"
    echo ""
else
    echo "⚠ conch_v1_5 already trained, skipping"
    echo ""
fi

# ============================================================================
# FASE 1: GENERAR META-FEATURES (in-sample) PARA LATE FUSION BASELINE
# ============================================================================

echo "============================================================================"
echo "FASE 1: Generar meta-features para Late Fusion Baseline"
echo "============================================================================"

for model in ctranspath virchow_v1 conch_v1_5; do
    echo ""
    echo "  Generando meta-features para $model..."
    
    python src/test_abmil.py \
        --foundational_model "$model" \
        --work_dir "$WORK_DIR" \
        --train_source "$DATASET" \
        --tissue_patching "$TISSUE_PATCHING" \
        --task_name "$TASK" \
        --epochs "$EPOCHS"
    
    echo "  ✓ $model meta-features done"
done

echo ""
echo "✓ All meta-features for Late Fusion generated"
echo ""

# ============================================================================
# FASE 2: ENTRENAR LATE FUSION + LOGISTIC REGRESSION (Baseline)
# ============================================================================

echo "============================================================================"
echo "FASE 2: Entrenando Late Fusion + LogisticRegression (Baseline)"
echo "============================================================================"

python src/ensemble4.py \
    --foundational_models ctranspath virchow_v1 conch_v1_5 \
    --work_dir "$WORK_DIR" \
    --train_source "$DATASET" \
    --tissue_patching "$TISSUE_PATCHING" \
    --task_name "$TASK" \
    --meta_model logreg \
    --results_subdir_suffix "_opcion_c_late_fusion_baseline"

echo ""
echo "✓ Late Fusion + LogReg training complete"
echo ""

# ============================================================================
# FASE 3: ENTRENAR EARLY FUSION (3 modelos concatenados)
# ============================================================================

echo "============================================================================"
echo "FASE 3: Entrenando Early Fusion (ctranspath ⊕ virchow_v1 ⊕ conch_v1_5)"
echo "============================================================================"

python src/ensemble5.py \
    --work_dir "$WORK_DIR" \
    --train_source "$DATASET" \
    --tissue_patching "$TISSUE_PATCHING" \
    --task_name "$TASK" \
    --fuse_models ctranspath virchow_v1 conch_v1_5 \
    --fold_start 0 \
    --fold_end 49 \
    --epochs "$EPOCHS"

echo ""
echo "✓ Early Fusion training complete"
echo ""

# ============================================================================
# FASE 4: GENERAR META-FEATURES PARA EARLY FUSION
# ============================================================================

echo "============================================================================"
echo "FASE 4: Generar meta-features para Early Fusion"
echo "============================================================================"

FUSED_MODEL="ctranspath_virchow_v1_conch_v1_5_fused"

# Crear symlink temporal para que test_abmil.py lo encuentre
FUSED_DIR="${WORK_DIR}/${DATASET}/${TASK}/abmil/${FUSED_MODEL}_${TISSUE_PATCHING}"

# Ajuste: usamos ensemble5.py que ya generó las meta-features
# Pero si necesitamos usar test_abmil.py, primero registramos el modelo

# Por ahora asumimos que ensemble5.py ya generó los directorios
# Si no, descomentar:
# python src/test_abmil.py --foundational_model "$FUSED_MODEL" ...

if [ -d "${FUSED_DIR}_train_eval/val_outputs" ]; then
    echo "✓ Early Fusion meta-features already generated by ensemble5.py"
else
    echo "⚠ Early Fusion meta-features not found, may need manual step"
fi

echo ""

# ============================================================================
# FASE 5: ENTRENAR EARLY FUSION + LOGISTIC REGRESSION (Hybrid)
# ============================================================================

echo "============================================================================"
echo "FASE 5: Entrenando Early Fusion + LogisticRegression (Hybrid)"
echo "============================================================================"

python src/ensemble_single_model.py \
    --work_dir "$WORK_DIR" \
    --train_source "$DATASET" \
    --tissue_patching "$TISSUE_PATCHING" \
    --task_name "$TASK" \
    --foundational_model "$FUSED_MODEL" \
    --meta_model logreg

echo ""
echo "✓ Early Fusion + LogReg training complete"
echo ""

# ============================================================================
# FASE 6: COMPARACIÓN PAREADA
# ============================================================================

echo "============================================================================"
echo "FASE 6: Comparación Pareada Late Fusion vs Early Fusion"
echo "============================================================================"

cat > /tmp/compare_opcion_c.py << 'PYTHON_EOF'
import json
import numpy as np
from sklearn.metrics import roc_auc_score
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
        # Try test_metrics/fold_k first (new structure), then fold_k (old structure)
        metrics_file = f"{metrics_dir}/test_metrics/fold_{fold_k}/metrics.json"
        if not os.path.exists(metrics_file):
            metrics_file = f"{metrics_dir}/fold_{fold_k}/metrics.json"

        if os.path.exists(metrics_file):
            with open(metrics_file) as f:
                metrics = json.load(f)
                auc = metrics["overall"]["macro-ovr-auc"]
                aucs[fold_k] = auc
    return aucs

# Load Late Fusion baseline
late_fusion_dir = f"{ABMIL_DIR}/ensemble4_opcion_c_late_fusion_baseline"
late_fusion_aucs = load_fold_metrics(late_fusion_dir)
print(f"✓ Loaded Late Fusion AUCs: {len(late_fusion_aucs)} folds")

# Load Early Fusion hybrid (fused model + logreg meta-learner)
fused_model_name = "ctranspath_virchow_v1_conch_v1_5_fused"
early_fusion_dir = f"{ABMIL_DIR}/ensemble4_{fused_model_name}_logreg"
early_fusion_aucs = load_fold_metrics(early_fusion_dir)
print(f"✓ Loaded Early Fusion AUCs: {len(early_fusion_aucs)} folds")

if len(late_fusion_aucs) == 0 or len(early_fusion_aucs) == 0:
    print("ERROR: No metrics found")
    exit(1)

print(f"\n{'='*80}")
print("COMPARACIÓN PAREADA: Early Fusion + LogReg vs Late Fusion + LogReg")
print(f"{'='*80}\n")

# Compute paired differences
paired_deltas = []
wins, ties, losses = 0, 0, 0

for fold_k in sorted(late_fusion_aucs.keys()):
    if fold_k in early_fusion_aucs:
        late_auc = late_fusion_aucs[fold_k]
        early_auc = early_fusion_aucs[fold_k]
        delta = early_auc - late_auc
        
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
        
        print(f"Fold {fold_k:2d}: Late={late_auc:.4f}  Early={early_auc:.4f}  Δ={delta:+.4f} {sign}")

paired_deltas = np.array(paired_deltas)
late_arr = np.array([late_fusion_aucs[k] for k in sorted(late_fusion_aucs.keys()) if k in early_fusion_aucs])
early_arr = np.array([early_fusion_aucs[k] for k in sorted(late_fusion_aucs.keys()) if k in early_fusion_aucs])

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
t_stat, p_value = sp_stats.ttest_rel(early_arr, late_arr)
print(f"Paired t-test: t={t_stat:.4f}, p={p_value:.4f}")

print(f"\n{'='*80}")
print("CONCLUSIÓN")
print(f"{'='*80}")
if ci_lower > 0:
    print(f"✅ MEJORA SIGNIFICATIVA: Early Fusion supera Late Fusion")
elif ci_upper < 0:
    print(f"❌ EMPEORAMIENTO SIGNIFICATIVO: Early Fusion peor que Late Fusion")
else:
    print(f"⚠️  SIN DIFERENCIA SIGNIFICATIVA")
    
print(f"Δ AUC: {paired_deltas.mean():+.4f} [{ci_lower:+.4f}, {ci_upper:+.4f}]")
print(f"Wins: {wins}/50")

# Save
result = {
    "method": "early_fusion_3models_vs_late_fusion_3models",
    "mean_delta_auc": float(paired_deltas.mean()),
    "ci_lower": float(ci_lower),
    "ci_upper": float(ci_upper),
    "wins_ties_losses": [wins, ties, losses],
    "p_value": float(p_value),
    "n_folds": len(paired_deltas),
}

with open("results_opcion_c_fair_comparison.json", "w") as f:
    json.dump(result, f, indent=2)

print(f"\n✓ Resultados guardados en results_opcion_c_fair_comparison.json")
PYTHON_EOF

python /tmp/compare_opcion_c.py

echo ""
echo "============================================================================"
echo "TODAS LAS FASES COMPLETADAS"
echo "Fecha: $(date)"
echo "============================================================================"
echo ""
echo "Resultados disponibles:"
echo "  Late Fusion + LogReg: ${WORK_DIR}/${DATASET}/${TASK}/abmil/ensemble4_opcion_c_late_fusion_baseline/"
echo "  Early Fusion + LogReg: ${WORK_DIR}/${DATASET}/${TASK}/abmil/ensemble4_ctranspath_virchow_v1_conch_v1_5_fused_logreg/"
echo "  Comparación pareada: results_opcion_c_fair_comparison.json"
echo ""
