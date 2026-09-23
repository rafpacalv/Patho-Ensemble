#!/bin/bash
#SBATCH --job-name=opcion_b_cmaes
#SBATCH --partition=main
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=04:00:00
#SBATCH --output=logs/opcion_b_cmaes_%j.out
#SBATCH --error=logs/opcion_b_cmaes_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=rafael.pachon.alvarez@gmail.com

# ============================================================================
# OPCIÓN B: Late Fusion + CMA-ES Weight Optimization
# ============================================================================
# Mismo problema que run_opcion_a_nelder_mead.sbatch (pesos de combinación
# lineal, optimizados en train_eval, evaluados en test), pero con CMA-ES en
# vez de Nelder-Mead — el optimizador que usa el post-hoc ensembling de
# Auto-Sklearn 2.0 para librerías de modelos (Purucker & Beel 2023,
# arXiv:2307.00286). Sin GPU: es optimización numérica sobre predicciones ya
# guardadas, igual que Nelder-Mead (que pedía GPU sin necesitarla).
#
# Se ejecuta sobre los dos tríos ya comparados en el proyecto, para que la
# tabla final sea una comparación pareada limpia optimizador-por-optimizador:
#   - trío ganador (ctranspath+uni_v2+conch_v1_5): logreg=0.7986, NM=0.7978
#   - trío histórico (ctranspath+virchow_v1+conch_v1_5): logreg=?, NM=0.8020
# ============================================================================

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
echo "OPCIÓN B: Late Fusion + CMA-ES Weight Optimization"
echo "Fecha: $(date)"
echo "============================================================================"

for MODELS in "ctranspath uni_v2 conch_v1_5" "ctranspath virchow_v1 conch_v1_5"; do
    TAG=$(echo "$MODELS" | tr ' ' '_')
    echo ""
    echo "============================================================================"
    echo "Modelos: $MODELS"
    echo "============================================================================"

    echo "--- piloto: 5 folds ---"
    python src/ensemble_cmaes.py \
        --work_dir "$WORK_DIR" \
        --train_source "$DATASET" \
        --tissue_patching "$TISSUE_PATCHING" \
        --task_name "$TASK" \
        --foundational_models $MODELS \
        --fold_start 0 \
        --fold_end 4

    echo "--- escala completa: 50 folds ---"
    python src/ensemble_cmaes.py \
        --work_dir "$WORK_DIR" \
        --train_source "$DATASET" \
        --tissue_patching "$TISSUE_PATCHING" \
        --task_name "$TASK" \
        --foundational_models $MODELS \
        --fold_start 0 \
        --fold_end 49

    echo "✓ $MODELS completado"
done

# ============================================================================
# COMPARACIÓN PAREADA: CMA-ES vs Nelder-Mead vs LogReg, en ambos tríos
# ============================================================================

echo ""
echo "============================================================================"
echo "Comparación pareada: CMA-ES vs Nelder-Mead vs LogReg"
echo "============================================================================"

cat > /tmp/compare_opcion_b.py << 'PYTHON_EOF'
import json
import os
import numpy as np
from scipy import stats as sp_stats

WORK_DIR = "/home/JKP6679/Patho-Ensemble/PARADIS/datos/patches"
DATASET = "cptac_brca"
TASK = "TP53_mutation"
ABMIL_DIR = f"{WORK_DIR}/{DATASET}/{TASK}/abmil"
N_FOLDS = 50


def load_fold_aucs(metrics_dir):
    aucs = {}
    for fold_k in range(N_FOLDS):
        metrics_file = f"{metrics_dir}/test_metrics/fold_{fold_k}/metrics.json"
        if not os.path.exists(metrics_file):
            metrics_file = f"{metrics_dir}/fold_{fold_k}/metrics.json"
        if os.path.exists(metrics_file):
            with open(metrics_file) as f:
                aucs[fold_k] = json.load(f)["overall"]["macro-ovr-auc"]
    return aucs


def paired_stats(a, b, name_a, name_b):
    common = sorted(set(a) & set(b))
    da = np.array([a[k] for k in common])
    db = np.array([b[k] for k in common])
    delta = da - db
    wins = int((delta > 0.001).sum())
    losses = int((delta < -0.001).sum())
    ties = len(delta) - wins - losses

    rng = np.random.default_rng(42)
    boot = np.array([rng.choice(delta, len(delta), replace=True).mean() for _ in range(1000)])
    ci_lo, ci_hi = np.percentile(boot, [2.5, 97.5])
    p = sp_stats.wilcoxon(delta).pvalue if not np.allclose(delta, 0) else 1.0

    print(f"\n{name_a} vs {name_b}  (n={len(common)} folds comunes)")
    print(f"  {name_a}: {da.mean():.4f}   {name_b}: {db.mean():.4f}")
    print(f"  Δ = {delta.mean():+.4f}  IC95%=[{ci_lo:+.4f},{ci_hi:+.4f}]  W/T/L={wins}/{ties}/{losses}  p={p:.4f}")
    return dict(a=name_a, b=name_b, n=len(common), mean_a=float(da.mean()), mean_b=float(db.mean()),
                mean_delta=float(delta.mean()), ci_95=[float(ci_lo), float(ci_hi)],
                wins=wins, ties=ties, losses=losses, p_value=float(p))


trios = {
    "ganador": ("ctranspath_uni_v2_conch_v1_5", "ensemble4_swap_conch"),
    "historico": ("ctranspath_virchow_v1_conch_v1_5", "ensemble4_opcion_c_late_fusion_baseline"),
}

results = {}
for trio_name, (tag, logreg_dir) in trios.items():
    print(f"\n{'='*80}\nTrío {trio_name} ({tag.replace('_', '+')})\n{'='*80}")
    cmaes = load_fold_aucs(f"{ABMIL_DIR}/ensemble4_cmaes_{tag}")
    nm = load_fold_aucs(f"{ABMIL_DIR}/ensemble4_nelder_mead_{tag}")
    logreg = load_fold_aucs(f"{ABMIL_DIR}/{logreg_dir}")
    print(f"cargados: cmaes={len(cmaes)}, nelder_mead={len(nm)}, logreg={len(logreg)} folds")

    results[trio_name] = dict(
        cmaes_vs_logreg=paired_stats(cmaes, logreg, "cmaes", "logreg") if cmaes and logreg else None,
        cmaes_vs_nelder_mead=paired_stats(cmaes, nm, "cmaes", "nelder_mead") if cmaes and nm else None,
        nelder_mead_vs_logreg=paired_stats(nm, logreg, "nelder_mead", "logreg") if nm and logreg else None,
    )

with open("results_opcion_b_cmaes.json", "w") as f:
    json.dump(results, f, indent=2)

print(f"\n{'='*80}")
print("✓ Resultados guardados en results_opcion_b_cmaes.json")
print(f"{'='*80}")
PYTHON_EOF

python /tmp/compare_opcion_b.py

echo ""
echo "============================================================================"
echo "TODO COMPLETADO — $(date)"
echo "============================================================================"
