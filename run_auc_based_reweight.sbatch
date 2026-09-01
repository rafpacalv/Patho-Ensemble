#!/bin/bash
#SBATCH --job-name=auc_based_reweight
#SBATCH --partition=main
#SBATCH --constraint=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=logs/auc_based_reweight_%j.out
#SBATCH --error=logs/auc_based_reweight_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=rafael.pachon.alvarez@gmail.com

# ============================================================================
# Opción B: AUC-based reweighting con MetricDistance
# ============================================================================
# Comparativa: SVM/KNN/NB con norm_ratio (baseline) vs. auc_based (nuevo)
#
# Uso:
#   DATASET=cptac_brca TASK=TP53_mutation sbatch run_auc_based_reweight.sbatch
# ============================================================================

set -uo pipefail

source $(conda info --base)/etc/profile.d/conda.sh
conda activate ensemble

PROJECT_DIR="/home/JKP6679/Patho-Ensemble"
cd "$PROJECT_DIR"

DATASET="${DATASET:-}"
TASK="${TASK:-}"
if [ -z "$DATASET" ] || [ -z "$TASK" ]; then
    read -r DATASET TASK < <(python3 src/experiment_config.py --list-active | head -1)
fi
eval "$(python3 src/experiment_config.py --emit-shell --dataset "$DATASET" --task "$TASK")"

LOG_DIR="$PROJECT_DIR/logs/${DATASET}_${TASK}"
mkdir -p "$LOG_DIR"

ABMIL_DIR="$WORK_DIR/$DATASET/$TASK/abmil"

# Comprobaciones previas
echo "============================================================================"
echo "COMPROBACIONES PREVIAS — $DATASET / $TASK"
echo "============================================================================"

STANDARD_OK=1
for MODEL in $MODELS; do
    B="$ABMIL_DIR/${MODEL}_${TISSUE_PATCHING}"
    if [ ! -d "${B}_train_eval/val_outputs/fold_0" ]; then
        echo "❌ falta ${MODEL}: _train_eval/val_outputs/fold_0"
        STANDARD_OK=0
    fi
done

if [ "$STANDARD_OK" -eq 1 ]; then
    echo "✓ bases in-sample disponibles"
else
    echo "❌ Faltan las bases in-sample. Ejecuta primero:"
    echo "   DATASET=$DATASET TASK=$TASK sbatch train_base_models.sbatch"
    exit 1
fi

echo ""
echo "============================================================================"
echo "PRUEBAS: SVM / KNN / NB — norm_ratio vs. auc_based — $DATASET / $TASK"
echo "============================================================================"
echo "Inicio: $(date)"
echo "modelos base: $MODELS"
echo "============================================================================"

TOTAL_START=$(date +%s)
FAILED=()
RAN=()

# Función para correr una configuración
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
        echo "[$(date)] ✓ $LABEL completado ($((DUR/60)) min $((DUR%60)) s)"
        RAN+=("$LABEL")
    else
        echo "[$(date)] ❌ $LABEL FALLÓ"
        FAILED+=("$LABEL")
    fi
}

# Correr todas las configuraciones
for META in svm knn nb; do
    run_config "$META" "norm_ratio" "_nr"      # baseline (norm_ratio)
    run_config "$META" "auc_based"  "_auc"     # new method (auc_based)
done

# Tabla comparativa con test pareado
echo ""
echo "╔════════════════════════════════════════════════════════════════════════╗"
echo "║ COMPARATIVA — norm_ratio (baseline) vs. auc_based (new)"
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

PAIRS = [
    ("SVM", "ensemble4_svm_nr", "ensemble4_svm_auc"),
    ("KNN", "ensemble4_knn_nr", "ensemble4_knn_auc"),
    ("NB",  "ensemble4_nb_nr",  "ensemble4_nb_auc"),
]

results = {}
for model_name, baseline_sub, auc_sub in PAIRS:
    baseline_dir = ABMIL_DIR / baseline_sub
    auc_dir = ABMIL_DIR / auc_sub

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

    results[model_name] = (baseline_data, auc_data)

print(f"\n{'='*120}")
print(f"{DATASET} / {TASK}   —   50 folds   —   norm_ratio (baseline) vs. auc_based (new)")
print(f"{'='*120}")

for key, label in METRICS:
    print(f"\n{label}   (norm_ratio vs. auc_based)")
    print("-" * 120)
    print(f"{'Modelo':<12}{'norm_ratio':>12}{'auc_based':>14}{'Δ':>10}{'IC 95%':>15}{'Wins':>6}{'Ties':>6}{'p-value':>10}")
    print("-" * 120)

    pvals = []
    for model_name in sorted(results.keys()):
        baseline_data, auc_data = results[model_name]

        b_vals = baseline_data.get(key, [])
        a_vals = auc_data.get(key, [])

        if len(b_vals) == 0 or len(a_vals) == 0 or len(b_vals) != len(a_vals):
            print(f"{model_name:<12}{'—':>12}{'—':>14}{'—':>10}{'—':>15}{'—':>6}{'—':>6}{'—':>10}")
            continue

        d = a_vals - b_vals
        mean_d = d.mean()
        sd_d = d.std(ddof=1)
        ci = 1.96 * sd_d / np.sqrt(len(d)) if sd_d > 0 else 0.0

        p = wilcoxon(a_vals, b_vals, alternative='two-sided').pvalue if sd_d > 0 else 1.0
        pvals.append(p)

        wins = int((d > 0).sum())
        ties = int((d == 0).sum())

        print(f"{model_name:<12}{b_vals.mean():>12.4f}{a_vals.mean():>14.4f}"
              f"{mean_d:>10.4f}{ci:>+14.4f}s  {wins:>5d}  {ties:>5d}  {p:>10.4f}")

    if pvals and len(pvals) > 0:
        pvals_arr = np.array(pvals)
        m = len(pvals_arr)
        order = np.argsort(pvals_arr)
        adj = np.empty(m)
        running = 0.0
        for rank, idx in enumerate(order):
            running = max(running, (m - rank) * pvals_arr[idx])
            adj[idx] = min(running, 1.0)

        print("-" * 120)
        print("Holm-corrected p-values:")
        for i, model_name in enumerate(sorted(results.keys())):
            if i < len(adj):
                sig = "✓ SIGNIFICANT (p<0.05)" if adj[i] < 0.05 else "  ns"
                print(f"  {model_name}: {adj[i]:.4f}  {sig}")

PYEOF

# Resumen
TOTAL_DUR=$(( $(date +%s) - TOTAL_START ))
echo ""
echo "╔════════════════════════════════════════════════════════════════════════╗"
echo "║ RESUMEN"
echo "╚════════════════════════════════════════════════════════════════════════╝"
echo "Completados: ${#RAN[@]}"
echo "Fallidos:    ${#FAILED[@]}"
if [ ${#FAILED[@]} -gt 0 ]; then
    echo "             ${FAILED[*]}"
fi
echo "Duración total: $((TOTAL_DUR/60)) min $((TOTAL_DUR%60)) s"
echo "Fin: $(date)"
