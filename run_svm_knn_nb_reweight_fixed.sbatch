#!/bin/bash
#SBATCH --job-name=svm_knn_nb_rw_fixed
#SBATCH --partition=main
#SBATCH --constraint=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=logs/svm_knn_nb_reweight_fixed_%j.out
#SBATCH --error=logs/svm_knn_nb_reweight_fixed_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=rafael.pachon.alvarez@gmail.com

# ============================================================================
# SVM / KNN / Naive Bayes — comparativa con/sin reweight (VERSIÓN FIJA)
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

# ---------------------------------------------------------------------------
# Comprobaciones previas
# ---------------------------------------------------------------------------
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
    echo ""
    echo "❌ Faltan las bases in-sample. Ejecuta primero:"
    echo "   DATASET=$DATASET TASK=$TASK sbatch train_base_models.sbatch"
    exit 1
fi

echo ""
echo "============================================================================"
echo "PRUEBAS: SVM / KNN / Naive Bayes — $DATASET / $TASK"
echo "============================================================================"
echo "Inicio: $(date)"
echo "modelos base: $MODELS"
echo "============================================================================"

TOTAL_START=$(date +%s)
FAILED=()
RAN=()

# Función para correr una configuración
function run_config() {
    local META=$1 REWEIGHT=$2 SUFFIX=$3
    local LABEL="$META"
    [ "$REWEIGHT" = "1" ] && LABEL="${META}_reweighted"

    echo ""
    echo "╔════════════════════════════════════════════════════════════════════════╗"
    echo "║ $LABEL (suffix: $SUFFIX)"
    echo "╚════════════════════════════════════════════════════════════════════════╝"

    REWEIGHT_FLAG=""
    [ "$REWEIGHT" = "1" ] && REWEIGHT_FLAG="--reweight_by_model_importance"

    START=$(date +%s)
    if python src/ensemble4.py \
        --foundational_models $MODELS \
        --work_dir "$WORK_DIR" \
        --train_source "$DATASET" \
        --tissue_patching "$TISSUE_PATCHING" \
        --task_name "$TASK" \
        --meta_model "$META" \
        --meta_features "insample" \
        --results_subdir_suffix "$SUFFIX" \
        $REWEIGHT_FLAG \
        2>&1 | tee -a "$LOG_DIR/${LABEL}.log"; then

        DUR=$(( $(date +%s) - START ))
        echo "[$(date)] ✓ $LABEL completado ($((DUR/60)) min $((DUR%60)) s)"
        RAN+=("$LABEL")
    else
        echo "[$(date)] ❌ $LABEL FALLÓ"
        FAILED+=("$LABEL")
    fi
}

# Correr todas las configuraciones con sufijos distintos para separar directorios
run_config "svm" "0" "_baseline"      # sin reweight
run_config "svm" "1" "_reweighted"    # con reweight
run_config "knn" "0" "_baseline"
run_config "knn" "1" "_reweighted"
run_config "nb"  "0" "_baseline"
run_config "nb"  "1" "_reweighted"

# ---------------------------------------------------------------------------
# Tabla comparativa con test pareado (Python inline)
# ---------------------------------------------------------------------------
echo ""
echo "╔════════════════════════════════════════════════════════════════════════╗"
echo "║ COMPARATIVA — baseline vs. reweighted"
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

def holm(pvals):
    """Corrección Holm-Bonferroni."""
    if not pvals:
        return []
    m = len(pvals)
    order = np.argsort(pvals)
    adj = np.empty(m)
    running = 0.0
    for rank, idx in enumerate(order):
        running = max(running, (m - rank) * pvals[idx])
        adj[idx] = min(running, 1.0)
    return adj

# Pares: (nombre, baseline_dir, reweighted_dir)
PAIRS = [
    ("SVM", "ensemble4_svm_baseline", "ensemble4_svm_reweighted"),
    ("KNN", "ensemble4_knn_baseline", "ensemble4_knn_reweighted"),
    ("NB",  "ensemble4_nb_baseline",  "ensemble4_nb_reweighted"),
]

results = {}
for model_name, baseline_sub, reweight_sub in PAIRS:
    baseline_dir = ABMIL_DIR / baseline_sub
    reweight_dir = ABMIL_DIR / reweight_sub

    baseline_data = {}
    reweight_data = {}

    # Cargar métricas por fold
    for sub, data_dict in [(baseline_dir, baseline_data), (reweight_dir, reweight_data)]:
        folds = sorted(sub.glob("test_metrics/fold_*"),
                      key=lambda p: int(p.name.split("_")[1]))
        for fd in folds:
            try:
                m = json.load(open(fd / "metrics.json"))["overall"]
                for k, _ in METRICS:
                    data_dict.setdefault(k, []).append(m[k])
            except:
                pass

    # Convertir a arrays
    for k in baseline_data:
        baseline_data[k] = np.array(baseline_data[k])
        reweight_data[k] = np.array(reweight_data[k])

    results[model_name] = (baseline_data, reweight_data)

n_folds = len(next(iter(results.values()))[0].get("macro-ovr-auc", []))
print(f"\n{'='*120}")
print(f"{DATASET} / {TASK}   —   {n_folds} folds")
print(f"{'='*120}")

for key, label in METRICS:
    print(f"\n{label}   (baseline vs. reweighted)")
    print("-" * 120)
    print(f"{'Modelo':<12}{'Baseline':>12}{'Reweighted':>14}{'Δ':>10}{'IC 95%':>15}{'Wins':>6}{'Ties':>6}{'p-value':>10}")
    print("-" * 120)

    pvals = []
    for model_name in sorted(results.keys()):
        baseline_data, reweight_data = results[model_name]

        b_vals = baseline_data.get(key, [])
        r_vals = reweight_data.get(key, [])

        if len(b_vals) == 0 or len(r_vals) == 0 or len(b_vals) != len(r_vals):
            print(f"{model_name:<12}{'—':>12}{'—':>14}{'—':>10}{'—':>15}{'—':>6}{'—':>6}{'—':>10}")
            continue

        d = r_vals - b_vals
        n = len(d)
        mean_d = d.mean()
        sd_d = d.std(ddof=1)
        ci = 1.96 * sd_d / np.sqrt(n) if sd_d > 0 else 0.0

        p = wilcoxon(r_vals, b_vals, alternative='two-sided').pvalue if sd_d > 0 else 1.0
        pvals.append(p)

        wins = (d > 0).sum()
        ties = (d == 0).sum()

        print(f"{model_name:<12}{b_vals.mean():>12.4f}{r_vals.mean():>14.4f}"
              f"{mean_d:>10.4f}{ci:>+14.4f}s  {wins:>5d}  {ties:>5d}  {p:>10.4f}")

    # Holm correction
    if pvals:
        adj_pvals = holm(np.array(pvals))
        print("-" * 120)
        print("Holm-corrected p-values:")
        for i, model_name in enumerate(sorted(results.keys())):
            if i < len(adj_pvals):
                sig = "✓ SIGNIFICANT" if adj_pvals[i] < 0.05 else "  ns"
                print(f"  {model_name}: {adj_pvals[i]:.4f}  {sig}")

    print()

PYEOF

# ---------------------------------------------------------------------------
# Resumen final
# ---------------------------------------------------------------------------
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
