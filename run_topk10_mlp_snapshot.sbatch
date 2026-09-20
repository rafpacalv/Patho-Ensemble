#!/bin/bash
#SBATCH --job-name=topk10_mlp_snapshot
#SBATCH --output=logs/topk10_mlp_snapshot_%j.out
#SBATCH --error=logs/topk10_mlp_snapshot_%j.err
#SBATCH --time=12:00:00
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --partition=main

# ============================================================================
# Idea nueva: pooling duro top-K de atención + SnapshotMLPMetaClassifier
# ============================================================================
# En vez de que cada ABMIL agregue con atención blanda sobre TODA la bolsa
# (N ~ 3000-5000 parches), se agregan sólo los K=10 parches de mayor peso,
# renormalizando su atención (abmil_engine.abmil_extract_topk_embeddings).
# Es la línea abierta "regularizar la atención con top-k de parches" del
# informe consolidado (prioridad baja, nunca antes probada) combinada con
# SnapshotMLPMetaClassifier (ya implementado y evaluado sobre probabilidades
# y sobre embeddings de atención blanda, nunca sobre top-k).
#
# Trío base: ctranspath + uni_v2 + conch_v1_5 (el ganador, no el histórico
# con virchow_v1 — ver CLAUDE.md, "Base model selection dominates").
#
# Tres configuraciones por dataset, todas sobre los mismos folds (comparación
# pareada):
#   insample:logreg       — baseline vigente, recomputado en esta misma pasada
#                            (nunca se reutiliza un ensemble4/ antiguo)
#   topk10:logreg          — aísla si el espacio de features top-10 ayuda/daña
#                            por sí solo, con el mismo combinador que el baseline
#   topk10:mlp_snapshot    — la propuesta: ¿aporta el snapshot MLP algo más
#                            que logreg DENTRO del espacio top-10?
#
# Uso:
#   sbatch run_topk10_mlp_snapshot.sbatch
#   DATASETS="cptac_brca" TASK_NAME=TP53_mutation K=20 sbatch run_topk10_mlp_snapshot.sbatch
# ============================================================================

set -uo pipefail   # sin -e: un dataset que falle no debe abortar el resto

source /home/JKP6679/miniconda3/etc/profile.d/conda.sh
conda activate ensemble

cd /home/JKP6679/Patho-Ensemble

WORK_DIR="/home/JKP6679/Patho-Ensemble/PARADIS/datos/patches"
TISSUE_PATCHING="20x_224px_0px_overlap"
MODELS="ctranspath uni_v2 conch_v1_5"
K="${K:-10}"
FEATS="topk${K}"
DATASETS="${DATASETS:-cptac_brca cptac_gbm}"
TASK_NAME="${TASK_NAME:-TP53_mutation}"

count_files() {
    # ls sobre un glob sin coincidencias sale con 2 bajo pipefail y set -e lo
    # mataría en silencio (CLAUDE.md, "Bash pitfalls in the sbatch scripts").
    local d="$1"
    if [ -d "$d" ]; then find "$d" -maxdepth 1 -name 'fold_*' -type d | wc -l; else echo 0; fi
}

echo "============================================================================"
echo "TOP-${K} ATTENTION POOLING + MLP SNAPSHOT — $(date)"
echo "============================================================================"
echo "Modelos base: $MODELS"
echo "Datasets:     $DATASETS"
echo "Task:         $TASK_NAME"
echo ""

FAILED=()
RAN_DATASETS=()

for DATASET in $DATASETS; do
    ABMIL_DIR="$WORK_DIR/$DATASET/$TASK_NAME/abmil"
    LOG_DIR="logs/${DATASET}_${TASK_NAME}"
    mkdir -p "$LOG_DIR"

    echo ""
    echo "╔════════════════════════════════════════════════════════════════════════╗"
    echo "║ $DATASET / $TASK_NAME"
    echo "╚════════════════════════════════════════════════════════════════════════╝"

    STANDARD_OK=1
    for MODEL in $MODELS; do
        B="$ABMIL_DIR/${MODEL}_${TISSUE_PATCHING}"
        N=$(count_files "$B/checkpoints")
        if [ "$N" -eq 0 ]; then
            echo "❌ falta $MODEL: sin checkpoints en $B/checkpoints"
            STANDARD_OK=0
        fi
    done
    if [ "$STANDARD_OK" -eq 0 ]; then
        echo "⚠ Se omite $DATASET: entrena antes el trío con train_base_models.sbatch"
        FAILED+=("$DATASET (sin checkpoints)")
        continue
    fi

    # ------------------------------------------------------------------
    # PASO 1: generar embeddings top-K (in-sample + val + test)
    # ------------------------------------------------------------------
    echo ""
    echo "PASO 1/3: generando embeddings top-${K} (${FEATS}) para cada modelo..."
    STEP1_OK=1
    for MODEL in $MODELS; do
        echo "  • $MODEL..."
        if ! python src/test_abmil.py \
            --foundational_model "$MODEL" \
            --work_dir "$WORK_DIR" \
            --train_source "$DATASET" \
            --tissue_patching "$TISSUE_PATCHING" \
            --task_name "$TASK_NAME" \
            --epochs 100 \
            --topk "$K" \
            2>&1 | tee -a "$LOG_DIR/topk${K}_${MODEL}.log"; then
            echo "  ❌ $MODEL falló generando ${FEATS}"
            STEP1_OK=0
        fi
    done
    if [ "$STEP1_OK" -eq 0 ]; then
        echo "⚠ Se omite $DATASET: fallo generando features ${FEATS}"
        FAILED+=("$DATASET (fallo en ${FEATS})")
        continue
    fi

    # ------------------------------------------------------------------
    # PASO 2: baseline vigente recomputado en esta misma pasada
    # (CLAUDE.md: nunca reutilizar un ensemble4_* antiguo para un contraste nuevo)
    # ------------------------------------------------------------------
    echo ""
    echo "PASO 2/3: baseline insample:logreg (recomputado ahora)..."
    if ! python src/ensemble4.py \
        --foundational_models $MODELS \
        --work_dir "$WORK_DIR" \
        --train_source "$DATASET" \
        --tissue_patching "$TISSUE_PATCHING" \
        --task_name "$TASK_NAME" \
        --meta_model logreg \
        --meta_features insample \
        2>&1 | tee -a "$LOG_DIR/insample_logreg.log"; then
        echo "❌ baseline insample:logreg falló, se omite $DATASET"
        FAILED+=("$DATASET (falló baseline)")
        continue
    fi

    # ------------------------------------------------------------------
    # PASO 3: las dos configuraciones sobre el nuevo espacio top-K
    # ------------------------------------------------------------------
    echo ""
    echo "PASO 3/3: meta-learners sobre ${FEATS}..."

    echo "  [1/2] logreg sobre ${FEATS} (control: ¿ayuda el espacio de features solo?)..."
    if ! python src/ensemble4.py \
        --foundational_models $MODELS \
        --work_dir "$WORK_DIR" \
        --train_source "$DATASET" \
        --tissue_patching "$TISSUE_PATCHING" \
        --task_name "$TASK_NAME" \
        --meta_model logreg \
        --meta_features "$FEATS" \
        2>&1 | tee -a "$LOG_DIR/${FEATS}_logreg.log"; then
        echo "  ❌ ${FEATS}:logreg falló"
        FAILED+=("$DATASET (falló ${FEATS}:logreg)")
    fi

    echo "  [2/2] mlp_snapshot sobre ${FEATS} (la propuesta)..."
    if ! python src/ensemble4.py \
        --foundational_models $MODELS \
        --work_dir "$WORK_DIR" \
        --train_source "$DATASET" \
        --tissue_patching "$TISSUE_PATCHING" \
        --task_name "$TASK_NAME" \
        --meta_model mlp_snapshot \
        --meta_features "$FEATS" \
        --meta_epochs 120 \
        --n_cycles 6 \
        --restart_lr 5e-3 \
        2>&1 | tee -a "$LOG_DIR/${FEATS}_mlp_snapshot.log"; then
        echo "  ❌ ${FEATS}:mlp_snapshot falló"
        FAILED+=("$DATASET (falló ${FEATS}:mlp_snapshot)")
    fi

    RAN_DATASETS+=("$DATASET")
done

# ============================================================================
# Comparación pareada por dataset (mismo protocolo que
# run_3_experiments_v2_improved.sbatch: Δ, IC 95%, W/T/L, MDE, Holm)
# ============================================================================
echo ""
echo "╔════════════════════════════════════════════════════════════════════════╗"
echo "║ COMPARATIVA PAREADA"
echo "╚════════════════════════════════════════════════════════════════════════╝"

WORK_DIR="$WORK_DIR" TASK_NAME="$TASK_NAME" FEATS="$FEATS" \
RAN_DATASETS="${RAN_DATASETS[*]:-}" python3 << 'PYEOF'
import json
import os
from pathlib import Path

import numpy as np
from scipy.stats import ttest_rel

WORK_DIR = Path(os.environ["WORK_DIR"])
TASK_NAME = os.environ["TASK_NAME"]
FEATS = os.environ["FEATS"]
RAN_DATASETS = os.environ.get("RAN_DATASETS", "").split()

CONFIGS = [
    ("insample:logreg", "ensemble4"),
    (f"{FEATS}:logreg", f"ensemble4_{FEATS}"),
    (f"{FEATS}:mlp_snapshot", f"ensemble4_mlp_snapshot_{FEATS}"),
]
METRICS = [("macro-ovr-auc", "AUC-ROC"), ("weighted_kappa", "Kappa"),
           ("macro-f1", "Macro-F1"), ("bacc", "Balanced acc"), ("acc", "Accuracy")]


def holm(pvals):
    m = len(pvals)
    order = np.argsort(pvals)
    adj = np.empty(m)
    running = 0.0
    for rank, idx in enumerate(order):
        running = max(running, (m - rank) * pvals[idx])
        adj[idx] = min(running, 1.0)
    return adj


def load_per_fold(sub_dir):
    folds = sorted(sub_dir.glob("fold_*"), key=lambda p: int(p.name.split("_")[1]))
    vals = {}
    for fd in folds:
        try:
            m = json.load(open(fd / "metrics.json"))["overall"]
        except Exception:
            continue
        for k, _ in METRICS:
            vals.setdefault(k, []).append(m[k])
    return {k: np.asarray(v) for k, v in vals.items()}


for dataset in RAN_DATASETS:
    abmil_dir = WORK_DIR / dataset / TASK_NAME / "abmil"
    per_fold = {}
    for name, sub in CONFIGS:
        d = abmil_dir / sub / "test_metrics"
        if d.exists():
            per_fold[name] = load_per_fold(d)

    baseline = "insample:logreg"
    base_v_auc = per_fold.get(baseline, {}).get("macro-ovr-auc")
    n_folds = len(base_v_auc) if base_v_auc is not None else 0

    print(f"\n{'=' * 100}")
    print(f"{dataset} / {TASK_NAME}   —   {len(per_fold)}/{len(CONFIGS)} configuraciones, "
          f"{n_folds} folds   —   baseline: {baseline}")
    print(f"{'=' * 100}")

    if baseline not in per_fold:
        print("  ❌ falta el baseline, no se puede comparar")
        continue

    for key, label in METRICS:
        base_v = per_fold[baseline].get(key)
        if base_v is None or len(base_v) == 0:
            continue

        rows, pvals = [], []
        for name, _ in CONFIGS:
            if name == baseline or name not in per_fold:
                continue
            v = per_fold[name].get(key)
            if v is None or len(v) != len(base_v):
                continue
            d = v - base_v
            sd = d.std(ddof=1)
            n = len(d)
            ci = 1.96 * sd / np.sqrt(n) if sd > 0 else 0.0
            mde = (1.96 + 0.842) * sd / np.sqrt(n) if sd > 0 else 0.0
            p = ttest_rel(v, base_v).pvalue if sd > 0 else 1.0
            rows.append([name, v.mean(), d.mean(), ci, mde,
                         int((d > 0).sum()), int((d == 0).sum()), int((d < 0).sum())])
            pvals.append(p)

        if not rows:
            continue
        adj = holm(np.asarray(pvals))

        print(f"\n{label}   —   baseline = {base_v.mean():.4f}")
        print("-" * 100)
        print(f"{'Configuración':<28}{'Media':>9}{'Δ':>10}{'IC 95% pareado':>22}"
              f"{'G/E/P':>12}{'p':>8}{'p Holm':>9}{'MDE':>8}")
        print("-" * 100)
        for (name, mean, dm, ci, mde, g, e, l), p, pa in zip(rows, pvals, adj):
            star = " *" if pa < 0.05 else ("  ~" if p < 0.05 else "")
            print(f"{name:<28}{mean:9.4f}{dm:+10.4f}"
                  f"{f'[{dm - ci:+.4f}, {dm + ci:+.4f}]':>22}"
                  f"{f'{g}/{e}/{l}':>12}{p:8.3f}{pa:9.3f}{mde:8.4f}{star}")
        print("-" * 100)

    print("\n  * = significativo tras corrección de Holm.  ~ = significativo sin corregir.")
    print("  MDE = efecto mínimo detectable al 80% de potencia: un Δ menor que esto")
    print("  no era detectable con este número de folds ('no significativo' = 'acotado').")
PYEOF

echo ""
echo "============================================================================"
echo "RESUMEN"
echo "============================================================================"
echo "Datasets completados (${#RAN_DATASETS[@]}): ${RAN_DATASETS[*]:-ninguno}"
if [ ${#FAILED[@]} -gt 0 ]; then
    echo "Con incidencias (${#FAILED[@]}):"
    printf '  ❌ %s\n' "${FAILED[@]}"
fi
echo ""
echo "Resultados en: \$WORK_DIR/{dataset}/$TASK_NAME/abmil/ensemble4*${FEATS}*/"
echo "Fin: $(date)"
