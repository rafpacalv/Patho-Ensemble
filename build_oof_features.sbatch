#!/bin/bash
#SBATCH --job-name=build_oof
#SBATCH --partition=main
#SBATCH --constraint=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --output=logs/build_oof_%j.out
#SBATCH --error=logs/build_oof_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=rafael.pachon.alvarez@gmail.com

# ============================================================================
# Fase 0 — Meta-features out-of-fold
# ============================================================================
# Corrige el sesgo in-sample de las features del meta-learner: hoy test_abmil.py
# predice sobre el propio train del fold, con lo que los modelos base valen
# AUC 0.94 en meta-train frente a 0.77 en test.
#
#   DATASET=cptac_brca TASK=TP53_mutation sbatch build_oof_features.sbatch
#
# Coste: n_inner veces el entrenamiento base. Medido en cptac_brca (3 modelos,
# 50 folds) el base tarda ~35 min, así que con n_inner=3 son ~1 h 45 min.
# El cap de 8 h es holgura.
#
# Al terminar ejecuta la verificación BLOQUEANTE: el AUC de los modelos base
# sobre las features OOF debe caer de ~0.94 a ~0.77-0.82. Si no cae, hay fuga
# y no tiene sentido seguir con la Fase 2.
# ============================================================================

set -euo pipefail

count_files() {
    local n
    n=$(ls -1d "$@" 2>/dev/null | wc -l) || n=0
    echo "$n"
}

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

N_INNER="${N_INNER:-3}"
# Etiqueta del arbol de salida. Cambiarla permite comparar varios n_inner:
#   N_INNER=5 OOF_TAG=oof5 sbatch build_oof_features.sbatch
OOF_TAG="${OOF_TAG:-oof}"
LOG_DIR="$PROJECT_DIR/logs/${DATASET}_${TASK}"
mkdir -p "$LOG_DIR"

ABMIL_DIR="$WORK_DIR/$DATASET/$TASK/abmil"
AVAIL_FOLDS=$(head -1 "$WORK_DIR/$DATASET/$TASK/k=all.tsv" \
              | tr '\t' '\n' | grep -c '^fold_' || echo 0)
EXPECTED_FOLDS=$(awk -v a="$AVAIL_FOLDS" -v m="${MAX_FOLDS:-0}" \
                 'BEGIN{print (m>0 && m<a) ? m : a}')

echo "============================================================================"
echo "FEATURES OUT-OF-FOLD — $DATASET / $TASK"
echo "============================================================================"
echo "Inicio:        $(date)"
echo "modelos:       $MODELS"
echo "folds:         $EXPECTED_FOLDS  (disponibles: $AVAIL_FOLDS)"
echo "folds internos: $N_INNER   (etiqueta: $OOF_TAG)"
echo "============================================================================"

TOTAL_START=$(date +%s)
FAILED=()

for MODEL in $MODELS; do
    echo ""
    echo "╔════════════════════════════════════════════════════════════════════════╗"
    echo "║ $MODEL"
    echo "╚════════════════════════════════════════════════════════════════════════╝"

    OOF_DIR="$ABMIL_DIR/${MODEL}_${TISSUE_PATCHING}_train_eval_${OOF_TAG}/val_outputs_${OOF_TAG}"
    DONE=$(count_files "$OOF_DIR"/fold_*)
    if [ "$DONE" -ge "$EXPECTED_FOLDS" ]; then
        echo "✓ ya generado ($DONE folds). Se salta."
        continue
    fi

    MSTART=$(date +%s)
    if python src/build_oof_features.py \
        --foundational_model "$MODEL" \
        --work_dir "$WORK_DIR" \
        --train_source "$DATASET" \
        --tissue_patching "$TISSUE_PATCHING" \
        --task_name "$TASK" \
        --epochs "$EPOCHS" \
        --n_inner "$N_INNER" \
        --oof_tag "$OOF_TAG" \
        ${MAX_FOLDS:+--max_folds "$MAX_FOLDS"} \
        2>&1 | tee -a "$LOG_DIR/${OOF_TAG}_${MODEL}.log"; then

        DUR=$(( $(date +%s) - MSTART ))
        N=$(count_files "$OOF_DIR"/fold_*)
        echo "[$(date)] ✓ $MODEL — $((DUR/60)) min, $N/$EXPECTED_FOLDS folds"
        [ "$N" -ge "$EXPECTED_FOLDS" ] || FAILED+=("$MODEL ($N/$EXPECTED_FOLDS folds)")
    else
        echo "[$(date)] ❌ $MODEL FALLÓ"
        FAILED+=("$MODEL")
    fi
done

# ---------------------------------------------------------------------------
# Verificación BLOQUEANTE: ¿desapareció el sesgo in-sample?
# ---------------------------------------------------------------------------
echo ""
echo "╔════════════════════════════════════════════════════════════════════════╗"
echo "║ VERIFICACIÓN: caída de AUC in-sample -> out-of-fold"
echo "╚════════════════════════════════════════════════════════════════════════╝"

ABMIL_DIR="$ABMIL_DIR" MODELS="$MODELS" TISSUE_PATCHING="$TISSUE_PATCHING" \
EXPECTED_FOLDS="$EXPECTED_FOLDS" OOF_TAG="$OOF_TAG" python3 << 'PYEOF'
import os
import numpy as np
from sklearn.metrics import roc_auc_score

ABMIL = os.environ["ABMIL_DIR"]
TP = os.environ["TISSUE_PATCHING"]
NF = int(os.environ["EXPECTED_FOLDS"])

def auc_over_folds(pattern):
    out = []
    for f in range(NF):
        d = pattern.format(f=f)
        try:
            y = np.load(f"{d}/labels.npy")
            p = np.load(f"{d}/preds.npy")
        except FileNotFoundError:
            continue
        if len(np.unique(y)) < 2:
            continue
        out.append(roc_auc_score(y, p[:, 1]) if p.shape[1] == 2
                   else roc_auc_score(y, p, multi_class="ovr"))
    return (np.mean(out), len(out)) if out else (float("nan"), 0)

print(f"\n{'modelo':14s} {'in-sample':>11s} {'out-of-fold':>13s} {'test':>8s} {'caída':>8s}")
print("-" * 60)

ok = True
for m in os.environ["MODELS"].split():
    b = f"{ABMIL}/{m}_{TP}"
    ins, _ = auc_over_folds(b + "_train_eval/val_outputs/fold_{f}")
    tag = os.environ["OOF_TAG"]
    oof, n_oof = auc_over_folds(b + f"_train_eval_{tag}/val_outputs_{tag}/fold_{{f}}")
    tst, _ = auc_over_folds(b + "/test_outputs/fold_{f}")
    drop = ins - oof
    print(f"{m:14s} {ins:11.4f} {oof:13.4f} {tst:8.4f} {drop:+8.4f}   ({n_oof} folds)")

    # El AUC out-of-fold debe quedar cerca del de test, no del in-sample. El
    # umbral es la mitad del camino entre ambos: por encima, las features OOF
    # siguen siendo optimistas y hay fuga en la partición interna.
    if not np.isnan(oof) and oof > (ins + tst) / 2:
        print(f"  ❌ {m}: el AUC out-of-fold sigue más cerca del in-sample que del test")
        ok = False

print()
if ok:
    print("✓ Las features out-of-fold ya no son optimistas. Fase 2 puede continuar.")
else:
    print("❌ VERIFICACIÓN FALLIDA: revisa el agrupamiento por paciente en _inner_folds.")
    raise SystemExit(1)
PYEOF

TOTAL_DUR=$(( $(date +%s) - TOTAL_START ))

echo ""
echo "============================================================================"
echo "RESUMEN — $DATASET / $TASK"
echo "============================================================================"
echo "Duración total: $((TOTAL_DUR/3600)) h $(( (TOTAL_DUR%3600)/60 )) min"
echo ""

for MODEL in $MODELS; do
    N=$(count_files "$ABMIL_DIR/${MODEL}_${TISSUE_PATCHING}_train_eval_${OOF_TAG}/val_outputs_${OOF_TAG}"/fold_*)
    echo "  $MODEL: $N/$EXPECTED_FOLDS folds"
done
echo ""

if [ ${#FAILED[@]} -eq 0 ]; then
    echo "✓ TODO CORRECTO"
    echo ""
    echo "Siguiente paso:"
    echo "  DATASET=$DATASET TASK=$TASK sbatch run_3_experiments_v2_improved.sbatch"
else
    echo "⚠ Incidencias:"
    printf '  ❌ %s\n' "${FAILED[@]}"
    exit 1
fi

echo ""
echo "Fin: $(date)"
echo "============================================================================"
