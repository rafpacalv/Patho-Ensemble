#!/bin/bash
#SBATCH --job-name=campaign
#SBATCH --partition=main
#SBATCH --constraint=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=logs/campaign_%j.out
#SBATCH --error=logs/campaign_%j.err

# ============================================================================
# run_full_campaign.sbatch — la campaña completa sobre UN dataset/tarea.
#
# Reproduce, de principio a fin, todo lo medido sobre cptac_brca/TP53_mutation
# en INFORME_SUITE_METACLASIFICADORES_20260822.md, pero parametrizado por
# dataset y tarea, usando el trío óptimo ctranspath + uni_v2 + conch_v1_5.
#
# Uso:
#   DATASET=cptac_gbm TASK=TP53_mutation sbatch run_full_campaign.sbatch
#   DATASET=bc_therapy TASK=er_status    sbatch run_full_campaign.sbatch
#   DATASET=cervical_subtype TASK=subtype sbatch run_full_campaign.sbatch
#
# Variables opcionales:
#   PATCHING   estrategia de parcheado      (defecto 20x_224px_0px_overlap)
#   MODELS     modelos base                 (defecto el trío óptimo)
#   EPOCHS     épocas del ABMIL             (defecto 100)
#   SKIP_BASE  =1 para saltar el reentreno base si ya está en disco
#   SKIP_FGE   =1 para saltar los snapshots a nivel base
#
# El número de folds NO se asume: se lee de las columnas fold_* del k=all.tsv.
# Asumirlo una vez marcó como fallido un entrenamiento correcto y canceló una
# cadena afterok entera.
# ============================================================================

set -euo pipefail

source $(conda info --base)/etc/profile.d/conda.sh
conda activate ensemble

PROJECT_DIR="/home/JKP6679/Patho-Ensemble"
cd "$PROJECT_DIR"

DATASET="${DATASET:?define DATASET, p.ej. DATASET=cptac_gbm}"
TASK="${TASK:?define TASK, p.ej. TASK=TP53_mutation}"
PATCHING="${PATCHING:-20x_224px_0px_overlap}"
MODELS="${MODELS:-ctranspath uni_v2 conch_v1_5}"
EPOCHS="${EPOCHS:-100}"
SKIP_BASE="${SKIP_BASE:-0}"
SKIP_FGE="${SKIP_FGE:-0}"

WORK_DIR="$PROJECT_DIR/PARADIS/datos/patches"
FEAT_DIR="$PROJECT_DIR/PARADIS/datos/features"
ABMIL_DIR="$WORK_DIR/$DATASET/$TASK/abmil"
TSV="$WORK_DIR/$DATASET/$TASK/k=all.tsv"
TAG="${DATASET}_${TASK}"

echo "================================================================================"
echo "CAMPAÑA COMPLETA — $DATASET / $TASK"
echo "Job ${SLURM_JOB_ID:-local} — $(date)"
echo "================================================================================"

# ============================================================================
# FASE 0 — Preflight. Falla aquí y barato, no a las 3 horas.
# ============================================================================
echo ""
echo "### FASE 0 — comprobaciones previas"

[ -f "$TSV" ] || { echo "❌ no existe $TSV"; exit 1; }

# Nº de folds y de clases, leídos del propio TSV.
read -r N_FOLDS N_CLASSES N_SLIDES N_CASES <<< "$(python - "$TSV" "$TASK" <<'PY'
import csv, sys
tsv, task = sys.argv[1], sys.argv[2]
rows = list(csv.DictReader(open(tsv), delimiter="\t"))
folds = [c for c in rows[0] if c.startswith("fold_")]
vals = {r[task] for r in rows if r.get(task) not in (None, "", "nan")}
cases = {r["case_id"] for r in rows}
print(len(folds), len(vals), len(rows), len(cases))
PY
)"

echo "  splits    : $TSV"
echo "  folds     : $N_FOLDS"
echo "  slides    : $N_SLIDES   pacientes: $N_CASES"
echo "  clases    : $N_CLASSES"

if [ "$N_FOLDS" -lt 2 ]; then
    echo "❌ $N_FOLDS fold(s): no hay comparación pareada posible."
    exit 1
fi

# Las features deben resolverse por la MISMA vía que usa el código
# (utils.get_features_dir prueba el sufijo _monai y luego sin sufijo).
echo "  features  :"
for M in $MODELS; do
    python - "$WORK_DIR" "$DATASET" "$M" <<'PY' || exit 1
import sys
sys.path.insert(0, "src")
from utils import get_features_dir, detect_latent_dim
work_dir, ds, model = sys.argv[1:4]
try:
    d = get_features_dir(work_dir, ds, model)
    print(f"    ✓ {model:<12} dim={detect_latent_dim(d)}  {d}")
except Exception as e:
    print(f"    ❌ {model}: {e}")
    raise SystemExit(1)
PY
done

echo "✓ Preflight OK"

# ============================================================================
# FASE 1 — ABMIL base, un modelo independiente por foundational model.
# ============================================================================
if [ "$SKIP_BASE" = "1" ]; then
    echo ""; echo "### FASE 1 — SALTADA (SKIP_BASE=1)"
else
    echo ""
    echo "### FASE 1 — entrenamiento ABMIL base"
    for M in $MODELS; do
        echo ""
        echo "--- [base] $M --- $(date)"
        # --latent_dim NO se pasa: se autodetecta del .h5. Cablearlo provocó un
        # desajuste silencioso 768/1536/2560.
        python src/train_abmil.py \
            --foundational_model "$M" \
            --work_dir "$WORK_DIR" \
            --train_source "$DATASET" \
            --tissue_patching "$PATCHING" \
            --task_name "$TASK" \
            --epochs "$EPOCHS" \
            --fold all
    done
fi

# ============================================================================
# FASE 2 — Meta-features in-sample (predicciones sobre el train del fold).
# ============================================================================
echo ""
echo "### FASE 2 — meta-features in-sample"
for M in $MODELS; do
    echo "--- [meta] $M --- $(date)"
    python src/test_abmil.py \
        --foundational_model "$M" \
        --work_dir "$WORK_DIR" \
        --train_source "$DATASET" \
        --tissue_patching "$PATCHING" \
        --task_name "$TASK" \
        --epochs "$EPOCHS"
done

# ============================================================================
# FASE 3 — Embeddings de slide (512-D post-atención) para el brazo §8.1.
# ============================================================================
echo ""
echo "### FASE 3 — embeddings de slide"
for M in $MODELS; do
    echo "--- [emb] $M --- $(date)"
    python src/test_abmil.py \
        --foundational_model "$M" \
        --work_dir "$WORK_DIR" \
        --train_source "$DATASET" \
        --tissue_patching "$PATCHING" \
        --task_name "$TASK" \
        --epochs "$EPOCHS" \
        --use_embeddings
done

# Los embeddings sólo valen frente a los checkpoints de los que salieron.
echo ""
echo "### FASE 3b — vigencia de los embeddings"
for M in $MODELS; do
    D="$ABMIL_DIR/${M}_${PATCHING}"
    CKPT=$(stat -c %Y "$D/checkpoints/fold_0/model.pt")
    EMB=$(stat -c %Y "${D}_train_eval_embeddings/val_outputs_embeddings/fold_0/preds.npy")
    if [ "$EMB" -le "$CKPT" ]; then
        echo "❌ $M: embeddings ($EMB) no son posteriores al checkpoint ($CKPT)"
        exit 1
    fi
    echo "  ✓ $M"
done

# ============================================================================
# FASE 4 — Snapshot ensembles a nivel base (FGE y SE sobre los ABMIL).
#   FGE  Garipov et al., ICLR 2018 — lr_1 2e-4  lr_2 2e-5  cycle_length 4
#   SE   Huang   et al., ICLR 2017 — lr_1 1e-3  lr_2 1e-6  cycle_length 15
# Hiperparámetros de fge_orig / se_orig en experiments.yaml, los validados.
# ============================================================================
SNAP_CONFIGS=""
if [ "$SKIP_FGE" = "1" ]; then
    echo ""; echo "### FASE 4 — SALTADA (SKIP_FGE=1)"
else
    echo ""
    echo "### FASE 4 — snapshot ensembles a nivel base"
    run_snapshot () {
        local SNAPTAG=$1 SCHED=$2 LR1=$3 LR2=$4 CLEN=$5
        echo ""
        echo "################################################################"
        echo "# $SNAPTAG (schedule=$SCHED lr_1=$LR1 lr_2=$LR2 cycle_length=$CLEN)"
        echo "################################################################"
        for M in $MODELS; do
            echo "--- [$SNAPTAG] ciclos: $M --- $(date)"
            python src/train_abmil_fge.py \
                --foundational_model "$M" \
                --work_dir "$WORK_DIR" \
                --train_source "$DATASET" \
                --tissue_patching "$PATCHING" \
                --task_name "$TASK" \
                --epochs "$EPOCHS" \
                --fold all \
                --n_cycles 6 \
                --cycle_length "$CLEN" \
                --lr_1 "$LR1" \
                --lr_2 "$LR2" \
                --cycle_patience 2 \
                --fge_tag "$SNAPTAG" \
                --schedule "$SCHED"

            echo "--- [$SNAPTAG] meta-features: $M --- $(date)"
            python src/test_abmil_fge.py \
                --foundational_model "$M" \
                --work_dir "$WORK_DIR" \
                --train_source "$DATASET" \
                --tissue_patching "$PATCHING" \
                --task_name "$TASK" \
                --epochs "$EPOCHS" \
                --fge_tag "$SNAPTAG"
        done
    }
    run_snapshot fge_trio fge 2e-4 2e-5 4
    run_snapshot se_trio  se  1e-3 1e-6 15
    SNAP_CONFIGS="fge_trio:logreg fge_trio:mlp_snapshot se_trio:logreg se_trio:mlp_snapshot"
fi

# ============================================================================
# FASE 5 — La suite completa, en UNA sola pasada.
#
# Todos los brazos comparten proceso, splits y predicciones base, así que el
# baseline queda recalculado dentro de la propia pasada y los contrastes son
# pareados. Es la regla operativa del proyecto: nunca reutilizar un resultado
# antiguo como referencia.
# ============================================================================
echo ""
echo "### FASE 5 — suite de metaclasificadores ($N_FOLDS folds)"
python src/run_meta_suite.py \
    --work_dir "$WORK_DIR" \
    --train_source "$DATASET" \
    --task_name "$TASK" \
    --tissue_patching "$PATCHING" \
    --foundational_models $MODELS \
    --n_folds "$N_FOLDS" \
    --baseline prob:logreg \
    --configs \
        prob:logreg prob:mlp prob:mlp_snapshot prob:svm prob:knn prob:nb \
        emb:logreg emb:mlp emb:mlp_snapshot emb:mlp_fge \
        $SNAP_CONFIGS \
    --out "$PROJECT_DIR/results_campaign_${TAG}.json"

echo ""
echo "================================================================================"
echo "✅ COMPLETADO — $DATASET / $TASK — $(date)"
echo "   Resultados: results_campaign_${TAG}.json"
echo "================================================================================"
