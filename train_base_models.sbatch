#!/bin/bash
#SBATCH --job-name=train_base_models
#SBATCH --partition=main
#SBATCH --constraint=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=logs/train_base_models_%j.out
#SBATCH --error=logs/train_base_models_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=rafael.pachon.alvarez@gmail.com

# ============================================================================
# Entrenamiento de los modelos base ABMIL (50 folds por modelo)
# ============================================================================
# Configuración: experiments.yaml (override con DATASET=... TASK=...)
#
#   sbatch train_base_models.sbatch
#   DATASET=cptac_coad TASK=MSI_H sbatch train_base_models.sbatch
#   MODELS_OVERRIDE="phikon_v2 uni_v1" sbatch train_base_models.sbatch
#
# --latent_dim NO se pasa: se auto-detecta del .h5 (evita el bug 768/1536/2560).
#
# Duración medida: ctranspath (768D) tarda ~6 min en los 50 folds. uni_v2 (1536D)
# y virchow_v1 (2560D) son proporcionalmente más lentos, así que los 3 modelos
# rondan la media hora. El cap de 12 h es holgura, no una estimación.
# ============================================================================

set -euo pipefail

# Cuenta ficheros que casan con un glob, devolviendo 0 si no casa ninguno.
# Imprescindible bajo `set -o pipefail`: si el glob no casa, bash pasa el patrón
# literal a `ls`, que sale con estado 2; pipefail propaga ese estado y `set -e`
# aborta el script sin escribir nada en stderr.
count_files() {
    local n
    n=$(ls -1d "$@" 2>/dev/null | wc -l) || n=0
    echo "$n"
}

source $(conda info --base)/etc/profile.d/conda.sh
conda activate ensemble

PROJECT_DIR="/home/JKP6679/Patho-Ensemble"
cd "$PROJECT_DIR"

# Cargar configuración del experimento
DATASET="${DATASET:-}"
TASK="${TASK:-}"
if [ -z "$DATASET" ] || [ -z "$TASK" ]; then
    read -r DATASET TASK < <(python3 src/experiment_config.py --list-active | head -1)
fi
eval "$(python3 src/experiment_config.py --emit-shell --dataset "$DATASET" --task "$TASK")"

# MODELS lo fija el eval de arriba desde experiments.yaml. MODELS_OVERRIDE
# permite entrenar un conjunto distinto sin tocar la matriz de experimentos
# (p. ej. ampliar la base a modelos que aún no están en `defaults.models`).
# Tiene que ir DESPUÉS del eval, o el eval lo pisa.
MODELS="${MODELS_OVERRIDE:-$MODELS}"

LOG_DIR="$PROJECT_DIR/logs/${DATASET}_${TASK}"
mkdir -p "$LOG_DIR"

ABMIL_DIR="$WORK_DIR/$DATASET/$TASK/abmil"

echo "============================================================================"
echo "Modelos base ABMIL — $DATASET / $TASK"
echo "============================================================================"
echo "Inicio:    $(date)"
echo "work_dir:  $WORK_DIR"
echo "modelos:   $MODELS"
echo "patching:  $TISSUE_PATCHING"
echo "epochs:    $EPOCHS"
echo "latent_dim: auto-detectado del .h5"
echo "============================================================================"
echo ""

TOTAL_START=$(date +%s)
FAILED=()
# Folds esperados = min(max_folds pedidos, folds que realmente tiene el TSV).
# El nº de columnas fold_* varía por dataset (cptac_brca: 50, cervical_subtype: 5),
# así que dar por hecho 50 —o el max_folds pedido— marca como fallo un
# entrenamiento que en realidad recorrió todos los folds disponibles.
AVAIL_FOLDS=$(head -1 "$WORK_DIR/$DATASET/$TASK/k=all.tsv" \
              | tr '\t' '\n' | grep -c '^fold_' || echo 0)
EXPECTED_FOLDS=$(awk -v a="$AVAIL_FOLDS" -v m="${MAX_FOLDS:-0}" \
                 'BEGIN{print (m>0 && m<a) ? m : a}')
echo "folds:     $EXPECTED_FOLDS  (disponibles en el TSV: $AVAIL_FOLDS${MAX_FOLDS:+, max_folds=$MAX_FOLDS})"

# ---------------------------------------------------------------------------
# Paso 1: entrenamiento
# ---------------------------------------------------------------------------
for MODEL in $MODELS; do
    echo ""
    echo "╔════════════════════════════════════════════════════════════════════════╗"
    echo "║ Modelo: $MODEL"
    echo "╚════════════════════════════════════════════════════════════════════════╝"

    MODEL_DIR="$ABMIL_DIR/${MODEL}_${TISSUE_PATCHING}"

    # Saltar sólo si existen los 50 model.pt REALES.
    # Comprobar el directorio no basta: hay folds con checkpoints legacy
    # (epoch_9.pt de patho_bench) que no sirven y provocarían un fallo silencioso.
    DONE=$(count_files "$MODEL_DIR"/checkpoints/fold_*/model.pt)
    if [ "$DONE" -ge "$EXPECTED_FOLDS" ]; then
        echo "✓ $MODEL ya entrenado ($DONE model.pt). Se salta."
        continue
    elif [ "$DONE" -gt 0 ]; then
        echo "⟳ $MODEL parcial ($DONE/$EXPECTED_FOLDS model.pt). Se reentrena."
    fi

    MODEL_START=$(date +%s)
    echo "[$(date)] Entrenando $MODEL..."

    if python src/train_abmil.py \
        --foundational_model "$MODEL" \
        --work_dir "$WORK_DIR" \
        --train_source "$DATASET" \
        --tissue_patching "$TISSUE_PATCHING" \
        --task_name "$TASK" \
        --epochs "$EPOCHS" \
        --fold all \
        ${MAX_FOLDS:+--max_folds "$MAX_FOLDS"} \
        2>&1 | tee -a "$LOG_DIR/train_${MODEL}.log"; then

        DUR=$(( $(date +%s) - MODEL_START ))
        NEW=$(count_files "$MODEL_DIR"/checkpoints/fold_*/model.pt)
        echo "[$(date)] ✓ $MODEL completado — $((DUR/60)) min, $NEW/$EXPECTED_FOLDS checkpoints"
        [ "$NEW" -ge "$EXPECTED_FOLDS" ] || FAILED+=("$MODEL (sólo $NEW/$EXPECTED_FOLDS checkpoints)")
    else
        echo "[$(date)] ❌ $MODEL FALLÓ"
        FAILED+=("$MODEL (entrenamiento)")
    fi
done

# ---------------------------------------------------------------------------
# Paso 2: predicciones sobre train (features del meta-learner)
# ---------------------------------------------------------------------------
echo ""
echo "╔════════════════════════════════════════════════════════════════════════╗"
echo "║ Paso 2: predicciones de train para el meta-learner"
echo "╚════════════════════════════════════════════════════════════════════════╝"

for MODEL in $MODELS; do
    if printf '%s\n' "${FAILED[@]:-}" | grep -q "^$MODEL"; then
        echo "⚠ $MODEL: se omite (falló el entrenamiento)"
        continue
    fi

    echo "[$(date)] Generando predicciones para $MODEL..."
    if python src/test_abmil.py \
        --foundational_model "$MODEL" \
        --work_dir "$WORK_DIR" \
        --train_source "$DATASET" \
        --tissue_patching "$TISSUE_PATCHING" \
        --task_name "$TASK" \
        --epochs "$EPOCHS" \
        2>&1 | tee -a "$LOG_DIR/test_${MODEL}.log"; then

        N=$(count_files "$ABMIL_DIR/${MODEL}_${TISSUE_PATCHING}_train_eval/val_outputs/fold_"*)
        echo "[$(date)] ✓ $MODEL — $N/$EXPECTED_FOLDS folds"
        [ "$N" -ge "$EXPECTED_FOLDS" ] || FAILED+=("$MODEL (predicciones: $N/$EXPECTED_FOLDS)")
    else
        echo "[$(date)] ❌ $MODEL: fallo generando predicciones"
        FAILED+=("$MODEL (predicciones)")
    fi
done

# ---------------------------------------------------------------------------
# Resumen
# ---------------------------------------------------------------------------
TOTAL_DUR=$(( $(date +%s) - TOTAL_START ))

echo ""
echo "============================================================================"
echo "RESUMEN — $DATASET / $TASK"
echo "============================================================================"
echo "Duración total: $((TOTAL_DUR/3600)) h $(( (TOTAL_DUR%3600)/60 )) min"
echo ""

for MODEL in $MODELS; do
    C=$(count_files "$ABMIL_DIR/${MODEL}_${TISSUE_PATCHING}"/checkpoints/fold_*/model.pt)
    E=$(count_files "$ABMIL_DIR/${MODEL}_${TISSUE_PATCHING}_train_eval/val_outputs/fold_"*)
    echo "  $MODEL: checkpoints=$C/$EXPECTED_FOLDS  train_eval=$E/$EXPECTED_FOLDS"
done
echo ""

if [ ${#FAILED[@]} -eq 0 ]; then
    echo "✓ TODO CORRECTO"
    echo ""
    echo "Siguiente paso:"
    echo "  DATASET=$DATASET TASK=$TASK sbatch train_base_models_fge_v2.sbatch"
else
    echo "⚠ Incidencias:"
    printf '  ❌ %s\n' "${FAILED[@]}"
    echo ""
    echo "Logs en: $LOG_DIR/"
    exit 1
fi

echo ""
echo "Fin: $(date)"
echo "============================================================================"
