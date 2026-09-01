#!/bin/bash
#SBATCH --job-name=train_base_models_fge
#SBATCH --partition=main
#SBATCH --constraint=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=16:00:00
#SBATCH --output=logs/train_base_models_fge_%j.out
#SBATCH --error=logs/train_base_models_fge_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=rafael.pachon.alvarez@gmail.com

# ============================================================================
# Modelos base ABMIL con Fast Geometric Ensembles (Garipov et al., ICLR 2018)
# ============================================================================
# Parte de los checkpoints ya convergidos y aplica ciclos cortos de LR
# piecewise-linear, tomando un snapshot al final de cada ciclo.
#
# Configuración: experiments.yaml (override con DATASET=... TASK=...)
#
#   DATASET=cptac_brca TASK=TP53_mutation sbatch train_base_models_fge_v2.sbatch
#
# Requiere: train_base_models.sbatch completado (checkpoints/fold_*/model.pt).
#
# Duración medida en cptac_brca/TP53_mutation: ~1 h para los 3 modelos
# (ctranspath 10 min, uni_v2 25 min, virchow_v1 22 min). El cap de 16 h es
# holgura para datasets mayores, no una estimación.
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

DATASET="${DATASET:-}"
TASK="${TASK:-}"
if [ -z "$DATASET" ] || [ -z "$TASK" ]; then
    read -r DATASET TASK < <(python3 src/experiment_config.py --list-active | head -1)
fi
eval "$(python3 src/experiment_config.py --emit-shell --dataset "$DATASET" --task "$TASK")"

LOG_DIR="$PROJECT_DIR/logs/${DATASET}_${TASK}"
mkdir -p "$LOG_DIR"

ABMIL_DIR="$WORK_DIR/$DATASET/$TASK/abmil"
# Folds esperados = min(max_folds pedidos, folds que realmente tiene el TSV).
# Ver comentario equivalente en train_base_models.sbatch: el nº de columnas
# fold_* depende del dataset (cptac_brca: 50, cervical_subtype: 5).
AVAIL_FOLDS=$(head -1 "$WORK_DIR/$DATASET/$TASK/k=all.tsv" \
              | tr '\t' '\n' | grep -c '^fold_' || echo 0)
EXPECTED_FOLDS=$(awk -v a="$AVAIL_FOLDS" -v m="${MAX_FOLDS:-0}" \
                 'BEGIN{print (m>0 && m<a) ? m : a}')

# ---------------------------------------------------------------------------
# Comprobaciones previas
# ---------------------------------------------------------------------------
echo "============================================================================"
echo "COMPROBACIONES PREVIAS"
echo "============================================================================"

ABORT=0
for MODEL in $MODELS; do
    # Exigir model.pt real, no sólo el directorio: los fold_* con checkpoints
    # legacy (epoch_9.pt) no sirven para arrancar FGE.
    N=$(count_files "$ABMIL_DIR/${MODEL}_${TISSUE_PATCHING}"/checkpoints/fold_*/model.pt)
    if [ "$N" -ge "$EXPECTED_FOLDS" ]; then
        echo "✓ $MODEL: $N checkpoints base"
    else
        echo "❌ $MODEL: sólo $N/$EXPECTED_FOLDS checkpoints base (model.pt)"
        ABORT=1
    fi
done

if [ "$ABORT" -eq 1 ]; then
    echo ""
    echo "❌ Faltan prerrequisitos. Ejecuta primero:"
    echo "   DATASET=$DATASET TASK=$TASK sbatch train_base_models.sbatch"
    exit 1
fi

echo ""
echo "============================================================================"
echo "FGE — $DATASET / $TASK"
echo "============================================================================"
echo "Inicio:   $(date)"
echo "modelos:  $MODELS"
echo "variantes: ${FGE_ONLY:-$FGE_VARIANTS}"
echo "============================================================================"

TOTAL_START=$(date +%s)
FAILED=()

# La variante puede acotarse con FGE_ONLY=fge_wbase (util para lanzarlas en
# paralelo como jobs SLURM independientes en vez de secuencialmente).
VARIANTS="${FGE_ONLY:-$FGE_VARIANTS}"

for VARIANT in $VARIANTS; do
    # Recuperar los parametros de esta variante: FGE_CFG_<NOMBRE en mayusculas>
    CFG_VAR="FGE_CFG_$(echo "$VARIANT" | tr '[:lower:]-' '[:upper:]_')"
    CFG="${!CFG_VAR:-}"
    if [ -z "$CFG" ]; then
        echo "❌ Variante '$VARIANT' sin configuracion ($CFG_VAR no definida)"
        FAILED+=("$VARIANT (sin config)")
        continue
    fi
    read -r V_LR1 V_LR2 V_CLEN V_NCYC V_PAT V_INCBASE V_SCHED <<< "$CFG"
    INC_FLAG=""
    [ "$V_INCBASE" = "1" ] && INC_FLAG="--include_base"
    # Variantes anteriores a la introducción de `schedule` no traen el campo.
    V_SCHED="${V_SCHED:-fge}"

    echo ""
    echo "############################################################################"
    echo "# VARIANTE: $VARIANT   [schedule=$V_SCHED]"
    echo "#   lr_1=$V_LR1 lr_2=$V_LR2 cycle_length=$V_CLEN n_cycles=$V_NCYC"
    echo "#   cycle_patience=$V_PAT include_base=$V_INCBASE"
    echo "############################################################################"

    for MODEL in $MODELS; do
        echo ""
        echo "╔════════════════════════════════════════════════════════════════════════╗"
        echo "║ $VARIANT / $MODEL"
        echo "╚════════════════════════════════════════════════════════════════════════╝"

        MODEL_DIR="$ABMIL_DIR/${MODEL}_${TISSUE_PATCHING}"

        DONE=$(count_files "$MODEL_DIR/checkpoints_${VARIANT}"/fold_*)
        if [ "$DONE" -ge "$EXPECTED_FOLDS" ]; then
            echo "✓ ya completo (50 folds). Se salta."
            continue
        fi

        MSTART=$(date +%s)
        if python src/train_abmil_fge.py \
            --foundational_model "$MODEL" \
            --work_dir "$WORK_DIR" \
            --train_source "$DATASET" \
            --tissue_patching "$TISSUE_PATCHING" \
            --task_name "$TASK" \
            --epochs "$EPOCHS" \
            --fold all \
            ${MAX_FOLDS:+--max_folds "$MAX_FOLDS"} \
            --n_cycles "$V_NCYC" \
            --cycle_length "$V_CLEN" \
            --lr_1 "$V_LR1" \
            --lr_2 "$V_LR2" \
            --cycle_patience "$V_PAT" \
            --fge_tag "$VARIANT" \
            --schedule "$V_SCHED" \
            $INC_FLAG \
            2>&1 | tee -a "$LOG_DIR/train_${VARIANT}_${MODEL}.log"; then

            DUR=$(( $(date +%s) - MSTART ))
            TOTS=$(count_files "$MODEL_DIR/checkpoints_${VARIANT}"/fold_*/snapshot_*.pt)
            NF=$(count_files "$MODEL_DIR/checkpoints_${VARIANT}"/fold_*)
            AVG=$(awk -v t="$TOTS" -v n="$NF" 'BEGIN{printf (n>0)?"%.1f":"0", t/(n?n:1)}')
            echo "[$(date)] ✓ $VARIANT/$MODEL — $((DUR/60)) min, $AVG snapshots/fold (media)"
        else
            echo "[$(date)] ❌ $VARIANT/$MODEL FALLÓ"
            FAILED+=("$VARIANT/$MODEL")
            continue
        fi

        # Predicciones de train con los snapshots de esta variante
        if python src/test_abmil_fge.py \
            --foundational_model "$MODEL" \
            --work_dir "$WORK_DIR" \
            --train_source "$DATASET" \
            --tissue_patching "$TISSUE_PATCHING" \
            --task_name "$TASK" \
            --fge_tag "$VARIANT" \
            2>&1 | tee -a "$LOG_DIR/test_${VARIANT}_${MODEL}.log"; then
            if [ ! -f "${MODEL_DIR}_train_eval_${VARIANT}/val_outputs_${VARIANT}/fold_0/preds.npy" ]; then
                echo "⚠ $VARIANT/$MODEL: no se generó fold_0/preds.npy"
                FAILED+=("$VARIANT/$MODEL (predicciones)")
            fi
        else
            echo "❌ $VARIANT/$MODEL: fallo en predicciones"
            FAILED+=("$VARIANT/$MODEL (predicciones)")
        fi
    done
done

# ---------------------------------------------------------------------------
# Resumen
# ---------------------------------------------------------------------------
TOTAL_DUR=$(( $(date +%s) - TOTAL_START ))

echo ""
echo "============================================================================"
echo "RESUMEN FGE — $DATASET / $TASK"
echo "============================================================================"
echo "Duración total: $((TOTAL_DUR/3600)) h $(( (TOTAL_DUR%3600)/60 )) min"
echo ""

for VARIANT in $VARIANTS; do
    echo "  [$VARIANT]"
    for MODEL in $MODELS; do
        B="$ABMIL_DIR/${MODEL}_${TISSUE_PATCHING}"
        TOTS=$(count_files "$B/checkpoints_${VARIANT}"/fold_*/snapshot_*.pt)
        NF=$(count_files "$B/checkpoints_${VARIANT}"/fold_*)
        S=$(awk -v t="$TOTS" -v n="$NF" 'BEGIN{printf (n>0)?"%.1f":"0", t/(n?n:1)}')
        F=$(count_files "${B}_train_eval_${VARIANT}/val_outputs_${VARIANT}/fold_"*)
        echo "    $MODEL: $S snapshots/fold (media) | train_eval=$F/$EXPECTED_FOLDS folds"
    done
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
    echo ""
    echo "Logs en: $LOG_DIR/"
    exit 1
fi

echo ""
echo "Fin: $(date)"
echo "============================================================================"
