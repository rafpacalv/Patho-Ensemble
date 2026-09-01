#!/bin/bash
#SBATCH --job-name=pipeline_submit
#SBATCH --partition=main
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=00:10:00
#SBATCH --output=logs/pipeline_submit_%j.out
#SBATCH --error=logs/pipeline_submit_%j.err

# ============================================================================
# Orquestador del pipeline completo
# ============================================================================
# Encadena las tres etapas con dependencias de SLURM (--dependency=afterok):
#
#   1. train_base_models.sbatch          modelos base ABMIL + predicciones train
#   2. train_base_models_fge_v2.sbatch   ciclos FGE + predicciones de snapshots
#   3. run_3_experiments_v2_improved.sbatch   meta-learners y comparativa
#
# Cada etapa se lanza como job independiente y sólo arranca si la anterior
# terminó bien. No se ocupa GPU esperando: este script sólo encola y sale.
#
# Uso (login node o sbatch, ambos válidos):
#
#   ./run_full_pipeline.sbatch                       # experimentos enabled: true
#   sbatch run_full_pipeline.sbatch
#   DATASET=cptac_coad TASK=MSI_H ./run_full_pipeline.sbatch
#   ALL_ACTIVE=1 ./run_full_pipeline.sbatch          # encola TODOS los enabled
#
# Seguimiento:
#   squeue -u $USER
#   tail -f logs/{dataset}_{task}/*.log
# ============================================================================

set -euo pipefail

PROJECT_DIR="/home/JKP6679/Patho-Ensemble"
cd "$PROJECT_DIR"

source $(conda info --base)/etc/profile.d/conda.sh
conda activate ensemble

mkdir -p "$PROJECT_DIR/logs"

# ---------------------------------------------------------------------------
# Determinar qué combinaciones (dataset, task) encolar
# ---------------------------------------------------------------------------
COMBOS=()
if [ -n "${DATASET:-}" ] && [ -n "${TASK:-}" ]; then
    COMBOS+=("$DATASET	$TASK")
elif [ "${ALL_ACTIVE:-0}" = "1" ]; then
    while IFS= read -r line; do
        [ -n "$line" ] && COMBOS+=("$line")
    done < <(python3 src/experiment_config.py --list-active)
else
    FIRST=$(python3 src/experiment_config.py --list-active | head -1)
    [ -n "$FIRST" ] && COMBOS+=("$FIRST")
fi

if [ ${#COMBOS[@]} -eq 0 ]; then
    echo "❌ No hay experimentos que lanzar."
    echo "   Marca alguno con 'enabled: true' en experiments.yaml,"
    echo "   o pasa DATASET=... TASK=... explícitamente."
    exit 1
fi

echo "============================================================================"
echo "ENCOLADO DEL PIPELINE"
echo "============================================================================"
echo "Fecha: $(date)"
echo "Experimentos a encolar: ${#COMBOS[@]}"
echo "============================================================================"

# ---------------------------------------------------------------------------
# Encolar cada combinación como una cadena de 3 jobs dependientes
# ---------------------------------------------------------------------------
for COMBO in "${COMBOS[@]}"; do
    IFS=$'\t' read -r DS TK <<< "$COMBO"

    echo ""
    echo "── $DS / $TK ──────────────────────────────────────────────────"

    mkdir -p "$PROJECT_DIR/logs/${DS}_${TK}"

    JID_BASE=$(sbatch --parsable \
        --export=ALL,DATASET="$DS",TASK="$TK" \
        --job-name="base_${DS}_${TK}" \
        train_base_models.sbatch)
    echo "  1. base ABMIL          job $JID_BASE"

    # Las variantes FGE son independientes entre sí: se lanzan en paralelo, una
    # por job. En secuencia sumarían el tiempo de todas y pasarían del cap.
    VARIANTS=$(python3 src/experiment_config.py --emit-shell --dataset "$DS" --task "$TK" \
               | sed -n "s/^FGE_VARIANTS='\(.*\)'$/\1/p")
    : "${VARIANTS:=fge}"

    FGE_JIDS=()
    for V in $VARIANTS; do
        J=$(sbatch --parsable \
            --dependency=afterok:"$JID_BASE" \
            --export=ALL,DATASET="$DS",TASK="$TK",FGE_ONLY="$V" \
            --job-name="fge_${V}_${DS}" \
            train_base_models_fge_v2.sbatch)
        FGE_JIDS+=("$J")
        echo "  2. FGE $(printf '%-12s' "$V")   job $J   (tras $JID_BASE)"
    done

    DEP_META=$(IFS=:; echo "${FGE_JIDS[*]}")
    JID_META=$(sbatch --parsable \
        --dependency=afterok:"$DEP_META" \
        --export=ALL,DATASET="$DS",TASK="$TK" \
        --job-name="meta_${DS}_${TK}" \
        run_3_experiments_v2_improved.sbatch)
    echo "  3. meta-learners       job $JID_META  (tras las ${#FGE_JIDS[@]} variantes)"

    echo "  logs: logs/${DS}_${TK}/"
done

echo ""
echo "============================================================================"
echo "✓ Encolado completo"
echo "============================================================================"
echo ""
echo "Estado de la cola:"
squeue -u "$USER" --format="%.10i %.24j %.10T %.12M %R" 2>/dev/null || true
echo ""
echo "Notas:"
echo "  • Una etapa que falle cancela las siguientes (dependencia afterok)."
echo "  • Etapas ya completadas se detectan y se saltan al reejecutar."
echo "  • Medido: ctranspath tarda ~6 min en 50 folds; los 3 modelos base, ~media"
echo "    hora. Los caps de tiempo de cada etapa son holgura, no estimaciones."
echo ""
echo "============================================================================"
