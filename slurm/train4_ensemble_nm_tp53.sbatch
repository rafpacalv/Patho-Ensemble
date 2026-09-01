#!/bin/bash
# Entrenamiento de 4 modelos ABMIL + ensemble promediado + ensemble Nelder-Mead AVANZADO
# Con validación automática (train/val/test) recreada en cada ejecución
#
# PASO 0: Crear/reemplazar splits con validación (train 85% / val 15%)
# PASO 1: Entrenar 4 modelos ABMIL (ctranspath, conch512, conch_v1_5, virchow_v1)
# PASO 2: Ensemble promediado (pesos por AUC-ROC en validación, aplicados en test)
# PASO 3: Ensemble Nelder-Mead AVANZADO (multi-start + análisis de sensibilidad)
# PASO 4: Comparación de resultados
#
# Smoke test (1 fold):  sbatch --export=ALL,FOLD=0 slurm/train4_ensemble_nm_tp53.sbatch
# Producción (50 folds): sbatch slurm/train4_ensemble_nm_tp53.sbatch
#
# Usar método diferente (opcional):
#   sbatch --export=ALL,NM_METHOD=differential-evolution slurm/train4_ensemble_nm_tp53.sbatch
#
#SBATCH --job-name=train4_ensemble_nm
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --gpus=1
#SBATCH --time=48:00:00
#SBATCH --mail-user=rafael.pachon.alvarez@gmail.com
#SBATCH --mail-type=END,FAIL
#SBATCH --output=train4_ensemble_nm_%j.out
#SBATCH --error=train4_ensemble_nm_%j.err

set -e

# Activar env correcto (ensemble, con patho_bench y torch)
source ~/miniconda3/etc/profile.d/conda.sh
conda activate ensemble

# Parámetros de entrenamiento (editables vía --export)
: "${WORK_DIR:=/shared/home/JKP6679/Patho-Ensemble/PARADIS/datos}"
: "${TRAIN_SOURCE:=cptac_brca}"
: "${TISSUE_PATCHING:=20x_224px_0px_overlap}"
: "${TASK_NAME:=TP53_mutation}"
: "${EPOCHS:=100}"
: "${FOLD:=all}"

# Parámetros de optimización Nelder-Mead AVANZADO
: "${NM_METHOD:=nelder-mead-multistart}"  # nelder-mead | nelder-mead-multistart | differential-evolution
: "${NM_N_STARTS:=10}"                    # Número de inicializaciones (para multi-start)
: "${NM_LAMBDA_REG:=0.01}"               # Coeficiente de regularización (0 = sin regularización)

MODELS=(ctranspath conch512 conch_v1_5 virchow_v1)
LATENT_DIMS=(768 768 768 2560)

echo "=========================================="
echo "ABMIL 4-model ensemble + Nelder-Mead AVANZADO"
echo "Dataset: ${TRAIN_SOURCE}/${TASK_NAME}"
echo "Tissue patching: ${TISSUE_PATCHING}"
echo "Epochs: ${EPOCHS}"
echo "Fold: ${FOLD}"
echo "NM Method: ${NM_METHOD} (n_starts=${NM_N_STARTS}, lambda_reg=${NM_LAMBDA_REG})"
echo "Work dir: ${WORK_DIR}"
echo "=========================================="
echo ""

cd /shared/home/JKP6679/Patho-Ensemble/src

# ========== PASO 0: Crear/reemplazar splits con validación ==========
echo "[PASO 0] Creando splits con validación (train/val/test)..."
SPLIT_FILE="$WORK_DIR/$TRAIN_SOURCE/$TASK_NAME/k=all.tsv"
SPLIT_WITH_VAL="$WORK_DIR/$TRAIN_SOURCE/$TASK_NAME/k=all_with_val.tsv"
VAL_FRAC=0.15
N_FOLDS=50

echo "  Input:  $SPLIT_FILE"
echo "  Output: $SPLIT_WITH_VAL"
echo "  Val frac: $VAL_FRAC (15% de train para validación)"
echo ""

python create_val_splits.py \
    -i "$SPLIT_FILE" \
    -o "$SPLIT_WITH_VAL" \
    --n-folds "$N_FOLDS" \
    --val-frac "$VAL_FRAC"

if [[ ! -f "$SPLIT_WITH_VAL" ]]; then
    echo "❌ Error: No se pudo crear splits con validación"
    exit 1
fi

# Reemplazar archivo original
cp "$SPLIT_WITH_VAL" "$SPLIT_FILE"
echo "✅ Splits con validación reemplazados en $SPLIT_FILE"
echo ""

# Verificar splits
TRAIN_COUNT=$(grep -c "train" "$SPLIT_FILE" || true)
VAL_COUNT=$(grep -c "val" "$SPLIT_FILE" || true)
TEST_COUNT=$(grep -c "test" "$SPLIT_FILE" || true)
echo "  Verificación - Train: $TRAIN_COUNT, Val: $VAL_COUNT, Test: $TEST_COUNT"
echo ""

# ========== PASO 1: Entrenar 4 modelos ==========
echo "[PASO 1] Entrenando 4 modelos ABMIL..."
for i in "${!MODELS[@]}"; do
    MODEL="${MODELS[$i]}"
    DIM="${LATENT_DIMS[$i]}"
    echo ""
    echo "  Entrenando ${MODEL} (dim=${DIM})..."
    python train_abmil.py \
        --foundational_model "$MODEL" \
        --latent_dim "$DIM" \
        --work_dir "$WORK_DIR" \
        --train_source "$TRAIN_SOURCE" \
        --tissue_patching "$TISSUE_PATCHING" \
        --task_name "$TASK_NAME" \
        --epochs "$EPOCHS" \
        --fold "$FOLD"
    echo "  ✅ ${MODEL} completado"
done

# ========== PASO 2: Ensemble promediado (MetricDistance en VALIDACIÓN) ==========
echo ""
echo "[PASO 2] Ensemble promediado (pesos por AUC-ROC en VALIDACIÓN)..."
python ensemble.py \
    --foundational_models "${MODELS[@]}" \
    --work_dir "$WORK_DIR" \
    --train_source "$TRAIN_SOURCE" \
    --tissue_patching "$TISSUE_PATCHING" \
    --task_name "$TASK_NAME" \
    --weights_type auc_roc
echo "✅ Ensemble promediado completado (pesos optimizados en validación, aplicados en test)"

# ========== PASO 3: Ensemble Nelder-Mead AVANZADO (optimización en VALIDACIÓN) ==========
echo ""
echo "[PASO 3] Ensemble Nelder-Mead AVANZADO (optimización en VALIDACIÓN)..."
echo "  Método: $NM_METHOD"
echo "  Multi-start inicializaciones: $NM_N_STARTS"
echo "  Regularización: $NM_LAMBDA_REG"
echo ""
python ensemble_nm_advanced.py \
    --foundational_models "${MODELS[@]}" \
    --work_dir "$WORK_DIR" \
    --train_source "$TRAIN_SOURCE" \
    --tissue_patching "$TISSUE_PATCHING" \
    --task_name "$TASK_NAME" \
    --method "$NM_METHOD" \
    --n_starts "$NM_N_STARTS" \
    --lambda_reg "$NM_LAMBDA_REG"
echo "✅ Ensemble Nelder-Mead AVANZADO completado (multi-start optimization)"

# ========== PASO 4: Comparación de resultados ==========
echo ""
echo "[PASO 4] Comparación de resultados (ambos evaluados en TEST)..."
echo ""
ENSEMBLE_METRICS="$WORK_DIR/$TRAIN_SOURCE/$TASK_NAME/abmil/ensemble/test_metrics_summary.json"
ENSEMBLE_NM_METRICS="$WORK_DIR/$TRAIN_SOURCE/$TASK_NAME/abmil/ensemble_nm_adv/test_metrics_summary.json"
COMPARISON_REPORT="$WORK_DIR/$TRAIN_SOURCE/$TASK_NAME/abmil/ensemble_nm_adv/comparison_report.txt"

echo "=== ENSEMBLE PROMEDIADO (AUC-ROC en validación) ==="
if [[ -f "$ENSEMBLE_METRICS" ]]; then
    python -m json.tool "$ENSEMBLE_METRICS" 2>/dev/null | grep -E "auc|mean|se" || cat "$ENSEMBLE_METRICS"
else
    echo "Archivo no encontrado: $ENSEMBLE_METRICS"
fi

echo ""
echo "=== ENSEMBLE NELDER-MEAD AVANZADO (${NM_METHOD}) ==="
if [[ -f "$ENSEMBLE_NM_METRICS" ]]; then
    python -m json.tool "$ENSEMBLE_NM_METRICS" 2>/dev/null | grep -E "auc|mean|se" || cat "$ENSEMBLE_NM_METRICS"
else
    echo "Archivo no encontrado: $ENSEMBLE_NM_METRICS"
fi

echo ""
echo "=== REPORTE COMPARATIVO ==="
if [[ -f "$COMPARISON_REPORT" ]]; then
    cat "$COMPARISON_REPORT"
else
    echo "Archivo no encontrado: $COMPARISON_REPORT"
fi

echo ""
echo "=========================================="
echo "✅ Pipeline completado exitosamente"
echo ""
echo "  Resultados ensemble promediado:"
echo "    $ENSEMBLE_METRICS"
echo ""
echo "  Resultados ensemble Nelder-Mead avanzado:"
echo "    $ENSEMBLE_NM_METRICS"
echo ""
echo "  Reporte comparativo:"
echo "    $COMPARISON_REPORT"
echo ""
echo "  Estadísticas de optimización:"
echo "    $WORK_DIR/$TRAIN_SOURCE/$TASK_NAME/abmil/optimization_stats.json"
echo "=========================================="
