#!/bin/bash
# Script para ejecutar todo el pipeline localmente (sin SLURM)
# Uso: ./run_all_steps.sh

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
PROJECT_DIR="/home/JKP6679/Patho-Ensemble"
WORK_DIR="$PROJECT_DIR/PARADIS/datos/patches"
ABMIL_DIR="$WORK_DIR/cptac_brca/TP53_mutation/abmil"

# Logging function
log_step() {
    echo -e "${BLUE}================================${NC}"
    echo -e "${BLUE}PASO: $1${NC}"
    echo -e "${BLUE}================================${NC}"
    echo ""
}

log_success() {
    echo -e "${GREEN}✅ $1${NC}"
    echo ""
}

log_error() {
    echo -e "${RED}❌ ERROR: $1${NC}"
    echo ""
}

log_info() {
    echo -e "${YELLOW}ℹ️  $1${NC}"
}

# Ensure logs directory exists
mkdir -p "$PROJECT_DIR/logs"

echo "=========================================="
echo "PATHO-ENSEMBLE: Pipeline Completo"
echo "=========================================="
echo "Inicio: $(date)"
echo "Directorio de trabajo: $WORK_DIR"
echo ""

# ============================================================================
# PASO 0: Limpieza de directorios obsoletos
# ============================================================================
log_step "0 - Limpieza de directorios obsoletos"

log_info "Removiendo symlink parche y directorios _train_eval antiguos..."

rm -f "$WORK_DIR/features/cptac_brca/features_uni_v2_monai" 2>/dev/null || true
rm -rf "$ABMIL_DIR/ctranspath_20x_224px_0px_overlap_train_eval" 2>/dev/null || true
rm -rf "$ABMIL_DIR/virchow_v1_20x_224px_0px_overlap_train_eval" 2>/dev/null || true
rm -rf "$ABMIL_DIR/uni_v2_20x_224px_0px_overlap_train_eval" 2>/dev/null || true
rm -f "$ABMIL_DIR/uni_v2_20x_224px_0px_overlap/split.csv" 2>/dev/null || true
rm -f "$WORK_DIR/cptac_brca/TP53_mutation/k=train.tsv" 2>/dev/null || true

log_success "Limpieza completada"

# ============================================================================
# PASO 1: Verificar que get_features_dir() funciona
# ============================================================================
log_step "1 - Verificar get_features_dir()"

cd "$PROJECT_DIR/src"

python3 -c "
from utils import get_features_dir
work_dir = '$WORK_DIR'

print('Verificando rutas de features...')
print()

# ctranspath (con sufijo _monai)
path_ctranspath = get_features_dir(work_dir, 'cptac_brca', 'ctranspath')
print(f'✅ ctranspath: {path_ctranspath}')

# uni_v2 (fallback sin sufijo)
path_uni_v2 = get_features_dir(work_dir, 'cptac_brca', 'uni_v2')
print(f'✅ uni_v2: {path_uni_v2}')

# virchow_v1 (con sufijo _monai)
path_virchow = get_features_dir(work_dir, 'cptac_brca', 'virchow_v1')
print(f'✅ virchow_v1: {path_virchow}')

print()
print('Todas las rutas resueltas correctamente.')
"

if [ $? -eq 0 ]; then
    log_success "get_features_dir() funciona correctamente"
else
    log_error "get_features_dir() falló"
    exit 1
fi

# ============================================================================
# PASO 2: Verificar que test_abmil.py se importa sin errores
# ============================================================================
log_step "2 - Verificar importación de test_abmil.py"

python3 -c "
import sys
import test_abmil
print('test_abmil importa correctamente')
" && log_success "test_abmil.py importa sin errores" || (log_error "test_abmil.py falla al importar"; exit 1)

# ============================================================================
# PASO 3: Smoke test — generar predicciones para ctranspath
# ============================================================================
log_step "3 - Smoke test: generar predicciones para ctranspath (50 folds)"

cd "$PROJECT_DIR"

log_info "Ejecutando test_abmil.py para ctranspath..."
python3 src/test_abmil.py \
    --foundational_model ctranspath \
    --latent_dim 768 \
    --work_dir "$WORK_DIR" \
    --train_source cptac_brca \
    --tissue_patching 20x_224px_0px_overlap \
    --task_name TP53_mutation \
    --epochs 100

# Verificar que se generaron 50 folds
FOLD_COUNT=$(ls -d "$ABMIL_DIR/ctranspath_20x_224px_0px_overlap_train_eval/val_outputs/fold_"* 2>/dev/null | wc -l)

if [ "$FOLD_COUNT" -eq 50 ]; then
    log_success "Se generaron 50 folds para ctranspath ($FOLD_COUNT directorios encontrados)"
else
    log_error "Se esperaban 50 folds pero se encontraron $FOLD_COUNT"
    exit 1
fi

# ============================================================================
# PASO 4: Re-entrenar uni_v2 con latent_dim=1536
# ============================================================================
log_step "4 - Re-entrenar uni_v2 (50 folds, latent_dim=1536)"

log_info "Este paso puede tomar 45-60 minutos..."
log_info "Monitorear con: tail -f logs/train_uni_v2.log"
echo ""

python3 src/train_abmil.py \
    --foundational_model uni_v2 \
    --latent_dim 1536 \
    --work_dir "$WORK_DIR" \
    --train_source cptac_brca \
    --tissue_patching 20x_224px_0px_overlap \
    --task_name TP53_mutation \
    --epochs 100 \
    --fold all

# Verificar que se completaron 50 folds
CKPT_COUNT=$(ls -d "$ABMIL_DIR/uni_v2_20x_224px_0px_overlap/checkpoints/fold_"* 2>/dev/null | wc -l)

if [ "$CKPT_COUNT" -eq 50 ]; then
    log_success "uni_v2 entrenado completamente ($CKPT_COUNT checkpoints)"
else
    log_error "Se esperaban 50 checkpoints pero se encontraron $CKPT_COUNT"
    exit 1
fi

# ============================================================================
# PASO 5: Generar predicciones de train para uni_v2 y virchow_v1
# ============================================================================
log_step "5 - Generar predicciones de train para uni_v2 y virchow_v1"

# uni_v2 (latent_dim=1536)
log_info "Generando predicciones para uni_v2..."
python3 src/test_abmil.py \
    --foundational_model uni_v2 \
    --latent_dim 1536 \
    --work_dir "$WORK_DIR" \
    --train_source cptac_brca \
    --tissue_patching 20x_224px_0px_overlap \
    --task_name TP53_mutation \
    --epochs 100

FOLD_COUNT_UNI=$(ls -d "$ABMIL_DIR/uni_v2_20x_224px_0px_overlap_train_eval/val_outputs/fold_"* 2>/dev/null | wc -l)
if [ "$FOLD_COUNT_UNI" -eq 50 ]; then
    log_success "uni_v2 predicciones generadas ($FOLD_COUNT_UNI folds)"
else
    log_error "uni_v2: se esperaban 50 folds pero se encontraron $FOLD_COUNT_UNI"
    exit 1
fi

# virchow_v1 (latent_dim=2560)
log_info "Generando predicciones para virchow_v1..."
python3 src/test_abmil.py \
    --foundational_model virchow_v1 \
    --latent_dim 2560 \
    --work_dir "$WORK_DIR" \
    --train_source cptac_brca \
    --tissue_patching 20x_224px_0px_overlap \
    --task_name TP53_mutation \
    --epochs 100

FOLD_COUNT_VIR=$(ls -d "$ABMIL_DIR/virchow_v1_20x_224px_0px_overlap_train_eval/val_outputs/fold_"* 2>/dev/null | wc -l)
if [ "$FOLD_COUNT_VIR" -eq 50 ]; then
    log_success "virchow_v1 predicciones generadas ($FOLD_COUNT_VIR folds)"
else
    log_error "virchow_v1: se esperaban 50 folds pero se encontraron $FOLD_COUNT_VIR"
    exit 1
fi

# ============================================================================
# PASO 6: Ejecutar los 3 experimentos de ensemble (ensemble4.py)
# ============================================================================
log_step "6 - Ejecutar ensemble4.py (3 experimentos)"

log_info "Entrenando meta-learner con los 3 modelos base..."
python3 src/ensemble4.py \
    --foundational_models ctranspath uni_v2 virchow_v1 \
    --work_dir "$WORK_DIR" \
    --train_source cptac_brca \
    --tissue_patching 20x_224px_0px_overlap \
    --task_name TP53_mutation

# Verificar que se generó el archivo de métricas
METRICS_FILE="$ABMIL_DIR/ensemble4/test_metrics_summary.json"
if [ -f "$METRICS_FILE" ]; then
    log_success "ensemble4 completado - métricas guardadas"
    echo ""
    log_info "Resultados de ensemble4:"
    cat "$METRICS_FILE" | python3 -m json.tool
else
    log_error "No se encontró archivo de métricas: $METRICS_FILE"
    exit 1
fi

# ============================================================================
# RESUMEN FINAL
# ============================================================================
echo ""
echo "=========================================="
echo -e "${GREEN}✅ PIPELINE COMPLETADO EXITOSAMENTE${NC}"
echo "=========================================="
echo "Fin: $(date)"
echo ""
echo "Resultados guardados en:"
echo "  - Checkpoints: $ABMIL_DIR/*/checkpoints/"
echo "  - Test outputs: $ABMIL_DIR/*/test_outputs/"
echo "  - Ensemble4: $ABMIL_DIR/ensemble4/"
echo ""
echo "Para ejecutar los 3 experimentos en cluster (opcional):"
echo "  sbatch -W run_3_experiments.sbatch"
echo ""
