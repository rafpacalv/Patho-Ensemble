#!/bin/bash
#SBATCH --job-name=test_spatial_gnn
#SBATCH --partition=main
#SBATCH --constraint=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=logs/test_spatial_gnn_%j.out
#SBATCH --error=logs/test_spatial_gnn_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=rafael.pachon.alvarez@gmail.com

# ============================================================================
# Puerta de la Etapa 1 del brazo espacial — EN GPU
# ============================================================================
# No entrena nada. Construye los grafos de slides reales y comprueba las
# propiedades de las que depende el resto del diseño.
#
#   sbatch test_spatial_gnn.sbatch
#   MODEL=ctranspath sbatch test_spatial_gnn.sbatch
#
# Por qué en GPU y no en el nodo de login: el nodo de login no tiene GPU, y las
# dos cifras que decide esta puerta sólo tienen sentido en el dispositivo donde
# se entrena de verdad.
#
#   1. El COSTE. En CPU (16 hilos) el brazo 'smooth' salió x1.78 frente a ABMIL,
#      no el x1.004 que predecía el conteo de FLOPs. El modelo de FLOPs es el
#      modelo equivocado: la agregación va limitada por ancho de banda con
#      accesos irregulares, mientras que el GEMM de la proyección (1536->512)
#      corre cerca del pico. En GPU el equilibrio entre ambos cambia, así que la
#      cifra de CPU no sirve para presupuestar el barrido.
#
#   2. El BACKEND. La agregación se despacha por dispositivo (graph_utils.
#      SlideGraph): index_add_ en CPU, donde es determinista y 2.1x más rápido
#      que la SpMM; SpMM en CUDA, donde el atomicAdd de index_add_ NO es
#      determinista y el protocolo de análisis cuenta empates como estadístico
#      de primera clase. Esta ejecución es la primera que recorre la rama CUDA.
#
# La equivalencia numérica entre ambas rutas la fija el propio test con
# allclose, así que el backend sólo puede cambiar el tiempo, no el resultado.
# ============================================================================

set -euo pipefail

mkdir -p logs

# El env del proyecto: 'base' no tiene patho_bench, del que depende src/utils.py
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate ensemble

cd "${SLURM_SUBMIT_DIR:-$(pwd)}"

WORK_DIR="${WORK_DIR:-/home/JKP6679/Patho-Ensemble/PARADIS/datos/patches}"
DATASET="${DATASET:-cptac_brca}"
TASK="${TASK:-TP53_mutation}"
MODEL="${MODEL:-uni_v2}"

echo "=============================================================================="
echo " Etapa 1 — puerta del brazo espacial"
echo "=============================================================================="
echo " job          : ${SLURM_JOB_ID:-local} en $(hostname)"
echo " work_dir     : ${WORK_DIR}"
echo " dataset/task : ${DATASET} / ${TASK}"
echo " modelo       : ${MODEL}"
echo "------------------------------------------------------------------------------"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader || true
echo "------------------------------------------------------------------------------"
python - <<'PY'
import torch
print(f" torch {torch.__version__} | cuda disponible: {torch.cuda.is_available()}"
      f" | dispositivo: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
PY
echo "=============================================================================="
echo

# --- la puerta en GPU (es la que decide) ------------------------------------
python test_spatial_gnn.py \
    --work_dir "${WORK_DIR}" \
    --train_source "${DATASET}" \
    --task_name "${TASK}" \
    --model "${MODEL}" \
    --device cuda
GPU_RC=$?

echo
echo "=============================================================================="
echo " Contraste en CPU (informativo: fija la comparación de backends)"
echo "=============================================================================="
# `|| true`: en CPU la puerta de coste puede fallar sin que eso invalide nada.
# Bajo `set -e` un rc distinto de 0 mataría el script antes del resumen.
python test_spatial_gnn.py \
    --work_dir "${WORK_DIR}" \
    --train_source "${DATASET}" \
    --task_name "${TASK}" \
    --model "${MODEL}" \
    --device cpu 2>&1 | sed -n '/Coste del forward/,$p' || true

echo
echo "=============================================================================="
if [ "${GPU_RC}" -eq 0 ]; then
    echo " PUERTA SUPERADA en GPU — listo para el cableado del motor (Etapa 2)"
else
    echo " PUERTA NO SUPERADA en GPU (rc=${GPU_RC}) — revisar antes de seguir"
fi
echo "=============================================================================="
exit "${GPU_RC}"
