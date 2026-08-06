#!/bin/bash
#SBATCH --job-name=copia_emb_rapido
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=48:00:00
#SBATCH --mail-user=rafael.pachon.alvarez@gmail.com
#SBATCH --mail-type=END,FAIL
#SBATCH --output=copia_embeddings_rapido_%A.out
#SBATCH --error=copia_embeddings_rapido_%A.err

set -e

echo "=========================================="
echo "⚡ COPIA RÁPIDA DE EMBEDDINGS (PARALELO MÁXIMO)"
echo "=========================================="
echo "Inicio: $(date)"
echo ""

# Directorio origen y destino
SRC="/shared/home/PARADIS/datos/features"
DST="/home/JKP6679/Patho-Ensemble/PARADIS/datos/features"

# Crear directorio destino si no existe
mkdir -p "$DST"

echo "📍 Origen:      $SRC"
echo "📍 Destino:     $DST"
echo "⚙️  Trabajos paralelos: 32 (MÁXIMO)"
echo ""

# Opción: Copia MÁS RÁPIDA paralelizada por modelo dentro de cada dataset
echo "🚀 Iniciando copia ultra-paralela (por modelo)..."
echo ""

# Copiar cada carpeta de modelo en paralelo
find "$SRC" -maxdepth 2 -mindepth 2 -type d | \
  parallel -j 32 --line-buffer \
    'MODEL=$(basename {}) && \
     DATASET=$(basename $(dirname {})) && \
     echo "[$(date +%H:%M:%S)] Copiando: $DATASET/$MODEL ..." && \
     rsync -av --ignore-existing --no-checksum --whole-file {} '"$DST"'/$DATASET/ && \
     echo "[$(date +%H:%M:%S)] ✓ $DATASET/$MODEL"' || true

echo ""
echo "=========================================="
echo "📊 VERIFICACIÓN DE COPIA"
echo "=========================================="

# Contar archivos
SRC_FEATURES=$(find "$SRC" -name "*.h5" | wc -l)
DST_FEATURES=$(find "$DST" -name "*.h5" 2>/dev/null | wc -l)

echo "Archivos .h5 copiados: $DST_FEATURES / $SRC_FEATURES"
echo ""

# Mostrar estado por dataset
echo "Estado por dataset:"
for dataset in $(find "$SRC" -maxdepth 1 -mindepth 1 -type d -exec basename {} \;); do
    if [ -d "$SRC/$dataset" ]; then
        src_count=$(find "$SRC/$dataset" -name "*.h5" 2>/dev/null | wc -l)
        dst_count=$(find "$DST/$dataset" -name "*.h5" 2>/dev/null | wc -l)
        pct=$((dst_count * 100 / src_count))

        if [ "$src_count" -eq "$dst_count" ]; then
            echo "  ✓ $dataset: $dst_count/$src_count (100%)"
        else
            echo "  ⏳ $dataset: $dst_count/$src_count ($pct%)"
        fi
    fi
done

echo ""
echo "=========================================="
echo "✅ COPIA COMPLETADA"
echo "=========================================="
echo "Fin: $(date)"
echo ""
