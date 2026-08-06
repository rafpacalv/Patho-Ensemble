#!/bin/bash
#SBATCH --job-name=copia_embeddings
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=48:00:00
#SBATCH --mail-user=rafael.pachon.alvarez@gmail.com
#SBATCH --mail-type=END,FAIL
#SBATCH --output=copia_embeddings_%A.out
#SBATCH --error=copia_embeddings_%A.err

set -e

echo "=========================================="
echo "📦 INICIO COPIA DE EMBEDDINGS (FEATURES)"
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
echo ""

# Opción 1: Copia rápida paralelizada por dataset
echo "🚀 Iniciando copia paralelizada de embeddings..."
echo "  (Copiando cada dataset en paralelo con 16 trabajos)"
echo ""

# Encontrar todos los directorios de datasets en features
# Paralelizar la copia de cada dataset
find "$SRC" -maxdepth 1 -type d ! -name features | \
  parallel -j 16 \
    'echo "  Copiando: {} ..." && \
     rsync -av --ignore-existing --no-checksum --whole-file {} '"$DST"'/ && \
     echo "  ✓ Completado: {}"' || true

echo ""
echo "=========================================="
echo "📊 VERIFICACIÓN DE COPIA"
echo "=========================================="

# Contar archivos copiados
SRC_FEATURES=$(find "$SRC" -name "*.h5" | wc -l)
DST_FEATURES=$(find "$DST" -name "*.h5" 2>/dev/null | wc -l)

echo "Archivos .h5 en origen:  $SRC_FEATURES"
echo "Archivos .h5 en destino: $DST_FEATURES"
echo ""

# Verificar por dataset
echo "Verificación por dataset:"
for dataset in $(ls "$SRC"); do
    if [ -d "$SRC/$dataset" ]; then
        src_count=$(find "$SRC/$dataset" -name "*.h5" 2>/dev/null | wc -l)
        dst_count=$(find "$DST/$dataset" -name "*.h5" 2>/dev/null | wc -l)

        if [ "$src_count" -eq "$dst_count" ]; then
            echo "  ✓ $dataset: $dst_count/$src_count"
        else
            echo "  ⚠️  $dataset: $dst_count/$src_count (INCOMPLETO)"
        fi
    fi
done

echo ""
echo "=========================================="
echo "✅ COPIA COMPLETADA"
echo "=========================================="
echo "Fin: $(date)"
echo ""
echo "📌 Próximo paso:"
echo "  Verifica que los embeddings se copiaron correctamente"
echo "  con: ls -lh $DST/"
echo ""
