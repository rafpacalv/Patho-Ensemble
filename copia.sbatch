#!/bin/bash
#SBATCH --job-name=copia_patches
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=48:00:00
#SBATCH --mail-user=rafael.pachon.alvarez@gmail.com
#SBATCH --mail-type=END,FAIL
#SBATCH --output=copia_patches_%A.out
#SBATCH --error=copia_patches_%A.err

set -e

echo "=========================================="
echo "📋 COPIA DE PATCHES (EXCLUYE PROBLEMAS)"
echo "=========================================="
echo "Inicio: $(date)"
echo ""

# Directorio origen y destino
SRC="/shared/home/PARADIS/datos/patches"
DST="/home/JKP6679/Patho-Ensemble/PARADIS/datos/patches"

# Crear directorio destino si no existe
mkdir -p "$DST"

echo "📍 Origen:      $SRC"
echo "📍 Destino:     $DST"
echo "⚙️  Trabajos paralelos: 16"
echo "🚫 Excluidas:"
echo "   - patches/mama/BCI_dataset_LMG (error de acceso)"
echo ""

# Copiar cada carpeta de dataset en paralelo, EXCLUYENDO la carpeta problemática
echo "🚀 Iniciando copia paralela (por dataset)..."
echo ""

find "$SRC" -maxdepth 1 -mindepth 1 -type d | \
  grep -v "BCI_dataset_LMG" | \
  parallel -j 16 --line-buffer \
    'DATASET=$(basename {}) && \
     echo "[$(date +%H:%M:%S)] Copiando: $DATASET ..." && \
     rsync -av --ignore-existing --no-checksum --whole-file \
       --exclude="BCI_dataset_LMG" \
       {} '"$DST"'/ && \
     echo "[$(date +%H:%M:%S)] ✓ $DATASET"' || true

echo ""
echo "=========================================="
echo "📊 VERIFICACIÓN DE COPIA"
echo "=========================================="

# Contar archivos
SRC_PATCHES=$(find "$SRC" -type f ! -path "*/BCI_dataset_LMG/*" 2>/dev/null | wc -l)
DST_PATCHES=$(find "$DST" -type f 2>/dev/null | wc -l)

echo "Archivos copiados: $DST_PATCHES / $SRC_PATCHES"
if [ "$SRC_PATCHES" -gt 0 ]; then
    PCT=$((DST_PATCHES * 100 / SRC_PATCHES))
    echo "Progreso: $PCT%"
fi
echo ""

# Mostrar estado por dataset
echo "Estado por dataset:"
for dataset_src in $(find "$SRC" -maxdepth 1 -mindepth 1 -type d ! -name "*BCI_dataset_LMG*" -exec basename {} \;); do
    if [ -d "$SRC/$dataset_src" ]; then
        src_count=$(find "$SRC/$dataset_src" -type f ! -path "*/BCI_dataset_LMG/*" 2>/dev/null | wc -l)
        dst_count=$(find "$DST/$dataset_src" -type f 2>/dev/null | wc -l)

        if [ "$src_count" -eq 0 ]; then
            echo "  ⏭️  $dataset_src: 0 archivos en origen"
        elif [ "$src_count" -eq "$dst_count" ]; then
            echo "  ✓ $dataset_src: $dst_count/$src_count (100%)"
        else
            pct=$((dst_count * 100 / src_count))
            echo "  ⏳ $dataset_src: $dst_count/$src_count ($pct%)"
        fi
    fi
done

echo ""
echo "=========================================="
echo "✅ COPIA COMPLETADA"
echo "=========================================="
echo "Fin: $(date)"
echo ""
echo "⚠️  NOTA: No se copió patches/mama/BCI_dataset_LMG (error de permiso)"
echo ""
