# 📦 GUÍA DE COPIA DE EMBEDDINGS

Dos scripts para copiar rápidamente los embeddings (features) a tu carpeta local.

---

## 🚀 OPCIÓN 1: Copia Estándar (Recomendada)

**Archivo:** `copia_embeddings.sbatch`

### Especificaciones
- **Trabajos paralelos:** 16
- **CPU:** 8 cores
- **Memoria:** 64GB
- **Tiempo:** 48 horas
- **Velocidad:** Moderada (buena relación velocidad/recursos)

### Uso
```bash
cd /shared/home/JKP6679/Patho-Ensemble
sbatch copia_embeddings.sbatch
```

### Monitorear progreso
```bash
# Ver trabajos SLURM
squeue -u $USER

# Ver salida en tiempo real
tail -f copia_embeddings_*.out
```

---

## ⚡ OPCIÓN 2: Copia Rápida (Máximo Paralelismo)

**Archivo:** `copia_embeddings_rapido.sbatch`

### Especificaciones
- **Trabajos paralelos:** 32 (MÁXIMO)
- **CPU:** 16 cores
- **Memoria:** 128GB
- **Tiempo:** 48 horas
- **Velocidad:** Muy rápida (consume más recursos)

### Uso
```bash
cd /shared/home/JKP6679/Patho-Ensemble
sbatch copia_embeddings_rapido.sbatch
```

### Ventajas
- ✓ Copia mucho más rápida
- ✓ Paraleliza a nivel de modelo (no solo dataset)
- ✓ Muestra timestamp de progreso

### Desventajas
- ✗ Consume más recursos del cluster
- ✗ Puedes esperar cola más tiempo

---

## 📊 DIFERENCIAS CON EL ORIGINAL

### Script Original (`copia.sbatch`)
```
Copia: Todos los directorios (patches, wsis, qc_tiles, etc.)
Trabajos paralelos: 12
Tiempo estimado: 24 horas
```

### Scripts Nuevos
```
Copia: SOLO embeddings/features
Trabajos paralelos: 16-32
Tiempo estimado: 12-24 horas (menos datos que original)
Monitoreo: Muestra progreso por dataset
```

---

## ✅ VERIFICACIÓN DESPUÉS DE LA COPIA

Una vez completada la copia, verifica que todo se copió correctamente:

```bash
# Ver cantidad de archivos
ls -lh /home/JKP6679/Patho-Ensemble/PARADIS/datos/features/

# Contar archivos .h5
find /home/JKP6679/Patho-Ensemble/PARADIS/datos/features -name "*.h5" | wc -l

# Verificar dataset específico
ls -lh /home/JKP6679/Patho-Ensemble/PARADIS/datos/features/cptac_brca/
```

---

## 🎯 PRÓXIMOS PASOS

**Una vez que los embeddings estén copiados:**

1. ✅ Verifica que todos los datasets de features estén en `/home/JKP6679/Patho-Ensemble/PARADIS/datos/features/`

2. ✅ Cambia el `work_dir` por defecto en los scripts:
   ```bash
   # Editar estos archivos:
   src/train_abmil.py       # Línea 57
   src/test_abmil.py        # Línea 52
   src/ensemble.py          # Línea 117
   src/ensemble4.py         # Línea 153
   
   # Cambiar de:
   default="/shared/home/PARADIS/datos"
   
   # A:
   default="/home/JKP6679/Patho-Ensemble/PARADIS/datos"
   ```

3. ✅ Ejecuta una prueba rápida:
   ```bash
   python src/train_abmil.py \
       --foundational_model ctranspath \
       --latent_dim 768 \
       --train_source cptac_brca \
       --tissue_patching 20x_224px_0px_overlap \
       --task_name TP53_mutation \
       --epochs 5 \
       --fold 0 \
       --create_val
   ```

---

## 📈 ESTIMACIONES DE TIEMPO

| Escenario | Tiempo |
|-----------|--------|
| Copia estándar (opción 1) | 12-18 horas |
| Copia rápida (opción 2) | 8-12 horas |
| Con cola SLURM | +2-4 horas |

---

## 💡 RECOMENDACIÓN

**Para la mayoría de usuarios:** Usa `copia_embeddings.sbatch` (OPCIÓN 1)
- Buen balance entre velocidad y recursos
- Menos competencia por recursos del cluster

**Si tienes urgencia:** Usa `copia_embeddings_rapido.sbatch` (OPCIÓN 2)
- Mucho más rápido
- Requiere esperar menos si hay cola
