# Implementación: SVM/KNN/Naive Bayes + Análisis de Importancia por Modelo

Fecha: 2026-08-19  
Rama: `pruebas_rafa`  
Commits: `8948a2d` (Parte A), `5ab90b3` (Parte B)

## Resumen

Se han implementado dos extensiones complementarias al sistema de stacking (`ensemble4.py`):

### **Parte A: Nuevos Meta-Modelos SVM, KNN, Naive Bayes**

Motivación: Con pocas muestras de meta-training, los clasificadores simples suelen comportarse mejor que las redes neuronales. No incluyen variante Snapshot Ensemble (mínimo único, sin landscape cíclico).

**Cambios:**

- **src/ensemble4.py**:
  - Imports: `SVC`, `KNeighborsClassifier`, `GaussianNB` (sklearn)
  - Tres ramas nuevas en instanciación (líneas 624-637):
    - `svm`: SVC(probability=True, kernel=linear por defecto, C=1.0)
    - `knn`: KNeighborsClassifier(n_neighbors=5, weights="distance")
    - `nb`: GaussianNB() (sin hiperparámetros expuestos)
  - Tres flags CLI: `--meta_svm_c`, `--meta_svm_kernel`, `--meta_knn_k`
  - `results_subdir_map` actualizado: `ensemble4_svm`, `ensemble4_knn`, `ensemble4_nb`

- **run_3_experiments_v2_improved.sbatch**:
  - SUBDIR map con las tres entradas nuevas

- **experiments.yaml**:
  - Seis filas nuevas (f19–f24): svm_ins/oof, knn_ins/oof, nb_ins/oof
  - Comentario explicativo: "clasificadores que funcionan mejor con pocas muestras"

### **Parte B: Análisis de Importancia + Reescalado de Features**

Motivación: Determinar qué modelo base pesa más en LogisticRegression y usar eso para reescalar features antes de entrenar cualquier meta-modelo.

**Cambios:**

- **src/utils.py**:
  - Nueva función `model_importance_from_coefs(coefs, num_models, model_names, temperature)`:
    - Agrega coeficientes por bloque de modelo base (norma L2)
    - Normaliza a media 1.0
    - Soporta temperature scaling
    - Retorna `{model_name: weight}`

- **src/analyze_meta_importance.py** (nuevo):
  - Script standalone para análisis post-hoc de importancia
  - Uso: `python src/analyze_meta_importance.py --results_dir .../ensemble4 --foundational_models ctranspath uni_v2 ...`
  - Imprime tabla de importancias
  - Guarda `model_importance.json`

- **src/ensemble4.py** (clase Ensemble):
  - `__init__`: parámetros `model_names`, `reweight_by_model_importance`, `reweight_temperature`
  - Método `_reweight_features()`: entrena LR auxiliar, calcula pesos, rescala features
  - En `forward()`: aplica reweighting tras `_transform`, antes de `_fit_meta_model`
  - Persiste pesos en `model_importance_weights.json`
  - Dos flags CLI: `--reweight_by_model_importance`, `--reweight_temperature`

- **src/experiment_config.py**:
  - `META_DEFAULTS`: añade `"reweight": False`
  - `emit_shell()`: emite reweight como 7.º campo del string META_CONFIGS
  - Retrocompatibilidad: filas antiguas sin campo = reweight=0

- **run_3_experiments_v2_improved.sbatch**:
  - Lee REWEIGHT como 7.º campo (con default '0')
  - Lógica REWEIGHT_FLAG (análogo a DROP_FLAG)
  - `subdir_for()`: añade parámetro `reweight`, sufijo `_reweighted`

## Verificación

### Smoke Tests

```bash
# Test 1: Imports
python -m py_compile src/ensemble4.py src/utils.py src/analyze_meta_importance.py

# Test 2: Dry run (si dataset pequeño está disponible)
python src/ensemble4.py \
    --foundational_models ctranspath uni_v2 virchow_v1 \
    --work_dir /home/JKP6679/Patho-Ensemble/PARADIS/datos/patches \
    --train_source cervical_subtype \
    --tissue_patching 20x_224px_0px_overlap \
    --task_name subtype \
    --meta_model svm \
    --fold 0  # Un fold solamente
```

### Post-hoc Analysis

```bash
# Tras correr ensemble4.py con --meta_model logreg (u otro lineal):
python src/analyze_meta_importance.py \
    --results_dir /home/JKP6679/Patho-Ensemble/PARADIS/datos/patches/cptac_brca/TP53_mutation/abmil/ensemble4 \
    --foundational_models ctranspath uni_v2 virchow_v1 \
    [--temperature 1.0]
```

## Formato del Archivo de Configuración (Retrocompatible)

META_CONFIGS ahora emite 7 campos separados por `:`:

```
name:meta_model:base_source:meta_features:feature_space:drop_redundant:reweight
```

Ejemplos (retrocompatibles):
- Fila antigua (3 campos): `f0_logreg_ins:logreg:standard` → se aplican defaults para feats/space/dropc/**reweight**
- Fila completa (7 campos): `f0_logreg_ins:logreg:standard:insample:prob:0:1` → reweight activo

## Archivos Modificados

| Archivo | Cambios |
|---------|---------|
| `src/ensemble4.py` | +3 imports sklearn, +3 ramas de instanciación, +9 parámetros CLI, clase Ensemble enriquecida |
| `src/utils.py` | +función `model_importance_from_coefs` (~80 líneas) |
| `src/analyze_meta_importance.py` | Nuevo archivo (~200 líneas) |
| `src/experiment_config.py` | META_DEFAULTS+reweight, emit_shell con 7 campos |
| `run_3_experiments_v2_improved.sbatch` | +3 SUBDIR entries, lectura REWEIGHT, subdir_for() mejorado |
| `experiments.yaml` | +6 filas (f19–f24: svm/knn/nb ins/oof) |

## Notas de Implementación

### SVM: por qué kernel='linear'
- Con ~20 features (6 modelos × 3–4 clases) y ~75 samples de meta-train, la complejidad es baja.
- Linear expone `.coef_` para análisis de importancia (Parte B).
- RBF requeriría más tuning de C/gamma con tan pocas muestras.

### KNN: weights='distance'
- Evita empates (importante con pocas muestras).
- Favorece vecinos cercanos.
- Default n_neighbors=5 (razonable para ~75 samples).

### Reweighting: LR auxiliar, no el meta-modelo
- Permite reutilizar con **cualquier** meta-modelo (svm/knn/nb/mlp/etc).
- No cambia el meta-modelo elegido: es puro rescalado de features.
- Temperature scaling da control sobre cuánto acentuar diferencias de peso.

### Retrocompatibilidad META_CONFIGS
- El 7.º campo tiene default '0' en bash (`${REWEIGHT:-0}`)
- `experiment_config.py` coloca `reweight` al final, donde no rompe parseo posicional antiguo
- Filas YAML sin campo `reweight:` heredan default `False` de META_DEFAULTS

## Próximos Pasos (Opcionales)

1. **Experimentos en experiments.yaml**: añadir pares `_reweighted` para comparar con/sin reweight:
   ```yaml
   - {name: f21_knn_ins_rw,  meta_model: knn, meta_features: insample, reweight: true}
   - {name: f22_knn_oof_rw,  meta_model: knn, meta_features: oof,      reweight: true}
   ```

2. **Análisis de coeficientes**: integrar `analyze_meta_importance.py` en el pipeline de reporting.

3. **Tuning de hiperparámetros SVM/KNN**: crear `grid_search_svm_knn.py` similar a `grid_search_mlp.py`.

## Referencias

- Plan detallado: `/home/JKP6679/.claude/plans/estaba-pensando-en-dos-sparkling-treehouse.md`
- CLAUDE.md: líneas 111–117 (explicación de snapshot ensemble), 279–304 (regla SUBDIR)
- `SNAPSHOT_ENSEMBLE_ROADMAP.md:270`: justificación por qué logreg/lightgbm no tienen snapshot
