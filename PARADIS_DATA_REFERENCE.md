# PARADIS — Referencia de la estructura de datos

Guía de la copia local de PARADIS en `/home/JKP6679/Patho-Ensemble/PARADIS/datos/`.

> **Lo más importante**: `--work_dir` debe apuntar a `.../PARADIS/datos/patches`.
> Ver [Por qué work_dir termina en `/patches`](#por-qué-work_dir-termina-en-patches).

---

## Estructura de directorios

```
/home/JKP6679/Patho-Ensemble/PARADIS/datos/
│
├── features/                       # Embeddings precalculados (HDF5)
│   └── {dataset}/
│       ├── features_{modelo}_monai/    # convención con sufijo
│       │   └── *.h5
│       └── features_{modelo}/          # convención sin sufijo (p.ej. uni_v2)
│           └── *.h5
│
├── patches/                        # ← RAÍZ DE work_dir
│   └── {dataset}/
│       ├── {patching_strategy}/        # p.ej. 20x_224px_0px_overlap
│       │   ├── patches/                #   coordenadas de los parches
│       │   └── visualization/          #   miniaturas JPG
│       │
│       └── {task}/                     # p.ej. TP53_mutation
│           ├── k=all.tsv               #   splits maestros (50 folds)
│           ├── config.yaml             #   metadatos de la tarea
│           └── abmil/
│               ├── {modelo}_{strategy}/
│               │   ├── checkpoints/fold_*/model.pt
│               │   ├── val_outputs/fold_*/
│               │   ├── test_outputs/fold_*/
│               │   ├── checkpoints_fge/fold_*/snapshot_*.pt
│               │   ├── val_outputs_fge/fold_*/
│               │   └── test_outputs_fge/fold_*/
│               ├── {modelo}_{strategy}_train_eval/
│               │   └── val_outputs/fold_*/       ← features del meta-learner
│               ├── {modelo}_{strategy}_train_eval_fge/
│               │   └── val_outputs_fge/fold_*/   ← ídem, versión FGE
│               └── ensemble4*/                    ← resultados de ensemble
│
├── wsis/                           # Whole-slide images originales
└── qc_tiles/                       # Control de calidad
```

---

## Por qué `work_dir` termina en `/patches`

Los embeddings y los splits **no cuelgan del mismo directorio**:

| Dato | Ruta |
|---|---|
| Embeddings | `datos/features/{dataset}/` |
| Splits y resultados | `datos/patches/{dataset}/{task}/` |

El código construye las rutas así:

- `utils.py::get_features_dir` → `{work_dir}/features/{dataset}` y, como alternativa, `{work_dir}/../features/{dataset}`
- `train_abmil.py` → `{work_dir}/{dataset}/{task}/k=all.tsv`
- salidas → `{work_dir}/{dataset}/{task}/abmil/{modelo}_{strategy}/`

Con `work_dir = .../datos/patches`:

- splits → `datos/patches/{dataset}/{task}/k=all.tsv` ✓
- resultados → `datos/patches/{dataset}/{task}/abmil/` ✓
- features → `datos/patches/../features/{dataset}` = `datos/features/{dataset}` ✓

> **Cuidado con `datos/{dataset}/`** (por ejemplo `datos/cptac_brca/`): es un árbol de
> pruebas de ejecuciones antiguas, con sus propios splits y checkpoints. Usar
> `work_dir = .../datos` hace que se lea de ahí en vez de los datos reales.

---

## Embeddings

**Patrón**: `datos/features/{dataset}/features_{modelo}[_monai]/*.h5`

Se prueban dos convenciones de nombre, en este orden: primero con sufijo `_monai`,
luego sin él. En `cptac_brca` conviven ambas — `features_ctranspath_monai` pero
`features_uni_v2` sin sufijo.

**Estructura del HDF5**:

```
{slide_id}.h5
├── coords     (N_parches, 2)  int64      coordenadas (x, y)
└── features   (N_parches, D)  float32    embeddings
```

### Dimensiones (verificadas en cptac_brca)

| Modelo | Directorio | Dimensión |
|---|---|---|
| ctranspath | `features_ctranspath_monai` | 768 |
| uni_v2 | `features_uni_v2` | **1536** |
| virchow_v1 | `features_virchow_v1_monai` | **2560** |
| uni_v1 | `features_uni_v1` | 1024 |
| conch512 | `features_conch512_monai` | 768 |
| phikon_v2 | `features_phikon_v2` | 1024 |
| virchow2 | `features_virchow2` | 2560 |

**Ya no hace falta pasar `--latent_dim`**: si se omite, se lee del propio `.h5`.
Esto elimina la clase de errores del tipo:

```
RuntimeError: size mismatch for projection.0.weight:
  copying a param with shape torch.Size([512, 1536]) from checkpoint,
  the shape in current model is torch.Size([512, 768])
```

Para comprobar una dimensión a mano:

```bash
cd src && python3 -c "
from utils import get_features_dir, detect_latent_dim
d = get_features_dir('/home/JKP6679/Patho-Ensemble/PARADIS/datos/patches', 'cptac_brca', 'uni_v2')
print(d, detect_latent_dim(d))"
```

---

## Splits

**Fichero maestro**: `datos/patches/{dataset}/{task}/k=all.tsv` (separado por tabuladores)

```
case_id    slide_id                 TP53_mutation    fold_0   fold_1  ...  fold_49
01BR001    01BR001-0684a407         1                train    train   ...  test
01BR002    01BR002-a1c2d4e5         0                train    val     ...  train
```

- `case_id`: paciente (la estratificación es a nivel de paciente)
- `slide_id`: identificador de slide, coincide con el nombre del `.h5`
- `fold_0 … fold_49`: `train`, `val` o `test`

**Reparto típico** (cptac_brca / TP53_mutation, 112 slides, 103 pacientes):
train 75 · val 15 · test 22

Los splits de PARADIS **ya incluyen partición de validación**; `--create_val` sólo
hace falta para datasets propios.

---

## Datasets y tareas disponibles

Cifras leídas del propio dato. Versiones anteriores de esta guía daban tamaños
inflados heredados de notas viejas — `cptac_coad` figuraba con 327 muestras cuando
en realidad es el dataset **más pequeño**.

| Dataset | Tareas | Slides | Pacientes | Folds | Dirs. features |
|---|---|---:|---:|---:|---:|
| **imp** | grade | 5333 | 5333 | **1** ⚠️ | 12 |
| **cervical_subtype** | subtype (4 clases) | 599 | 599 | **5** | 12 |
| **hancook** | 7 tareas (grading, invasión…) | 387 | 383 | **5** | 1 ⚠️ |
| **cptac_luad** | EGFR, KRAS, STK11, TP53, OS, Immune_class | 324 | 108 | 50 | 1 ⚠️ |
| **cptac_lscc** | ARID1A, KEAP1, Histologic_Grade, Immune_class | 304 | 108 | 50 | 8 |
| **cptac_hnsc** | CASP8_mutation, Histologic_Grade, Immune_class | 256 | 107 | 50 | 8 |
| **cptac_ccrcc** | VHL, PBRM1, BAP1, Immune_class | 245 | 103 | 50 | 8 |
| **cptac_gbm** | EGFR_mutation, TP53_mutation, Immune_class | 243 | 99 | 50 | 15 |
| **bc_therapy** | er_status, her2_status, grade, RCB | 166 | 166 | 50 | 12 |
| **crc_outcomes** | 5 tareas (braf_*) | 136 | 134 | 50 | 1 ⚠️ |
| **cptac_brca** | TP53_mutation, PIK3CA_mutation, Immune_class | 112 | 103 | 50 | 12 |
| **cptac_coad** | TP53, KRAS, APC, ARID1A, ACVR2A, PIK3CA, SETD1B, MSI_H, Immune_class | 98 | 94 | 50 | 12 |

⚠️ Con un solo directorio de features no se pueden montar los 3 modelos base por
defecto; hay que sobrescribir `models` en la fila correspondiente de
`experiments.yaml`.

**Tres columnas que engañan si se leen deprisa:**

- **Pacientes, no slides**, es el tamaño muestral efectivo: la estratificación es a
  nivel de paciente, así que los `cptac_*` tienen mucho menos dato independiente
  del que sugieren sus slides (`cptac_luad`: 324 slides pero 108 pacientes).
- **El nº de folds no es 50 en todos**: `cervical_subtype` y `hancook` tienen 5, e
  `imp` tiene **1**. Dar por hecho 50 hace que una validación del tipo
  `[ "$N" -eq 50 ]` marque como fallo un entrenamiento correcto.
- ⚠️ **`imp`, con un solo fold**, toma otra rama en `utils.Metrics`: bootstrap sobre
  las predicciones agrupadas, con claves `lower`/`upper` en vez de `se`.

### Cómo se calculan los intervalos de confianza

En multi-fold, `utils.Metrics` reporta **`se = std(scores por fold) / √n_folds`**;
la rama de bootstrap sólo se usa con `num_folds == 1`. De ahí dos cosas poco
intuitivas:

- **El nº de pacientes no entra en la fórmula.** Elegir un dataset con más
  pacientes no estrecha el intervalo por sí solo.
- Lo que manda es el **nº de folds** y la **estabilidad del score de cada fold**,
  que tiran en sentidos opuestos: más folds implica test más pequeños y por tanto
  scores por fold más ruidosos.

Medido sobre estos datos:

| | std entre folds | folds | SE | anchura IC |
|---|---:|---:|---:|---:|
| cptac_brca (test=22/fold) | 0.092 | 50 | 0.013 | 0.051 |
| cervical_subtype (test=120/fold) | 0.020 | 5 | 0.009 | **0.035** |

cervical_subtype acaba con el intervalo **más estrecho** pese a tener una décima
parte de folds, porque cada fold estima su AUC con 120 muestras en vez de 22.

---

## Comandos útiles

```bash
# Datasets disponibles
ls PARADIS/datos/patches/

# Tareas de un dataset (las que tienen k=all.tsv)
ls PARADIS/datos/patches/cptac_brca/

# Modelos fundacionales de un dataset
ls PARADIS/datos/features/cptac_brca/ | grep features_

# Resultados ABMIL de una tarea
ls PARADIS/datos/patches/cptac_brca/TP53_mutation/abmil/

# Checkpoints reales (model.pt, no basta con que exista el directorio)
ls PARADIS/datos/patches/cptac_brca/TP53_mutation/abmil/ctranspath_20x_224px_0px_overlap/checkpoints/fold_*/model.pt | wc -l
```

```bash
# Resumen de un split
python3 -c "
import pandas as pd
df = pd.read_csv('PARADIS/datos/patches/cptac_brca/TP53_mutation/k=all.tsv', sep='\t')
print('slides:', len(df), '| pacientes:', df.case_id.nunique())
print('etiquetas:', df.TP53_mutation.value_counts().to_dict())
print('fold_0:', df.fold_0.value_counts().to_dict())"
```

---

## Problemas frecuentes

### `FileNotFoundError` al buscar features

El mensaje enumera todas las rutas probadas. Comprueba que `--work_dir` termina en
`/patches` y que el modelo existe:

```bash
ls PARADIS/datos/features/{dataset}/ | grep features_
```

### Checkpoints legacy que no cargan

Algunos folds contienen `epoch_9.pt` (formato antiguo de `patho_bench`, con otra
arquitectura) en vez de `model.pt`. No son compatibles con el pipeline actual y hay
que reentrenar. Los `.sbatch` ya comprueban la presencia de `model.pt`, no sólo la
del directorio.

### Los resultados aparecen en un sitio inesperado

Casi siempre es `--work_dir` apuntando a `.../datos` en lugar de `.../datos/patches`,
lo que dirige las escrituras al árbol de pruebas `datos/{dataset}/`.

---

## Ejemplo completo

```bash
# 1. Comprobar que las rutas resuelven (barato, sin entrenar nada)
cd src && python3 -c "
from utils import get_features_dir, detect_latent_dim
W = '/home/JKP6679/Patho-Ensemble/PARADIS/datos/patches'
for m in ['ctranspath', 'uni_v2', 'virchow_v1']:
    d = get_features_dir(W, 'cptac_brca', m)
    print(f'{m:12s} dim={detect_latent_dim(d):5d}  {d}')"

# 2. Lanzar el pipeline completo (encola 3 jobs encadenados)
cd .. && DATASET=cptac_brca TASK=TP53_mutation ./run_full_pipeline.sbatch
```

---

## Referencias

- **README.md** — documentación del pipeline y ejemplos
- **experiments.yaml** — matriz de experimentos (dataset × tarea)
- **src/experiment_config.py** — lector de la matriz
- **src/utils.py** — `get_features_dir`, `detect_latent_dim`, `resolve_latent_dim`
