#!/bin/bash
#SBATCH --job-name=stage2_spatial
#SBATCH --partition=main
#SBATCH --constraint=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=logs/stage2_spatial_%j.out
#SBATCH --error=logs/stage2_spatial_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=rafael.pachon.alvarez@gmail.com

# ============================================================================
# Etapa 2 del brazo espacial: inercia del motor -> primer brazo -> puerta
# ============================================================================
#   sbatch stage2_spatial_gnn.sbatch
#   MODEL=uni_v2 ARCH=sage_bn sbatch stage2_spatial_gnn.sbatch
#
# Tres bloques, y el primero es bloqueante.
#
# A. INERCIA (bloqueante). Se editó abmil_engine.py para aceptar `graphs=` y
#    `model_factory=`. Hay que demostrar que con graphs=None NADA cambia.
#
#    La puerta NO es "reentrenar el fold 0 y comparar": el entrenamiento en GPU
#    puede no ser bit-reproducible por razones ajenas a la edición (orden de
#    reducción en cuDNN), y entonces un fallo no distinguiría "he roto algo" de
#    "la GPU no es determinista". Peor: reentrenar sobre el directorio del
#    baseline destruiría la referencia contra la que se compara.
#
#    En su lugar se aísla la edición exacta, y en DOS pruebas separadas, porque
#    la primera versión las confundía y falló por el motivo equivocado:
#
#    A1. La decisiva: los dos caminos de código en el MISMO proceso, sobre la
#        MISMA GPU y el mismo checkpoint — `model(x)` contra
#        `_apply(model, feats, None, s, device)`. Aquí sí se exige igualdad bit
#        a bit: es literalmente la misma expresión y no hay ninguna excusa.
#
#    A2. Informativa: contra los preds.npy guardados hace meses. Aquí NO se
#        exige igualdad bit a bit. La primera versión sí lo hacía y falló con
#        maxdiff de 6e-8 a 1.2e-7 sobre uni_v2 — que es el épsilon de float32
#        (1.19e-7), no un cambio de comportamiento: cuBLAS elige kernels
#        distintos según la arquitectura de GPU, y con eso cambia el orden de
#        reducción. ctranspath pasó bit a bit por la suerte de haber caído en
#        una GPU compatible, no porque fuese más inerte. Lo que sí se exige es
#        que no se mueva ningún argmax, que es lo único que puede alterar una
#        matriz de confusión o un recuento de empates.
#
# B. PRIMER BRAZO. ctranspath --arch smooth (2 parámetros) a 50 folds, hacia un
#    directorio propio (ctranspath_gnn_...), sin tocar el baseline.
#
# C. PUERTA DE CORRELACIÓN. La que decide si el proyecto sigue. El informe de
#    campaña midió que la diversificación a nivel base falló por SATURACIÓN DE
#    DIVERSIDAD: los snapshots correlacionan r = 0.92-0.98, muy por encima del
#    suelo inter-modelo r = 0.797 que ya aportan tres modelos fundacionales.
#      r >= 0.92  -> mismo callejón sin salida. PARAR, con un nulo acotado y con
#                    mecanismo, por ~1 GPU-hora.
#      r <= 0.85  -> eje de diversidad nuevo. Seguir a los controles (Etapa 3).
#      intermedio -> marginal; correr los controles antes de ampliar.
# ============================================================================

set -euo pipefail
mkdir -p logs

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate ensemble
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"

WORK_DIR="${WORK_DIR:-/home/JKP6679/Patho-Ensemble/PARADIS/datos/patches}"
DATASET="${DATASET:-cptac_brca}"
TASK="${TASK:-TP53_mutation}"
TISSUE="${TISSUE:-20x_224px_0px_overlap}"
# uni_v2 y no ctranspath, aunque ctranspath sea 2x más rápido (768D vs 1536D).
# Medido sobre las 112 slides de la tarea, con la MISMA retícula de 448:
#   uni_v2            grado 7.75, 0 nodos aislados de 923403
#   ctranspath_monai  grado 5.34, 5151 aislados de 536784 (0.96 %, y hasta un
#                     13 % en la slide más pequeña)
# La extracción _monai conserva 1.7x menos parches, así que su grafo tiene
# muchos más huecos. Probar la hipótesis espacial sobre el sustrato con PEOR
# conectividad sesga hacia el nulo — y el nulo ya es el desenlace más probable,
# así que saldría un resultado no informativo por construcción.
MODEL="${MODEL:-uni_v2}"
ARCH="${ARCH:-smooth}"
NLAYERS="${NLAYERS:-2}"
TAG="${TAG:-gnn}"
EPOCHS="${EPOCHS:-100}"
# Valor inicial de alpha en el brazo 'smooth'. 0.0 hace que el modelo arranque
# siendo ABMIL exacto, que es lo que permite demostrar la inercia — pero deja el
# resultado ambiguo: un alpha final ~0 no distingue "el espacio no aporta" de
# "el gradiente nunca movió alpha". Arrancar en 0.3 y ver si el entrenamiento lo
# empuja de vuelta a 0 sí separa las dos hipótesis.
ALPHA_INIT="${ALPHA_INIT:-0.0}"

BASE_DIR="${WORK_DIR}/${DATASET}/${TASK}/abmil/${MODEL}_${TISSUE}"
GNN_DIR="${WORK_DIR}/${DATASET}/${TASK}/abmil/${MODEL}_${TAG}_${TISSUE}"

echo "=============================================================================="
echo " Etapa 2 — ${MODEL} / ${ARCH} (K=${NLAYERS}) sobre ${DATASET}/${TASK}"
echo " job ${SLURM_JOB_ID:-local} en $(hostname)"
echo "=============================================================================="
nvidia-smi --query-gpu=name --format=csv,noheader || true
echo

# ── A. puerta de inercia ────────────────────────────────────────────────────
echo "=============================================================================="
echo " A. INERCIA DEL MOTOR (bloqueante)"
echo "=============================================================================="
if [ "${SKIP_INERT:-0}" = "1" ]; then
  echo "  omitida (SKIP_INERT=1): ya demostrada en el job 70017, 50/50 folds,"
  echo "  los dos caminos bit-idénticos y ningún argmax desplazado."
else
python - <<PY
import sys; sys.path.insert(0, "src")
from pathlib import Path
import numpy as np, pandas as pd, torch
import abmil_engine
from train_abmil import load_config_yaml, load_features_cached, create_fold_split
from utils import resolve_latent_dim

W, DS, TK = "${WORK_DIR}", "${DATASET}", "${TASK}"
BASE = Path("${BASE_DIR}")
dev = "cuda" if torch.cuda.is_available() else "cpu"

task_col, n_cls = load_config_yaml(f"{W}/{DS}/{TK}/config.yaml")
dim = resolve_latent_dim(W, DS, "${MODEL}", None)
df = pd.read_csv(f"{W}/{DS}/{TK}/k=all.tsv", sep="\t").dropna(subset=[task_col]).reset_index(drop=True)
feats = load_features_cached(W, DS, "${MODEL}", set(df.slide_id.unique()), dim)

n_folds = len([c for c in df.columns if c.startswith("fold_")])
checked = skipped = 0
ab_diff = 0            # A1: los dos caminos en el mismo proceso
flips = 0              # A2: argmax que se mueven contra lo guardado
maxdiff = 0.0
for k in range(n_folds):
    ck, ref = BASE / f"checkpoints/fold_{k}/model.pt", BASE / f"test_outputs/fold_{k}/preds.npy"
    if not (ck.exists() and ref.exists()):
        skipped += 1; continue
    _, _, te = create_fold_split(df, k, task_col)
    if te is None or te.empty:
        skipped += 1; continue
    stems = te.slide_id.tolist()
    m = abmil_engine.ABMIL(in_dim=dim, num_classes=n_cls, hidden_dim=256,
                           proj_dim=512, dropout=0.25).to(dev)
    m.load_state_dict(torch.load(ck, map_location=dev)); m.eval()

    # --- A1: camino viejo vs camino nuevo, mismo proceso, misma GPU ---------
    with torch.no_grad():
        old = np.vstack([torch.softmax(m(feats[s].to(dev))[0], -1).cpu().numpy()
                         for s in stems])
    new = abmil_engine.abmil_predict(m, feats, stems, dev, None)
    if not np.array_equal(old, new):
        ab_diff += 1
        print(f"  fold {k}: A1 DIFIERE maxdiff={np.abs(old-new).max():.3e}")

    # --- A2: contra lo guardado (tolerante a kernels de otra GPU) -----------
    exp = np.load(ref)
    d = float(np.abs(new - exp).max())
    maxdiff = max(maxdiff, d)
    flips += int((new.argmax(1) != exp.argmax(1)).sum())
    checked += 1

print(f"\n  folds comparados: {checked}  (omitidos {skipped})")
print(f"  A1  caminos de código que difieren : {ab_diff}")
print(f"  A2  maxdiff vs preds guardados     : {maxdiff:.3e}  (eps float32 = 1.19e-07)")
print(f"  A2  argmax desplazados             : {flips}")
if checked == 0:
    print("\n  NO CONCLUYENTE: no había checkpoints que comparar"); sys.exit(2)
if ab_diff:
    print("\n  MOTOR NO INERTE — los dos caminos difieren en el mismo proceso. Parar.")
    sys.exit(1)
if maxdiff > 1e-5 or flips:
    print("\n  DERIVA REAL vs lo guardado: excede el ruido de float32. Revisar.")
    sys.exit(1)
print("\n  MOTOR INERTE — los dos caminos son bit-idénticos en el mismo proceso;")
print("  la diferencia contra lo guardado es ruido de float32 y no mueve ningún argmax.")
PY
fi
echo

# ── A2. ¿es determinista el entrenamiento? ──────────────────────────────────
# Un bug en la recursión de abmil_train_test hizo que el job 70005 reentrenara
# ctranspath como ABMIL PLANO encima del baseline (los kwargs de arquitectura no
# se reenviaban a la llamada por fold). El bug está corregido; queda medir si
# ese pisado cambió algo. Se reentrena el fold 0 hacia un tag de usar y tirar y
# se compara con el baseline actual:
#   idéntico -> el entrenamiento es determinista, el pisado reprodujo el
#               original bit a bit y no se perdió nada.
#   distinto -> los preds del baseline ya no son los que produjeron las cifras
#               publicadas, y hay que regenerar ctranspath entero.
if [ "${SKIP_DET:-0}" != "1" ]; then
echo "=============================================================================="
echo " A2. DETERMINISMO DEL ENTRENAMIENTO (mide el daño del pisado de 70005)"
echo "=============================================================================="
DET_MODEL=ctranspath
DET_TAG=dettest
python src/train_abmil.py \
    --foundational_model "${DET_MODEL}" --work_dir "${WORK_DIR}" \
    --train_source "${DATASET}" --tissue_patching "${TISSUE}" \
    --task_name "${TASK}" --epochs "${EPOCHS}" --fold 0 \
    --arch abmil --arch_tag "${DET_TAG}" > /dev/null 2>&1 || true

python - <<PY
import numpy as np
from pathlib import Path
W, DS, TK, TI = "${WORK_DIR}", "${DATASET}", "${TASK}", "${TISSUE}"
a = Path(f"{W}/{DS}/{TK}/abmil/ctranspath_{TI}/test_outputs/fold_0/preds.npy")
b = Path(f"{W}/{DS}/{TK}/abmil/ctranspath_dettest_{TI}/test_outputs/preds.npy")
if not b.exists():
    b = Path(f"{W}/{DS}/{TK}/abmil/ctranspath_dettest_{TI}/test_outputs/fold_0/preds.npy")
if not (a.exists() and b.exists()):
    print(f"  NO CONCLUYENTE: falta {a if not a.exists() else b}")
else:
    x, y = np.load(a), np.load(b)
    if np.array_equal(x, y):
        print("  ENTRENAMIENTO DETERMINISTA — el pisado de 70005 reprodujo el")
        print("  baseline bit a bit. No se perdió nada.")
    else:
        print(f"  NO DETERMINISTA — maxdiff={np.abs(x-y).max():.3e}")
        print("  Los preds de ctranspath ya NO son los que produjeron las cifras")
        print("  publicadas. Regenerar ctranspath antes de usarlo como baseline.")
PY
echo
fi

# ── B. primer brazo ─────────────────────────────────────────────────────────
echo "=============================================================================="
echo " B. BRAZO ESPACIAL: --arch ${ARCH} --arch_tag ${TAG}"
echo "=============================================================================="
python src/train_abmil.py \
    --foundational_model "${MODEL}" \
    --work_dir "${WORK_DIR}" \
    --train_source "${DATASET}" \
    --tissue_patching "${TISSUE}" \
    --task_name "${TASK}" \
    --epochs "${EPOCHS}" \
    --fold all \
    --arch "${ARCH}" \
    --n_layers "${NLAYERS}" \
    --arch_tag "${TAG}" \
    --alpha_init "${ALPHA_INIT}"

echo
echo " meta-features in-sample del brazo espacial"
python src/test_abmil.py \
    --foundational_model "${MODEL}" \
    --work_dir "${WORK_DIR}" \
    --train_source "${DATASET}" \
    --tissue_patching "${TISSUE}" \
    --task_name "${TASK}" \
    --arch "${ARCH}" \
    --arch_tag "${TAG}"
echo

# ── C. puerta de correlación ────────────────────────────────────────────────
echo "=============================================================================="
echo " C. PUERTA DE CORRELACIÓN (decide si el proyecto sigue)"
echo "=============================================================================="
python - <<PY
import sys; sys.path.insert(0, "src")
from pathlib import Path
import numpy as np
from sklearn.metrics import roc_auc_score

A, B = Path("${BASE_DIR}"), Path("${GNN_DIR}")
rs, d_auc, wins, ties = [], [], 0, 0
for k in range(200):
    pa, pb = A / f"test_outputs/fold_{k}", B / f"test_outputs/fold_{k}"
    if not (pa / "preds.npy").exists() or not (pb / "preds.npy").exists():
        continue
    a, b = np.load(pa / "preds.npy"), np.load(pb / "preds.npy")
    ya, yb = np.load(pa / "labels.npy"), np.load(pb / "labels.npy")
    assert np.array_equal(ya, yb), f"fold {k}: labels distintos entre brazos"
    if a.shape != b.shape or len(np.unique(ya)) < 2:
        continue
    # correlación sobre la probabilidad de la clase positiva (o todas si multiclase)
    x, y = (a[:, 1], b[:, 1]) if a.shape[1] == 2 else (a.ravel(), b.ravel())
    if x.std() > 0 and y.std() > 0:
        rs.append(np.corrcoef(x, y)[0, 1])
    sa = roc_auc_score(ya, a[:, 1]) if a.shape[1] == 2 else roc_auc_score(ya, a, multi_class="ovr", average="macro")
    sb = roc_auc_score(yb, b[:, 1]) if b.shape[1] == 2 else roc_auc_score(yb, b, multi_class="ovr", average="macro")
    d_auc.append(sb - sa)
    wins += sb > sa; ties += sb == sa

rs, d = np.array(rs), np.array(d_auc)
n = len(d)
print(f"  folds pareados: {n}")
print(f"  correlación media base<->gnn : r = {rs.mean():.4f}  (min {rs.min():.3f}, max {rs.max():.3f})")
print(f"  suelo inter-modelo publicado : r = 0.797")
print(f"  snapshots FGE/SE publicados  : r = 0.92-0.98")
print()
sd = d.std(ddof=1)
mde = (1.96 + 0.842) * sd / np.sqrt(n)
ci = 1.96 * sd / np.sqrt(n)
print(f"  AUC pareado  delta = {d.mean():+.4f}   IC95 [{d.mean()-ci:+.4f}, {d.mean()+ci:+.4f}]")
print(f"  G/E/P = {wins}/{ties}/{n-wins-ties}     MDE(80%) = {mde:.4f}")
if abs(d.mean()) < mde:
    print(f"  -> efecto NO resoluble: acotado por +-{mde:.4f}, no ausente")
print()
r = rs.mean()
if r >= 0.92:
    print(f"  PUERTA: r={r:.3f} >= 0.92 — misma saturación que los snapshots.")
    print("  PARAR. Nulo acotado con mecanismo; no correr el barrido completo.")
elif r <= 0.85:
    print(f"  PUERTA: r={r:.3f} <= 0.85 — eje de diversidad NUEVO. Seguir a Etapa 3.")
else:
    print(f"  PUERTA: r={r:.3f} en zona marginal (0.85-0.92).")
    print("  Correr los controles de Etapa 3 sobre ctranspath antes de ampliar.")
PY

echo
echo "=============================================================================="
echo " D. TRAYECTORIA DE ALPHA"
echo "=============================================================================="
ALPHA_LOG="logs/stage2_spatial_${SLURM_JOB_ID:-local}.out"
ALPHA_INIT="${ALPHA_INIT}" ALPHA_LOG="${ALPHA_LOG}" python - <<'PYA'
import os, re
import numpy as np

init = float(os.environ["ALPHA_INIT"])
try:
    txt = open(os.environ["ALPHA_LOG"]).read()
except OSError:
    txt = ""

a = np.array([float(x)
              for m in re.findall(r"alphas=\[([^\]]*)\]", txt)
              for x in m.split(",")])

if not len(a):
    print("  sin alphas en el log (arch != smooth, o el log aun no esta volcado)")
else:
    print(f"  alpha inicial   : {init:+.4f}")
    print(f"  alpha aprendido : mediana {np.median(a):+.5f}   media {a.mean():+.5f}"
          f"   sd {a.std():.5f}")
    print(f"  rango           : [{a.min():+.5f}, {a.max():+.5f}]   n={len(a)}")
    print(f"  |alpha| > 0.05  : {(np.abs(a) > 0.05).sum()}/{len(a)}")
    print()
    if init <= 0.05:
        print("  Arranque en ~0: este run NO puede distinguir 'el espacio no aporta'")
        print("  de 'el gradiente no llego a mover alpha'. Repetir con ALPHA_INIT=0.3.")
    else:
        med = float(np.median(a))
        if med < init * 0.25:
            print("  -> El entrenamiento EMPUJA alpha hacia 0 desde un arranque sustantivo.")
            print("     Evidencia ACTIVA de que el modelo rechaza el promediado espacial:")
            print("     el nulo no era falta de gradiente. Queda cerrado.")
        elif med > init * 0.75:
            print("  -> El entrenamiento MANTIENE alpha. El arranque en 0 del run previo")
            print("     SI era el limitante, y aquel nulo no era concluyente.")
        else:
            print("  -> Alpha baja parcialmente. Caso intermedio: decidir por el AUC pareado")
            print("     y la correlacion de arriba, no por alpha.")
PYA

echo
echo "=============================================================================="
echo " Etapa 2 completada"
echo "=============================================================================="
