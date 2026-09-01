#!/bin/bash
#SBATCH --job-name=stage3_shuffle
#SBATCH --partition=main
#SBATCH --constraint=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=logs/stage3_shuffle_%j.out
#SBATCH --error=logs/stage3_shuffle_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=rafael.pachon.alvarez@gmail.com

# ============================================================================
# Etapa 3 — control negativo de vinculación espacial
# ============================================================================
#   sbatch stage3_control_shuffle.sbatch
#   ALPHA_INIT=0.5 TAG_REAL=gnn05 TAG_SHUF=gnn05shuf sbatch stage3_control_shuffle.sbatch
#
# CONTEXTO. El brazo con alpha=0.3 (job 70020) es el primero que se mueve de
# verdad: r = 0.966 contra el baseline (frente a 0.9998 con alpha~0), G/E/P
# 22/14/14, y delta AUC +0.0053 — positivo pero por debajo de su MDE de 0.0114.
#
# Ese +0.0053 NO distingue dos explicaciones:
#   (a) el CONTEXTO ESPACIAL ayuda: importa qué tiles son vecinos;
#   (b) SUAVIZAR ayuda, con los vecinos que sean — un regularizador cualquiera.
#
# Este job entrena el mismo brazo sobre un grafo RELABELADO: se construye la
# vecindad real y luego se reasignan los tiles a los nodos con una permutación
# fija por slide. El grafo resultante es ISOMORFO — mismo grado, mismo número
# de aristas, mismos FLOPs, mismos parámetros — y lo ÚNICO que se destruye es
# la correspondencia entre cada tile y su posición.
#
# El contraste decisivo es por tanto REAL menos SHUFFLE, no contra el baseline:
# comparte fold, semilla, arquitectura y alpha, así que la única variable es si
# "vecino" significa "pegado en la slide". Si el real no le gana al shuffle por
# encima del MDE, el efecto no es espacial.
#
# El otro control del plan (--graph_mode self) NO se corre aquí: con 'smooth',
# los self-loops hacen que la agregación sea la identidad, o sea
# h + alpha*(h - h) = h, que es EXACTAMENTE el baseline. Sería un job entero
# para reproducir una fila que ya tenemos.
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
MODEL="${MODEL:-uni_v2}"
ARCH="${ARCH:-smooth}"
NLAYERS="${NLAYERS:-2}"
EPOCHS="${EPOCHS:-100}"
ALPHA_INIT="${ALPHA_INIT:-0.3}"
TAG_REAL="${TAG_REAL:-gnn03}"      # ya entrenado por el job 70020
TAG_SHUF="${TAG_SHUF:-gnn03shuf}"  # lo que entrena este job

BASE_DIR="${WORK_DIR}/${DATASET}/${TASK}/abmil/${MODEL}_${TISSUE}"
REAL_DIR="${WORK_DIR}/${DATASET}/${TASK}/abmil/${MODEL}_${TAG_REAL}_${TISSUE}"
SHUF_DIR="${WORK_DIR}/${DATASET}/${TASK}/abmil/${MODEL}_${TAG_SHUF}_${TISSUE}"

echo "=============================================================================="
echo " Etapa 3 — control de vinculación espacial"
echo " ${MODEL} / ${ARCH} alpha=${ALPHA_INIT} sobre ${DATASET}/${TASK}"
echo " job ${SLURM_JOB_ID:-local} en $(hostname)"
echo "=============================================================================="
nvidia-smi --query-gpu=name --format=csv,noheader || true
echo

if [ ! -d "${REAL_DIR}/test_outputs" ]; then
    echo " ERROR: falta el brazo real en ${REAL_DIR}"
    echo " Correr antes: ALPHA_INIT=${ALPHA_INIT} TAG=${TAG_REAL} sbatch stage2_spatial_gnn.sbatch"
    exit 1
fi

# ── A. entrenar el control ──────────────────────────────────────────────────
echo "=============================================================================="
echo " A. BRAZO CONTROL: grafo relabelado (--shuffle_coords)"
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
    --arch_tag "${TAG_SHUF}" \
    --alpha_init "${ALPHA_INIT}" \
    --shuffle_coords

echo
echo " meta-features in-sample del control"
python src/test_abmil.py \
    --foundational_model "${MODEL}" \
    --work_dir "${WORK_DIR}" \
    --train_source "${DATASET}" \
    --tissue_patching "${TISSUE}" \
    --task_name "${TASK}" \
    --arch "${ARCH}" \
    --arch_tag "${TAG_SHUF}" \
    --shuffle_coords
echo

# ── B. el contraste decisivo ────────────────────────────────────────────────
echo "=============================================================================="
echo " B. REAL vs SHUFFLE — ¿el efecto es espacial?"
echo "=============================================================================="
BASE_DIR="${BASE_DIR}" REAL_DIR="${REAL_DIR}" SHUF_DIR="${SHUF_DIR}" \
TAG_REAL="${TAG_REAL}" TAG_SHUF="${TAG_SHUF}" python - <<'PYB'
import os
from pathlib import Path
import numpy as np
from scipy import stats
from sklearn.metrics import roc_auc_score

A = Path(os.environ["BASE_DIR"])
R = Path(os.environ["REAL_DIR"])
S = Path(os.environ["SHUF_DIR"])


def auc(y, p):
    if p.shape[1] == 2:
        return roc_auc_score(y, p[:, 1])
    return roc_auc_score(y, p, multi_class="ovr", average="macro")


rows = []
for k in range(200):
    try:
        pa, pr, ps = (d / f"test_outputs/fold_{k}" for d in (A, R, S))
        if not all((d / "preds.npy").exists() for d in (pa, pr, ps)):
            continue
        y = np.load(pa / "labels.npy")
        for d in (pr, ps):
            assert np.array_equal(y, np.load(d / "labels.npy")), f"fold {k}: labels distintos"
        if len(np.unique(y)) < 2:
            continue
        a, r, s = (np.load(d / "preds.npy") for d in (pa, pr, ps))
        rows.append((auc(y, a), auc(y, r), auc(y, s), r, s))
    except Exception as e:
        print(f"  fold {k}: {e}")

n = len(rows)
if n == 0:
    print("  sin folds comparables"); raise SystemExit(2)

base = np.array([x[0] for x in rows])
real = np.array([x[1] for x in rows])
shuf = np.array([x[2] for x in rows])

# correlación real<->shuffle: si es ~1, el grafo real no aporta NADA distinto
cs = []
for _, _, _, pr, ps in rows:
    x, y_ = (pr[:, 1], ps[:, 1]) if pr.shape[1] == 2 else (pr.ravel(), ps.ravel())
    if x.std() > 0 and y_.std() > 0:
        cs.append(np.corrcoef(x, y_)[0, 1])
cs = np.array(cs)

print(f"  folds pareados: {n}\n")
print(f"  {'brazo':<22}{'AUC medio':>11}")
print(f"  {'-' * 33}")
print(f"  {'baseline (sin grafo)':<22}{base.mean():>11.4f}")
print(f"  {'real (vecinos reales)':<22}{real.mean():>11.4f}")
print(f"  {'shuffle (relabelado)':<22}{shuf.mean():>11.4f}")
print()
print(f"  correlacion real<->shuffle : r = {cs.mean():.4f}")
print()


def contrast(d, name, note=""):
    m, sd = d.mean(), d.std(ddof=1)
    ci = 1.96 * sd / np.sqrt(len(d))
    mde = (1.96 + 0.842) * sd / np.sqrt(len(d))
    w, t = int((d > 0).sum()), int((d == 0).sum())
    p = stats.ttest_rel(d, np.zeros_like(d)).pvalue if sd > 0 else 1.0
    star = "*" if abs(m) > mde else " "
    print(f"  {name:<26} delta = {m:+.4f}  IC95 [{m - ci:+.4f}, {m + ci:+.4f}]")
    print(f"  {'':<26} G/E/P = {w}/{t}/{len(d) - w - t}   MDE = {mde:.4f}   p = {p:.3f} {star}")
    if note:
        print(f"  {'':<26} {note}")
    print()
    return m, mde


print("  " + "=" * 72)
print("  EL CONTRASTE DECISIVO")
print("  " + "=" * 72)
m_sp, mde_sp = contrast(real - shuf, "real - shuffle",
                        "aisla la VINCULACION ESPACIAL: misma topologia, mismos")
print(f"  {'':<26} parametros, solo cambia quien es vecino de quien\n")

print("  contexto (no deciden por si solos):")
contrast(real - base, "real - baseline")
contrast(shuf - base, "shuffle - baseline")

"""
El veredicto se lee de DOS cosas que responden preguntas distintas, y
confundirlas fue un error de la primera version de este script:

  - el IC / la p dicen si ESTE resultado es distinguible de cero;
  - el MDE dice que efecto se habria podido detectar con fiabilidad.

Un efecto por debajo del MDE puede aun asi salir significativo: eso es
justo el regimen de potencia < 80 %. Asi que no se puede concluir "no hay
efecto" solo porque |delta| < MDE.

Y sobre todo: "real ~ shuffle" NO implica "lo que ayuda es suavizar". Eso
solo se sostiene si ademas shuffle mejora sobre el baseline. Si shuffle
EMPEORA, suavizar no es el mecanismo, y hay que decirlo.
"""
d_sh = shuf - base
sig = (m_sp - 1.96 * (real - shuf).std(ddof=1) / np.sqrt(n)) > 0

print("  " + "=" * 72)
if m_sp > 0 and sig:
    print(f"  El grafo real GANA al relabelado: {m_sp:+.4f}, IC95 excluye el cero.")
    if abs(m_sp) < mde_sp:
        print(f"  Pero por debajo del MDE ({mde_sp:.4f}): potencia < 80 %, asi que es")
        print(f"  SUGERENTE, no concluyente. Replicar antes de afirmarlo.")
    else:
        print(f"  Y por encima del MDE ({mde_sp:.4f}): el efecto ES espacial.")
elif m_sp > 0:
    print(f"  El grafo real apunta por encima del relabelado ({m_sp:+.4f}) pero el")
    print(f"  IC95 incluye el cero. Efecto espacial acotado en +-{mde_sp:.4f}.")
else:
    print(f"  El grafo real NO supera al relabelado ({m_sp:+.4f}).")

print()
if d_sh.mean() < 0:
    print(f"  Nota: el relabelado EMPEORA sobre el baseline ({d_sh.mean():+.4f}), asi que")
    print(f"  'suavizar' por si solo no es el mecanismo — mezclar tiles al azar hace dano.")
else:
    print(f"  Nota: el relabelado tambien mejora sobre el baseline ({d_sh.mean():+.4f}):")
    print(f"  parte del efecto es suavizado generico, no vinculacion espacial.")
print("  " + "=" * 72)
PYB

echo
echo "=============================================================================="
echo " Etapa 3 completada"
echo "=============================================================================="
