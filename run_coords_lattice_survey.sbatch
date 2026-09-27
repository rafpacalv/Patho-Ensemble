#!/bin/bash
#SBATCH --job-name=coords_survey
#SBATCH --partition=main
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=logs/coords_survey_%j.out
#SBATCH --error=logs/coords_survey_%j.err

# ============================================================================
# ¿En qué datasets sirven los `coords` para una vecindad 3×3 (Moore-8)?
# ============================================================================
# Requisito previo de `--topk_ctx` (top-K de atención + promedio 3×3 de cada
# parche seleccionado con sus vecinos, ver abmil_engine.
# abmil_extract_topk_ctx_embeddings). No todos los datasets tienen coords
# utilizables: los de parches sueltos (Kather, LC25000...) no tienen rejilla, y
# una extracción puede traer coords sin el attr del paso o fuera de retícula.
#
# Por dataset × modelo del trío, sobre hasta 20 slides: ¿hay `coords`? ¿alineados
# fila a fila con `features`? ¿caen en la retícula de `patch_size_level0`? ¿qué
# fracción de parches se queda sin vecinos? Mismo criterio de rechazo que
# train_abmil.load_graphs_cached (>50 % de slides fuera de retícula o >25 % de
# parches aislados), para que el sondeo y la extracción digan lo mismo.
# Sin `patch_size_level0` el paso se deduce por slide del MCD de los coords
# (graph_utils.infer_stride), igual que hace --topk_ctx. La primera versión
# asumía 448 y marcaba cptac_gbm/bc_therapy como no utilizables cuando su
# paso real es 224 (job 72685 → 72687).
# Sólo lee .h5: no entrena ni predice nada.
# ============================================================================
set -euo pipefail
source $(conda info --base)/etc/profile.d/conda.sh
conda activate ensemble
cd /home/JKP6679/Patho-Ensemble

python - <<'PY'
import sys, os
import numpy as np, h5py
sys.path.insert(0, "src")
import graph_utils
from utils import get_features_dir

W = "/home/JKP6679/Patho-Ensemble/PARADIS/datos/patches"
FEAT_ROOT = "/home/JKP6679/Patho-Ensemble/PARADIS/datos/features"
MODELS = ["ctranspath", "uni_v2", "conch_v1_5"]
N_SLIDES = 20

datasets = sorted(d for d in os.listdir(FEAT_ROOT)
                  if not d.startswith("_") and os.path.isdir(f"{FEAT_ROOT}/{d}"))
print("stride: * = deducido del MCD de los coords (el .h5 no trae patch_size_level0)")
print(f"{'dataset':<18}{'modelo':<12}{'n':>4}{'coords':>8}{'alin.':>7}"
      f"{'stride':>16}{'retíc.':>8}{'aisl.%':>8}{'deg':>6}  veredicto")
for ds in datasets:
    for m in MODELS:
        try:
            fd = get_features_dir(W, ds, m)
        except Exception:
            print(f"{ds:<18}{m:<12}   -  (sin features)")
            continue
        if not fd or not os.path.isdir(fd):
            print(f"{ds:<18}{m:<12}   -  (sin features)")
            continue
        files = sorted(f for f in os.listdir(fd) if f.endswith(".h5"))[:N_SLIDES]
        n = has = aligned = on_lat = 0
        strides, iso, tiles, degs = set(), 0, 0, []
        for f in files:
            n += 1
            try:
                with h5py.File(f"{fd}/{f}", "r") as h:
                    if "coords" not in h:
                        continue
                    has += 1
                    c = h["coords"][:]
                    nf = h["features"].shape[0]
                    st = h["coords"].attrs.get("patch_size_level0", None)
            except Exception:
                continue
            if c.ndim != 2 or c.shape[0] != nf:
                continue
            aligned += 1
            stride = int(st) if st is not None else (graph_utils.infer_stride(c) or 448)
            strides.add(stride if st is not None else f"{stride}*")
            _, _, info = graph_utils.build_graph(c, stride, mode="lattice")
            if info["method"] == "lattice":
                on_lat += 1
            iso += info["n_isolated"]; tiles += info["n"]; degs.append(info["mean_deg"])
        iso_pct = 100 * iso / tiles if tiles else float("nan")
        deg = float(np.mean(degs)) if degs else float("nan")
        ok = (n > 0 and aligned == n and on_lat >= 0.5 * n and iso_pct <= 25)
        verdict = "✅ UTILIZABLE" if ok else "❌ NO"
        print(f"{ds:<18}{m:<12}{n:>4}{has:>8}{aligned:>7}"
              f"{','.join(map(str, sorted(strides, key=str))) or '-':>16}"
              f"{on_lat:>8}{iso_pct:>8.1f}{deg:>6.2f}  {verdict}")
PY
echo "✅ $(date)"
