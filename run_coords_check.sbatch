#!/bin/bash
#SBATCH --job-name=coords_chk
#SBATCH --partition=main
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=logs/coords_chk_%j.out
#SBATCH --error=logs/coords_chk_%j.err

# ¿En qué datasets comparten rejilla de parches ctranspath y conch_v1_5?
# Es el requisito para poder compartir/transferir el vector de atención.
set -euo pipefail
source $(conda info --base)/etc/profile.d/conda.sh
conda activate ensemble
cd /home/JKP6679/Patho-Ensemble

python - <<'PY'
import sys, numpy as np, h5py, csv, hashlib
sys.path.insert(0,"src")
from utils import get_features_dir
W="/home/JKP6679/Patho-Ensemble/PARADIS/datos/patches"
CASES=[("cptac_gbm","TP53_mutation"),("bc_therapy","er_status"),
       ("cervical_subtype","subtype"),("cptac_brca","TP53_mutation")]
MODELS=["ctranspath","conch_v1_5","uni_v2"]
for ds,task in CASES:
    rows=list(csv.DictReader(open(f"{W}/{ds}/{task}/k=all.tsv"),delimiter="\t"))
    stems=[r["slide_id"].replace(".svs","").replace(".tiff","") for r in rows][:30]
    fds={m:get_features_dir(W,ds,m) for m in MODELS}
    pair=trio=n=0; ex=[]
    for s in stems:
        hs={}; ok=True
        for m in MODELS:
            try:
                with h5py.File(f"{fds[m]}/{s}.h5","r") as f: c=np.ascontiguousarray(f["coords"][:])
            except Exception: ok=False; break
            hs[m]=(c.shape, hashlib.md5(c.tobytes()).hexdigest())
        if not ok: continue
        n+=1
        if len(ex)<2: ex.append({m:hs[m][0][0] for m in MODELS})
        if hs["ctranspath"]==hs["conch_v1_5"]: pair+=1
        if len(set(hs.values()))==1: trio+=1
    verdict = "SONDA VIABLE" if n and pair==n else "NO VIABLE"
    print(f"{ds}/{task}: n={n}  ctranspath==conch_v1_5: {pair}/{n}  los tres: {trio}/{n}  -> {verdict}")
    for e in ex: print(f"    parches: {e}")
PY
echo "✅ $(date)"
