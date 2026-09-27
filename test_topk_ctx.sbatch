#!/bin/bash
#SBATCH --job-name=test_topk_ctx
#SBATCH --partition=main
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=logs/test_topk_ctx_%j.out
#SBATCH --error=logs/test_topk_ctx_%j.err

# Prueba de humo de abmil_extract_topk_ctx_embeddings antes de lanzar
# run_topk_ctx.sbatch: checkpoint REAL (uni_v2, cptac_brca, fold 0), 4 slides.
# Invariantes que deben cumplirse por construcción:
#   1. Con un grafo de sólo self-loops, "ctx" == topk (sin vecinos no hay contexto).
#   2. "match" con M == k equivale a topk.
#   3. La media 3×3 de un centro coincide con un cálculo a mano.
#   4. El control "shuffle" usa EXACTAMENTE el mismo nº de vecinos por centro
#      que "ctx" (el relabelado de P6 no lo hacía: ver docstring del motor).
set -euo pipefail
source $(conda info --base)/etc/profile.d/conda.sh
conda activate ensemble
cd /home/JKP6679/Patho-Ensemble

python - <<'PY'
import sys
import numpy as np, pandas as pd, torch
sys.path.insert(0, "src")
import abmil_engine, graph_utils
from train_abmil import load_features_cached, load_graphs_cached, load_config_yaml
from utils import resolve_latent_dim

W = "/home/JKP6679/Patho-Ensemble/PARADIS/datos/patches"
DS, TASK, M = "cptac_brca", "TP53_mutation", "uni_v2"
task_col, C = load_config_yaml(f"{W}/{DS}/{TASK}/config.yaml")
df = pd.read_csv(f"{W}/{DS}/{TASK}/k=all.tsv", sep="\t")
stems = df["slide_id"].tolist()[:4]
D = resolve_latent_dim(W, DS, M, None)
feats = load_features_cached(W, DS, M, set(stems), D)
real = load_graphs_cached(W, DS, M, set(stems), mode="lattice", shuffle=False)

model = abmil_engine.ABMIL_EMBEDDING(in_dim=D, num_classes=C, hidden_dim=256,
                                     proj_dim=512, dropout=0.25)
ck = f"{W}/{DS}/{TASK}/abmil/{M}_20x_224px_0px_overlap/checkpoints/fold_0/model.pt"
model.load_state_dict(torch.load(ck, map_location="cpu"), strict=False)
model.eval()

K = 10
top = abmil_engine.abmil_extract_topk_embeddings(model, feats, stems, "cpu", k=K)
ctx, st = abmil_engine.abmil_extract_topk_ctx_embeddings(model, feats, stems, "cpu", real, k=K)
sh, st_sh = abmil_engine.abmil_extract_topk_ctx_embeddings(model, feats, stems, "cpu", real, k=K, mode="shuffle")
sh2, _ = abmil_engine.abmil_extract_topk_ctx_embeddings(model, feats, stems, "cpu", real, k=K, mode="shuffle")
assert np.allclose(sh, sh2), "shuffle no determinista"
mt, _ = abmil_engine.abmil_extract_topk_ctx_embeddings(model, feats, stems, "cpu", real, k=K, mode="match")
print("shapes", top.shape, ctx.shape, sh.shape, mt.shape)
print("ctx_stats real   [M, frac_en_top, vec/centro]:\n", np.round(st, 3))
print("ctx_stats barajado:\n", np.round(st_sh, 3))

# 1. self-loops -> ctx == topk
selfg = {s: graph_utils.slide_graph(np.zeros((feats[s].shape[0], 2)), mode="self")[0] for s in stems}
c1, _ = abmil_engine.abmil_extract_topk_ctx_embeddings(model, feats, stems, "cpu", selfg, k=K)
assert np.allclose(c1, top, atol=1e-5), "self-loops: ctx != topk"
print("✓ 1. self-loops: ctx == topk")

# 2. match con M == k -> topk. Grafo self => región = k centros
m2, s2 = abmil_engine.abmil_extract_topk_ctx_embeddings(model, feats, stems, "cpu", selfg, k=K, mode="match")
assert (s2[:, 0] == K).all() and np.allclose(m2, top, atol=1e-5), "match(M=k) != topk"
print("✓ 2. match con M=k == topk")

# 3. a mano, primera slide
s = stems[0]
with torch.no_grad():
    h = model.projection(feats[s]); a = model.attention(h).squeeze(-1)
    ti = torch.topk(a, K).indices
    src, dst = real[s].edge_index
    vecs = []
    for c in ti.tolist():
        nb = src[(dst == c) & (src != c)].tolist()
        vecs.append(h[[c] + nb].mean(0))
    at = a[ti] / a[ti].sum()
    z = (at.unsqueeze(-1) * torch.stack(vecs)).sum(0).numpy()
assert np.allclose(z, ctx[0], atol=1e-5), "cálculo a mano no coincide"
print("✓ 3. media 3×3 == cálculo a mano")

# 4. mismo nº de vecinos por centro en ctx y shuffle; shuffle no toca centros
assert np.allclose(st[:, 2], st_sh[:, 2]), "shuffle con distinto nº de vecinos"
assert (st_sh[:, 1] == 0).all(), "shuffle sorteó centros como vecinos"
assert not np.allclose(sh, ctx), "shuffle == ctx"
print("✓ 4. shuffle: mismo nº de vecinos por centro, determinista, distinto de ctx")

# 5. paso deducido en cptac_gbm/ctranspath (sin attr): debe ser 224 y en retícula
gd = "cptac_gbm"
gdf = pd.read_csv(f"{W}/{gd}/{TASK}/k=all.tsv", sep="\t")
gs = set(gdf["slide_id"].tolist()[:5])
gg = load_graphs_cached(W, gd, "ctranspath", gs, mode="lattice", infer_stride=True)
print("✓ 5. cptac_gbm/ctranspath con paso deducido: grafos construidos")
print("diferencia relativa ||ctx-topk||/||topk|| =",
      np.round(np.linalg.norm(ctx - top, axis=1) / np.linalg.norm(top, axis=1), 3))
print("OK")
PY
