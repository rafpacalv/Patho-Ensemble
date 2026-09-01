#!/usr/bin/env python
"""
shared_attention_probe.py — ¿Merece la pena compartir el vector de atención?

Pregunta: en el brazo de embeddings, cada ABMIL agrega sus parches con SU
propia atención. ¿Y si se promediaran las atenciones y se aplicara el mismo
vector a todos los modelos?

Sólo es posible entre modelos que compartan rejilla de parches. Verificado
sobre 25 slides de cptac_brca: `ctranspath` y `conch_v1_5` tienen `coords`
byte-idénticos en 25/25, y `uni_v2` en 0/25 (rejilla distinta: 5487 parches
donde los otros tienen 4101). Por eso la sonda es del PAR, no del trío.

Cuatro regímenes de agregación, todos sobre los MISMOS checkpoints ya
entrenados — esto es sólo inferencia, no reentrena nada:

    own    z_m = Σ (a_m · proj_m(h_m))          statu quo, cada uno su atención
    avg    z_m = Σ (ā   · proj_m(h_m))          ā = media de las atenciones
    conch  z_m = Σ (a_conch · proj_m(h_m))      la atención del mejor modelo
    mean   z_m = Σ (1/N · proj_m(h_m))          CONTROL NEGATIVO: mean pooling

`mean` es el brazo que más informa y el que faltaba en todos los informes: si
empata con `own`, la atención no está aportando nada en este pipeline y la
pregunta de cómo combinarla es discutible. Hay motivo para sospecharlo — la
atención de uni_v2 tiene entropía normalizada 0.9939, es casi uniforme ya.

Nótese que la atención se comparte pero la PROYECCIÓN no: cada modelo proyecta
sus propias features con sus propios pesos. Lo único que se unifica es el peso
de cada parche.

Escribe en el layout que espera run_meta_suite.py, así que el análisis
estadístico (alineamiento, contrastes pareados, Holm, MDE) se reutiliza tal cual:

    {model}_{patching}_train_eval_attn_{reg}/val_outputs_attn_{reg}/fold_k/
    {model}_{patching}/val_outputs_attn_{reg}/fold_k/
    {model}_{patching}/test_outputs_attn_{reg}/fold_k/
"""

import argparse
import sys
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).parent))

import abmil_engine  # noqa: E402
from train_abmil import create_fold_split  # noqa: E402
from utils import get_features_dir, resolve_latent_dim  # noqa: E402

REGIMES = ["own", "avg", "conch", "mean"]


def load_pair_features(work_dir, dataset, models, slide_ids):
    """Carga features y coords de los dos modelos, verificando la rejilla.

    La comprobación de `coords` no es decorativa: si las rejillas no coinciden,
    un vector de atención compartido estaría ponderando parches DISTINTOS en
    cada modelo, y el resultado parecería plausible siendo basura.
    """
    feats = {m: {} for m in models}
    dirs = {m: get_features_dir(work_dir, dataset, m) for m in models}
    for m, d in dirs.items():
        print(f"  {m}: {d}")
    for i, s in enumerate(slide_ids):
        ref_coords = None
        for m in models:
            with h5py.File(f"{dirs[m]}/{s}.h5", "r") as h:
                x = h["features"][:]
                c = np.ascontiguousarray(h["coords"][:])
            if ref_coords is None:
                ref_coords = c
            elif c.shape != ref_coords.shape or not np.array_equal(c, ref_coords):
                raise SystemExit(
                    f"ABORTADO: {s}: '{m}' no comparte rejilla con '{models[0]}' "
                    f"({c.shape} vs {ref_coords.shape}). Una atención compartida "
                    f"ponderaría parches distintos en cada modelo."
                )
            feats[m][s] = torch.tensor(x, dtype=torch.float32)
        if (i + 1) % 25 == 0:
            print(f"    {i + 1}/{len(slide_ids)} slides", flush=True)
    return feats


def load_model(abmil_dir, model, patching, fold, in_dim, device):
    ck = torch.load(f"{abmil_dir}/{model}_{patching}/checkpoints/fold_{fold}/model.pt",
                    map_location=device)
    state = ck["state_dict"] if isinstance(ck, dict) and "state_dict" in ck else ck
    mdl = abmil_engine.ABMIL_EMBEDDING(in_dim=in_dim, num_classes=2, hidden_dim=256,
                                       proj_dim=512, dropout=0.25).to(device)
    inc = mdl.load_state_dict(state, strict=False)
    # strict=False silencia claves ausentes: sin esta comprobación, unos pesos de
    # atención que no cargaran darían embeddings de una proyección aleatoria y
    # nadie se enteraría.
    if inc.missing_keys:
        raise SystemExit(f"ABORTADO: {model} fold {fold}: claves sin cargar "
                         f"{inc.missing_keys}")
    mdl.eval()
    return mdl


@torch.no_grad()
def embeddings_for(models, mdls, feats, stems, device, attn_model_idx):
    """Devuelve {regimen: [N, 512*len(models)]} para esos stems."""
    out = {r: [] for r in REGIMES}
    for s in stems:
        proj, att = {}, {}
        for m in models:
            h = feats[m][s].to(device)
            p = mdls[m].projection(h)                  # (N, 512)
            a = mdls[m].attention(p)                   # (N, 1), softmax sobre parches
            proj[m], att[m] = p, a
        n = proj[models[0]].shape[0]
        shared = {
            "avg": torch.stack([att[m] for m in models], 0).mean(0),
            "conch": att[models[attn_model_idx]],
            "mean": torch.full((n, 1), 1.0 / n, device=device),
        }
        for r in REGIMES:
            z = [ (att[m] if r == "own" else shared[r]) * proj[m] for m in models ]
            out[r].append(torch.cat([t.sum(0) for t in z]).cpu().numpy())
    return {r: np.vstack(v) if v else np.zeros((0, 512 * len(models)))
            for r, v in out.items()}


def save(abmil_dir, model_dir, fold, y, arr, split, tag):
    """Escribe en el layout que lee run_meta_suite.py."""
    p = Path(abmil_dir) / model_dir / f"{split}_outputs_{tag}" / f"fold_{fold}"
    p.mkdir(parents=True, exist_ok=True)
    np.save(p / "preds.npy", arr)
    np.save(p / "labels.npy", y)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--work_dir", default="/home/JKP6679/Patho-Ensemble/PARADIS/datos/patches")
    ap.add_argument("--train_source", default="cptac_brca")
    ap.add_argument("--task_name", default="TP53_mutation")
    ap.add_argument("--tissue_patching", default="20x_224px_0px_overlap")
    ap.add_argument("--models", nargs="+", default=["ctranspath", "conch_v1_5"],
                    help="Deben compartir rejilla de parches; se verifica y aborta si no.")
    ap.add_argument("--attn_model", default="conch_v1_5",
                    help="Modelo cuya atención usa el régimen 'conch'.")
    ap.add_argument("--n_folds", type=int, default=50)
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    models = args.models
    if args.attn_model not in models:
        raise SystemExit(f"--attn_model {args.attn_model} no está en {models}")
    ai = models.index(args.attn_model)

    W = args.work_dir
    abmil_dir = f"{W}/{args.train_source}/{args.task_name}/abmil"
    df = pd.read_csv(f"{W}/{args.train_source}/{args.task_name}/k=all.tsv", sep="\t")
    task_col = args.task_name
    df = df.dropna(subset=[task_col]).reset_index(drop=True)
    slide_ids = df["slide_id"].tolist()

    print("=" * 78)
    print("SONDA DE ATENCIÓN COMPARTIDA")
    print("=" * 78)
    print(f"Par        : {' ⊕ '.join(models)}")
    print(f"Tarea      : {args.train_source} / {args.task_name}")
    print(f"Folds      : {args.n_folds}   slides: {len(slide_ids)}")
    print(f"Regímenes  : {REGIMES}   (atención de '{args.attn_model}' para 'conch')")
    print(f"Dispositivo: {device}")

    dims = {m: resolve_latent_dim(W, args.train_source, m) for m in models}
    print(f"Dims       : {dims}")

    print("\nCargando features y verificando rejilla común...")
    feats = load_pair_features(W, args.train_source, models, slide_ids)
    print("✓ Rejilla de parches idéntica en todas las slides.")

    ent = {m: [] for m in models}
    for k in range(args.n_folds):
        mdls = {m: load_model(abmil_dir, m, args.tissue_patching, k, dims[m], device)
                for m in models}
        tr, va, te = create_fold_split(df, k, task_col)
        for split, sub in (("train", tr), ("val", va), ("test", te)):
            if sub is None or sub.empty:
                continue
            stems = sub["slide_id"].tolist()
            y = sub[task_col].values.astype(int)
            E = embeddings_for(models, mdls, feats, stems, device, ai)
            d = 512
            for r in REGIMES:
                tag = f"attn_{r}"
                for j, m in enumerate(models):
                    block = E[r][:, j * d:(j + 1) * d]
                    md = f"{m}_{args.tissue_patching}"
                    if split == "train":
                        save(abmil_dir, f"{md}_train_eval_{tag}", k, y, block,
                             "val", tag)
                    else:
                        save(abmil_dir, md, k, y, block, split, tag)
        # entropía de la atención, para caracterizar los modelos
        if k == 0:
            with torch.no_grad():
                for m in models:
                    for s in slide_ids[:20]:
                        h = feats[m][s].to(device)
                        a = mdls[m].attention(mdls[m].projection(h)).cpu().numpy().ravel()
                        ent[m].append(float(-(a * np.log(a + 1e-12)).sum() / np.log(len(a))))
        if (k + 1) % 5 == 0:
            print(f"  fold {k + 1}/{args.n_folds}", flush=True)

    print("\nEntropía normalizada de la atención (fold 0, 20 slides; 1.0 = uniforme):")
    for m in models:
        e = np.array(ent[m])
        print(f"  {m:<14} {e.mean():.4f} ± {e.std():.4f}")

    print("\n✅ Escritos los 4 regímenes. Analizar con:")
    cfg = " ".join(f"attn_{r}:logreg" for r in REGIMES)
    print(f"""
python src/run_meta_suite.py \\
    --train_source {args.train_source} --task_name {args.task_name} \\
    --foundational_models {' '.join(models)} \\
    --n_folds {args.n_folds} --baseline attn_own:logreg \\
    --configs prob:logreg {cfg} \\
    --out results_shared_attention_{args.train_source}_{args.task_name}.json
""")


if __name__ == "__main__":
    main()
