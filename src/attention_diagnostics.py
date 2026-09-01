#!/usr/bin/env python
"""
attention_diagnostics.py — ¿Por qué uni_v2 no aprende a atender?

Observación de partida: la entropía normalizada de la atención de `uni_v2` es
0.9939 ± 0.0025 (1.0 = uniforme = mean pooling), frente a 0.8835 de conch_v1_5
y 0.9216 de ctranspath. Es decir, su ABMIL apenas selecciona parches.

Hipótesis a discriminar:

  A. DIMENSIÓN — `uni_v2` proyecta 1536→512 (compresión 3×) donde los de 768
     comprimen 1.5×. Si la compresión destruye la estructura entre parches, la
     atención se queda sin señal. Predicción falsable: la entropía debería
     crecer monótonamente con `in_dim`, y `virchow_v1` (2560 D, compresión 5×)
     debería ser el más plano de todos.

  B. Nº DE PARCHES — `uni_v2` tiene ~5250 parches/slide frente a ~3009. La
     entropía se normaliza por log(N), así que N no la infla por construcción,
     pero sí diluye el gradiente por parche.

  C. GEOMETRÍA — sus parches son intrínsecamente más homogéneos, y entonces la
     atención plana es la respuesta CORRECTA, no un fallo.

Nótese que la capacidad del módulo de atención NO es una hipótesis viable: la
atención opera sobre el espacio proyectado (512 D) y es GatedAttention(512,256)
— 262 400 parámetros idénticos en los tres modelos. Lo único que escala con
`in_dim` es la proyección, y ahí `uni_v2` tiene el DOBLE (787k vs 394k). Si la
capacidad fuera el problema, sería por exceso, no por defecto.

Métricas por modelo (sobre las slides muestreadas, fold 0):
  entropy_norm     entropía de la atención / log(N).  1.0 = uniforme
  ess_norm         tamaño efectivo de muestra 1/Σa² / N.  1.0 = uniforme
  logit_std        desviación de los logits PRE-softmax. Si ≈0, el modelo
                   directamente no discrimina; si es grande y la entropía sigue
                   alta, discrimina pero de forma difusa
  cos_var_raw      varianza de la similitud coseno entre parches, features CRUDAS
  cos_var_proj     ídem tras la proyección. La caída raw→proj mide cuánta
                   estructura destruye el cuello de botella (hipótesis A)
"""

import argparse
import json
import sys
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).parent))

import abmil_engine  # noqa: E402
from utils import get_features_dir, detect_latent_dim  # noqa: E402


def pairwise_cos_var(x, n_sample=400, seed=0):
    """Varianza de la similitud coseno entre pares de parches.

    Se submuestrean parches porque la matriz completa es O(N²) y N ~5000.
    Mide cuánta estructura hay ENTRE parches: si todos apuntan a lo mismo, la
    atención no tiene nada que distinguir.
    """
    rng = np.random.default_rng(seed)
    if x.shape[0] > n_sample:
        x = x[rng.choice(x.shape[0], n_sample, replace=False)]
    x = x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-9)
    g = x @ x.T
    iu = np.triu_indices(len(g), k=1)
    return float(np.var(g[iu]))


@torch.no_grad()
def diagnose(model, feats_dir, stems, in_dim, device):
    rows = []
    for s in stems:
        with h5py.File(f"{feats_dir}/{s}.h5", "r") as f:
            raw = f["features"][:]
        h = torch.tensor(raw, dtype=torch.float32, device=device)
        p = model.projection(h)
        # logits pre-softmax: la rama gated antes de w(), luego w()
        g = torch.tanh(model.attention.V(p)) * torch.sigmoid(model.attention.U(p))
        logits = model.attention.w(g).squeeze(-1)
        a = torch.softmax(logits, dim=0).cpu().numpy()
        n = len(a)
        rows.append(dict(
            n_patches=n,
            entropy_norm=float(-(a * np.log(a + 1e-12)).sum() / np.log(n)),
            ess_norm=float((1.0 / (a ** 2).sum()) / n),
            logit_std=float(logits.std().item()),
            cos_var_raw=pairwise_cos_var(raw),
            cos_var_proj=pairwise_cos_var(p.cpu().numpy()),
        ))
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--work_dir", default="/home/JKP6679/Patho-Ensemble/PARADIS/datos/patches")
    ap.add_argument("--train_source", default="cptac_brca")
    ap.add_argument("--task_name", default="TP53_mutation")
    ap.add_argument("--tissue_patching", default="20x_224px_0px_overlap")
    ap.add_argument("--models", nargs="+", default=[
        "ctranspath", "uni_v2", "conch_v1_5", "virchow_v1",
        "uni_v1", "phikon_v2", "hoptimus1", "virchow2"])
    ap.add_argument("--n_slides", type=int, default=25)
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--out", default="results_attention_diagnostics.json")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    W = args.work_dir
    abmil = f"{W}/{args.train_source}/{args.task_name}/abmil"
    df = pd.read_csv(f"{W}/{args.train_source}/{args.task_name}/k=all.tsv", sep="\t")
    stems = df["slide_id"].tolist()[:args.n_slides]

    print("=" * 100)
    print("DIAGNÓSTICO DE ATENCIÓN — ¿por qué unos modelos atienden y otros no?")
    print("=" * 100)
    print(f"{args.train_source}/{args.task_name} · fold {args.fold} · "
          f"{len(stems)} slides · {device}\n")

    out = {}
    for m in args.models:
        ck_path = f"{abmil}/{m}_{args.tissue_patching}/checkpoints/fold_{args.fold}/model.pt"
        if not Path(ck_path).exists():
            print(f"  ⚠ {m}: sin checkpoint, se omite")
            continue
        try:
            fd = get_features_dir(W, args.train_source, m)
            in_dim = detect_latent_dim(fd)
            ck = torch.load(ck_path, map_location=device)
            state = ck["state_dict"] if isinstance(ck, dict) and "state_dict" in ck else ck
            mdl = abmil_engine.ABMIL_EMBEDDING(in_dim=in_dim, num_classes=2, hidden_dim=256,
                                               proj_dim=512, dropout=0.25).to(device)
            inc = mdl.load_state_dict(state, strict=False)
            if inc.missing_keys:
                # strict=False silencia claves ausentes; sin esto mediríamos una
                # atención aleatoria y el diagnóstico sería basura plausible.
                print(f"  ⚠ {m}: claves sin cargar {inc.missing_keys[:3]}, se omite")
                continue
            mdl.eval()
            rows = diagnose(mdl, fd, stems, in_dim, device)
        except Exception as e:
            print(f"  ⚠ {m}: {type(e).__name__}: {e}")
            continue
        agg = {k: float(np.mean([r[k] for r in rows])) for k in rows[0]}
        agg["entropy_sd"] = float(np.std([r["entropy_norm"] for r in rows]))
        agg["in_dim"] = in_dim
        agg["compression"] = in_dim / 512.0
        out[m] = agg
        print(f"  ✓ {m:<12} in_dim={in_dim:<5} entropía={agg['entropy_norm']:.4f}")

    if not out:
        raise SystemExit("ningún modelo diagnosticado")

    order = sorted(out, key=lambda m: out[m]["in_dim"])
    print("\n" + "=" * 100)
    print(f"{'modelo':<13}{'in_dim':>7}{'compr.':>8}{'parches':>9}"
          f"{'entropía':>10}{'±':>8}{'ess_norm':>10}{'logit_sd':>10}"
          f"{'cos_var_raw':>13}{'cos_var_proj':>14}")
    print("-" * 100)
    for m in order:
        a = out[m]
        print(f"{m:<13}{a['in_dim']:>7}{a['compression']:>8.1f}x{a['n_patches']:>8.0f}"
              f"{a['entropy_norm']:>10.4f}{a['entropy_sd']:>8.4f}{a['ess_norm']:>10.4f}"
              f"{a['logit_std']:>10.4f}{a['cos_var_raw']:>13.5f}{a['cos_var_proj']:>14.5f}")
    print("-" * 100)

    # ---- el contraste que decide la hipótesis A ----------------------------
    d = np.array([out[m]["in_dim"] for m in order], dtype=float)
    e = np.array([out[m]["entropy_norm"] for m in order])
    npz = np.array([out[m]["n_patches"] for m in order])
    if len(order) > 2:
        from scipy.stats import pearsonr, spearmanr
        r_d, p_d = pearsonr(d, e)
        rs_d, ps_d = spearmanr(d, e)
        r_n, p_n = pearsonr(npz, e)
        print("\nHIPÓTESIS A — ¿la entropía crece con la dimensión de entrada?")
        print(f"  Pearson  in_dim vs entropía : r={r_d:+.3f}  p={p_d:.3f}")
        print(f"  Spearman in_dim vs entropía : ρ={rs_d:+.3f}  p={ps_d:.3f}")
        print("\nHIPÓTESIS B — ¿o crece con el número de parches?")
        print(f"  Pearson  parches vs entropía: r={r_n:+.3f}  p={p_n:.3f}")
        print(f"\n  (n={len(order)} modelos: la correlación es orientativa, no un test)")
        out["_correlations"] = dict(pearson_in_dim=[float(r_d), float(p_d)],
                                    spearman_in_dim=[float(rs_d), float(ps_d)],
                                    pearson_n_patches=[float(r_n), float(p_n)])

    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\n✅ {args.out}")


if __name__ == "__main__":
    main()
