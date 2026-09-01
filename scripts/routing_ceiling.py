"""¿Merece la pena elegir el comité por tarea, en vez de usar uno fijo?

Mide el TECHO del enrutado dinámico. La selección de aquí conoce la tarea y usa
su propia validación, cosa que un enrutado ciego sobre una WSI no tiene. Por eso
es una cota superior: si conocer la tarea NO ayuda, enrutar a ciegas tampoco
puede ayudar, y la hipótesis queda falsada por el camino barato.

Tres estimaciones por tarea:
  fijo    - los 4 modelos comunes, siempre. Ninguna selección.
  honesta - leave-one-fold-out: para cada fold k se elige el subconjunto con
            mejor AUC de VALIDACIÓN en los folds != k, y se reporta su test en k.
  oráculo - el mejor subconjunto en test. Inalcanzable; marca cuánto se deja
            sobre la mesa y cuánto de la diferencia es sesgo de selección.
"""
import itertools
import os

import numpy as np
from scipy import stats as sp
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

W = "/home/JKP6679/Patho-Ensemble/PARADIS/datos/patches"
PAT = "20x_224px_0px_overlap"
COMMON = ["hoptimus1", "uni_v1", "uni_v2", "virchow2"]
TASKS = [
    ("cptac_brca", "PIK3CA_mutation"), ("cptac_brca", "TP53_mutation"),
    ("bc_therapy", "er_status"), ("bc_therapy", "grade"), ("bc_therapy", "her2_status"),
    ("cptac_coad", "APC_mutation"), ("cptac_coad", "KRAS_mutation"),
    ("cptac_coad", "MSI_H"), ("cptac_coad", "TP53_mutation"),
    ("cptac_coad", "ARID1A_mutation"), ("cptac_coad", "PIK3CA_mutation"),
    ("cptac_coad", "ACVR2A_mutation"), ("cptac_coad", "SETD1B_mutation"),
]
ORGANO = {"cptac_brca": "mama", "bc_therapy": "mama", "cptac_coad": "colon"}


def load(ds, tk, m, k):
    b = f"{W}/{ds}/{tk}/abmil/{m}_{PAT}"
    out = []
    for p in (f"{b}_train_eval/val_outputs", f"{b}/val_outputs", f"{b}/test_outputs"):
        d = f"{p}/fold_{k}"
        out += [np.load(f"{d}/preds.npy"), np.load(f"{d}/labels.npy")]
    return out


subs = [c for r in range(1, len(COMMON) + 1) for c in itertools.combinations(COMMON, r)]
FULL = tuple(COMMON)

V, T, ok = {}, {}, []
for ds, tk in TASKS:
    try:
        td = f"{W}/{ds}/{tk}/abmil/{COMMON[0]}_{PAT}/test_outputs"
        nf = len([p for p in os.listdir(td) if p.startswith("fold_")])
        P = {m: {k: load(ds, tk, m, k) for k in range(nf)} for m in COMMON}
    except Exception as e:
        print(f"  salto {ds}/{tk}: {type(e).__name__}")
        continue

    # binaria, y modelos alineados fila a fila en los tres splits
    if len(np.unique(P[COMMON[0]][0][5])) != 2:
        print(f"  salto {ds}/{tk}: no binaria")
        continue
    if any(not np.array_equal(P[COMMON[0]][k][si], P[m][k][si])
           for k in range(nf) for si in (1, 3, 5) for m in COMMON[1:]):
        print(f"  salto {ds}/{tk}: modelos no alineados")
        continue

    for c in subs:
        av, at = [], []
        for k in range(nf):
            Xtr = np.concatenate([P[m][k][0] for m in c], axis=1)
            Xva = np.concatenate([P[m][k][2] for m in c], axis=1)
            Xte = np.concatenate([P[m][k][4] for m in c], axis=1)
            f = LogisticRegression(max_iter=1000).fit(Xtr, P[c[0]][k][1])
            av.append(roc_auc_score(P[c[0]][k][3], f.predict_proba(Xva)[:, 1]))
            at.append(roc_auc_score(P[c[0]][k][5], f.predict_proba(Xte)[:, 1]))
        V[(ds, tk, c)] = np.array(av)
        T[(ds, tk, c)] = np.array(at)
    ok.append((ds, tk, nf))

print(f"\n{len(ok)} tareas binarias válidas, {len(subs)} subconjuntos de {COMMON}\n")
hdr = f"{'tarea':34} {'org':>5} {'fijo(4)':>8} {'honesta':>8} {'oráculo':>8} {'Δ hon':>8} {'sprd k=2':>9}"
print(hdr); print("-" * len(hdr))

dsel, dorac, filas = [], [], []
for ds, tk, nf in ok:
    hon = np.array([
        T[(ds, tk, max(subs, key=lambda c: V[(ds, tk, c)][[i for i in range(nf) if i != k]].mean()))][k]
        for k in range(nf)
    ])
    orac = max(subs, key=lambda c: T[(ds, tk, c)].mean())
    pares = [c for c in subs if len(c) == 2]
    sp2 = (max(T[(ds, tk, c)].mean() for c in pares)
           - min(T[(ds, tk, c)].mean() for c in pares))
    fijo = T[(ds, tk, FULL)]
    d = hon - fijo
    dsel.append(d.mean())
    dorac.append(T[(ds, tk, orac)].mean() - fijo.mean())
    filas.append((ds, tk, "+".join(orac)))
    print(f"{ds+'/'+tk:34} {ORGANO[ds]:>5} {fijo.mean():>8.4f} {hon.mean():>8.4f} "
          f"{T[(ds,tk,orac)].mean():>8.4f} {d.mean():>+8.4f} {sp2:>9.4f}")

dsel, dorac = np.array(dsel), np.array(dorac)
print(f"\n=== Selección honesta por tarea vs comité fijo de 4 ({len(ok)} tareas) ===")
print(f"  Δ medio  = {dsel.mean():+.4f}    gana en {(dsel>0).sum()}/{len(dsel)} tareas"
      f"    Wilcoxon p={sp.wilcoxon(dsel).pvalue:.4f}")
print(f"  oráculo  = {dorac.mean():+.4f}  <- techo inalcanzable; la diferencia con"
      f" {dsel.mean():+.4f} es sesgo de selección")

print("\n=== ¿el mejor comité depende del órgano? (oráculo por tarea) ===")
for ds, tk, o in filas:
    print(f"  {ORGANO[ds]:>5}  {ds+'/'+tk:34} {o}")

# ============================================================================
# Enrutado por ÓRGANO con leave-one-task-out.
#   A fijo   - los 4, siempre
#   B global - un comité para todo, elegido por val de las OTRAS tareas
#   C órgano - comité elegido por val de las otras tareas DEL MISMO órgano
# B y C no miran nunca la tarea evaluada. C es el enrutado ciego realizable:
# el órgano se lee del embedding, no de la etiqueta del dataset.
# ============================================================================
print("\n" + "=" * 86)
print("ENRUTADO POR ÓRGANO (leave-one-task-out)")
print("=" * 86)

tareas = [(ds, tk) for ds, tk, _ in ok]
vmean = {(ds, tk, c): V[(ds, tk, c)].mean() for ds, tk in tareas for c in subs}

hdr2 = f"{'tarea':34} {'org':>5} {'A fijo':>8} {'B global':>9} {'C órgano':>9} {'C elige':>26}"
print(hdr2); print("-" * len(hdr2))
dB, dC, dCB = [], [], []
for ds, tk in tareas:
    otras_all = [(d, t) for d, t in tareas if (d, t) != (ds, tk)]
    otras_org = [(d, t) for d, t in otras_all if ORGANO[d] == ORGANO[ds]]
    pick_B = max(subs, key=lambda c: np.mean([vmean[(d, t, c)] for d, t in otras_all]))
    pick_C = max(subs, key=lambda c: np.mean([vmean[(d, t, c)] for d, t in otras_org]))
    a, b, cc = T[(ds, tk, FULL)].mean(), T[(ds, tk, pick_B)].mean(), T[(ds, tk, pick_C)].mean()
    dB.append(b - a); dC.append(cc - a); dCB.append(cc - b)
    print(f"{ds+'/'+tk:34} {ORGANO[ds]:>5} {a:>8.4f} {b:>9.4f} {cc:>9.4f} {'+'.join(pick_C):>26}")

for nombre, d in (("B global vs A fijo  ", dB), ("C órgano vs A fijo  ", dC),
                  ("C órgano vs B global", dCB)):
    d = np.array(d)
    p = sp.wilcoxon(d).pvalue if not np.allclose(d, 0) else 1.0
    print(f"\n  {nombre}: Δ={d.mean():+.4f}  gana {(d>0).sum()}/{len(d)}  "
          f"pierde {(d<0).sum()}/{len(d)}  Wilcoxon p={p:.4f}")
