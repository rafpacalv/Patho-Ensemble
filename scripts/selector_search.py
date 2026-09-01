"""¿Hay algún criterio SIN etiquetas de test que ordene los comités mejor que el
AUC de validación (Spearman +0.75)?

Ese +0.75 es el cuello de botella de la campaña: la información para elegir
existe (el oráculo deja +0.0141 sobre la mesa) pero el criterio disponible no la
lee. Aquí se comparan criterios alternativos, todos calculables en producción.

Dos medidas por criterio:
  - Spearman(criterio, AUC test) sobre los 255 subconjuntos: ¿ordena mejor?
  - AUC test de la selección honesta LOFO guiada por ese criterio: ¿sirve?
La segunda decide: un criterio puede ordenar mejor el conjunto y no mover la
selección, porque para elegir sólo importa quién queda en la cima.

Nota de implementación: las métricas por fold se precalculan UNA vez y el
leave-one-fold-out se deriva por diferencia. Las agrupadas (AUC y average
precision sobre los folds concatenados) no son descomponibles y se recalculan,
pero sólo 255x50 veces, no 255x50x8.
"""
import itertools
import os

import numpy as np
from scipy.stats import spearmanr
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score

W = "/home/JKP6679/Patho-Ensemble/PARADIS/datos/patches"
DS, TK, PAT = "cptac_brca", "TP53_mutation", "20x_224px_0px_overlap"
MODELS = ["conch_v1_5", "ctranspath", "hoptimus1", "phikon_v2",
          "uni_v1", "uni_v2", "virchow2", "virchow_v1"]
NF = 50
CACHE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "selector_cache.npz")

subs = [c for r in range(1, len(MODELS) + 1) for c in itertools.combinations(MODELS, r)]
NS = len(subs)


def load(m, k):
    b = f"{W}/{DS}/{TK}/abmil/{m}_{PAT}"
    o = []
    for p in (f"{b}_train_eval/val_outputs", f"{b}/val_outputs", f"{b}/test_outputs"):
        d = f"{p}/fold_{k}"
        o += [np.load(f"{d}/preds.npy"), np.load(f"{d}/labels.npy")]
    return o


if os.path.exists(CACHE):
    z = np.load(CACHE)
    VPROB, VY, TE_AUC, TR_AUC, VLEN = z["vprob"], z["vy"], z["te"], z["tr"], z["vlen"]
    print(f"cache: {CACHE}\n")
else:
    P = {m: {k: load(m, k) for k in range(NF)} for m in MODELS}
    vlen = [len(P[MODELS[0]][k][3]) for k in range(NF)]
    VLEN = np.array(vlen)
    VPROB = np.zeros((NS, int(VLEN.sum())))
    VY = np.concatenate([P[MODELS[0]][k][3] for k in range(NF)])
    TE_AUC = np.zeros((NS, NF))
    TR_AUC = np.zeros((NS, NF))
    off = np.concatenate([[0], np.cumsum(VLEN)])
    for i, c in enumerate(subs):
        for k in range(NF):
            Xtr = np.concatenate([P[m][k][0] for m in c], axis=1); ytr = P[c[0]][k][1]
            Xva = np.concatenate([P[m][k][2] for m in c], axis=1)
            Xte = np.concatenate([P[m][k][4] for m in c], axis=1); yte = P[c[0]][k][5]
            f = LogisticRegression(max_iter=1000).fit(Xtr, ytr)
            VPROB[i, off[k]:off[k + 1]] = f.predict_proba(Xva)[:, 1]
            TE_AUC[i, k] = roc_auc_score(yte, f.predict_proba(Xte)[:, 1])
            TR_AUC[i, k] = roc_auc_score(ytr, f.predict_proba(Xtr)[:, 1])
        if (i + 1) % 50 == 0:
            print(f"  ajustados {i+1}/{NS} subconjuntos", flush=True)
    np.savez(CACHE, vprob=VPROB, vy=VY, te=TE_AUC, tr=TR_AUC, vlen=VLEN)
    print(f"cache guardada en {CACHE}\n")

off = np.concatenate([[0], np.cumsum(VLEN)]).astype(int)
sl = [slice(off[k], off[k + 1]) for k in range(NF)]
test = TE_AUC.mean(axis=1)

# --- por fold, una sola vez ---
AUC_F = np.zeros((NS, NF))
for i in range(NS):
    for k in range(NF):
        AUC_F[i, k] = roc_auc_score(VY[sl[k]], VPROB[i, sl[k]])
Pc = np.clip(VPROB, 1e-9, 1 - 1e-9)
BRI_S = np.array([[((Pc[i, sl[k]] - VY[sl[k]]) ** 2).sum() for k in range(NF)] for i in range(NS)])
LL_S = np.array([[-(VY[sl[k]] * np.log(Pc[i, sl[k]])
                    + (1 - VY[sl[k]]) * np.log(1 - Pc[i, sl[k]])).sum()
                  for k in range(NF)] for i in range(NS)])
NMOD = np.array([len(c) for c in subs], dtype=float)

# --- agrupadas, no descomponibles: 255x50 ---
AUC_P = np.zeros((NS, NF))   # AUC agrupado excluyendo el fold k
AP_P = np.zeros((NS, NF))
for k in range(NF):
    keep = np.ones(len(VY), bool); keep[sl[k]] = False
    y = VY[keep]
    for i in range(NS):
        AUC_P[i, k] = roc_auc_score(y, VPROB[i, keep])
        AP_P[i, k] = average_precision_score(y, VPROB[i, keep])
AUC_P_ALL = np.array([roc_auc_score(VY, VPROB[i]) for i in range(NS)])
AP_ALL = np.array([average_precision_score(VY, VPROB[i]) for i in range(NS)])

NTOT, NK = len(VY), VLEN.astype(float)


def valores(nombre, k=None):
    """Criterio para los 255 subconjuntos. k=None usa todos los folds; si no, los != k."""
    if k is None:
        m, s = AUC_F.mean(axis=1), AUC_F.std(axis=1)
        return {"auc_val_medio": m, "auc_val_agrupado": AUC_P_ALL,
                "brier_val": -BRI_S.sum(1) / NTOT, "logloss_val": -LL_S.sum(1) / NTOT,
                "avg_precision": AP_ALL, "auc_val_penaliz": m - s,
                "n_modelos": NMOD, "auc_metatrain": TR_AUC.mean(axis=1)}[nombre]
    keep = [j for j in range(NF) if j != k]
    m, s = AUC_F[:, keep].mean(1), AUC_F[:, keep].std(1)
    n = NTOT - NK[k]
    return {"auc_val_medio": m, "auc_val_agrupado": AUC_P[:, k],
            "brier_val": -(BRI_S.sum(1) - BRI_S[:, k]) / n,
            "logloss_val": -(LL_S.sum(1) - LL_S[:, k]) / n,
            "avg_precision": AP_P[:, k], "auc_val_penaliz": m - s,
            "n_modelos": NMOD, "auc_metatrain": TR_AUC[:, keep].mean(1)}[nombre]


CRITS = ["auc_val_medio", "auc_val_agrupado", "brier_val", "logloss_val",
         "avg_precision", "auc_val_penaliz", "n_modelos", "auc_metatrain"]

print(f"{'criterio':20} {'Spearman':>9} {'sel.honesta':>12} {'elegido mayoritario':>36}")
print("-" * 82)
res = []
for nm in CRITS:
    rho = spearmanr(valores(nm), test).statistic
    hon, eleg = [], []
    for k in range(NF):
        pick = int(np.argmax(valores(nm, k)))
        hon.append(TE_AUC[pick, k]); eleg.append("+".join(subs[pick]))
    hon = np.array(hon)
    top = max(set(eleg), key=eleg.count)
    res.append((nm, rho, hon.mean()))
    print(f"{nm:20} {rho:>+9.3f} {hon.mean():>12.4f} {top[:34]:>34} {eleg.count(top):>2}/{NF}")

base = test[subs.index(("conch_v1_5", "ctranspath", "uni_v2"))]
orac = test.max()
print(f"\n  trio baseline .......... {base:.4f}")
print(f"  oraculo (techo) ........ {orac:.4f}   margen = {orac-base:+.4f}")
mj = max(res, key=lambda r: r[2])
print(f"  mejor criterio ......... {mj[2]:.4f}  ({mj[0]}, rho={mj[1]:+.3f})")
print(f"  -> recupera {100*(mj[2]-base)/(orac-base):.0f}% del margen disponible")
