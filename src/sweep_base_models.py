#!/usr/bin/env python3
"""Barrido de subconjuntos de modelos base con el stacking de `ensemble4.py`.

Responde a "¿qué modelos fundación deben entrar en el ensemble?", que empíricamente
pesa más que la elección de meta-modelo: en cptac_brca/TP53_mutation cambiar un
modelo base valió +0.037 AUC frente a +0.005 de la mejor optimización de pesos.

Replica el camino de `ensemble4.py --meta_model logreg` (feature_space="prob",
sin drop_redundant_class): X = probabilidades concatenadas, LogisticRegression
sobre `{model}_train_eval/val_outputs`, evaluación en `test_outputs`.

Dos estimaciones, y la diferencia entre ellas importa:

  - **naive**: el mejor subconjunto según el AUC medio de test. Sobreestima,
    porque el subconjunto se elige mirando el mismo test con el que se reporta.
    Con 2^n − 1 subconjuntos y 50 folds el sesgo no es pequeño.
  - **honesta**: selección leave-one-fold-out. Para cada fold k se elige el
    subconjunto que maximiza el AUC de *validación* en los folds ≠ k, y se
    reporta su AUC de test en el fold k. Estima el rendimiento del
    procedimiento completo (elegir + aplicar), que es lo que se desplegaría.

Uso:
    python src/sweep_base_models.py \
        --work_dir /home/JKP6679/Patho-Ensemble/PARADIS/datos/patches \
        --train_source cptac_brca --task_name TP53_mutation \
        --tissue_patching 20x_224px_0px_overlap \
        --models ctranspath uni_v2 virchow_v1 conch_v1_5 \
        --baseline ctranspath uni_v2 virchow_v1 \
        --out results_sweep.json
"""
import argparse
import json
from itertools import combinations
from pathlib import Path

import numpy as np
from scipy import stats as sp
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

N_BOOT = 10000


def load_model(abmil_dir, model, patching, n_folds):
    """{fold: (Xtr, ytr, Xva, yva, Xte, yte)} con las probabilidades del modelo."""
    d = f"{model}_{patching}"
    xd = abmil_dir / f"{d}_train_eval" / "val_outputs"
    vd = abmil_dir / d / "val_outputs"
    td = abmil_dir / d / "test_outputs"
    out = {}
    for k in range(n_folds):
        parts = []
        for base in (xd, vd, td):
            f = base / f"fold_{k}"
            if not (f / "preds.npy").exists():
                raise FileNotFoundError(f"falta {f}/preds.npy")
            parts += [np.load(f / "preds.npy"), np.load(f / "labels.npy")]
        out[k] = tuple(parts)
    return out


def check_alignment(P, models, n_folds):
    """Aborta si dos modelos no coinciden fila a fila: es la comprobación que
    separa un barrido válido de uno que compara sobre poblaciones distintas.
    Ha pasado de verdad — una campaña evaluó a nivel paciente (103 filas) y otra
    a nivel slide (112), y los AUC parecían comparables sin serlo."""
    ref_model = models[0]
    for k in range(n_folds):
        for si, name in ((1, "train_eval"), (3, "val"), (5, "test")):
            ref = P[ref_model][k][si]
            for m in models[1:]:
                other = P[m][k][si]
                if len(ref) != len(other):
                    raise SystemExit(
                        f"ABORTADO: fold {k}, split {name}: {ref_model} tiene "
                        f"{len(ref)} filas y {m} tiene {len(other)}. Los modelos "
                        f"provienen de campañas distintas y no son comparables; "
                        f"reentrena el que sobre con el motor actual."
                    )
                if not np.array_equal(ref, other):
                    raise SystemExit(
                        f"ABORTADO: fold {k}, split {name}: etiquetas distintas "
                        f"entre {ref_model} y {m}."
                    )


def eval_subset(P, subset, n_folds):
    """(auc_val, auc_test) por fold para el stacking logreg de este subconjunto."""
    av, at = [], []
    for k in range(n_folds):
        Xtr = np.concatenate([P[m][k][0] for m in subset], axis=1)
        ytr = P[subset[0]][k][1]
        Xva = np.concatenate([P[m][k][2] for m in subset], axis=1)
        yva = P[subset[0]][k][3]
        Xte = np.concatenate([P[m][k][4] for m in subset], axis=1)
        yte = P[subset[0]][k][5]
        clf = LogisticRegression(max_iter=1000).fit(Xtr, ytr)
        av.append(roc_auc_score(yva, clf.predict_proba(Xva)[:, 1]))
        at.append(roc_auc_score(yte, clf.predict_proba(Xte)[:, 1]))
    return np.array(av), np.array(at)


def paired(delta, rng):
    """Δ medio, IC95% bootstrap, W/T/L y p de Wilcoxon."""
    if np.allclose(delta, 0):
        return dict(mean_delta=0.0, ci_95=[0.0, 0.0], wins=0,
                    ties=len(delta), losses=0, p_value=1.0)
    bs = np.array([rng.choice(delta, len(delta), replace=True).mean()
                   for _ in range(N_BOOT)])
    lo, hi = np.percentile(bs, [2.5, 97.5])
    return dict(mean_delta=float(delta.mean()), ci_95=[float(lo), float(hi)],
                wins=int((delta > 0).sum()), ties=int((delta == 0).sum()),
                losses=int((delta < 0).sum()),
                p_value=float(sp.wilcoxon(delta).pvalue))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--work_dir", required=True)
    ap.add_argument("--train_source", required=True)
    ap.add_argument("--task_name", required=True)
    ap.add_argument("--tissue_patching", required=True)
    ap.add_argument("--models", nargs="+", required=True)
    ap.add_argument("--baseline", nargs="+", required=True,
                    help="subconjunto de referencia para el Δ pareado")
    ap.add_argument("--max_subset", type=int, default=None,
                    help="tamaño máximo de subconjunto (por defecto, todos)")
    ap.add_argument("--out", default=None, help="fichero JSON de salida")
    args = ap.parse_args()

    abmil = Path(args.work_dir) / args.train_source / args.task_name / "abmil"
    ref_dir = abmil / f"{args.models[0]}_{args.tissue_patching}" / "test_outputs"
    n_folds = len([p for p in ref_dir.iterdir() if p.name.startswith("fold_")])
    print(f"{len(args.models)} modelos, {n_folds} folds\n")

    P = {m: load_model(abmil, m, args.tissue_patching, n_folds) for m in args.models}
    check_alignment(P, args.models, n_folds)
    print("✓ filas alineadas entre todos los modelos en todos los folds\n")

    top = args.max_subset or len(args.models)
    subsets = [c for r in range(1, top + 1) for c in combinations(args.models, r)]
    V, T = {}, {}
    for c in subsets:
        V[c], T[c] = eval_subset(P, list(c), n_folds)
    print(f"evaluados {len(subsets)} subconjuntos\n")

    # Las tuplas de `combinations` conservan el orden de --models, así que un
    # baseline correcto pero escrito en otro orden no se encontraba y abortaba.
    # Se reordena en vez de exigir que el usuario adivine el orden interno.
    base = tuple(m for m in args.models if m in set(args.baseline))
    desconocidos = set(args.baseline) - set(args.models)
    if desconocidos:
        raise SystemExit(f"el baseline nombra modelos que no están en --models: {sorted(desconocidos)}")
    if base not in T:
        raise SystemExit(f"el baseline {base} no está entre los subconjuntos evaluados "
                         f"(¿--max_subset lo deja fuera?)")
    rng = np.random.default_rng(0)

    ranking = sorted(subsets, key=lambda c: -T[c].mean())
    print(f'{"subconjunto":58} {"AUC test":>9} {"AUC val":>9}')
    print("-" * 80)
    for c in ranking[:15]:
        mark = "  <- baseline" if c == base else ""
        print(f'{"+".join(c):58} {T[c].mean():>9.4f} {V[c].mean():>9.4f}{mark}')
    if base not in ranking[:15]:
        print(f'{"+".join(base):58} {T[base].mean():>9.4f} {V[base].mean():>9.4f}  <- baseline')

    # --- estimación honesta: selección leave-one-fold-out sobre validación ---
    honest, chosen = [], []
    for k in range(n_folds):
        others = [i for i in range(n_folds) if i != k]
        pick = max(subsets, key=lambda c: V[c][others].mean())
        honest.append(T[pick][k])
        chosen.append("+".join(pick))
    honest = np.array(honest)
    freq = {s: chosen.count(s) for s in set(chosen)}

    naive_best = ranking[0]
    print(f'\n{"="*80}')
    print(f'naive  (mejor en test, OPTIMISTA): {"+".join(naive_best):40} {T[naive_best].mean():.4f}')
    print(f'honesta (selección LOFO por val):  {"":40} {honest.mean():.4f}')
    print(f'baseline:                          {"+".join(base):40} {T[base].mean():.4f}')
    print(f'\nsesgo de selección (naive − honesta): {T[naive_best].mean() - honest.mean():+.4f}')
    print("\nsubconjunto elegido por la selección LOFO:")
    for s, n in sorted(freq.items(), key=lambda kv: -kv[1]):
        print(f"   {n:3}/{n_folds}  {s}")

    d_honest = paired(honest - T[base], rng)
    print(f'\nΔ pareado honesta vs baseline: {d_honest["mean_delta"]:+.4f} '
          f'[{d_honest["ci_95"][0]:+.4f},{d_honest["ci_95"][1]:+.4f}] '
          f'W/T/L={d_honest["wins"]}/{d_honest["ties"]}/{d_honest["losses"]} '
          f'p={d_honest["p_value"]:.5f}')

    if args.out:
        res = dict(
            dataset=args.train_source, task=args.task_name, n_folds=n_folds,
            models=args.models, baseline=list(base), meta_model="logreg",
            baseline_auc=float(T[base].mean()),
            naive_best=dict(subset=list(naive_best), auc=float(T[naive_best].mean()),
                            nota="OPTIMISTA: subconjunto elegido sobre el mismo test que reporta"),
            honest=dict(auc=float(honest.mean()), seleccion="leave-one-fold-out sobre val",
                        elegidos=freq, delta_vs_baseline=d_honest),
            sesgo_seleccion=float(T[naive_best].mean() - honest.mean()),
            ranking=[dict(subset=list(c), auc_test=float(T[c].mean()),
                          auc_val=float(V[c].mean()),
                          delta_vs_baseline=paired(T[c] - T[base], rng))
                     for c in ranking],
        )
        Path(args.out).write_text(json.dumps(res, indent=2, ensure_ascii=False))
        print(f"\n✓ guardado en {args.out}")


if __name__ == "__main__":
    main()
