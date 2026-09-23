#!/usr/bin/env python3
"""LogME como criterio de selección de modelos base — sin entrenar el ABMIL.

Motivación: N14 (INFORME_CONSOLIDADO_TESIS_20260901.md) probó 8 criterios de
selección, todos calculados *post-hoc* sobre el ABMIL ya entrenado
(auc_val_medio, brier_val, logloss_val...), y ninguno mejoró la selección
honesta. LogME (You et al., ICML 2021 — ver `logme_utils.py`) es distinto en
un punto: se calcula *pre-hoc*, directamente sobre los embeddings congelados
del foundational model y la etiqueta, sin pasar por el ABMIL. Responde a una
pregunta distinta ("¿merece la pena entrenar un ABMIL con este modelo?") en
vez de "¿qué tal generalizó el ABMIL que ya entrené?".

Requiere los embeddings de atención blanda por slide
(`test_abmil.py --use_embeddings`, GPU), generados aparte para los 8 modelos.

Dos comprobaciones, ambas sin mirar las etiquetas de test:

  1. Ranking: LogME medio por modelo (sobre los 50 folds) frente al AUC de
     test medio de ESE MODELO SOLO (no del ensemble) — Spearman sobre 8
     puntos, más la distribución de Spearman por-fold (50 estimaciones de 8
     puntos cada una, más potencia que el único punto agregado).
  2. Selección honesta LOFO: para cada fold k, se promedia el LogME de cada
     modelo sobre los folds != k, se toman los --topk mejores, y se entrena
     el stacking logreg de sweep_base_models.py sobre esos modelos para el
     fold k. Se compara contra el trío ganador ya conocido
     (ctranspath+uni_v2+conch_v1_5, AUC 0.7986).
"""
import argparse
import json
from pathlib import Path

import numpy as np
from scipy import stats as sp
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

from logme_utils import LogME

N_BOOT = 10000


def load_embeddings(abmil_dir, model, patching, n_folds):
    """{fold: (X, y)} — embeddings de atención blanda del split train_eval."""
    d = abmil_dir / f"{model}_{patching}_train_eval_embeddings" / "val_outputs_embeddings"
    out = {}
    for k in range(n_folds):
        f = d / f"fold_{k}"
        if not (f / "preds.npy").exists():
            raise FileNotFoundError(
                f"faltan embeddings en {f} — corre test_abmil.py --use_embeddings "
                f"--foundational_model {model} primero (ver run_logme_embeddings.sbatch)")
        out[k] = (np.load(f / "preds.npy"), np.load(f / "labels.npy"))
    return out


def load_probs(abmil_dir, model, patching, n_folds):
    """{fold: (Xtr, ytr, Xva, yva, Xte, yte)} — predicciones de clase estándar,
    mismo formato que sweep_base_models.load_model."""
    d = f"{model}_{patching}"
    xd = abmil_dir / f"{d}_train_eval" / "val_outputs"
    vd = abmil_dir / d / "val_outputs"
    td = abmil_dir / d / "test_outputs"
    out = {}
    for k in range(n_folds):
        parts = []
        for base in (xd, vd, td):
            f = base / f"fold_{k}"
            parts += [np.load(f / "preds.npy"), np.load(f / "labels.npy")]
        out[k] = tuple(parts)
    return out


def compute_logme_scores(embeddings, models, n_folds):
    """logme[model][fold] = evidencia LogME (float)."""
    logme = {m: {} for m in models}
    for m in models:
        for k in range(n_folds):
            X, y = embeddings[m][k]
            logme[m][k] = LogME(regression=False).fit(X, y.astype(int))
    return logme


def single_model_test_auc(probs, models, n_folds):
    """auc[model][fold] = AUC de test de ESE modelo solo (no del ensemble)."""
    auc = {m: {} for m in models}
    for m in models:
        for k in range(n_folds):
            _, _, _, _, Xte, yte = probs[m][k]
            p = Xte[:, 1] if Xte.ndim == 2 else Xte
            auc[m][k] = roc_auc_score(yte, p)
    return auc


def eval_subset_logreg(probs, subset, k):
    Xtr = np.concatenate([probs[m][k][0] for m in subset], axis=1)
    ytr = probs[subset[0]][k][1]
    Xte = np.concatenate([probs[m][k][4] for m in subset], axis=1)
    yte = probs[subset[0]][k][5]
    clf = LogisticRegression(max_iter=1000).fit(Xtr, ytr)
    return roc_auc_score(yte, clf.predict_proba(Xte)[:, 1])


def paired(delta, rng):
    if np.allclose(delta, 0):
        return dict(mean_delta=0.0, ci_95=[0.0, 0.0], wins=0, ties=len(delta),
                    losses=0, p_value=1.0)
    bs = np.array([rng.choice(delta, len(delta), replace=True).mean() for _ in range(N_BOOT)])
    lo, hi = np.percentile(bs, [2.5, 97.5])
    return dict(mean_delta=float(delta.mean()), ci_95=[float(lo), float(hi)],
                wins=int((delta > 0).sum()), ties=int((delta == 0).sum()),
                losses=int((delta < 0).sum()), p_value=float(sp.wilcoxon(delta).pvalue))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--work_dir", required=True)
    ap.add_argument("--train_source", required=True)
    ap.add_argument("--task_name", required=True)
    ap.add_argument("--tissue_patching", required=True)
    ap.add_argument("--models", nargs="+", required=True)
    ap.add_argument("--baseline", nargs="+", required=True)
    ap.add_argument("--topk", type=int, default=3)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    abmil = Path(args.work_dir) / args.train_source / args.task_name / "abmil"
    ref_dir = abmil / f"{args.models[0]}_{args.tissue_patching}" / "test_outputs"
    n_folds = len([p for p in ref_dir.iterdir() if p.name.startswith("fold_")])
    print(f"{len(args.models)} modelos, {n_folds} folds\n")

    embeddings = {m: load_embeddings(abmil, m, args.tissue_patching, n_folds) for m in args.models}
    probs = {m: load_probs(abmil, m, args.tissue_patching, n_folds) for m in args.models}
    print("✓ embeddings y predicciones cargados\n")

    logme = compute_logme_scores(embeddings, args.models, n_folds)
    single_auc = single_model_test_auc(probs, args.models, n_folds)

    # --- comprobación 1: ranking agregado y por-fold ---
    mean_logme = {m: np.mean(list(logme[m].values())) for m in args.models}
    mean_auc = {m: np.mean(list(single_auc[m].values())) for m in args.models}
    order = sorted(args.models, key=lambda m: -mean_logme[m])

    print(f'{"modelo":16} {"LogME medio":>12} {"AUC solo (medio)":>18}')
    print("-" * 50)
    for m in order:
        print(f"{m:16} {mean_logme[m]:>12.4f} {mean_auc[m]:>18.4f}")

    rho_agg, p_agg = sp.spearmanr([mean_logme[m] for m in args.models],
                                   [mean_auc[m] for m in args.models])
    print(f"\nSpearman(LogME medio, AUC-solo medio) sobre {len(args.models)} modelos: "
          f"rho={rho_agg:+.3f}  p={p_agg:.3f}")

    per_fold_rho = []
    for k in range(n_folds):
        l = [logme[m][k] for m in args.models]
        a = [single_auc[m][k] for m in args.models]
        r, _ = sp.spearmanr(l, a)
        if not np.isnan(r):
            per_fold_rho.append(r)
    per_fold_rho = np.array(per_fold_rho)
    print(f"Spearman por-fold (n={len(per_fold_rho)} folds válidos): "
          f"media={per_fold_rho.mean():+.3f}  mediana={np.median(per_fold_rho):+.3f}  "
          f"desviación={per_fold_rho.std():.3f}")

    # --- comprobación 2: selección honesta LOFO con top-K por LogME ---
    honest, chosen = [], []
    for k in range(n_folds):
        others = [i for i in range(n_folds) if i != k]
        avg_logme_others = {m: np.mean([logme[m][o] for o in others]) for m in args.models}
        topk_models = tuple(sorted(avg_logme_others, key=lambda m: -avg_logme_others[m])[:args.topk])
        honest.append(eval_subset_logreg(probs, topk_models, k))
        chosen.append("+".join(topk_models))
    honest = np.array(honest)
    freq = {s: chosen.count(s) for s in set(chosen)}

    base = tuple(m for m in args.models if m in set(args.baseline))
    base_auc = np.array([eval_subset_logreg(probs, base, k) for k in range(n_folds)])

    rng = np.random.default_rng(0)
    d_vs_base = paired(honest - base_auc, rng)

    print(f"\nSelección honesta LOFO, top-{args.topk} por LogME:")
    for s, n in sorted(freq.items(), key=lambda kv: -kv[1]):
        print(f"   {n:3}/{n_folds}  {s}")
    print(f"\nAUC LogME top-{args.topk} (honesta): {honest.mean():.4f}")
    print(f"AUC baseline ({'+'.join(base)}):        {base_auc.mean():.4f}")
    print(f'Δ pareado LogME vs baseline: {d_vs_base["mean_delta"]:+.4f} '
          f'[{d_vs_base["ci_95"][0]:+.4f},{d_vs_base["ci_95"][1]:+.4f}] '
          f'W/T/L={d_vs_base["wins"]}/{d_vs_base["ties"]}/{d_vs_base["losses"]} '
          f'p={d_vs_base["p_value"]:.5f}')

    if args.out:
        res = dict(
            dataset=args.train_source, task=args.task_name, n_folds=n_folds,
            models=args.models, baseline=list(base), baseline_auc=float(base_auc.mean()),
            topk=args.topk,
            ranking=[dict(model=m, logme_medio=float(mean_logme[m]), auc_solo_medio=float(mean_auc[m]))
                     for m in order],
            spearman_agregado=dict(rho=float(rho_agg), p=float(p_agg)),
            spearman_por_fold=dict(media=float(per_fold_rho.mean()),
                                    mediana=float(np.median(per_fold_rho)),
                                    desviacion=float(per_fold_rho.std()),
                                    n=int(len(per_fold_rho))),
            honest_logme_topk=dict(auc=float(honest.mean()), elegidos=freq,
                                    delta_vs_baseline=d_vs_base),
        )
        Path(args.out).write_text(json.dumps(res, indent=2, ensure_ascii=False))
        print(f"\n✓ guardado en {args.out}")


if __name__ == "__main__":
    main()
