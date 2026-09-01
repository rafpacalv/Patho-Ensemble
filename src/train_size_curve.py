#!/usr/bin/env python
"""
train_size_curve.py — ¿Es el tamaño del train lo que decide embeddings vs. probabilidades?

Entre datasets se observó que `emb:logreg` pierde con 75 slides de entrenamiento
por fold (`cptac_brca`, −0.0160) y gana con 164 (`cptac_gbm`, +0.0230 ✱). Eso es
una **correlación entre datasets**, y los datasets difieren en mucho más que en n:
tejido, tarea, balance de clases, parches por slide.

Este script convierte esa correlación en una relación causal **dentro de un solo
dataset**: submuestrea las filas de entrenamiento del metaclasificador y mide el
Δ a cada tamaño. Todo lo demás —modelos base, folds, test, código— queda fijo.

Qué se submuestrea y por qué SÓLO eso:

    ABMIL base           → NO se toca. Se reutilizan los checkpoints entrenados
                           con el train completo.
    filas del meta-train → SÍ, a n filas por fold.
    val, test            → NO se tocan. El test debe ser idéntico a todos los
                           tamaños o los AUC no serían comparables.

Es deliberado. Reentrenar también los ABMIL con menos slides mediría dos cosas a
la vez (modelos base peores + menos filas para el meta) y no diría cuál manda.
La hipótesis concreta a falsar es «una logreg sobre 1536 columnas con 75 filas
sobreajusta», y quien ve esas filas es el metaclasificador.

Predicción falsable: `emb:logreg − prob:logreg` debe ser **negativo con n pequeña
y cruzar el cero** al aumentar n. Si el Δ es plano en n, la explicación no es el
tamaño del train y hay que buscarla en las diferencias entre datasets.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

from run_meta_suite import (  # noqa: E402
    build_xy, check_alignment, check_spaces_aligned, fit_meta,
    fold_metrics, holm, load_space, make_meta, paired,
)

METRICS = ["auc", "ap", "bacc", "macro_f1", "kappa"]


def stratified_subsample(y, n, rng):
    """n índices conservando la proporción de clases del train completo.

    Estratificar importa: un submuestreo simple a n=50 puede dejar una clase con
    3 ejemplos y hundir la logreg por una razón que no es la que se investiga.
    Se garantiza al menos 1 ejemplo por clase presente.
    """
    if n >= len(y):
        return np.arange(len(y))
    classes, counts = np.unique(y, return_counts=True)
    quota = np.maximum(1, np.round(counts / counts.sum() * n).astype(int))
    # ajustar por redondeo
    while quota.sum() > n:
        quota[quota.argmax()] -= 1
    while quota.sum() < n:
        quota[quota.argmin()] += 1
    idx = []
    for c, q in zip(classes, quota):
        pool = np.flatnonzero(y == c)
        idx.append(rng.choice(pool, min(q, len(pool)), replace=False))
    return np.sort(np.concatenate(idx))


def run_at_size(spaces, models, n_folds, space, meta_name, classes, n_train,
                repeats, device, seed0=0):
    """Métricas por fold con el meta-train recortado a `n_train` filas.

    Se promedian `repeats` submuestreos distintos por fold: con n pequeña el
    resultado depende mucho de qué filas toquen, y una sola extracción mediría
    esa lotería en vez del efecto del tamaño.
    """
    P = spaces[space]
    per_fold = {m: [] for m in METRICS}
    used_n = []
    for k in range(n_folds):
        Xtr, ytr, Xva, yva, Xte, yte = build_xy(P, models, k)
        reps = {m: [] for m in METRICS}
        n_eff = min(n_train, len(ytr)) if n_train else len(ytr)
        for r in range(repeats):
            rng = np.random.default_rng(seed0 + 1000 * k + r)
            sel = stratified_subsample(ytr, n_eff, rng)
            if len(np.unique(ytr[sel])) < 2:
                continue
            meta = make_meta(meta_name, k, device)
            fit_meta(meta, Xtr[sel], ytr[sel], Xva, yva)
            proba = np.asarray(meta.predict_proba(Xte))
            if proba.ndim == 1:
                proba = np.column_stack([1 - proba, proba])
            fm = fold_metrics(yte, proba, classes)
            for m in METRICS:
                reps[m].append(fm[m])
            # si no hay submuestreo real, repetir es idéntico
            if n_eff >= len(ytr):
                break
        used_n.append(n_eff)
        for m in METRICS:
            per_fold[m].append(np.nanmean(reps[m]) if reps[m] else np.nan)
    out = {m: np.asarray(per_fold[m], dtype=float) for m in METRICS}
    out["_n"] = float(np.mean(used_n))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--work_dir", default="/home/JKP6679/Patho-Ensemble/PARADIS/datos/patches")
    ap.add_argument("--train_source", default="cptac_gbm")
    ap.add_argument("--task_name", default="TP53_mutation")
    ap.add_argument("--tissue_patching", default="20x_224px_0px_overlap")
    ap.add_argument("--foundational_models", nargs="+",
                    default=["ctranspath", "uni_v2", "conch_v1_5"])
    ap.add_argument("--n_folds", type=int, default=50)
    ap.add_argument("--sizes", nargs="+", type=int,
                    default=[40, 55, 75, 95, 120, 145, 0],
                    help="Filas de meta-train. 0 = sin submuestrear (train completo).")
    ap.add_argument("--repeats", type=int, default=5,
                    help="Submuestreos por fold y tamaño, promediados.")
    ap.add_argument("--meta_model", default="logreg")
    ap.add_argument("--spaces", nargs="+", default=["prob", "emb"])
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    abmil = Path(args.work_dir) / args.train_source / args.task_name / "abmil"
    models = args.foundational_models

    print("=" * 96)
    print("CURVA DE TAMAÑO DE ENTRENAMIENTO — ¿manda n, o manda el dataset?")
    print("=" * 96)
    print(f"Dataset   : {args.train_source} / {args.task_name}")
    print(f"Modelos   : {' + '.join(models)}")
    print(f"Meta-model: {args.meta_model}   espacios: {args.spaces}")
    print(f"Folds     : {args.n_folds}   submuestreos por fold: {args.repeats}")

    spaces = {}
    for sp in args.spaces:
        P = {m: load_space(abmil, m, args.tissue_patching, args.n_folds, sp)
             for m in models}
        check_alignment(P, models, args.n_folds, sp)
        spaces[sp] = P
        print(f"  ✓ '{sp}': {P[models[0]][0][0].shape[1]} col/modelo, "
              f"{len(models) * P[models[0]][0][0].shape[1]} features")
    check_spaces_aligned(spaces, models, args.n_folds)

    ref = spaces[args.spaces[0]][models[0]]
    classes = sorted({int(v) for k in range(args.n_folds)
                      for si in (1, 3, 5) for v in ref[k][si]})
    tr_sizes = [len(ref[k][1]) for k in range(args.n_folds)]
    print(f"  ✓ alineado. Clases: {classes}. "
          f"Train por fold: min={min(tr_sizes)} max={max(tr_sizes)} "
          f"media={np.mean(tr_sizes):.0f}")

    rng = np.random.default_rng(0)
    curve, results = [], {}
    for n in args.sizes:
        label = "full" if n == 0 else str(n)
        row = {"requested": n}
        for sp in args.spaces:
            r = run_at_size(spaces, models, args.n_folds, sp, args.meta_model,
                            classes, n, args.repeats, args.device)
            results[f"{sp}@{label}"] = r
            row[sp] = {m: float(np.nanmean(r[m])) for m in METRICS}
            row["n_eff"] = r["_n"]
        if len(args.spaces) == 2:
            a, b = args.spaces[1], args.spaces[0]   # emb - prob
            st = {}
            for m in METRICS:
                st[m] = paired(results[f"{a}@{label}"][m] - results[f"{b}@{label}"][m], rng)
            adj = holm([st[m]["p_value"] for m in METRICS])
            for m, p in zip(METRICS, adj):
                st[m]["p_holm"] = float(p)
            row["delta"] = st
        curve.append(row)
        d = row.get("delta", {}).get("auc")
        extra = f"   Δ AUC={d['mean_delta']:+.4f} [{d['ci_95'][0]:+.4f},{d['ci_95'][1]:+.4f}]" if d else ""
        print(f"  n={label:>4} (efectivo {row['n_eff']:.0f}): "
              + "  ".join(f"{sp} AUC={row[sp]['auc']:.4f}" for sp in args.spaces)
              + extra, flush=True)

    # ---- tabla ---------------------------------------------------------------
    if len(args.spaces) == 2:
        a, b = args.spaces[1], args.spaces[0]
        print("\n" + "=" * 96)
        print(f"CURVA — Δ AUC de '{a}' menos '{b}' según filas de meta-train")
        print("=" * 96)
        print(f"{'n meta-train':>13}{b+' AUC':>13}{a+' AUC':>13}{'Δ AUC':>10}"
              f"{'IC 95%':>24}{'G/E/P':>11}{'p Holm':>9}")
        print("-" * 96)
        for row in curve:
            d = row["delta"]["auc"]
            ci = f"[{d['ci_95'][0]:+.4f}, {d['ci_95'][1]:+.4f}]"
            gep = f"{d['wins']}/{d['ties']}/{d['losses']}"
            star = " *" if d["p_holm"] < 0.05 else ""
            print(f"{row['n_eff']:>13.0f}{row[b]['auc']:>13.4f}{row[a]['auc']:>13.4f}"
                  f"{d['mean_delta']:>+10.4f}{ci:>24}{gep:>11}{d['p_holm']:>9.4f}{star}")
        print("-" * 96)

        # ---- cruce ----------------------------------------------------------
        xs = np.array([r["n_eff"] for r in curve])
        ys = np.array([r["delta"]["auc"]["mean_delta"] for r in curve])
        o = np.argsort(xs); xs, ys = xs[o], ys[o]
        cross = None
        for i in range(len(xs) - 1):
            if ys[i] < 0 <= ys[i + 1]:
                t = -ys[i] / (ys[i + 1] - ys[i])
                cross = float(xs[i] + t * (xs[i + 1] - xs[i]))
                break
        print("\nVEREDICTO")
        if cross is not None:
            print(f"  El Δ cruza el cero en n ≈ {cross:.0f} filas de meta-train.")
            print("  El signo depende del tamaño del train DENTRO de un mismo dataset:")
            print("  la explicación es n, no las diferencias entre datasets.")
        elif np.all(ys > 0):
            print("  El Δ es positivo en TODOS los tamaños, incluido el más pequeño.")
            print("  ⇒ n NO explica el signo. La diferencia con cptac_brca viene de")
            print("     otra cosa (tejido, tarea, parches/slide, calidad de los modelos base).")
        elif np.all(ys < 0):
            print("  El Δ es negativo en todos los tamaños: los embeddings no ganan aquí.")
        else:
            print("  El Δ cambia de signo pero no de forma monótona; ver la tabla.")
        print(f"  Rango del Δ: {ys.min():+.4f} (n={xs[np.argmin(ys)]:.0f}) "
              f"→ {ys.max():+.4f} (n={xs[np.argmax(ys)]:.0f})")

    payload = {
        "dataset": args.train_source, "task": args.task_name,
        "models": models, "n_folds": args.n_folds, "meta_model": args.meta_model,
        "repeats": args.repeats, "classes": classes, "metrics": METRICS,
        "curve": curve,
        "per_fold": {k: {m: v[m].tolist() for m in METRICS} for k, v in results.items()},
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"\n✅ {args.out}")


if __name__ == "__main__":
    main()