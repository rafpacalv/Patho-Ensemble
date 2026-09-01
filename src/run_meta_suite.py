#!/usr/bin/env python
"""
run_meta_suite.py — Suite de metaclasificadores sobre el trío ganador.

Evalúa en UNA SOLA PASADA, sobre los mismos folds y las mismas filas, todas las
combinaciones de:

  * espacio de features del metaclasificador: `prob` (2 col/modelo) o
    `emb` (512 col/modelo, embedding de slide post-atención)
  * metaclasificador: logreg, mlp, mlp_snapshot, svm, knn, nb, mlp_fge

Existe porque la regla operativa del proyecto (CLAUDE.md §"Stored ensemble4_*
results go stale") obliga a recalcular el baseline en la misma pasada que
cualquier contraste nuevo. Ejecutar `ensemble4.py` una vez por configuración
deja cada resultado en un directorio distinto y en un momento distinto; aquí
todas las configuraciones comparten proceso, splits y predicciones base.

Los constructores de los metaclasificadores replican literalmente los de
`ensemble4.py` (líneas 858-985), incluida la semilla `42 + fold`.

Salidas:
  * `<out>.json`  — métricas por fold, agregados y contrastes pareados
  * stdout        — tablas legibles
"""

import argparse
import inspect
import json
import sys
from pathlib import Path

import numpy as np
import scipy.stats as sp
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    cohen_kappa_score,
    f1_score,
    roc_auc_score,
)

sys.path.insert(0, str(Path(__file__).parent))

from meta_models import (  # noqa: E402
    FGEMLPMetaClassifier,
    MLPMetaClassifier,
    SnapshotMLPMetaClassifier,
)

N_BOOT = 10000

# Métricas reportadas. La muestra está desbalanceada (67 wild-type / 45 mutado,
# 40.2 % positivos), así que `accuracy` por sí sola es engañosa: un clasificador
# que prediga siempre la clase mayoritaria saca 0.598. Por eso se reportan
# además bacc, macro-F1 y kappa, que sí penalizan ese degenerado.
RANK_METRICS = ["auc", "ap"]                                   # no dependen del umbral
THR_METRICS = ["acc", "bacc", "macro_f1", "f1_pos", "kappa"]   # sí dependen del umbral

# Cada métrica de umbral se reporta dos veces:
#   `<m>`    → umbral 0.5 (argmax), que es lo que hace ensemble4.py
#   `<m>_t`  → umbral elegido en VALIDACIÓN maximizando balanced accuracy
# La distinción importa aquí: con 40.2 % de positivos, un metaclasificador puede
# ordenar bien (AUC alto) y aun así clasificar mal en 0.5, porque el óptimo de
# decisión no está en 0.5 cuando las clases están desbalanceadas. Sin las dos
# columnas no se puede saber si un brazo falla al ordenar o solo al cortar.
METRICS = RANK_METRICS + THR_METRICS + [f"{m}_t" for m in THR_METRICS]


# --------------------------------------------------------------------------- #
# Carga de predicciones base
# --------------------------------------------------------------------------- #
def load_space(abmil_dir, model, patching, n_folds, space):
    """{fold: (Xtr, ytr, Xva, yva, Xte, yte)} para un modelo base y un espacio.

    `space='prob'`      → salida softmax del ABMIL base, [N, 2]
    `space='emb'`       → embedding de slide post-atención, [N, 512]
    `space='<tag>'`     → salida de una variante snapshot (FGE/SE) escrita por
                          test_abmil_fge.py con --fge_tag <tag>, [N, 2]

    Los tres splits salen de directorios distintos y ese detalle es el que
    decide si el resultado significa algo (ver §6.2 del estado del arte):
      * train_eval → predicciones sobre el propio train del fold (in-sample)
      * val        → early stopping del ABMIL y del metaclasificador
      * test       → única superficie de reporte

    Todos los espacios etiquetados comparten el mismo esquema de rutas, así que
    'embeddings' y un tag de snapshot se resuelven con el mismo código: lo que
    cambia es qué escribió esos ficheros, no dónde están.
    """
    d = f"{model}_{patching}"
    if space == "prob":
        xd = abmil_dir / f"{d}_train_eval" / "val_outputs"
        vd = abmil_dir / d / "val_outputs"
        td = abmil_dir / d / "test_outputs"
    else:
        tag = "embeddings" if space == "emb" else space
        xd = abmil_dir / f"{d}_train_eval_{tag}" / f"val_outputs_{tag}"
        vd = abmil_dir / d / f"val_outputs_{tag}"
        td = abmil_dir / d / f"test_outputs_{tag}"

    out = {}
    for k in range(n_folds):
        parts = []
        for base in (xd, vd, td):
            f = base / f"fold_{k}"
            if not (f / "preds.npy").exists():
                if space == "prob":
                    hint = "correr antes test_abmil.py"
                elif space == "emb":
                    hint = "correr antes test_abmil.py --use_embeddings"
                else:
                    hint = f"correr antes test_abmil_fge.py --fge_tag {space}"
                raise FileNotFoundError(f"falta {f}/preds.npy — {hint}")
            parts += [np.load(f / "preds.npy"), np.load(f / "labels.npy")]
        out[k] = tuple(parts)
    return out


def check_alignment(P, models, n_folds, tag):
    """Aborta si dos modelos no coinciden fila a fila.

    Es la comprobación que separa un barrido válido de uno que compara sobre
    poblaciones distintas. Ha pasado de verdad: una campaña evaluó a nivel
    paciente (103 filas) y otra a nivel slide (112), y los AUC parecían
    comparables sin serlo.
    """
    ref_model = models[0]
    for k in range(n_folds):
        for si, name in ((1, "train_eval"), (3, "val"), (5, "test")):
            ref = P[ref_model][k][si]
            for m in models[1:]:
                other = P[m][k][si]
                if len(ref) != len(other):
                    raise SystemExit(
                        f"ABORTADO [{tag}]: fold {k}, split {name}: {ref_model} "
                        f"tiene {len(ref)} filas y {m} tiene {len(other)}."
                    )
                if not np.array_equal(ref, other):
                    raise SystemExit(
                        f"ABORTADO [{tag}]: fold {k}, split {name}: etiquetas "
                        f"distintas entre {ref_model} y {m}."
                    )


def check_spaces_aligned(spaces, models, n_folds):
    """Los espacios prob y emb deben describir las mismas filas.

    Sin esto, comparar `logreg-prob` con `mlp_snapshot-emb` podría estar
    comparando sobre poblaciones distintas si una de las dos ramas se generó
    con otro k=all.tsv.
    """
    names = list(spaces)
    if len(names) < 2:
        return
    ref_space = names[0]
    for k in range(n_folds):
        for si, name in ((1, "train_eval"), (3, "val"), (5, "test")):
            ref = spaces[ref_space][models[0]][k][si]
            for s in names[1:]:
                other = spaces[s][models[0]][k][si]
                if len(ref) != len(other) or not np.array_equal(ref, other):
                    raise SystemExit(
                        f"ABORTADO: fold {k}, split {name}: el espacio "
                        f"'{ref_space}' y el espacio '{s}' no describen las "
                        f"mismas filas ({len(ref)} vs {len(other)}). Las ramas "
                        f"vienen de splits distintos y no son comparables."
                    )


def build_xy(P, models, k):
    """Concatena los bloques por modelo igual que ensemble4.py (model-major)."""
    Xtr = np.concatenate([P[m][k][0] for m in models], axis=1)
    ytr = P[models[0]][k][1]
    Xva = np.concatenate([P[m][k][2] for m in models], axis=1)
    yva = P[models[0]][k][3]
    Xte = np.concatenate([P[m][k][4] for m in models], axis=1)
    yte = P[models[0]][k][5]
    return Xtr, ytr, Xva, yva, Xte, yte


# --------------------------------------------------------------------------- #
# Metaclasificadores — constructores idénticos a ensemble4.py
# --------------------------------------------------------------------------- #
def make_meta(name, fold, device):
    seed = 42 + fold
    # ensemble4.py (línea 711) resuelve el `None` de --meta_epochs a 120 para
    # mlp_snapshot y 100 para el resto; replicarlo aquí es lo que hace que estos
    # brazos sean los mismos que los de la configuración de referencia.
    epochs = 120 if name == "mlp_snapshot" else 100
    if name == "logreg":
        return LogisticRegression(max_iter=1000)
    if name == "mlp":
        return MLPMetaClassifier(hidden_dim=16, dropout=0.1, epochs=epochs,
                                 lr=1e-3, wd=1e-4, patience=8,
                                 device=device, seed=seed)
    if name == "mlp_snapshot":
        return SnapshotMLPMetaClassifier(hidden_dim=16, dropout=0.1, epochs=epochs,
                                         restart_lr=5e-3, n_cycles=6,
                                         n_snapshots=None, wd=1e-4,
                                         device=device, seed=seed)
    if name == "mlp_fge":
        return FGEMLPMetaClassifier(hidden_dim=16, dropout=0.1,
                                    warmup_epochs=epochs, warmup_lr=1e-3,
                                    warmup_wd=1e-4, warmup_patience=8,
                                    n_cycles=6, cycle_length=3,
                                    lr_1=1e-3, lr_2=1e-5, wd=1e-4,
                                    cycle_patience=2, device=device, seed=seed)
    if name == "svm":
        return SVC(probability=True, kernel="linear", C=1.0, random_state=seed)
    if name == "knn":
        return KNeighborsClassifier(n_neighbors=5, weights="distance")
    if name == "nb":
        return GaussianNB()
    raise ValueError(f"meta-modelo desconocido: {name}")


def fit_meta(meta, Xtr, ytr, Xva, yva):
    """Pasa el set de validación solo a los modelos que lo aceptan."""
    params = inspect.signature(meta.fit).parameters
    if "X_val" in params and "y_val" in params:
        meta.fit(Xtr, ytr, X_val=Xva, y_val=yva)
    else:
        meta.fit(Xtr, ytr)


# --------------------------------------------------------------------------- #
# Métricas
# --------------------------------------------------------------------------- #
def _thr_scores(y_true, yhat, binary):
    """Métricas de decisión. `f1_pos` sólo existe en binario."""
    return {
        "acc": float(accuracy_score(y_true, yhat)),
        "bacc": float(balanced_accuracy_score(y_true, yhat)),
        "macro_f1": float(f1_score(y_true, yhat, average="macro", zero_division=0)),
        "f1_pos": (float(f1_score(y_true, yhat, pos_label=1, zero_division=0))
                   if binary else float("nan")),
        "kappa": float(cohen_kappa_score(y_true, yhat)),
    }


def pick_threshold(y_val, p_val):
    """Umbral que maximiza balanced accuracy en VALIDACIÓN.

    Se elige en `val`, nunca en `test`: val ya se usa para el early stopping del
    ABMIL y del metaclasificador, así que no añade ninguna fuga nueva, mientras
    que ajustarlo en test invalidaría la única superficie de reporte honesta.
    Empate → se prefiere el umbral más cercano a 0.5, para no moverse del
    criterio por defecto sin motivo.

    Sólo binario: en multiclase la decisión no es un corte escalar, así que el
    bloque C se reporta como NaN en vez de inventar una generalización.
    """
    if len(np.unique(y_val)) < 2:
        return 0.5
    cands = np.unique(np.concatenate([p_val, [0.5]]))
    best_thr, best_score = 0.5, -np.inf
    for t in cands:
        s = balanced_accuracy_score(y_val, (p_val >= t).astype(int))
        if s > best_score or (s == best_score and abs(t - 0.5) < abs(best_thr - 0.5)):
            best_thr, best_score = t, s
    return float(best_thr)


def _rank_scores(y_true, proba, classes, binary):
    """AUC y precisión media. En multiclase, ambas macro sobre one-vs-rest.

    Un fold cuyo test no contiene todas las clases deja el macro-OvR indefinido:
    se propaga NaN en vez de inventar un 0.5, que sesgaría la media hacia el
    azar. `paired()` descarta esos folds, así que el contraste sigue siendo
    pareado sobre los folds en que ambas ramas están definidas.
    """
    present = np.unique(y_true)
    if len(present) < 2:
        return {"auc": float("nan"), "ap": float("nan")}
    try:
        if binary:
            auc = float(roc_auc_score(y_true, proba[:, 1]))
            ap = float(average_precision_score(y_true, proba[:, 1]))
        else:
            auc = float(roc_auc_score(y_true, proba, multi_class="ovr",
                                      average="macro", labels=classes))
            onehot = (y_true[:, None] == np.asarray(classes)[None, :]).astype(int)
            ap = float(average_precision_score(onehot, proba, average="macro"))
    except ValueError:
        # p.ej. macro-OvR con una clase ausente del test de este fold
        return {"auc": float("nan"), "ap": float("nan")}
    return {"auc": auc, "ap": ap}


def fold_metrics(y_true, proba, classes, thr=0.5):
    """Métricas de un fold. `proba` es [N, K]; en binario la positiva es la 1.

    `thr` es el umbral ajustado en validación; las métricas sin sufijo usan
    siempre argmax, que es el criterio de ensemble4.py.
    """
    binary = len(classes) == 2
    out = _rank_scores(y_true, proba, classes, binary)
    out.update(_thr_scores(y_true, proba.argmax(axis=1), binary))
    if binary:
        tuned = _thr_scores(y_true, (proba[:, 1] >= thr).astype(int), binary)
    else:
        tuned = {k: float("nan") for k in THR_METRICS}
    out.update({f"{k}_t": v for k, v in tuned.items()})
    return out


def degenerate_baseline(ref, n_folds, classes):
    """Métricas del clasificador que siempre predice la clase mayoritaria.

    Es la vara con la que se lee `accuracy` en una tarea desbalanceada: sin
    ella, un 0.60 parece un resultado cuando puede ser el degenerado. La clase
    mayoritaria se toma del TRAIN de cada fold (no del test), que es lo que un
    clasificador real podría saber, y se promedia sobre folds.
    """
    binary = len(classes) == 2
    acc, bacc, mf1, f1p, kap = [], [], [], [], []
    for k in range(n_folds):
        ytr, yte = ref[k][1], ref[k][5]
        major = int(np.bincount(ytr.astype(int), minlength=len(classes)).argmax())
        yhat = np.full_like(yte, major)
        sc = _thr_scores(yte, yhat, binary)
        acc.append(sc["acc"]); bacc.append(sc["bacc"])
        mf1.append(sc["macro_f1"]); f1p.append(sc["f1_pos"]); kap.append(sc["kappa"])
    return dict(acc=float(np.mean(acc)), bacc=float(np.mean(bacc)),
                macro_f1=float(np.mean(mf1)), f1_pos=float(np.nanmean(f1p)) if binary else float("nan"),
                kappa=float(np.mean(kap)))


def run_config(spaces, models, n_folds, space, meta_name, device, classes, verbose=True):
    """Entrena y evalúa una configuración en los n_folds. Devuelve dict de arrays."""
    P = spaces[space]
    per_fold = {m: [] for m in METRICS}
    thresholds = []
    binary = len(classes) == 2

    def _proba(meta, X):
        """Alinea las columnas de predict_proba con `classes`.

        Un fold cuyo train no contiene alguna clase produce un predict_proba con
        menos columnas; sin este realineado, argmax señalaría la clase
        equivocada en silencio.
        """
        p = np.asarray(meta.predict_proba(X))
        if p.ndim == 1:
            p = np.column_stack([1 - p, p])
        seen = getattr(meta, "classes_", None)
        if seen is None or len(seen) == len(classes):
            return p
        full = np.zeros((p.shape[0], len(classes)))
        for j, c in enumerate(seen):
            full[:, list(classes).index(c)] = p[:, j]
        return full

    for k in range(n_folds):
        Xtr, ytr, Xva, yva, Xte, yte = build_xy(P, models, k)
        meta = make_meta(meta_name, k, device)
        fit_meta(meta, Xtr, ytr, Xva, yva)
        thr = pick_threshold(yva, _proba(meta, Xva)[:, 1]) if binary else float("nan")
        thresholds.append(thr)
        fm = fold_metrics(yte, _proba(meta, Xte), classes, thr=thr)
        for m in METRICS:
            per_fold[m].append(fm[m])
        if verbose and (k + 1) % 10 == 0:
            print(f"    fold {k + 1}/{n_folds}", flush=True)
    res = {m: np.asarray(per_fold[m], dtype=float) for m in METRICS}
    res["_thresholds"] = np.asarray(thresholds, dtype=float)
    return res


# --------------------------------------------------------------------------- #
# Contraste estadístico pareado
# --------------------------------------------------------------------------- #
def paired(delta, rng):
    """Δ medio, IC95 % bootstrap, G/E/P, p de Wilcoxon y MDE al 80 % de potencia.

    Los empates se cuentan explícitamente: con 22 slides de test por fold, kappa
    ha llegado a empatar en 22 de 50 folds, y ocultarlos hace que un nulo
    parezca una victoria estrecha.
    """
    delta = delta[~np.isnan(delta)]
    n = len(delta)
    if n == 0:
        # p = 1.0 (no NaN) para que Holm no propague NaN por toda la familia
        return dict(n=0, mean_delta=float("nan"), ci_95=[float("nan")] * 2,
                    wins=0, ties=0, losses=0, p_value=1.0, mde=float("nan"))
    sd = float(delta.std(ddof=1)) if n > 1 else 0.0
    mde = (1.96 + 0.842) * sd / np.sqrt(n) if sd > 0 else 0.0
    if np.allclose(delta, 0):
        return dict(n=n, mean_delta=0.0, ci_95=[0.0, 0.0], wins=0, ties=n,
                    losses=0, p_value=1.0, mde=float(mde))
    bs = np.array([rng.choice(delta, n, replace=True).mean() for _ in range(N_BOOT)])
    lo, hi = np.percentile(bs, [2.5, 97.5])
    return dict(
        n=n,
        mean_delta=float(delta.mean()),
        ci_95=[float(lo), float(hi)],
        wins=int((delta > 0).sum()),
        ties=int((delta == 0).sum()),
        losses=int((delta < 0).sum()),
        p_value=float(sp.wilcoxon(delta).pvalue),
        mde=float(mde),
    )


def holm(pvals):
    """Corrección de Holm-Bonferroni sobre la familia de métricas."""
    p = np.asarray(pvals, dtype=float)
    order = np.argsort(p)
    adj = np.empty_like(p)
    running = 0.0
    for rank, idx in enumerate(order):
        val = (len(p) - rank) * p[idx]
        running = max(running, val)
        adj[idx] = min(running, 1.0)
    return adj


# --------------------------------------------------------------------------- #
# Presentación
# --------------------------------------------------------------------------- #
def _block(results, order, cols, title, note):
    width = 30 + 12 * len(cols)
    print("\n" + "=" * width)
    print(title)
    print("=" * width)
    print(f"{'Configuración':<30}" + "".join(f"{c:>12}" for c in cols))
    print("-" * width)
    for name in order:
        print(f"{name:<30}"
              + "".join(f"{np.nanmean(results[name][c]):>12.4f}" for c in cols))
    print("-" * width)
    print(note)


def print_summary_table(results, order, degenerate_note, binary):
    _block(results, order, RANK_METRICS,
           "RESUMEN A — capacidad de ORDENACIÓN (independiente del umbral)",
           "auc=macro-OvR AUC · ap=average precision (PR-AUC, la métrica de "
           "ordenación\nque más pesa bajo desbalance porque ignora los "
           "verdaderos negativos)")
    _block(results, order, THR_METRICS,
           "RESUMEN B — DECISIÓN con umbral 0.5 (argmax; el criterio de ensemble4.py)",
           degenerate_note)
    if binary:
        _block(results, order, [f"{m}_t" for m in THR_METRICS],
               "RESUMEN C — DECISIÓN con umbral ajustado en VALIDACIÓN (máx. bacc)",
               "Mismas métricas, con el umbral elegido en val en vez de fijado a 0.5.\n"
               "La diferencia B→C separa un fallo de ORDENACIÓN de un fallo de CORTE.")
        print("\nUmbral medio elegido en validación:")
        for name in order:
            thr = results[name].get("_thresholds")
            if thr is not None and not np.all(np.isnan(thr)):
                print(f"  {name:<30} {np.nanmean(thr):.3f}  "
                      f"(mediana {np.nanmedian(thr):.3f})")
    else:
        print("\nRESUMEN C omitido: sólo aplica a tareas binarias.")


def print_paired_table(name, baseline_name, stats):
    print("\n" + "=" * 104)
    print(f"PAREADO: {name}  vs  {baseline_name}  (baseline recalculado en esta misma pasada)")
    print("=" * 104)
    print(f"{'Métrica':<12}{'Δ medio':>10}{'IC 95%':>24}{'G/E/P':>12}"
          f"{'p':>9}{'p Holm':>9}{'MDE':>9}")
    print("-" * 104)
    for m in METRICS:
        s = stats[m]
        ci = f"[{s['ci_95'][0]:+.4f}, {s['ci_95'][1]:+.4f}]"
        gep = f"{s['wins']}/{s['ties']}/{s['losses']}"
        star = " *" if s["p_holm"] < 0.05 else ""
        print(f"{m:<12}{s['mean_delta']:>+10.4f}{ci:>24}{gep:>12}"
              f"{s['p_value']:>9.4f}{s['p_holm']:>9.4f}{s['mde']:>9.4f}{star}")
    print("-" * 104)


# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--work_dir", default="/home/JKP6679/Patho-Ensemble/PARADIS/datos/patches")
    ap.add_argument("--train_source", default="cptac_brca")
    ap.add_argument("--task_name", default="TP53_mutation")
    ap.add_argument("--tissue_patching", default="20x_224px_0px_overlap")
    ap.add_argument("--foundational_models", nargs="+",
                    default=["ctranspath", "uni_v2", "conch_v1_5"])
    ap.add_argument("--n_folds", type=int, default=50)
    ap.add_argument("--configs", nargs="+", required=True,
                    help="Configuraciones 'espacio:meta', p.ej. prob:logreg emb:mlp_snapshot")
    ap.add_argument("--baseline", default="prob:logreg",
                    help="Configuración de referencia para los contrastes pareados")
    ap.add_argument("--out", required=True, help="Ruta del JSON de resultados")
    ap.add_argument("--device", default=None)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    import torch
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    abmil_dir = Path(args.work_dir) / args.train_source / args.task_name / "abmil"
    models = args.foundational_models

    configs = []
    for c in args.configs:
        space, meta = c.split(":", 1)
        configs.append((c, space, meta))
    if args.baseline not in [c[0] for c in configs]:
        raise SystemExit(f"el baseline {args.baseline} no está en --configs")

    needed_spaces = sorted({s for _, s, _ in configs})

    print("=" * 104)
    print("SUITE DE METACLASIFICADORES — trío ganador")
    print("=" * 104)
    print(f"Modelos base : {' + '.join(models)}")
    print(f"Dataset/tarea: {args.train_source} / {args.task_name}")
    print(f"Folds        : {args.n_folds}")
    print(f"Espacios     : {needed_spaces}")
    print(f"Dispositivo  : {device}")

    # ---- carga y verificación de alineamiento --------------------------------
    spaces = {}
    for space in needed_spaces:
        print(f"\nCargando espacio '{space}' ...", flush=True)
        P = {m: load_space(abmil_dir, m, args.tissue_patching, args.n_folds, space)
             for m in models}
        check_alignment(P, models, args.n_folds, space)
        shapes = {m: P[m][0][0].shape[1] for m in models}
        print(f"  ✓ alineado. Columnas por modelo: {shapes}")
        print(f"  ✓ filas fold_0: train={len(P[models[0]][0][1])} "
              f"val={len(P[models[0]][0][3])} test={len(P[models[0]][0][5])}")
        spaces[space] = P
    check_spaces_aligned(spaces, models, args.n_folds)
    print("\n✓ Todos los brazos comparten exactamente los mismos folds y filas "
          "de train/val/test.")

    # ---- cardinalidad de la tarea -------------------------------------------
    # Se toma de la UNIÓN de las etiquetas sobre todos los folds y splits: un
    # fold suelto puede no contener las K clases, y fijar K por un solo fold
    # descuadraría el macro-OvR en los demás.
    ref = spaces[needed_spaces[0]][models[0]]
    classes = sorted({int(v) for k in range(args.n_folds)
                      for si in (1, 3, 5) for v in ref[k][si]})
    binary = len(classes) == 2
    prior = np.concatenate([ref[k][5] for k in range(args.n_folds)])
    counts = {c: int((prior == c).sum()) for c in classes}
    major = max(counts, key=counts.get)
    print(f"\nClases         : K={len(classes)} {classes} "
          f"({'binaria' if binary else 'multiclase'})")
    print(f"Reparto en test: {counts}  →  mayoritaria={major} "
          f"({counts[major] / len(prior):.1%})")
    if not binary:
        print("⚠ Multiclase: AUC y AP son macro-OvR, `f1_pos` no aplica y el "
              "bloque C (umbral ajustado) no se calcula —\n"
              "  la decisión multiclase no es un corte escalar.")

    deg = degenerate_baseline(ref, args.n_folds, classes)
    degenerate_note = (
        "acc=accuracy · bacc=balanced accuracy · macro_f1=F1 macro · "
        "f1_pos=F1 de la clase positiva · kappa=Cohen.\n"
        f"REFERENCIA DEGENERADA (predecir siempre la mayoritaria del train): "
        f"acc={deg['acc']:.3f}, bacc={deg['bacc']:.3f}, "
        f"macro_f1={deg['macro_f1']:.3f}, kappa={deg['kappa']:.3f}.\n"
        "Cualquier accuracy que no supere claramente ese valor no dice nada."
    )
    print("\n" + degenerate_note)

    # ---- ejecución -----------------------------------------------------------
    results = {}
    for name, space, meta in configs:
        print(f"\n>>> {name}", flush=True)
        results[name] = run_config(spaces, models, args.n_folds, space, meta, device, classes)
        print(f"    AUC={np.nanmean(results[name]['auc']):.4f}  "
              f"bacc={np.nanmean(results[name]['bacc']):.4f}  "
              f"macro-F1={np.nanmean(results[name]['macro_f1']):.4f}")

    order = [c[0] for c in configs]
    print_summary_table(results, order, degenerate_note, binary)

    # ---- contrastes pareados -------------------------------------------------
    rng = np.random.default_rng(args.seed)
    base = results[args.baseline]
    comparisons = {}
    for name in order:
        if name == args.baseline:
            continue
        stats = {}
        for m in METRICS:
            stats[m] = paired(results[name][m] - base[m], rng)
        adj = holm([stats[m]["p_value"] for m in METRICS])
        for m, pa in zip(METRICS, adj):
            stats[m]["p_holm"] = float(pa)
        comparisons[name] = stats
        print_paired_table(name, args.baseline, stats)

    # ---- volcado -------------------------------------------------------------
    payload = {
        "dataset": args.train_source,
        "task": args.task_name,
        "foundational_models": models,
        "n_folds": args.n_folds,
        "baseline": args.baseline,
        "classes": classes,
        "binary": binary,
        "degenerate_baseline": deg,
        "metrics": METRICS,
        "aggregate": {
            name: {m: float(np.nanmean(results[name][m])) for m in METRICS}
            for name in order
        },
        "aggregate_std": {
            name: {m: float(np.nanstd(results[name][m], ddof=1)) for m in METRICS}
            for name in order
        },
        "per_fold": {
            name: {m: results[name][m].tolist() for m in METRICS} for name in order
        },
        "val_thresholds": {
            name: results[name]["_thresholds"].tolist() for name in order
        },
        "paired_vs_baseline": comparisons,
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"\n✅ Resultados escritos en {args.out}")


if __name__ == "__main__":
    main()
