#!/bin/bash
#SBATCH --job-name=bag_size_sweep
#SBATCH --partition=main
#SBATCH --constraint=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=16:00:00
#SBATCH --output=logs/bag_size_sweep_%j.out
#SBATCH --error=logs/bag_size_sweep_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=rafael.pachon.alvarez@gmail.com

# ============================================================================
# Barrido de bag_size — augmentación de bolsa sobre el trío ganador
# ============================================================================
# Hipótesis: el cuello de botella del pipeline son 75 slides de entrenamiento
# contra 0.4-1.4 M de parámetros (ver §6.2 y §7.2 de ESTADO_DEL_ARTE_20260822).
# Muestrear N parches por forward convierte cada slide en una bolsa distinta
# por época, lo que multiplica los ejemplos efectivos sin recomputar features.
#
# ATENCIÓN — el muestreo SOLO actúa en el bucle de entrenamiento. La evaluación,
# la predicción y las meta-features usan la bolsa completa. Si el muestreo se
# colara en `_train_eval` cambiaría la distribución de entrada del
# metaclasificador y contaminaría la comparación de §6.2.
#
# El baseline (bag_size=0, bolsa completa) se REENTRENA en esta misma pasada.
# Es la regla de §6.5: un ensemble4_* antiguo no es comparable con uno nuevo.
# Por eso no se reutiliza el 0.7986 guardado.
#
#   sbatch run_bag_size_sweep.sbatch
#   BAGS="256 512" sbatch run_bag_size_sweep.sbatch
# ============================================================================

set -euo pipefail

count_files() { local n; n=$(ls -1d "$@" 2>/dev/null | wc -l) || n=0; echo "$n"; }

source $(conda info --base)/etc/profile.d/conda.sh
conda activate ensemble
cd /shared/home/JKP6679/Patho-Ensemble

W=/home/JKP6679/Patho-Ensemble/PARADIS/datos/patches
DATASET=${DATASET:-cptac_brca}
TASK=${TASK:-TP53_mutation}
PATCHING=${PATCHING:-20x_224px_0px_overlap}
MODELS=${MODELS:-"ctranspath uni_v2 conch_v1_5"}
BAGS=${BAGS:-"0 256 512 1024"}
EPOCHS=${EPOCHS:-100}
OUT=results_bag_size_sweep_${DATASET}_${TASK}.json
ABMIL_DIR=$W/$DATASET/$TASK/abmil

echo "============================================================================"
echo "Barrido de bag_size — $DATASET / $TASK"
echo "============================================================================"
echo "Inicio:   $(date)"
echo "modelos:  $MODELS"
echo "bag_size: $BAGS   (0 = bolsa completa, el baseline reentrenado)"
echo "salida:   $OUT"
echo "============================================================================"
echo

# ── Fase 1: entrenar + meta-features para cada bag_size ─────────────────────
for BAG in $BAGS; do
    if [ "$BAG" = "0" ]; then TAG="bagfull"; else TAG="bag${BAG}"; fi
    echo "############################################################"
    echo "### bag_size=$BAG   (tag: $TAG)"
    echo "############################################################"

    for M in $MODELS; do
        echo "[$(date '+%H:%M:%S')] $M / $TAG — entrenando 50 folds..."
        python src/train_abmil.py \
            --foundational_model "$M" \
            --work_dir "$W" \
            --train_source "$DATASET" \
            --tissue_patching "$PATCHING" \
            --task_name "$TASK" \
            --epochs "$EPOCHS" \
            --fold all \
            --bag_size "$BAG" \
            --arch_tag "$TAG"

        echo "[$(date '+%H:%M:%S')] $M / $TAG — meta-features (bolsa COMPLETA)..."
        python src/test_abmil.py \
            --foundational_model "$M" \
            --work_dir "$W" \
            --train_source "$DATASET" \
            --tissue_patching "$PATCHING" \
            --task_name "$TASK" \
            --arch_tag "$TAG"

        NC=$(count_files "$ABMIL_DIR/${M}_${TAG}_${PATCHING}/checkpoints/fold_"*)
        NT=$(count_files "$ABMIL_DIR/${M}_${TAG}_${PATCHING}_train_eval/val_outputs/fold_"*)
        echo "[$(date '+%H:%M:%S')] ✓ $M / $TAG — checkpoints=$NC  train_eval=$NT"
    done
    echo
done

# ── Fase 2: stacking + comparación pareada, todo en la misma pasada ─────────
echo "============================================================================"
echo "FASE 2: stacking LogReg y comparación pareada"
echo "============================================================================"

python - "$W" "$DATASET" "$TASK" "$PATCHING" "$OUT" "$MODELS" "$BAGS" <<'PYEOF'
import sys, json, itertools
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, balanced_accuracy_score, f1_score, cohen_kappa_score

W, DS, TASK, PATCHING, OUT, MODELS, BAGS = sys.argv[1:8]
MODELS = MODELS.split()
BAGS = BAGS.split()
ROOT = f"{W}/{DS}/{TASK}/abmil"
RNG = np.random.default_rng(0)


def tag_of(b):
    return "bagfull" if b == "0" else f"bag{b}"


def load(model, tag, sub, k):
    d = f"{ROOT}/{model}_{tag}_{PATCHING}{sub}/fold_{k}"
    return np.load(f"{d}/preds.npy"), np.load(f"{d}/labels.npy")


def evaluate(tag):
    """Stacking LogReg por fold. Devuelve dict fold -> métricas."""
    per = {}
    for k in range(50):
        try:
            Xtr = np.hstack([load(m, tag, "_train_eval/val_outputs", k)[0] for m in MODELS])
            ytr = load(MODELS[0], tag, "_train_eval/val_outputs", k)[1]
            Xte = np.hstack([load(m, tag, "/test_outputs", k)[0] for m in MODELS])
            yte = load(MODELS[0], tag, "/test_outputs", k)[1]
        except FileNotFoundError:
            continue
        if len(np.unique(ytr)) < 2 or len(np.unique(yte)) < 2:
            continue
        lr = LogisticRegression(max_iter=2000).fit(Xtr, ytr)
        p = lr.predict_proba(Xte)[:, 1]
        yh = (p > 0.5).astype(int)
        per[k] = dict(auc=roc_auc_score(yte, p),
                      bacc=balanced_accuracy_score(yte, yh),
                      f1=f1_score(yte, yh, average="macro"),
                      kappa=cohen_kappa_score(yte, yh, weights="linear"))
    return per


def paired(a, b, key):
    ks = sorted(set(a) & set(b))
    d = np.array([a[k][key] - b[k][key] for k in ks])
    if len(d) == 0:
        return None
    bo = np.array([RNG.choice(d, len(d)).mean() for _ in range(10000)])
    sd = d.std(ddof=1)
    return dict(mean_delta=float(d.mean()),
                ci_95=[float(np.percentile(bo, 2.5)), float(np.percentile(bo, 97.5))],
                wins=int((d > 0).sum()), ties=int((d == 0).sum()), losses=int((d < 0).sum()),
                # efecto mínimo detectable al 80 % de potencia, test pareado
                mde_80=float(2.8 * sd / np.sqrt(len(d))) if sd > 0 else 0.0,
                n=len(d))


res = {t: evaluate(t) for t in map(tag_of, BAGS)}
base = tag_of("0")
KEYS = ["auc", "bacc", "f1", "kappa"]

print(f"\n{'config':<14}{'n':>4}" + "".join(f"{k:>10}" for k in KEYS))
print("-" * 58)
for b in BAGS:
    t = tag_of(b)
    if not res[t]:
        print(f"{t:<14}{0:>4}   (sin datos)")
        continue
    m = {k: np.mean([v[k] for v in res[t].values()]) for k in KEYS}
    mark = "  <- baseline" if t == base else ""
    print(f"{t:<14}{len(res[t]):>4}" + "".join(f"{m[k]:>10.4f}" for k in KEYS) + mark)

print(f"\n{'='*78}\nComparación pareada contra bolsa completa ({base})\n{'='*78}")
comps = {}
for b in BAGS:
    t = tag_of(b)
    if t == base or not res[t]:
        continue
    comps[t] = {}
    print(f"\n{t}:")
    for k in KEYS:
        c = paired(res[t], res[base], k)
        if c is None:
            continue
        comps[t][k] = c
        sig = "SÍ" if (c["ci_95"][0] > 0 or c["ci_95"][1] < 0) else "no"
        print(f"  {k:<6} Δ={c['mean_delta']:+.4f}  "
              f"IC95%[{c['ci_95'][0]:+.4f},{c['ci_95'][1]:+.4f}]  "
              f"G/E/P={c['wins']}/{c['ties']}/{c['losses']}  "
              f"MDE={c['mde_80']:.4f}  signif={sig}")

json.dump(dict(dataset=DS, task=TASK, models=MODELS, bags=BAGS, meta_model="logreg",
               baseline=base,
               nota="baseline reentrenado en la misma pasada (regla §6.5). "
                    "El muestreo solo actúa en entrenamiento; meta-features y "
                    "test usan la bolsa completa.",
               medias={t: {k: float(np.mean([v[k] for v in r.values()])) for k in KEYS}
                       for t, r in res.items() if r},
               n_folds={t: len(r) for t, r in res.items()},
               comparaciones=comps),
          open(OUT, "w"), indent=2, ensure_ascii=False)
print(f"\n✓ guardado en {OUT}")
PYEOF

echo
echo "Fin: $(date)"
echo "============================================================================"
