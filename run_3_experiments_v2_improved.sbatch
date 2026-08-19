#!/bin/bash
#SBATCH --job-name=run_meta_experiments
#SBATCH --partition=main
#SBATCH --constraint=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=logs/run_meta_experiments_%j.out
#SBATCH --error=logs/run_meta_experiments_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=rafael.pachon.alvarez@gmail.com

# ============================================================================
# Configuraciones de meta-learner (stacking sobre los modelos base)
# ============================================================================
# Las configuraciones se definen en experiments.yaml (meta_configs). Por defecto:
#
#   exp0_baseline    logreg + bases estándar   -> ensemble4/
#   exp1_tabpfn_std  TabPFN + bases estándar   -> ensemble4_tabpfn/
#   exp2_tabpfn_fge  TabPFN + bases FGE        -> ensemble4_tabpfn_fge/
#   exp3_logreg_fge  logreg + bases FGE        -> ensemble4_fge/
#
#   DATASET=cptac_brca TASK=TP53_mutation sbatch run_3_experiments_v2_improved.sbatch
#
# Requiere: train_base_models.sbatch (y train_base_models_fge_v2.sbatch para fge).
# Duración estimada: 3-4 h.
# ============================================================================

set -uo pipefail   # sin -e: un experimento que falle no debe abortar el resto

source $(conda info --base)/etc/profile.d/conda.sh
conda activate ensemble

PROJECT_DIR="/home/JKP6679/Patho-Ensemble"
cd "$PROJECT_DIR"

DATASET="${DATASET:-}"
TASK="${TASK:-}"
if [ -z "$DATASET" ] || [ -z "$TASK" ]; then
    read -r DATASET TASK < <(python3 src/experiment_config.py --list-active | head -1)
fi
eval "$(python3 src/experiment_config.py --emit-shell --dataset "$DATASET" --task "$TASK")"

LOG_DIR="$PROJECT_DIR/logs/${DATASET}_${TASK}"
mkdir -p "$LOG_DIR"

ABMIL_DIR="$WORK_DIR/$DATASET/$TASK/abmil"

# ---------------------------------------------------------------------------
# Comprobaciones previas
# ---------------------------------------------------------------------------
echo "============================================================================"
echo "COMPROBACIONES PREVIAS — $DATASET / $TASK"
echo "============================================================================"

STANDARD_OK=1

for MODEL in $MODELS; do
    B="$ABMIL_DIR/${MODEL}_${TISSUE_PATCHING}"
    [ -d "${B}_train_eval/val_outputs/fold_0" ] || { echo "❌ falta ${MODEL}: _train_eval/val_outputs/fold_0"; STANDARD_OK=0; }
done

[ "$STANDARD_OK" -eq 1 ] && echo "✓ bases estándar disponibles"

if [ "$STANDARD_OK" -eq 0 ]; then
    echo ""
    echo "❌ Faltan las bases estándar. Ejecuta primero:"
    echo "   DATASET=$DATASET TASK=$TASK sbatch train_base_models.sbatch"
    exit 1
fi

# Features out-of-fold: las genera build_oof_features.py, no train_base_models.
# Puede haber varias etiquetas conviviendo (oof, oof5, oof10 = distintos n_inner),
# así que se comprueba cada una por separado.
MISSING_FEATS=""
CHECKED_FEATS=""
for CFG in $META_CONFIGS; do
    IFS=':' read -r _ _ _ FEATS _ _ <<< "$CFG"
    [ "$FEATS" = "insample" ] && continue
    case " $CHECKED_FEATS " in *" $FEATS "*) continue ;; esac
    CHECKED_FEATS="$CHECKED_FEATS $FEATS"
    for MODEL in $MODELS; do
        if [ ! -d "$ABMIL_DIR/${MODEL}_${TISSUE_PATCHING}_train_eval_${FEATS}/val_outputs_${FEATS}/fold_0" ]; then
            echo "⚠ falta ${MODEL}: _train_eval_${FEATS}/val_outputs_${FEATS}/fold_0"
            MISSING_FEATS="$MISSING_FEATS $FEATS"
            break
        fi
    done
    case " $MISSING_FEATS " in
        *" $FEATS "*) echo "  genéralas con: N_INNER=<k> OOF_TAG=$FEATS sbatch build_oof_features.sbatch" ;;
        *) echo "✓ features '$FEATS' disponibles" ;;
    esac
done

# Bases no estándar: cada base_source es una etiqueta de variante FGE (fge_orig,
# fge_wbase, ...) y tiene su propio árbol {..}_train_eval_<v>/val_outputs_<v>/.
# Comprobar el literal "fge" no vale: ninguna variante se llama así.
MISSING_BASES=""
CHECKED_BASES=""
for CFG in $META_CONFIGS; do
    IFS=':' read -r _ _ BASE _ _ _ <<< "$CFG"
    [ "$BASE" = "standard" ] && continue
    case " $CHECKED_BASES " in *" $BASE "*) continue ;; esac
    CHECKED_BASES="$CHECKED_BASES $BASE"
    for MODEL in $MODELS; do
        if [ ! -d "$ABMIL_DIR/${MODEL}_${TISSUE_PATCHING}_train_eval_${BASE}/val_outputs_${BASE}/fold_0" ]; then
            echo "⚠ falta ${MODEL}: _train_eval_${BASE}/val_outputs_${BASE}/fold_0"
            MISSING_BASES="$MISSING_BASES $BASE"
            break
        fi
    done
    case " $MISSING_BASES " in *" $BASE "*) ;; *) echo "✓ bases $BASE disponibles" ;; esac
done

echo ""
echo "============================================================================"
echo "META-LEARNERS — $DATASET / $TASK"
echo "============================================================================"
echo "Inicio:  $(date)"
echo "modelos: $MODELS"
echo "configs: $META_CONFIGS"
echo "============================================================================"

TOTAL_START=$(date +%s)
FAILED=()
RAN=()

# ---------------------------------------------------------------------------
# Ejecutar cada configuración de meta-learner
# ---------------------------------------------------------------------------
for CFG in $META_CONFIGS; do
    IFS=':' read -r NAME META BASE FEATS SPACE DROPC REWEIGHT <<< "$CFG"
    REWEIGHT="${REWEIGHT:-0}"  # default a 0 si la fila trae menos campos (retrocompatibilidad)

    echo ""
    echo "╔════════════════════════════════════════════════════════════════════════╗"
    echo "║ $NAME — meta=$META, bases=$BASE, features=$FEATS, espacio=$SPACE"
    echo "╚════════════════════════════════════════════════════════════════════════╝"

    case " $MISSING_BASES " in
        *" $BASE "*)
            echo "⚠ Se omite: no hay bases '$BASE'."
            FAILED+=("$NAME (sin bases $BASE)")
            continue
            ;;
    esac

    case " $MISSING_FEATS " in
        *" $FEATS "*)
            echo "⚠ Se omite: no hay features '$FEATS'."
            FAILED+=("$NAME (sin features $FEATS)")
            continue
            ;;
    esac

    DROP_FLAG=""
    [ "$DROPC" = "1" ] && DROP_FLAG="--drop_redundant_class"

    REWEIGHT_FLAG=""
    [ "$REWEIGHT" = "1" ] && REWEIGHT_FLAG="--reweight_by_model_importance"

    START=$(date +%s)
    if python src/ensemble4.py \
        --foundational_models $MODELS \
        --work_dir "$WORK_DIR" \
        --train_source "$DATASET" \
        --tissue_patching "$TISSUE_PATCHING" \
        --task_name "$TASK" \
        --meta_model "$META" \
        --base_source "$BASE" \
        --meta_features "$FEATS" \
        --feature_space "$SPACE" \
        $DROP_FLAG \
        $REWEIGHT_FLAG \
        2>&1 | tee -a "$LOG_DIR/${NAME}.log"; then

        DUR=$(( $(date +%s) - START ))
        echo "[$(date)] ✓ $NAME completado ($((DUR/60)) min $((DUR%60)) s)"
        RAN+=("$NAME")
    else
        echo "[$(date)] ❌ $NAME FALLÓ"
        FAILED+=("$NAME")
    fi
done

# ---------------------------------------------------------------------------
# Tabla comparativa
# ---------------------------------------------------------------------------
echo ""
echo "╔════════════════════════════════════════════════════════════════════════╗"
echo "║ COMPARATIVA DE RESULTADOS"
echo "╚════════════════════════════════════════════════════════════════════════╝"

ABMIL_DIR="$ABMIL_DIR" META_CONFIGS="$META_CONFIGS" RAN="${RAN[*]:-}" \
BASELINE="${BASELINE:-}" DATASET="$DATASET" TASK="$TASK" python3 << 'PYEOF'
import json
import os
from pathlib import Path

import numpy as np
from scipy.stats import ttest_rel

ABMIL_DIR = Path(os.environ["ABMIL_DIR"])
DATASET = os.environ["DATASET"]
TASK = os.environ["TASK"]

# Mismo mapeo que results_subdir_map en ensemble4.py. Si se añade un
# meta-modelo allí hay que añadirlo aquí: leer el directorio equivocado produce
# una tabla plausible y falsa, que es peor que un error.
SUBDIR = {
    "logreg": "ensemble4",
    "mlp": "ensemble4_mlp",
    "mlp_snapshot": "ensemble4_mlp_snapshot",
    "mlp_fge": "ensemble4_mlp_fge",
    "tabpfn": "ensemble4_tabpfn",
    "logit_avg": "ensemble4_logit_avg",
    "mlp_deepens": "ensemble4_mlp_deepens",
    "gating": "ensemble4_gating",
    "lightgbm": "ensemble4_lightgbm",
    "svm": "ensemble4_svm",
    "knn": "ensemble4_knn",
    "nb": "ensemble4_nb",
}

METRICS = [("macro-ovr-auc", "AUC-ROC"), ("weighted_kappa", "Kappa"),
           ("macro-f1", "Macro-F1"), ("bacc", "Balanced acc"), ("acc", "Accuracy")]


def subdir_for(meta, base, feats, space, dropc, reweight="0"):
    """Reproduce la construcción del subdirectorio que hace ensemble4.py."""
    sub = SUBDIR.get(meta, f"ensemble4_{meta}")
    if base != "standard":
        sub += f"_{base}"
    if feats != "insample":
        sub += f"_{feats}"
    if space != "prob":
        sub += f"_{space}"
    if dropc == "1":
        sub += "_dropc"
    if reweight == "1":
        sub += "_reweighted"
    return sub


configs = []
for cfg in os.environ["META_CONFIGS"].split():
    parts = cfg.split(":")
    # Filas anteriores a la ampliación del formato traen sólo 3 campos.
    name, meta, base = parts[:3]
    feats = parts[3] if len(parts) > 3 else "insample"
    space = parts[4] if len(parts) > 4 else "prob"
    dropc = parts[5] if len(parts) > 5 else "0"
    reweight = parts[6] if len(parts) > 6 else "0"
    configs.append((name, subdir_for(meta, base, feats, space, dropc, reweight)))

# Sólo se leen resultados de configuraciones que han corrido en ESTA ejecución.
# Un test_metrics_summary.json antiguo se leería como si fuera de ahora.
RAN = set(os.environ.get("RAN", "").split())

summary, per_fold = {}, {}
for name, sub in configs:
    if name not in RAN:
        summary[name] = {"__error__": "no completado en esta ejecución"}
        continue
    f = ABMIL_DIR / sub / "test_metrics_summary.json"
    try:
        summary[name] = json.load(open(f))
    except Exception as e:
        summary[name] = {"__error__": str(e)}
        continue

    # Métricas por fold: son las que permiten el contraste pareado. Sin ellas
    # sólo se pueden comparar IC independientes, que es un test mucho más débil
    # cuando todas las configuraciones se evalúan sobre los mismos folds.
    folds = sorted((ABMIL_DIR / sub / "test_metrics").glob("fold_*"),
                   key=lambda p: int(p.name.split("_")[1]))
    vals = {}
    for fd in folds:
        try:
            m = json.load(open(fd / "metrics.json"))["overall"]
        except Exception:
            continue
        for k, _ in METRICS:
            vals.setdefault(k, []).append(m[k])
    per_fold[name] = {k: np.asarray(v) for k, v in vals.items()}

ok = [n for n, r in summary.items() if "__error__" not in r]
baseline = os.environ.get("BASELINE") or (configs[0][0] if configs else None)
if baseline not in ok and ok:
    baseline = ok[0]

n_folds = len(per_fold.get(baseline, {}).get("macro-ovr-auc", []))

print(f"\n{'=' * 100}")
print(f"{DATASET} / {TASK}   —   {len(ok)}/{len(configs)} configuraciones, "
      f"{n_folds} folds   —   baseline: {baseline}")
print(f"{'=' * 100}")


def holm(pvals):
    """Corrección de Holm-Bonferroni; devuelve los p ajustados en el orden dado.

    Se aplica por familia (una por métrica): con ~19 configuraciones el número
    de contrastes hace esperable algún p<0.05 por azar, y ya nos costó un falso
    positivo en cervical_subtype.
    """
    m = len(pvals)
    order = np.argsort(pvals)
    adj = np.empty(m)
    running = 0.0
    for rank, idx in enumerate(order):
        running = max(running, (m - rank) * pvals[idx])
        adj[idx] = min(running, 1.0)
    return adj


for key, label in METRICS:
    base_v = per_fold.get(baseline, {}).get(key)
    if base_v is None or len(base_v) == 0:
        continue

    rows, pvals = [], []
    for name, _ in configs:
        if name == baseline or name not in per_fold:
            continue
        v = per_fold[name].get(key)
        if v is None or len(v) != len(base_v):
            continue
        d = v - base_v
        sd = d.std(ddof=1)
        n = len(d)
        ci = 1.96 * sd / np.sqrt(n) if sd > 0 else 0.0
        # Efecto mínimo detectable al 80 % de potencia: convierte un resultado
        # no significativo en una cota superior del efecto, en vez de dejarlo
        # indeterminado.
        mde = (1.96 + 0.842) * sd / np.sqrt(n) if sd > 0 else 0.0
        p = ttest_rel(v, base_v).pvalue if sd > 0 else 1.0
        rows.append([name, v.mean(), d.mean(), ci, mde,
                     int((d > 0).sum()), int((d == 0).sum()), int((d < 0).sum())])
        pvals.append(p)

    adj = holm(np.asarray(pvals)) if pvals else []

    print(f"\n{label}   —   baseline = {base_v.mean():.4f}")
    print("-" * 100)
    print(f"{'Configuración':<24}{'Media':>9}{'Δ':>10}{'IC 95% pareado':>22}"
          f"{'G/E/P':>12}{'p':>8}{'p Holm':>9}{'MDE':>8}")
    print("-" * 100)
    for (name, mean, dm, ci, mde, g, e, l), p, pa in zip(rows, pvals, adj):
        star = " *" if pa < 0.05 else ("  ~" if p < 0.05 else "")
        print(f"{name:<24}{mean:9.4f}{dm:+10.4f}"
              f"{f'[{dm - ci:+.4f}, {dm + ci:+.4f}]':>22}"
              f"{f'{g}/{e}/{l}':>12}{p:8.3f}{pa:9.3f}{mde:8.4f}{star}")

    print("-" * 100)
    print("  G/E/P = folds ganados/empatados/perdidos frente al baseline. "
          "Los empates importan:")
    print("  con pocas muestras de test por fold muchas configuraciones producen "
          "la misma matriz de confusión.")
    print("  * = significativo tras corrección de Holm.  ~ = significativo sin "
          "corregir (no concluyente).")
    print("  MDE = efecto mínimo detectable al 80 % de potencia: un Δ menor que "
          "esto no era detectable")
    print("  con este número de folds, así que 'no significativo' aquí significa "
          "'acotado', no 'sin datos'.")

# Resumen
print(f"\n{'=' * 100}")
print("RESUMEN")
print(f"{'=' * 100}\n")

fallidos = [n for n, r in summary.items() if "__error__" in r]
if fallidos:
    print(f"Sin resultado ({len(fallidos)}): {', '.join(fallidos)}\n")

key = "macro-ovr-auc"
base_v = per_fold.get(baseline, {}).get(key)
if base_v is not None and len(base_v):
    ranked = sorted(
        ((n, per_fold[n][key].mean()) for n, _ in configs
         if n in per_fold and key in per_fold[n]),
        key=lambda t: -t[1])
    print(f"AUC del baseline ({baseline}): {base_v.mean():.4f}\n")
    print("Ranking por AUC:")
    for i, (n, v) in enumerate(ranked[:8], 1):
        print(f"  {i}. {n:<24} {v:.4f}  ({v - base_v.mean():+.4f})")
    print()
    print("Una diferencia de medias sin significación tras Holm no es concluyente,")
    print("por muy limpio que parezca el ranking.")
PYEOF

# ---------------------------------------------------------------------------
# Resumen final
# ---------------------------------------------------------------------------
TOTAL_DUR=$(( $(date +%s) - TOTAL_START ))

echo ""
echo "============================================================================"
echo "RESUMEN DE EJECUCIÓN"
echo "============================================================================"
echo "Duración total: $((TOTAL_DUR/60)) min $((TOTAL_DUR%60)) s"
echo ""
echo "Completados (${#RAN[@]}):"
printf '  ✓ %s\n' "${RAN[@]:-(ninguno)}"

if [ ${#FAILED[@]} -gt 0 ]; then
    echo ""
    echo "Con incidencias (${#FAILED[@]}):"
    printf '  ❌ %s\n' "${FAILED[@]}"
fi

echo ""
echo "Resultados en: $ABMIL_DIR/ensemble4*/"
echo "Logs en:       $LOG_DIR/"
echo ""
echo "Fin: $(date)"
echo "============================================================================"
