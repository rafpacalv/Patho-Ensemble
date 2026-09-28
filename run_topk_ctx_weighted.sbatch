#!/bin/bash
#SBATCH --job-name=topk_ctx_w
#SBATCH --output=logs/topk_ctx_w_%j.out
#SBATCH --error=logs/topk_ctx_w_%j.err
#SBATCH --time=12:00:00
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --partition=main

# ============================================================================
# Top-K + contexto 3×3 con MÁS PESO AL CENTRO: curva de λ y filtro bilateral
# ============================================================================
# Por qué: job 72690 midió la media 3×3 UNIFORME (el centro pesa 1/9) y salió
# nula en cptac_brca (Δ −0.002 vs topk10) y dañina en cptac_gbm (Δ −0.039 ✱),
# con `match` (mismos parches, ponderados por atención) conservando la
# ganancia de P7: el daño es diluir los centros, no usar más parches.
# Pregunta ahora: ¿existe un peso del centro intermedio que mejore a topk10?
#
#   Curva λ (ctx_c = λ·h_c + (1−λ)·media vecinos), de ~1/9 a 1:
#     topk10_ctx (uniforme, ≈λ 0.11–0.17 según nº de vecinos) · lam050 ·
#     lam075 · lam090 · topk10 (λ=1). Expectativa previa (baja): máximo en
#     λ=1, es decir, curva creciente → el entorno no aporta nada distinto.
#     Si hay un pico interior por encima de topk10, λ se habrá elegido
#     mirando test: sería HIPÓTESIS para una réplica con λ fijado, no un
#     resultado.
#   topk10_sim — cada vecino con peso max(0, corr(h_j, h_c)) (coseno tras
#     centrar por la media de la slide), centro con peso 1. Sin parámetros.
#     Motivo: los top-10 están en bordes de tejido (~5 vecinos vs 7.7), y
#     sólo deberían sumar los vecinos del mismo lado del borde.
#     Contraste primario pre-registrado: topk10_sim vs topk10.
#
# Todo se recalcula en esta pasada (baseline, topk10 y ctx incluidos).
#
# Uso:
#   sbatch run_topk_ctx_weighted.sbatch
#   DATASETS="cptac_gbm" sbatch run_topk_ctx_weighted.sbatch
# ============================================================================

set -uo pipefail   # sin -e: un dataset que falle no debe abortar el resto

source $(conda info --base)/etc/profile.d/conda.sh
conda activate ensemble

cd /home/JKP6679/Patho-Ensemble

WORK_DIR="/home/JKP6679/Patho-Ensemble/PARADIS/datos/patches"
TISSUE_PATCHING="20x_224px_0px_overlap"
MODELS="ctranspath uni_v2 conch_v1_5"
K="${K:-10}"
DATASETS="${DATASETS:-cptac_brca cptac_gbm}"
TASK_NAME="${TASK_NAME:-TP53_mutation}"
# "modo:λ" — λ vacío = sin --ctx_centre_weight
CTX_VARIANTS="ctx: ctx:0.5 ctx:0.75 ctx:0.9 sim:"

count_files() {
    local d="$1"
    if [ -d "$d" ]; then find "$d" -maxdepth 1 -name 'fold_*' -type d | wc -l; else echo 0; fi
}

run_test_abmil() {   # $1=dataset $2=model $3=log  resto=flags extra
    local ds="$1" model="$2" log="$3"; shift 3
    python src/test_abmil.py \
        --foundational_model "$model" --work_dir "$WORK_DIR" \
        --train_source "$ds" --tissue_patching "$TISSUE_PATCHING" \
        --task_name "$TASK_NAME" --epochs 100 "$@" 2>&1 | tee -a "$log"
    return "${PIPESTATUS[0]}"
}

run_meta() {         # $1=dataset $2=meta_features $3=log
    python src/ensemble4.py \
        --foundational_models $MODELS --work_dir "$WORK_DIR" \
        --train_source "$1" --tissue_patching "$TISSUE_PATCHING" \
        --task_name "$TASK_NAME" --meta_model logreg --meta_features "$2" \
        2>&1 | tee -a "$3"
    return "${PIPESTATUS[0]}"
}

echo "============================================================================"
echo "TOP-${K} + CONTEXTO 3×3 — $(date)"
echo "Modelos: $MODELS | Datasets: $DATASETS | Task: $TASK_NAME"
echo "============================================================================"

FAILED=()
RAN_DATASETS=()

for DATASET in $DATASETS; do
    ABMIL_DIR="$WORK_DIR/$DATASET/$TASK_NAME/abmil"
    LOG_DIR="logs/${DATASET}_${TASK_NAME}"
    mkdir -p "$LOG_DIR"
    echo ""
    echo "╔══ $DATASET / $TASK_NAME"

    OK=1
    for MODEL in $MODELS; do
        if [ "$(count_files "$ABMIL_DIR/${MODEL}_${TISSUE_PATCHING}/checkpoints")" -eq 0 ]; then
            echo "❌ falta $MODEL: sin checkpoints"; OK=0
        fi
    done
    [ "$OK" -eq 1 ] || { FAILED+=("$DATASET (sin checkpoints)"); continue; }

    # PASO 1: variantes con contexto primero — si los coords no sirven, se
    # omite el dataset antes de gastar nada más en él.
    echo "PASO 1/3: embeddings con contexto (${CTX_VARIANTS})..."
    for VAR in $CTX_VARIANTS; do
        MODE="${VAR%%:*}"; LAM="${VAR#*:}"
        EXTRA=(--topk "$K" --topk_ctx "$MODE")
        [ -n "$LAM" ] && EXTRA+=(--ctx_centre_weight "$LAM")
        for MODEL in $MODELS; do
            echo "  • $MODEL [$VAR]"
            if ! run_test_abmil "$DATASET" "$MODEL" "$LOG_DIR/topk${K}_${MODE}${LAM}_${MODEL}.log" \
                    "${EXTRA[@]}"; then
                echo "  ❌ $MODEL [$VAR] falló (¿coords no utilizables?)"; OK=0; break 2
            fi
        done
    done
    [ "$OK" -eq 1 ] || { FAILED+=("$DATASET (coords/contexto)"); continue; }

    echo "PASO 2/3: topk${K} sin contexto (regenerado ahora)..."
    for MODEL in $MODELS; do
        run_test_abmil "$DATASET" "$MODEL" "$LOG_DIR/topk${K}_${MODEL}.log" --topk "$K" \
            || { echo "  ❌ $MODEL [topk${K}] falló"; OK=0; break; }
    done
    [ "$OK" -eq 1 ] || { FAILED+=("$DATASET (topk${K})"); continue; }

    echo "PASO 3/3: meta-learner logreg sobre cada espacio + baseline insample..."
    for FEATS in insample "topk${K}" "topk${K}_ctx" "topk${K}_lam050" "topk${K}_lam075" \
                 "topk${K}_lam090" "topk${K}_sim"; do
        echo "  • $FEATS:logreg"
        run_meta "$DATASET" "$FEATS" "$LOG_DIR/${FEATS}_logreg_ctxwrun.log" \
            || { echo "  ❌ $FEATS:logreg falló"; FAILED+=("$DATASET ($FEATS:logreg)"); }
    done
    RAN_DATASETS+=("$DATASET")
done

echo ""
echo "╔══ COMPARATIVA PAREADA: CURVA λ + FILTRO BILATERAL"

WORK_DIR="$WORK_DIR" TASK_NAME="$TASK_NAME" K="$K" MODELS="$MODELS" \
TISSUE_PATCHING="$TISSUE_PATCHING" RAN_DATASETS="${RAN_DATASETS[*]:-}" python3 << 'PYEOF'
import json, os
from pathlib import Path
import numpy as np
from scipy.stats import ttest_rel, wilcoxon

W = Path(os.environ["WORK_DIR"]); TASK = os.environ["TASK_NAME"]; K = os.environ["K"]
MODELS = os.environ["MODELS"].split(); TP = os.environ["TISSUE_PATCHING"]
RAN = os.environ.get("RAN_DATASETS", "").split()
T = f"topk{K}"
LAMS = ["ctx", "lam050", "lam075", "lam090"]
SUB = {"insample": "ensemble4", T: f"ensemble4_{T}", f"{T}_sim": f"ensemble4_{T}_sim",
       **{f"{T}_{l}": f"ensemble4_{T}_{l}" for l in LAMS}}
# Familia 1 (pre-registrada): sim vs topk10 y sim vs ctx uniforme.
# Familia 2 (curva λ, descriptiva): cada punto vs topk10 (λ=1), Holm sobre 4.
PRIMARY = [(f"{T}_sim", T), (f"{T}_sim", f"{T}_ctx")]
SECONDARY = [(f"{T}_{l}", T) for l in LAMS]
METRICS = [("macro-ovr-auc", "AUC"), ("bacc", "bacc"), ("weighted_kappa", "kappa"),
           ("macro-f1", "macro-F1")]


def holm(p):
    p = np.asarray(p); m = len(p); o = np.argsort(p); adj = np.empty(m); run = 0.0
    for r, i in enumerate(o):
        run = max(run, (m - r) * p[i]); adj[i] = min(run, 1.0)
    return adj


def per_fold(d):
    out = {}
    for fd in sorted(d.glob("fold_*"), key=lambda p: int(p.name.split("_")[1])):
        try:
            m = json.load(open(fd / "metrics.json"))["overall"]
        except Exception:
            continue
        for k, _ in METRICS:
            out.setdefault(k, []).append(m[k])
    return {k: np.asarray(v) for k, v in out.items()}


def contrast(a, b):
    d = a - b; n = len(d); sd = d.std(ddof=1)
    ci = 1.96 * sd / np.sqrt(n) if sd > 0 else 0.0
    mde = 2.802 * sd / np.sqrt(n) if sd > 0 else 0.0
    p = ttest_rel(a, b).pvalue if sd > 0 else 1.0
    try:
        pw = wilcoxon(a, b).pvalue if sd > 0 else 1.0
    except ValueError:
        pw = 1.0
    return d.mean(), ci, mde, p, pw, int((d > 0).sum()), int((d == 0).sum()), int((d < 0).sum())


summary = {}
for ds in RAN:
    ab = W / ds / TASK / "abmil"
    pf = {n: per_fold(ab / s / "test_metrics") for n, s in SUB.items()
          if (ab / s / "test_metrics").exists()}
    print(f"\n{'=' * 104}\n{ds}/{TASK} — configs: {', '.join(pf)}")
    for n in SUB:
        if n in pf and "macro-ovr-auc" in pf[n]:
            print(f"  {n:<20} AUC {pf[n]['macro-ovr-auc'].mean():.4f}  "
                  f"({len(pf[n]['macro-ovr-auc'])} folds)")
    summary[ds] = {}
    print("\n  CURVA λ (AUC medio):  " + "  ".join(
        f"{lab} {pf[n]['macro-ovr-auc'].mean():.4f}"
        for lab, n in [("uniforme", f"{T}_ctx"), ("λ0.50", f"{T}_lam050"),
                       ("λ0.75", f"{T}_lam075"), ("λ0.90", f"{T}_lam090"), ("λ1=topk", T)]
        if n in pf and "macro-ovr-auc" in pf[n]))
    for fam, pairs in (("SIMILITUD, pre-registrado (Holm sobre 2)", PRIMARY),
                       ("CURVA λ vs topk10 (Holm sobre 4)", SECONDARY)):
        print(f"\n  {fam}")
        for key, lab in METRICS:
            rows = []
            for a, b in pairs:
                if a in pf and b in pf and key in pf[a] and key in pf[b] \
                        and len(pf[a][key]) == len(pf[b][key]):
                    rows.append((a, b, *contrast(pf[a][key], pf[b][key])))
            if not rows:
                continue
            adj = holm([r[5] for r in rows])
            for r, pa in zip(rows, adj):
                a, b, dm, ci, mde, p, pw, g, e, l = r
                star = " *" if pa < 0.05 else ("  ~" if p < 0.05 else "")
                print(f"    {lab:<9}{a + ' vs ' + b:<34}Δ {dm:+.4f} "
                      f"[{dm - ci:+.4f},{dm + ci:+.4f}]  W/T/L {g}/{e}/{l}  "
                      f"p {p:.3f}  pW {pw:.3f}  Holm {pa:.3f}  MDE {mde:.4f}{star}")
                summary[ds][f"{lab}|{a}|{b}"] = dict(delta=dm, ci=[dm - ci, dm + ci], p=p,
                                                     p_wilcoxon=pw, p_holm=float(pa), mde=mde,
                                                     wtl=[g, e, l], family=fam)

    # Diagnóstico: ctx_stats.npy = [M, frac vecinos que ya son centros, vecinos/centro]
    print(f"\n  DIAGNÓSTICO 3×3 (test, mediana sobre slides y folds)")
    for m in MODELS:
        for mode in ("ctx",):
            files = sorted((ab / f"{m}_{TP}" / f"test_outputs_{T}_{mode}").glob("fold_*/ctx_stats.npy"))
            if not files:
                continue
            st = np.vstack([np.load(f) for f in files])
            print(f"    {m:<12}[{mode:<7}] M={np.median(st[:, 0]):6.0f}  "
                  f"vecinos ya en top-{K}: {100 * np.nanmedian(st[:, 1]):5.1f} %  "
                  f"vecinos/centro: {np.median(st[:, 2]):.2f}")
            summary[ds][f"diag|{m}|{mode}"] = dict(
                M_median=float(np.median(st[:, 0])),
                frac_neigh_in_topk_median=float(np.nanmedian(st[:, 1])),
                neigh_per_centre_median=float(np.median(st[:, 2])))

    print("\n  * = significativo tras Holm; ~ = p<0.05 sin corregir. MDE al 80 % de potencia.")

out = Path(f"results_topk_ctx_weighted_{TASK}.json")
json.dump(summary, open(out, "w"), indent=2)
print(f"\nResultados → {out}")
PYEOF

echo ""
echo "Datasets completados: ${RAN_DATASETS[*]:-ninguno}"
if [ ${#FAILED[@]} -gt 0 ]; then printf '  ❌ %s\n' "${FAILED[@]}"; fi
echo "Fin: $(date)"
