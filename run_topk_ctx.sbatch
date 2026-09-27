#!/bin/bash
#SBATCH --job-name=topk_ctx
#SBATCH --output=logs/topk_ctx_%j.out
#SBATCH --error=logs/topk_ctx_%j.err
#SBATCH --time=12:00:00
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --partition=main

# ============================================================================
# Top-K de atención + contexto espacial 3×3 (average pooling tipo CNN)
# ============================================================================
# Por qué: P7 (top-10 duro) dio +0.0388 AUC en cptac_gbm y nulo acotado en
# cptac_brca; P6 mostró que la vecindad espacial lleva señal (GNN real vs grafo
# barajado +0.0119) pero que pagarla con 1.05 M de parámetros se come la
# ganancia. Aquí la vecindad entra SIN parámetros: cada uno de los K centros de
# P7 se sustituye por la media uniforme de su ventana 3×3 (él + vecinos Moore-8
# de la retícula de coords), y los centros se combinan con su atención
# renormalizada, igual que en P7. Sin reentrenar: sólo cambia la extracción.
#
# Hipótesis PRE-REGISTRADA (un solo punto, sin barrido): K=10, ventana 3×3,
# vecinos con peso uniforme. Contraste primario: topk10_ctx vs topk10. Sólo se
# atribuye a la vecindad si además gana a sus dos controles:
#   topk10_shuffle — mismos centros y mismo nº de vecinos POR CENTRO, pero
#                    sorteados al azar en la slide: descarta "promediar más
#                    parches cualesquiera". No es el grafo barajado de P6:
#                    ese sólo conserva el grado global, y los top-10 están
#                    en bordes de tejido (4.8 vecinos/centro vs 7.7 de media
#                    en la prueba de humo, job 72686), así que promediaba
#                    ~60 % más parches que ctx.
#   topk10_match   — top-M por atención, M = tamaño de la región de ctx en esa
#                    slide: descarta "simplemente un K mayor" (P7/P8 ya
#                    muestran que K importa).
# Diagnóstico previo guardado por slide (ctx_stats.npy): fracción de vecinos de
# los centros que ya son centros. Si es alta, los top-10 forman una mancha
# contigua y lo esperable es ctx ≈ topk10.
#
# Expectativa por P7/P8: efecto, si lo hay, en cptac_gbm; nulo en cptac_brca.
# Se lanzan juntos para que cada uno quede pareado internamente.
#
# Coords: no todos los datasets tienen retícula utilizable (ver
# run_coords_lattice_survey.sbatch). test_abmil.py --topk_ctx falla alto si los
# coords no sirven, y aquí ese dataset se OMITE entero (no se mezcla con otro
# modo). Todo, incluido topk10 y el baseline, se recalcula en esta pasada.
#
# Uso:
#   sbatch run_topk_ctx.sbatch
#   DATASETS="cptac_gbm" K=10 sbatch run_topk_ctx.sbatch
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
CTX_MODES="ctx shuffle match"

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

    # PASO 1: contexto primero — si los coords no sirven, se omite el dataset
    # antes de gastar nada más en él.
    echo "PASO 1/3: embeddings topk${K}_{${CTX_MODES// /,}}..."
    for MODE in $CTX_MODES; do
        for MODEL in $MODELS; do
            echo "  • $MODEL [$MODE]"
            if ! run_test_abmil "$DATASET" "$MODEL" "$LOG_DIR/topk${K}_${MODE}_${MODEL}.log" \
                    --topk "$K" --topk_ctx "$MODE"; then
                echo "  ❌ $MODEL [$MODE] falló (¿coords no utilizables?)"; OK=0; break 2
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
    for FEATS in insample "topk${K}" "topk${K}_ctx" "topk${K}_shuffle" "topk${K}_match"; do
        echo "  • $FEATS:logreg"
        run_meta "$DATASET" "$FEATS" "$LOG_DIR/${FEATS}_logreg_ctxrun.log" \
            || { echo "  ❌ $FEATS:logreg falló"; FAILED+=("$DATASET ($FEATS:logreg)"); }
    done
    RAN_DATASETS+=("$DATASET")
done

echo ""
echo "╔══ COMPARATIVA PAREADA + DIAGNÓSTICO DE SOLAPAMIENTO"

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
SUB = {"insample": "ensemble4", T: f"ensemble4_{T}", f"{T}_ctx": f"ensemble4_{T}_ctx",
       f"{T}_shuffle": f"ensemble4_{T}_shuffle", f"{T}_match": f"ensemble4_{T}_match"}
# Contrastes primarios (Holm dentro de cada métrica sobre estos 3) y
# secundario (vs baseline insample, para situarlo frente a P7).
PRIMARY = [(f"{T}_ctx", T), (f"{T}_ctx", f"{T}_shuffle"), (f"{T}_ctx", f"{T}_match")]
SECONDARY = [(T, "insample"), (f"{T}_ctx", "insample")]
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
    for fam, pairs in (("PRIMARIO (Holm sobre 3)", PRIMARY), ("SECUNDARIO (Holm sobre 2)", SECONDARY)):
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
        for mode in ("ctx", "shuffle"):
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

out = Path(f"results_topk_ctx_{TASK}.json")
json.dump(summary, open(out, "w"), indent=2)
print(f"\nResultados → {out}")
PYEOF

echo ""
echo "Datasets completados: ${RAN_DATASETS[*]:-ninguno}"
if [ ${#FAILED[@]} -gt 0 ]; then printf '  ❌ %s\n' "${FAILED[@]}"; fi
echo "Fin: $(date)"
