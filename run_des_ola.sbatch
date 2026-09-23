#!/bin/bash
#SBATCH --job-name=des_ola
#SBATCH --partition=main
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=01:30:00
#SBATCH --output=logs/des_ola_%j.out
#SBATCH --error=logs/des_ola_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=rafael.pachon.alvarez@gmail.com

# ============================================================================
# DYNAMIC ENSEMBLE SELECTION (DES-OLA) vs logreg — 2026-09-23
# ============================================================================
# Nueva opción --meta_model des_ola en ensemble4.py (clase DESOLAMetaClassifier
# en src/meta_models.py): selección dinámica de ensembles clásica, no
# paramétrica (Ko, Sabourin & Britto 2008 — Overall Local Accuracy; ver
# también Cruz et al., META-DES, arXiv:1810.01270; librería deslib). Para cada
# instancia de consulta, estima la competencia de cada modelo base por
# accuracy local entre sus k vecinos más cercanos en el DSEL (aquí, el propio
# meta-train), y combina las predicciones base con esos pesos locales
# (mezcla convexa, nunca sale del envolvente convexo de las predicciones
# base). Distinto de --meta_model gating (mixture-of-experts paramétrico,
# entrenado por gradiente): des_ola es perezoso (k-NN), sin entrenamiento.
#
# Régimen OBLIGATORIO: --meta_features oof. Estimar competencia local sobre
# features in-sample (AUC~0.94) describiría una vecindad donde casi todos los
# modelos aciertan casi siempre — la estimación quedaría inflada y sin poder
# discriminar, el mismo fallo de stacking que build_oof_features.py ya
# resolvió para el ajuste de coeficientes (ver CLAUDE.md, "Meta-features:
# in-sample vs out-of-fold").
#
# Freshness check (2026-09-23, antes de lanzar este job): OOF existe para los
# 4 modelos necesarios (ctranspath, uni_v2, virchow_v1, conch_v1_5) bajo
# .../abmil/{model}_20x_224px_0px_overlap_train_eval_oof/val_outputs_oof/.
# Comparación fold-a-fold de mtime(OOF fold_k/preds.npy) vs
# mtime(checkpoints/fold_k/model.pt) sobre los 50 folds:
#   ctranspath  45/50 folds con OOF posterior al checkpoint, 5/50 anterior
#   uni_v2      50/50 OOF posterior
#   virchow_v1  50/50 OOF posterior
#   conch_v1_5  50/50 OOF posterior (calculado 2026-09-01, el más reciente)
# build_oof_features.py entrena sus propios modelos internos (CV anidada
# independiente); no depende de los checkpoints externos de train_abmil.py,
# así que esa comparación es una cota, no una prueba directa de staleness.
# Los inputs de los que SÍ depende (k=all.tsv, embeddings .h5) no se han
# tocado desde sept-2025, antes de toda esta campaña. Los 5/50 folds
# "invertidos" de ctranspath son consistentes con jobs paralelos
# (entrenamiento base + OOF corriendo a la vez), no con un reentrenamiento
# posterior que invalide el OOF. Ver el informe final para el detalle.
#
# Barrido, por trío (histórico y ganador, en la misma tanda, k=10 fijo — ver
# CLAUDE.md: k conservador porque el DSEL tiene ~75 filas/fold):
#   logreg   --meta_features oof                  (baseline, recalculado aquí)
#   des_ola  --meta_features oof --meta_des_k 10
# = 2 configs x 2 tríos = 4 runs de 50 folds. CPU-only.
#
# Análisis pareado (Holm-Bonferroni, aunque con una sola comparación por
# trío Holm no cambia el p) reutilizando analyze_feature_engineering_ladder_v2.py
# tal cual (agnóstico al significado del meta-modelo). Además, un bloque
# Python al final calcula wins/ties/losses y el MDE al 80% de potencia con la
# misma fórmula que src/run_meta_suite.py, porque el script de la escalera no
# los reporta y el protocolo de esta sesión los pide siempre.
# ============================================================================

set -uo pipefail   # sin -e: una config que falle no debe abortar el resto

source $(conda info --base)/etc/profile.d/conda.sh
conda activate ensemble

PROJECT_DIR="/home/JKP6679/Patho-Ensemble"
cd "$PROJECT_DIR"

DATASET="cptac_brca"
TASK="TP53_mutation"
WORK_DIR="/home/JKP6679/Patho-Ensemble/PARADIS/datos/patches"
TISSUE_PATCHING="20x_224px_0px_overlap"

MODELS_HIST="ctranspath uni_v2 virchow_v1"
MODELS_WIN="ctranspath uni_v2 conch_v1_5"

echo "============================================================================"
echo "DES-OLA (Dynamic Ensemble Selection) vs logreg — $DATASET / $TASK"
echo "Fecha: $(date)"
echo "============================================================================"

run_config() {
    local LABEL=$1
    local MODELS=$2
    local META_MODEL=$3
    local EXTRA_ARGS=$4
    local SUFFIX=$5

    echo "============================================================================"
    echo "CONFIG: $LABEL"
    echo "  Modelos: $MODELS | meta_model: $META_MODEL | extra: $EXTRA_ARGS | suffix: $SUFFIX"
    echo "  Inicio: $(date)"
    echo "============================================================================"

    CMD="python src/ensemble4.py"
    CMD="$CMD --foundational_models $MODELS"
    CMD="$CMD --work_dir $WORK_DIR"
    CMD="$CMD --train_source $DATASET"
    CMD="$CMD --tissue_patching $TISSUE_PATCHING"
    CMD="$CMD --task_name $TASK"
    CMD="$CMD --meta_model $META_MODEL"
    CMD="$CMD --results_subdir_suffix $SUFFIX"
    CMD="$CMD $EXTRA_ARGS"

    if eval "$CMD"; then
        echo "✓ $LABEL completado"
        echo ""
    else
        echo "❌ $LABEL falló con código $?"
        echo ""
        return 1
    fi
}

for TRIO_TAG in hist win; do
    if [ "$TRIO_TAG" = "hist" ]; then
        MODELS="$MODELS_HIST"
    else
        MODELS="$MODELS_WIN"
    fi

    run_config "logreg oof ($TRIO_TAG)"  "$MODELS" "logreg"  "--meta_features oof" "_$TRIO_TAG"
    run_config "des_ola oof ($TRIO_TAG)" "$MODELS" "des_ola" "--meta_features oof --meta_des_k 10 --meta_des_smoothing 1.0" "_$TRIO_TAG"
done

echo "============================================================================"
echo "ANÁLISIS PAREADO (AUC, vía analyze_feature_engineering_ladder_v2.py)"
echo "============================================================================"

for TRIO_TAG in hist win; do
    python src/analyze_feature_engineering_ladder_v2.py \
        --dataset "$DATASET" \
        --task "$TASK" \
        --work_dir "$WORK_DIR" \
        --subdirs "logreg baseline (oof)=ensemble4_oof_${TRIO_TAG}" \
                  "des_ola k:10 (oof)=ensemble4_des_ola_oof_${TRIO_TAG}" \
        --output "$PROJECT_DIR/results_des_ola_${TRIO_TAG}_${DATASET}_${TASK}.json"
done

echo "============================================================================"
echo "W/T/L + MDE (protocolo completo, misma fórmula que src/run_meta_suite.py)"
echo "============================================================================"

python3 - "$WORK_DIR" "$DATASET" "$TASK" "$PROJECT_DIR" <<'PYEOF'
import json
import sys
from pathlib import Path

import numpy as np
import scipy.stats as sp

work_dir, dataset, task, project_dir = sys.argv[1:5]
abmil_dir = Path(work_dir) / dataset / task / "abmil"


def load_fold_aucs(subdir):
    d = abmil_dir / subdir / "test_metrics"
    out = {}
    for fold_dir in sorted(d.glob("fold_*")):
        mf = fold_dir / "metrics.json"
        if not mf.exists():
            continue
        fold_num = int(fold_dir.name.split("_")[1])
        with open(mf) as f:
            m = json.load(f)
        auc = m.get("overall", {}).get("macro-ovr-auc", float("nan"))
        if auc != auc:  # nan
            auc = m.get("auc", float("nan"))
        out[fold_num] = auc
    return out


def paired(delta):
    delta = delta[~np.isnan(delta)]
    n = len(delta)
    if n == 0:
        return dict(n=0, mean_delta=float("nan"), ci_95=[float("nan")] * 2,
                    wins=0, ties=0, losses=0, p_value=1.0, mde=float("nan"))
    sd = float(delta.std(ddof=1)) if n > 1 else 0.0
    mde = (1.96 + 0.842) * sd / np.sqrt(n) if sd > 0 else 0.0
    wins = int((delta > 0).sum())
    ties = int((delta == 0).sum())
    losses = int((delta < 0).sum())
    if np.allclose(delta, 0):
        return dict(n=n, mean_delta=0.0, ci_95=[0.0, 0.0], wins=wins, ties=ties,
                    losses=losses, p_value=1.0, mde=float(mde))
    rng = np.random.default_rng(42)
    bs = np.array([rng.choice(delta, n, replace=True).mean() for _ in range(10000)])
    lo, hi = np.percentile(bs, [2.5, 97.5])
    try:
        p = float(sp.wilcoxon(delta).pvalue)
    except ValueError:
        p = 1.0
    return dict(n=n, mean_delta=float(delta.mean()), ci_95=[float(lo), float(hi)],
                wins=wins, ties=ties, losses=losses, p_value=p, mde=float(mde))


results = {}
for trio_tag, label in [("hist", "historico (ctranspath+uni_v2+virchow_v1)"),
                         ("win", "ganador (ctranspath+uni_v2+conch_v1_5)")]:
    base = load_fold_aucs(f"ensemble4_oof_{trio_tag}")
    new = load_fold_aucs(f"ensemble4_des_ola_oof_{trio_tag}")
    common = sorted(set(base) & set(new))
    delta = np.array([new[k] - base[k] for k in common])
    stats = paired(delta)
    stats["trio"] = label
    stats["n_folds_logreg"] = len(base)
    stats["n_folds_des_ola"] = len(new)
    stats["mean_auc_logreg"] = float(np.mean([base[k] for k in common])) if common else float("nan")
    stats["mean_auc_des_ola"] = float(np.mean([new[k] for k in common])) if common else float("nan")
    results[trio_tag] = stats

    print(f"\n=== {label} ===")
    print(f"  logreg  AUC = {stats['mean_auc_logreg']:.4f} (n={stats['n_folds_logreg']})")
    print(f"  des_ola AUC = {stats['mean_auc_des_ola']:.4f} (n={stats['n_folds_des_ola']})")
    print(f"  Delta AUC (des_ola - logreg) = {stats['mean_delta']:+.4f} "
          f"[{stats['ci_95'][0]:+.4f}, {stats['ci_95'][1]:+.4f}]")
    print(f"  W/T/L = {stats['wins']}/{stats['ties']}/{stats['losses']}  "
          f"p(Wilcoxon) = {stats['p_value']:.4f}  MDE(80% power) = {stats['mde']:.4f}")

out_path = Path(project_dir) / f"results_des_ola_wtl_{dataset}_{task}.json"
with open(out_path, "w") as f:
    json.dump(results, f, indent=2)
print(f"\nGuardado: {out_path}")
PYEOF

echo "============================================================================"
echo "COMPLETADO — Fin: $(date)"
echo "============================================================================"
echo ""
echo "Resultados:"
echo "  $PROJECT_DIR/results_des_ola_hist_${DATASET}_${TASK}.json"
echo "  $PROJECT_DIR/results_des_ola_win_${DATASET}_${TASK}.json"
echo "  $PROJECT_DIR/results_des_ola_wtl_${DATASET}_${TASK}.json"
