#!/usr/bin/env python
"""
Análisis de escalera Feature Engineering (Idea 1) — versión simplificada.

Lee métricas JSON directamente de los directorios ensemble4_fea_e{0,1,2,3}.
No requiere patho_bench ni utils.Metrics.

Uso:
    python src/analyze_feature_engineering_ladder_v2.py \
        --dataset cptac_brca \
        --task TP53_mutation \
        --work_dir PARADIS/datos/patches
"""

import os
import json
import argparse
from pathlib import Path
import numpy as np
from scipy import stats
from typing import Dict, List, Tuple


def load_fold_metrics(ensemble_dir: str) -> Dict[int, Dict]:
    """
    Carga métricas de cada fold de un directorio ensemble4_*.

    Espera estructura:
        {ensemble_dir}/test_metrics/fold_{k}/metrics.json

    Retorna:
        {fold_k: {'test_auc': float}, ...}
    """
    metrics = {}
    test_metrics_dir = Path(ensemble_dir) / "test_metrics"

    if not test_metrics_dir.exists():
        return {}

    for fold_dir in sorted(test_metrics_dir.glob("fold_*")):
        try:
            fold_num = int(fold_dir.name.split("_")[1])
        except (ValueError, IndexError):
            continue

        metrics_file = fold_dir / "metrics.json"
        if metrics_file.exists():
            try:
                with open(metrics_file) as f:
                    fold_metrics = json.load(f)
                    # Try different possible AUC keys
                    auc = fold_metrics.get('overall', {}).get('macro-ovr-auc', np.nan)
                    if np.isnan(auc):
                        auc = fold_metrics.get('auc', np.nan)
                    if not np.isnan(auc):
                        metrics[fold_num] = {'test_auc': float(auc)}
            except (json.JSONDecodeError, KeyError):
                pass

    return metrics


def compute_bootstrap_ci(auc_values: np.ndarray, n_bootstrap: int = 1000, ci: float = 0.95) -> Tuple[float, float]:
    """Calcula intervalo de confianza bootstrap para AUC."""
    n_folds = len(auc_values)
    bootstraps = []

    np.random.seed(42)
    for _ in range(n_bootstrap):
        sample = np.random.choice(auc_values, size=n_folds, replace=True)
        bootstraps.append(np.mean(sample))

    alpha = 1 - ci
    lower = np.percentile(bootstraps, 100 * alpha / 2)
    upper = np.percentile(bootstraps, 100 * (1 - alpha / 2))

    return lower, upper


def paired_test_fold_auc(auc_baseline: np.ndarray, auc_new: np.ndarray) -> Tuple[float, Tuple[float, float], float]:
    """Prueba pareada (paired t-test) del AUC."""
    deltas = auc_new - auc_baseline
    mean_delta = np.mean(deltas)

    ci_lower, ci_upper = compute_bootstrap_ci(deltas, n_bootstrap=1000, ci=0.95)

    t_stat, p_value = stats.ttest_rel(auc_new, auc_baseline)

    return mean_delta, (ci_lower, ci_upper), p_value


def holm_bonferroni_correction(p_values: List[float]) -> Tuple[List[float], List[str]]:
    """Aplica corrección de Holm-Bonferroni."""
    n_tests = len(p_values)
    sorted_indices = np.argsort(p_values)
    sorted_p = np.array(p_values)[sorted_indices]

    adjusted_p = sorted_p * (n_tests - np.arange(n_tests))
    adjusted_p = np.minimum(adjusted_p, 1.0)

    adjusted_p_original = np.empty_like(adjusted_p)
    adjusted_p_original[sorted_indices] = adjusted_p

    labels = []
    for p in adjusted_p_original:
        if p < 0.05:
            labels.append("**")
        elif p < 0.10:
            labels.append("*")
        else:
            labels.append("")

    return list(adjusted_p_original), labels


def main():
    parser = argparse.ArgumentParser(description="Análisis de escalera Feature Engineering (Idea 1)")
    parser.add_argument("--dataset", required=True, help="Dataset name (e.g., cptac_brca)")
    parser.add_argument("--task", required=True, help="Task name (e.g., TP53_mutation)")
    parser.add_argument("--work_dir", required=True, help="Base work directory")
    parser.add_argument(
        "--subdirs", nargs="+", required=True, metavar="LABEL=SUBDIR",
        help="Escalones a comparar, en orden; el primero es el baseline. "
             "Formato 'Etiqueta=nombre_de_subdirectorio', p.ej. "
             "'E0 (Baseline)=ensemble4_fea_e0' 'E1 (+attn)=ensemble4_idea3_attn_e1'.",
    )
    parser.add_argument(
        "--output", default=None,
        help="Fichero JSON de salida. Por defecto results_ladder_{dataset}_{task}.json, "
             "que distintas escaleras se pisan entre sí: pásalo explícitamente.",
    )
    args = parser.parse_args()

    dataset = args.dataset
    task = args.task
    work_dir = args.work_dir

    abmil_dir = Path(work_dir) / dataset / task / "abmil"

    # Los subdirectorios llegan por CLI, no hardcodeados: cuando la escalera que
    # se ejecuta escribe en `ensemble4_idea3_attn_e*` y aquí se leía siempre
    # `ensemble4_fea_e*`, el script no fallaba — imprimía la tabla de la escalera
    # anterior y la guardaba como si fuese la nueva (jobs 70119 y 70169).
    # Una LABEL repetida (p.ej. porque contiene su propio '=' y el split se
    # comió parte del valor, como "L1 C=0.01" -> label="L1 C") sobreescribe en
    # silencio la entrada anterior: sin este chequeo, 3 de 4 escalones L1
    # desaparecían del análisis sin ningún error (job 72576).
    escalones = {}
    for item in args.subdirs:
        if "=" not in item:
            parser.error(f"--subdirs espera 'LABEL=SUBDIR', recibido: {item!r}")
        label, subdir = item.split("=", 1)
        label = label.strip()
        if label in escalones:
            parser.error(
                f"--subdirs tiene la etiqueta {label!r} repetida (antes apuntaba a "
                f"{escalones[label]!r}, ahora a {subdir.strip()!r}). Si la etiqueta "
                f"contiene '=' (p.ej. 'C=0.01'), usa otro separador como 'C:0.01' — "
                f"split('=', 1) corta en el primer '=' y se come el resto de la etiqueta."
            )
        escalones[label] = subdir.strip()

    print("=" * 90)
    print(f"ANÁLISIS: Feature Engineering Ladder (Idea 1)")
    print(f"Dataset: {dataset}, Task: {task}")
    print("=" * 90)
    print()

    # 1. Cargar métricas de cada escalón
    # Un escalón que falta aborta el análisis. Saltárselo dejaba que el primer
    # escalón superviviente pasase a ser el "baseline" de la comparación, con
    # una tabla perfectamente plausible y equivocada.
    all_metrics = {}
    for label, subdir in escalones.items():
        ensemble_dir = abmil_dir / subdir
        if not ensemble_dir.exists():
            raise FileNotFoundError(
                f"Escalón {label!r}: no existe {ensemble_dir}. "
                f"¿Coincide --subdirs con lo que escribió el .sbatch?"
            )

        fold_metrics = load_fold_metrics(str(ensemble_dir))
        if not fold_metrics:
            raise RuntimeError(f"Escalón {label!r}: sin métricas de fold en {ensemble_dir}")

        auc_values = np.array([fold_metrics[k].get('test_auc', np.nan) for k in sorted(fold_metrics.keys())])
        auc_values = auc_values[~np.isnan(auc_values)]

        if len(auc_values) == 0:
            raise RuntimeError(f"Escalón {label!r}: sin valores de AUC válidos en {ensemble_dir}")

        all_metrics[label] = auc_values
        mean_auc = np.mean(auc_values)
        ci_lower, ci_upper = compute_bootstrap_ci(auc_values)

        print(f"✓ {label:35} | n_folds={len(auc_values):2d} | AUC={mean_auc:.4f} [{ci_lower:.4f}, {ci_upper:.4f}]")

    print()
    if len(all_metrics) < 2:
        print("❌ Insuficientes escalones con métricas válidas para comparación")
        return

    # 2. Comparaciones pareadas E0 vs E1, E2, E3
    baseline_label = list(all_metrics.keys())[0]
    auc_baseline = all_metrics[baseline_label]

    other_labels = list(all_metrics.keys())[1:]
    comparisons = []

    print("=" * 90)
    print("COMPARACIONES PAREADAS vs BASELINE")
    print("=" * 90)
    print(f"Baseline: {baseline_label}")
    print()

    p_values = []
    for other_label in other_labels:
        auc_new = all_metrics[other_label]

        # Alinear folds
        min_folds = min(len(auc_baseline), len(auc_new))
        auc_baseline_aligned = auc_baseline[:min_folds]
        auc_new_aligned = auc_new[:min_folds]

        mean_delta, ci, p_val = paired_test_fold_auc(auc_baseline_aligned, auc_new_aligned)
        p_values.append(p_val)

        comparisons.append({
            'comparison': other_label,
            'mean_delta': mean_delta,
            'ci_lower': ci[0],
            'ci_upper': ci[1],
            'p_value': p_val,
        })

    # 3. Holm-Bonferroni correction
    adjusted_p_values, sig_labels = holm_bonferroni_correction(p_values)

    for i, comp in enumerate(comparisons):
        comp['adjusted_p_value'] = adjusted_p_values[i]
        comp['significance'] = sig_labels[i]

    # 4. Tabla de comparaciones
    print(f"{'Comparison':<35} | {'Δ AUC':>8} | {'95% CI':>25} | {'p-value':>10} | {'Adj. p':>10} | {'Sig':>3}")
    print("-" * 110)

    for comp in comparisons:
        delta_str = f"{comp['mean_delta']:+.6f}"
        ci_str = f"[{comp['ci_lower']:.4f}, {comp['ci_upper']:.4f}]"
        p_str = f"{comp['p_value']:.6f}"
        adj_p_str = f"{comp['adjusted_p_value']:.6f}"
        sig = comp['significance']

        print(f"{comp['comparison']:<35} | {delta_str:>8} | {ci_str:>25} | {p_str:>10} | {adj_p_str:>10} | {sig:>3}")

    print()

    # 5. Resumen
    print("=" * 90)
    print("INTERPRETACIÓN")
    print("=" * 90)
    # '**' = adj. p < 0.05 (el α declarado abajo); '*' = 0.05 <= adj. p < 0.10,
    # marginal, no cuenta como significativo a ese α. Antes este bloque trataba
    # cualquier marca no vacía como "significativo (α=0.05)" y además llamaba
    # "mejora" a cualquier cambio significativo sin mirar el signo de Δ — un
    # escalón que empeora significativamente se imprimía como mejora.
    sig_005 = [c for c in comparisons if c['significance'] == "**"]
    marginal = [c for c in comparisons if c['significance'] == "*"]
    if not sig_005:
        print("⚠️  Ningún escalón es significativamente diferente del baseline (Holm-Bonferroni α=0.05)")
        print()
        print("   El presupuesto estadístico (MDE ≈ ±0.016) es mayor que cualquier efecto real.")
        print("   Las features extras no capturan señal adicional significativa.")
        if marginal:
            print()
            print(f"   ({len(marginal)} escalón(es) marginal(es), 0.05 ≤ adj. p < 0.10 — no cruzan α=0.05):")
            for comp in marginal:
                direction = "mejora" if comp['mean_delta'] > 0 else "empeora"
                print(f"     - {comp['comparison']}: Δ={comp['mean_delta']:+.6f} ({direction})")
        print()
    else:
        print(f"✓ {len(sig_005)} escalón(es) significativamente distinto(s) del baseline (α=0.05):")
        for comp in sig_005:
            direction = "mejora" if comp['mean_delta'] > 0 else "empeora"
            print(f"  - {comp['comparison']}: Δ={comp['mean_delta']:+.6f} ({direction})")
        print()

    # 6. Guardar resultados
    results = {
        'dataset': dataset,
        'task': task,
        'baseline': baseline_label,
        'escalones': {
            label: {
                'mean_auc': float(np.mean(auc_values)),
                'std_auc': float(np.std(auc_values)),
                'n_folds': len(auc_values),
            }
            for label, auc_values in all_metrics.items()
        },
        'comparisons': [
            {
                'comparison': c['comparison'],
                'mean_delta': float(c['mean_delta']),
                'ci_95': [float(c['ci_lower']), float(c['ci_upper'])],
                'p_value': float(c['p_value']),
                'adjusted_p_value': float(c['adjusted_p_value']),
                # '**' = adj. p < 0.05 only. '*' (0.05<=adj. p<0.10) is marginal,
                # not significant at the declared alpha — see 'marginal' below.
                'significant': c['significance'] == "**",
                'marginal': c['significance'] == "*",
            }
            for c in comparisons
        ],
    }

    output_file = args.output or f"results_ladder_{dataset}_{task}.json"
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"💾 Resultados guardados en: {output_file}")
    print()


if __name__ == "__main__":
    main()
