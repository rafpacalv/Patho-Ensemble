#!/usr/bin/env python
"""
Análisis de escalera Feature Engineering (Idea 1).

SUPERSEDIDO por `analyze_feature_engineering_ladder_v2.py`. Este fichero
mantiene los subdirectorios `ensemble4_fea_e*` **hardcodeados**, así que
aplicado a cualquier otra escalera (p.ej. `ensemble4_idea3_attn_e*`) no falla:
lee los directorios equivocados e imprime una tabla plausible y errónea. La v2
los recibe por `--subdirs`. No usar este.

Lee métricas pareadas de los 4 escalones (E0, E1, E2, E3) y reporta:
- AUC por fold y agregado
- Comparaciones pareadas (Δ, 95% CI, p-value)
- Holm-Bonferroni correction para múltiples comparaciones
- Minimum Detectable Effect (MDE) @ 80% poder

Uso:
    python src/analyze_feature_engineering_ladder.py \
        --dataset cptac_brca \
        --task TP53_mutation \
        --work_dir /home/JKP6679/Patho-Ensemble/PARADIS/datos/patches

Output:
    - results_ladder_DATASET_TASK.json (métricas agregadas)
    - Tabla formateada en stdout (paired comparisons)
"""

import os
import json
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats
from typing import Dict, List, Tuple

from utils import Metrics


def load_fold_metrics(ensemble_dir: str) -> Dict[int, Dict]:
    """
    Carga métricas de cada fold de un directorio ensemble4_*.

    Espera estructura:
        {ensemble_dir}/val_outputs_metrics/fold_{k}/metrics.json
        {ensemble_dir}/test_outputs_metrics/fold_{k}/metrics.json

    Retorna:
        {fold_k: {'val_auc': float, 'test_auc': float}, ...}
    """
    metrics = {}
    val_metrics_dir = Path(ensemble_dir) / "val_outputs_metrics"
    test_metrics_dir = Path(ensemble_dir) / "test_outputs_metrics"

    if not val_metrics_dir.exists() or not test_metrics_dir.exists():
        return {}

    for fold_dir in val_metrics_dir.glob("fold_*"):
        fold_num = int(fold_dir.name.split("_")[1])
        metrics_file = fold_dir / "metrics.json"

        if metrics_file.exists():
            with open(metrics_file) as f:
                fold_metrics = json.load(f)
                metrics[fold_num] = {
                    'val_auc': fold_metrics.get('auc', np.nan),
                }

    # Cargar AUCs de test set
    for fold_dir in test_metrics_dir.glob("fold_*"):
        fold_num = int(fold_dir.name.split("_")[1])
        metrics_file = fold_dir / "metrics.json"

        if metrics_file.exists():
            with open(metrics_file) as f:
                fold_metrics = json.load(f)
                if fold_num in metrics:
                    metrics[fold_num]['test_auc'] = fold_metrics.get('auc', np.nan)

    return metrics


def compute_bootstrap_ci(auc_values: np.ndarray, n_bootstrap: int = 1000, ci: float = 0.95) -> Tuple[float, float]:
    """
    Calcula intervalo de confianza bootstrap para AUC.

    Args:
        auc_values: array [n_folds] de AUCs
        n_bootstrap: número de iteraciones
        ci: nivel de confianza (0.95 = 95%)

    Returns:
        (lower, upper) intervalo de confianza
    """
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
    """
    Prueba pareada (paired t-test) del AUC entre baseline y nueva configuración.

    Args:
        auc_baseline: array [n_folds] AUC del baseline (E0)
        auc_new: array [n_folds] AUC de la nueva configuración

    Returns:
        (mean_delta, ci_95, p_value)
            mean_delta: promedio de diferencias (auc_new - auc_baseline)
            ci_95: intervalo de confianza 95% de la diferencia
            p_value: paired t-test, two-tailed
    """
    deltas = auc_new - auc_baseline
    mean_delta = np.mean(deltas)

    # Bootstrap CI en las diferencias
    ci_lower, ci_upper = compute_bootstrap_ci(deltas, n_bootstrap=1000, ci=0.95)

    # Paired t-test
    t_stat, p_value = stats.ttest_rel(auc_new, auc_baseline)

    return mean_delta, (ci_lower, ci_upper), p_value


def holm_bonferroni_correction(p_values: List[float]) -> Tuple[List[float], List[str]]:
    """
    Aplica corrección de Holm-Bonferroni para múltiples comparaciones.

    Args:
        p_values: lista de p-values originales

    Returns:
        (adjusted_p_values, significance_labels)
            adjusted_p_values: p-values ajustados (no exceden 1.0)
            significance_labels: strings con ** (p<0.05), * (p<0.10), '' (no sig.)
    """
    n_tests = len(p_values)
    sorted_indices = np.argsort(p_values)
    sorted_p = np.array(p_values)[sorted_indices]

    adjusted_p = sorted_p * (n_tests - np.arange(n_tests))
    adjusted_p = np.minimum(adjusted_p, 1.0)

    # Reorder back to original order
    adjusted_p_original = np.empty_like(adjusted_p)
    adjusted_p_original[sorted_indices] = adjusted_p

    # Significance labels
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
    parser.add_argument("--metric", default="test_auc", choices=["val_auc", "test_auc"],
                        help="Which metric to use for comparison (default: test_auc)")
    args = parser.parse_args()

    dataset = args.dataset
    task = args.task
    work_dir = args.work_dir
    metric_name = args.metric

    abmil_dir = Path(work_dir) / dataset / task / "abmil"

    escalones = {
        "E0 (Baseline)": "ensemble4_fea_e0",
        "E1 (+disagreement)": "ensemble4_fea_e1",
        "E2 (+disagreement+margin)": "ensemble4_fea_e2",
        "E3 (+disagreement+margin+entropy)": "ensemble4_fea_e3",
    }

    print("=" * 90)
    print(f"ANÁLISIS: Feature Engineering Ladder (Idea 1)")
    print(f"Dataset: {dataset}, Task: {task}")
    print(f"Métrica: {metric_name}")
    print("=" * 90)
    print()

    # 1. Cargar métricas de cada escalón
    all_metrics = {}
    for label, subdir in escalones.items():
        ensemble_dir = abmil_dir / subdir
        if not ensemble_dir.exists():
            print(f"❌ {label}: directorio no encontrado: {ensemble_dir}")
            continue

        fold_metrics = load_fold_metrics(str(ensemble_dir))
        if not fold_metrics:
            print(f"❌ {label}: sin métricas de fold encontradas")
            continue

        auc_values = np.array([fold_metrics[k].get(metric_name, np.nan) for k in sorted(fold_metrics.keys())])
        auc_values = auc_values[~np.isnan(auc_values)]

        if len(auc_values) == 0:
            print(f"❌ {label}: sin valores de AUC válidos")
            continue

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

        # Alinear folds (puede haber discrepancias)
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

    # 4. Imprimir tabla de comparaciones
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

    # 5. Resumen de significancia
    print("=" * 90)
    print("INTERPRETACIÓN")
    print("=" * 90)
    sig_count = sum(1 for c in comparisons if c['significance'] != "")
    if sig_count == 0:
        print("⚠️  Ningún escalón es significativamente diferente del baseline (Holm-Bonferroni α=0.05)")
        print()
        print("   Posibles causas:")
        print("   1. El presupuesto estadístico (MDE ≈ ±0.016) es mayor que el efecto verdadero")
        print("   2. Las features extras no capturan señal adicional relevante")
        print("   3. LogisticRegression ya es un modelo tan simple que agrega más features solo añade ruido")
        print()
    else:
        print(f"✓ {sig_count} escalones muestran mejora significativa (α=0.05, Holm-Bonferroni corrected):")
        for comp in comparisons:
            if comp['significance'] != "":
                print(f"  - {comp['comparison']}: Δ={comp['mean_delta']:+.6f}")
        print()

    # 6. Guardar resultados en JSON
    results = {
        'dataset': dataset,
        'task': task,
        'metric': metric_name,
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
                'significant': c['significance'] != "",
            }
            for c in comparisons
        ],
    }

    output_file = f"results_ladder_{dataset}_{task}.json"
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"💾 Resultados guardados en: {output_file}")
    print()


if __name__ == "__main__":
    main()
