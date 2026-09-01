"""
Ensemble con pesos optimizados por Nelder-Mead MEJORADO.

Mejoras implementadas:
1. Multi-start Nelder-Mead: Múltiples inicializaciones
2. Differential Evolution: Método global opcional
3. Regularización: Penalización de pesos extremos
4. Análisis de sensibilidad: Detecta si el espacio es plano
5. Comparación de métodos: Ve cuál mejora más

Reutiliza: Ensemble (de ensemble.py) y Metrics (de utils.py)
"""
import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
import os
import argparse
import json
from scipy.optimize import minimize, differential_evolution
from sklearn.metrics import roc_auc_score

from ensemble import Ensemble
from utils import Metrics
from format_metrics import enrich_metrics_summary, print_metrics_table, generate_comparison_report


def auroc(y, pp, nC):
    """Calcula AUROC binaria o multiclase según nC."""
    try:
        return (
            roc_auc_score(y, pp[:, 1])
            if nC == 2
            else roc_auc_score(y, pp, multi_class="ovr", average="macro", labels=list(range(nC)))
        )
    except ValueError:
        return 0.5


def objective(x, val_preds_stacked, val_labels, nC, lambda_reg=0.0):
    """Objetivo a minimizar: 1 - AUROC + regularización.

    x: vector de pesos crudos (uno por modelo)
    val_preds_stacked: (n_models, n_samples, nC)
    val_labels: (n_samples,)
    nC: número de clases
    lambda_reg: coeficiente de regularización (0 = sin regularización)
    """
    # Normalizar pesos con softmax
    w = torch.softmax(torch.tensor(x, dtype=torch.float32), dim=0).numpy()

    # Promedio ponderado
    weighted_preds = np.tensordot(w, val_preds_stacked, axes=([0], [0]))

    # AUROC
    auc = auroc(val_labels, weighted_preds, nC)

    # Regularización: penalizar pesos muy extremos
    # Desviación estándar de los pesos (si todos = 0.25 → desv = 0)
    reg_term = lambda_reg * np.std(w)

    return (1.0 - auc) + reg_term


def objective_for_diff_evolution(x, val_preds_stacked, val_labels, nC, lambda_reg=0.0):
    """Wrapper para Differential Evolution (bounds: [0, 1])."""
    w = x / x.sum()  # Normalizar a suma=1
    weighted_preds = np.tensordot(w, val_preds_stacked, axes=([0], [0]))
    auc = auroc(val_labels, weighted_preds, nC)
    reg_term = lambda_reg * np.std(w)
    return (1.0 - auc) + reg_term


def analyze_sensitivity(val_preds_stacked, val_labels, nC, x_opt, delta=0.01):
    """Analiza cuán sensible es el AUC a cambios en los pesos.

    Si los cambios pequeños en pesos no afectan AUC → espacio es plano.
    """
    w_opt = torch.softmax(torch.tensor(x_opt, dtype=torch.float32), dim=0).numpy()
    auc_opt = auroc(val_labels,
                    np.tensordot(w_opt, val_preds_stacked, axes=([0], [0])),
                    nC)

    sensitivities = []
    for i in range(len(x_opt)):
        # Perturbar peso i en ±delta
        x_plus = x_opt.copy()
        x_plus[i] += delta
        w_plus = torch.softmax(torch.tensor(x_plus, dtype=torch.float32), dim=0).numpy()
        auc_plus = auroc(val_labels,
                        np.tensordot(w_plus, val_preds_stacked, axes=([0], [0])),
                        nC)

        # Cambio en AUC
        delta_auc = auc_plus - auc_opt
        sensitivities.append(delta_auc)

    return np.array(sensitivities), auc_opt


def multistart_nelder_mead(val_preds_stacked, val_labels, nC,
                          n_starts=5, lambda_reg=0.0):
    """Ejecuta Nelder-Mead desde múltiples inicializaciones.

    Devuelve: (x_best, auc_best, results_dict)
    """
    results = []

    for start in range(n_starts):
        # Inicialización aleatoria o predefinida
        if start == 0:
            # Primer intento: equal weights
            x0 = np.zeros(len(val_preds_stacked))
        else:
            # Inicializaciones aleatorias
            x0 = np.random.randn(len(val_preds_stacked)) * (start * 0.1)

        result = minimize(
            objective,
            x0,
            args=(val_preds_stacked, val_labels, nC, lambda_reg),
            method="Nelder-Mead",
            options={"maxiter": 500, "xatol": 1e-6, "fatol": 1e-7, "adaptive": True}
        )

        # Calcular AUC del resultado
        w = torch.softmax(torch.tensor(result.x, dtype=torch.float32), dim=0).numpy()
        auc = auroc(val_labels,
                   np.tensordot(w, val_preds_stacked, axes=([0], [0])),
                   nC)

        results.append({
            'start': start,
            'x': result.x,
            'auc': auc,
            'nfev': result.nfev,
            'success': result.success
        })

    # Elegir mejor resultado
    best_idx = np.argmax([r['auc'] for r in results])
    best = results[best_idx]

    return best['x'], best['auc'], results


def _run_with_equal_weights(foundational_models, work_dir, train_source, tissue_patching, task_name, n_folds, test_dirs):
    """Fallback: Use equal weights for ensemble when validation data is missing."""

    all_preds = []
    all_labels = []

    print(f"Evaluating ensemble with equal weights ({n_folds} folds)...")
    with tqdm(total=n_folds, unit="fold") as pbar:
        for f in range(n_folds):
            pbar.set_description(f"Fold {f+1}/{n_folds}")

            # Load test predictions from all models
            test_preds_list = []
            test_labels_fold = None

            for idx, test_dir in enumerate(test_dirs):
                test_fold_path = (
                    os.path.join(test_dir, f'fold_{f}') if n_folds > 1 else test_dir
                )
                test_preds = np.load(os.path.join(test_fold_path, 'preds.npy'))
                test_labels = np.load(os.path.join(test_fold_path, 'labels.npy'))

                test_preds_list.append(test_preds)
                if test_labels_fold is None:
                    test_labels_fold = test_labels

            test_preds_stacked = np.stack(test_preds_list, axis=0)

            # Use equal weights
            n_models = len(foundational_models)
            w_equal = np.ones(n_models) / n_models
            test_preds_combined = np.tensordot(w_equal, test_preds_stacked, axes=([0], [0]))

            all_preds.append(test_preds_combined)
            all_labels.append(test_labels_fold)

            pbar.update(1)

    # Save equal weights
    weights_nm_array = np.full((n_folds, len(foundational_models)), 1.0/len(foundational_models))
    weights_nm_path = os.path.join(work_dir, train_source, task_name, 'abmil', 'weights_nm_adv.npy')
    np.save(weights_nm_path, weights_nm_array)
    print(f"✅ Equal weights saved to {weights_nm_path}")

    # Compute metrics on test
    print("\n🔎 Computing test metrics (equal weights)...")
    results_dir = os.path.join(work_dir, train_source, task_name, 'abmil', 'ensemble_nm_adv')
    num_classes = len(set(all_labels[0]))

    Metrics(
        task_type='classification',
        model_kwargs={'num_classes': num_classes},
        num_bootstraps=100,
        results_dir=results_dir,
        split='test',
        num_folds=n_folds,
        all_labels_across_folds=all_labels,
        all_preds_across_folds=all_preds
    ).run()

    # Enrich metrics
    metrics_summary_path = os.path.join(results_dir, 'test_metrics_summary.json')
    enrich_metrics_summary(metrics_summary_path, all_labels, all_preds, num_classes)

    # Print metrics
    import json
    with open(metrics_summary_path, 'r') as f:
        summary = json.load(f)
    if 'all_metrics' in summary:
        print_metrics_table(summary['all_metrics'])

    print("✅ Metrics completed")


def main(foundational_models, work_dir, train_source, tissue_patching, task_name,
         method='nelder-mead-multistart', lambda_reg=0.0, n_starts=5):
    """Optimiza pesos con método elegido (Nelder-Mead mejorado o Differential Evolution).

    method: 'nelder-mead' | 'nelder-mead-multistart' | 'differential-evolution'
    lambda_reg: coeficiente de regularización (default: 0.0)
    n_starts: número de inicializaciones para multi-start (default: 5)
    """

    dirs = [
        os.path.join(work_dir, train_source, task_name, 'abmil', f'{f}_{tissue_patching}')
        for f in foundational_models
    ]

    # Detectar número de folds y si tenemos validación
    folds = []
    val_dirs = []
    test_dirs = []
    has_validation = True

    for idx, d in enumerate(dirs):
        val_path = os.path.join(d, 'val_outputs')
        test_path = os.path.join(d, 'test_outputs')

        if not os.path.exists(test_path):
            raise FileNotFoundError(f"Missing path: {test_path}")

        if not os.path.exists(val_path):
            print(f"  ⚠️  Missing validation outputs in {d}")
            has_validation = False
        else:
            val_dirs.append(val_path)

        has_fold = any(name.startswith("fold_") for name in os.listdir(test_path))
        if has_fold:
            n_test_folds = len([n for n in os.listdir(test_path) if n.startswith("fold_")])
            folds.append(n_test_folds)
        else:
            folds.append(1)

        test_dirs.append(test_path)

    if not has_validation:
        print("\n⚠️  No validation outputs found - skipping optimization")
        _run_with_equal_weights(foundational_models, work_dir, train_source, tissue_patching, task_name, folds[0], test_dirs)
        return

    assert len(set(folds)) == 1, "Mismatch in number of folds"
    n_folds = folds[0]
    device = "cuda" if torch.cuda.is_available() else "cpu"

    all_preds = []
    all_labels = []
    weights_nm = []
    optimization_stats = []  # Para guardar estadísticas de optimización

    print(f"\n\n{'='*80}")
    print(f"Optimizando pesos con: {method.upper()} ({n_folds} folds)")
    print(f"Regularización (lambda): {lambda_reg}")
    if 'multistart' in method:
        print(f"Número de inicializaciones: {n_starts}")
    print(f"{'='*80}\n")

    with tqdm(total=n_folds, unit="fold") as pbar:
        for f in range(n_folds):
            pbar.set_description(f"Fold {f+1}/{n_folds}")

            # Check if all models have validation data for this fold
            fold_has_val = True
            for idx, val_dir in enumerate(val_dirs):
                val_fold_path = (
                    os.path.join(val_dir, f'fold_{f}') if n_folds > 1 else val_dir
                )
                if not os.path.exists(os.path.join(val_fold_path, 'preds.npy')):
                    fold_has_val = False
                    break

            # Determinar pesos a usar
            if not fold_has_val:
                # Si no hay validación, usar equal weights
                print(f"  ⚠️  Fold {f}: validation data missing, using equal weights")
                x_opt = np.zeros(len(foundational_models))
                weights_nm.append(x_opt)
                optimization_stats.append({'fold': f, 'method': 'equal_weights', 'auc_val': 0, 'avg_sensitivity': 0})
                auc_opt = 0
                opt_method_used = 'equal_weights'
            else:
                # Si hay validación, optimizar
                # Cargar predicciones de validación
                val_preds_list = []
                val_labels_fold = None

                for idx, val_dir in enumerate(val_dirs):
                    val_fold_path = (
                        os.path.join(val_dir, f'fold_{f}') if n_folds > 1 else val_dir
                    )
                    val_preds = np.load(os.path.join(val_fold_path, 'preds.npy'))
                    val_labels = np.load(os.path.join(val_fold_path, 'labels.npy'))

                    val_preds_list.append(val_preds)
                    if val_labels_fold is None:
                        val_labels_fold = val_labels
                    else:
                        assert np.array_equal(val_labels_fold, val_labels), \
                            f"Labels differ between models in fold {f}"

                val_preds_stacked = np.stack(val_preds_list, axis=0)
                nC = val_preds_stacked.shape[2]

                # Seleccionar método de optimización
                if method == 'nelder-mead':
                    x0 = np.zeros(len(foundational_models))
                    result = minimize(
                        objective,
                        x0,
                        args=(val_preds_stacked, val_labels_fold, nC, lambda_reg),
                        method="Nelder-Mead",
                        options={"maxiter": 500, "xatol": 1e-6, "fatol": 1e-7, "adaptive": True}
                    )
                    x_opt = result.x
                    w_opt = torch.softmax(torch.tensor(x_opt, dtype=torch.float32), dim=0).numpy()
                    auc_opt = auroc(val_labels_fold,
                                   np.tensordot(w_opt, val_preds_stacked, axes=([0], [0])),
                                   nC)
                    opt_method_used = 'nelder-mead'

                elif method == 'nelder-mead-multistart':
                    x_opt, auc_opt, results = multistart_nelder_mead(
                        val_preds_stacked, val_labels_fold, nC,
                        n_starts=n_starts, lambda_reg=lambda_reg
                    )
                    opt_method_used = f'nelder-mead-multistart({n_starts})'

                elif method == 'differential-evolution':
                    bounds = [(0, 1)] * len(foundational_models)
                    result = differential_evolution(
                        objective_for_diff_evolution,
                        bounds,
                        args=(val_preds_stacked, val_labels_fold, nC, lambda_reg),
                        seed=42 + f,
                        maxiter=500,
                        atol=1e-7,
                        tol=1e-7,
                        workers=1
                    )
                    x_opt = result.x / result.x.sum()  # Normalizar
                    # Convertir a escala softmax (aproximado)
                    x_opt = np.log(x_opt + 1e-10)
                    auc_opt = auroc(val_labels_fold,
                                   np.tensordot(x_opt, val_preds_stacked, axes=([0], [0])),
                                   nC)
                    opt_method_used = 'differential-evolution'

                # Análisis de sensibilidad
                sensitivities, _ = analyze_sensitivity(val_preds_stacked, val_labels_fold, nC, x_opt)
                avg_sensitivity = np.mean(np.abs(sensitivities))

                optimization_stats.append({
                    'fold': f,
                    'method': opt_method_used,
                    'auc_val': auc_opt,
                    'avg_sensitivity': avg_sensitivity
                })

                weights_nm.append(x_opt)

            # Predicciones en test
            test_preds_list = []
            test_labels_fold = None

            for idx, test_dir in enumerate(test_dirs):
                test_fold_path = (
                    os.path.join(test_dir, f'fold_{f}') if n_folds > 1 else test_dir
                )
                test_preds = np.load(os.path.join(test_fold_path, 'preds.npy'))
                test_labels = np.load(os.path.join(test_fold_path, 'labels.npy'))

                test_preds_list.append(test_preds)
                if test_labels_fold is None:
                    test_labels_fold = test_labels
                else:
                    assert np.array_equal(test_labels_fold, test_labels), \
                        f"Labels differ between models in test fold {f}"

            test_preds_stacked = np.stack(test_preds_list, axis=0)

            # Aplicar pesos óptimos
            w_opt = torch.softmax(torch.tensor(x_opt, dtype=torch.float32), dim=0).numpy()
            test_preds_combined = np.tensordot(w_opt, test_preds_stacked, axes=([0], [0]))

            all_preds.append(test_preds_combined)
            all_labels.append(test_labels_fold)

            pbar.update(1)

    # Guardar pesos
    weights_nm_array = np.array(weights_nm)
    weights_nm_path = os.path.join(work_dir, train_source, task_name, 'abmil', 'weights_nm_adv.npy')
    np.save(weights_nm_path, weights_nm_array)
    print(f"\n✅ Pesos guardados en {weights_nm_path}")

    # Guardar estadísticas de optimización
    stats_path = os.path.join(work_dir, train_source, task_name, 'abmil', 'optimization_stats.json')
    with open(stats_path, 'w') as f:
        json.dump(optimization_stats, f, indent=4)
    print(f"✅ Estadísticas de optimización guardadas en {stats_path}")

    # Análisis de sensibilidad agregado
    avg_sensitivity_all = np.mean([s['avg_sensitivity'] for s in optimization_stats if 'avg_sensitivity' in s])
    print(f"\n📊 Sensibilidad promedio: {avg_sensitivity_all:.6f}")
    if avg_sensitivity_all < 0.001:
        print("   ⚠️  ESPACIO PLANO: Los cambios en pesos afectan poco el AUC")
        print("   → Nelder-Mead no puede mejorar mucho al promedio ponderado")
    else:
        print("   ✓ Espacio sensible: Nelder-Mead puede mejorar")

    # Métricas en test
    print("\n🔎 Calculando métricas de test...")
    results_dir = os.path.join(work_dir, train_source, task_name, 'abmil', 'ensemble_nm_adv')
    num_classes = len(set(all_labels[0]))

    Metrics(
        task_type='classification',
        model_kwargs={'num_classes': num_classes},
        num_bootstraps=100,
        results_dir=results_dir,
        split='test',
        num_folds=n_folds,
        all_labels_across_folds=all_labels,
        all_preds_across_folds=all_preds
    ).run()

    # Enriquecer métricas
    metrics_summary_path = os.path.join(results_dir, 'test_metrics_summary.json')
    enrich_metrics_summary(metrics_summary_path, all_labels, all_preds, num_classes)

    # Imprimir métricas
    with open(metrics_summary_path, 'r') as f:
        summary = json.load(f)
    if 'all_metrics' in summary:
        print_metrics_table(summary['all_metrics'])

    # Generar reporte comparativo
    print("\n" + "="*80)
    print("GENERANDO REPORTE COMPARATIVO...")
    print("="*80)
    ensemble_metrics_path = os.path.join(work_dir, train_source, task_name, 'abmil', 'ensemble', 'test_metrics_summary.json')
    if os.path.exists(ensemble_metrics_path):
        generate_comparison_report(
            ensemble_metrics_path,
            metrics_summary_path,
            output_path=os.path.join(results_dir, 'comparison_report.txt')
        )
    else:
        print("⚠️  Ensemble promediado no encontrado.")

    print("✅ Optimización completada")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ensemble con optimización avanzada")
    parser.add_argument("--foundational_models", nargs='+', type=str, required=True)
    parser.add_argument("--work_dir", type=str, default="/home/JKP6679/Patho-Ensemble/PARADIS/datos/patches")
    parser.add_argument("--train_source", type=str, required=True)
    parser.add_argument("--tissue_patching", type=str, required=True)
    parser.add_argument("--task_name", type=str, required=True)
    parser.add_argument("--method", type=str, default="nelder-mead-multistart",
                       choices=["nelder-mead", "nelder-mead-multistart", "differential-evolution"],
                       help="Método de optimización")
    parser.add_argument("--lambda_reg", type=float, default=0.0,
                       help="Coeficiente de regularización (default: 0.0)")
    parser.add_argument("--n_starts", type=int, default=5,
                       help="Número de inicializaciones para multi-start (default: 5)")
    args = parser.parse_args()

    main(**vars(args))
