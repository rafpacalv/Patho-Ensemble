"""
Ensemble con pesos optimizados por Nelder-Mead (alternativa a promediado por MetricDistance).

Usa scipy.optimize.minimize(method='Nelder-Mead') para maximizar AUROC de validación,
luego aplica los pesos óptimos al test.

Reutiliza: Ensemble (de ensemble.py) para la combinación, Metrics (de utils.py) para métricas finales.
"""
import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
import os
import argparse
from scipy.optimize import minimize
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
    weights_nm_path = os.path.join(work_dir, train_source, task_name, 'abmil', 'weights_nm.npy')
    np.save(weights_nm_path, weights_nm_array)
    print(f"✅ Equal weights saved to {weights_nm_path}")

    # Compute metrics on test
    print("\n🔎 Computing test metrics (equal weights)...")
    results_dir = os.path.join(work_dir, train_source, task_name, 'abmil', 'ensemble_nm')
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

    # Generate comparison report
    print("\n" + "="*80)
    print("GENERATING COMPARISON REPORT...")
    print("="*80)
    ensemble_metrics_path = os.path.join(work_dir, train_source, task_name, 'abmil', 'ensemble', 'test_metrics_summary.json')
    if os.path.exists(ensemble_metrics_path):
        generate_comparison_report(
            ensemble_metrics_path,
            metrics_summary_path,
            output_path=os.path.join(results_dir, 'comparison_report.txt')
        )

    print("✅ Metrics completed")


def objective(x, val_preds_stacked, val_labels, nC):
    """Objetivo a minimizar: 1 - AUROC en validación.

    x: vector de pesos crudos (uno por modelo)
    val_preds_stacked: (n_models, n_samples, nC)
    val_labels: (n_samples,) o (n_samples, nC) si multiclase
    nC: número de clases
    """
    # Normalizar pesos con softmax
    w = torch.softmax(torch.tensor(x, dtype=torch.float32), dim=0).numpy()

    # Promedio ponderado
    weighted_preds = np.tensordot(w, val_preds_stacked, axes=([0], [0]))  # (n_samples, nC)

    # AUROC
    auc = auroc(val_labels, weighted_preds, nC)
    return 1.0 - auc  # minimizar 1-AUROC = maximizar AUROC


def main(foundational_models, work_dir, train_source, tissue_patching, task_name):
    """Optimiza pesos con Nelder-Mead, por fold.

    Si no hay val_outputs (no explicit validation splits), usa igual weights.
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

        # Check test_outputs (required)
        if not os.path.exists(test_path):
            raise FileNotFoundError(f"Missing path: {test_path}")

        # Check val_outputs (optional)
        if not os.path.exists(val_path):
            print(f"  ⚠️  Missing validation outputs in {d}")
            has_validation = False
        else:
            val_dirs.append(val_path)

        # Count folds from test_outputs
        has_fold = any(name.startswith("fold_") for name in os.listdir(test_path))
        if has_fold:
            n_test_folds = len([n for n in os.listdir(test_path) if n.startswith("fold_")])
            folds.append(n_test_folds)
        else:
            folds.append(1)

        test_dirs.append(test_path)

    # If no validation data found, skip Nelder-Mead optimization
    if not has_validation:
        print("\n⚠️  No validation outputs found - skipping Nelder-Mead optimization")
        print("    (This happens when split file has no explicit validation splits)")
        print("    Using equal weights for ensemble instead.\n")
        _run_with_equal_weights(foundational_models, work_dir, train_source, tissue_patching, task_name, folds[0], test_dirs)
        return

    assert len(set(folds)) == 1, "Mismatch in number of folds"
    n_folds = folds[0]
    device = "cuda" if torch.cuda.is_available() else "cpu"

    all_preds = []
    all_labels = []
    weights_nm = []

    print(f"\n\nOptimizando pesos con Nelder-Mead ({n_folds} folds)...")
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

            # Si no hay validación, usar equal weights para este fold
            if not fold_has_val:
                print(f"  ⚠️  Fold {f}: validation data missing, using equal weights")
                x_opt = np.zeros(len(foundational_models))  # Equal weights after softmax
                weights_nm.append(x_opt)
            else:
                # Cargar predicciones de validación (para optimización)
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

                val_preds_stacked = np.stack(val_preds_list, axis=0)  # (n_models, n_samples, nC)
                nC = val_preds_stacked.shape[2]

                # Nelder-Mead: encontrar pesos óptimos
                x0 = np.zeros(len(foundational_models))
                result = minimize(
                    objective,
                    x0,
                    args=(val_preds_stacked, val_labels_fold, nC),
                    method="Nelder-Mead",
                    options={"maxiter": 200, "xatol": 1e-4, "fatol": 1e-5}
                )
                x_opt = result.x
                weights_nm.append(x_opt)

            # Predicciones en test con pesos óptimos
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
    weights_nm_path = os.path.join(work_dir, train_source, task_name, 'abmil', 'weights_nm.npy')
    np.save(weights_nm_path, weights_nm_array)
    print(f"✅ Pesos Nelder-Mead guardados en {weights_nm_path}")

    # Métricas en test
    print("\n🔎 Calculando métricas de test (Nelder-Mead)...")
    results_dir = os.path.join(work_dir, train_source, task_name, 'abmil', 'ensemble_nm')
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

    # Enriquecer métricas con cálculos explícitos
    metrics_summary_path = os.path.join(results_dir, 'test_metrics_summary.json')
    enrich_metrics_summary(metrics_summary_path, all_labels, all_preds, num_classes)

    # Imprimir métricas detalladas
    import json
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
        print("⚠️  Ensemble promediado no encontrado. Ejecuta ensemble.py primero.")

    print("✅ Métricas completadas")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ensemble con Nelder-Mead")
    parser.add_argument("--foundational_models", nargs='+', type=str, required=True)
    parser.add_argument("--work_dir", type=str, default="/home/JKP6679/Patho-Ensemble/PARADIS/datos/patches")
    parser.add_argument("--train_source", type=str, required=True)
    parser.add_argument("--tissue_patching", type=str, required=True)
    parser.add_argument("--task_name", type=str, required=True)
    args = parser.parse_args()

    main(**vars(args))
