#!/usr/bin/env python3
"""
Opción A: Late fusion con optimización de pesos via Nelder-Mead.

En lugar de aprender pesos implícitamente con LogReg, optimiza explícitamente
la combinación lineal de predicciones de múltiples modelos:

  P_final = w_ct * P_ctranspath + w_vir * P_virchow + w_conch * P_conch

La optimización minimiza -AUC en las predicciones in-sample (train_eval),
evaluando los pesos óptimos en test.

Compara directamente contra Late Fusion + LogReg baseline.
"""

import argparse
import json
import os
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, f1_score
from scipy.optimize import minimize
import torch

sys.path.insert(0, str(Path(__file__).parent))
from utils import Metrics


def load_fold_predictions(work_dir, train_source, tissue_patching, task_name,
                          model_name, fold_k, split="val"):
    """
    Carga predicciones de un modelo para un fold específico.

    Retorna:
      preds: (n_samples,) array con probabilidades clase 1
      labels: (n_samples,) array con labels
    """
    if split == "val":
        # in-sample (train_eval)
        pred_dir = f"{work_dir}/{train_source}/{task_name}/abmil/{model_name}_{tissue_patching}_train_eval/val_outputs/fold_{fold_k}"
    else:
        # test
        pred_dir = f"{work_dir}/{train_source}/{task_name}/abmil/{model_name}_{tissue_patching}/test_outputs/fold_{fold_k}"

    pred_file = f"{pred_dir}/preds.npy"
    label_file = f"{pred_dir}/labels.npy"

    if not os.path.exists(pred_file) or not os.path.exists(label_file):
        return None, None

    preds = np.load(pred_file)  # shape (n,) o (n, 2)
    labels = np.load(label_file)  # shape (n,)

    # Si es (n, 2), extraer probabilidad de clase 1
    if preds.ndim == 2:
        preds = preds[:, 1]

    return preds, labels


def optimize_fold_weights(P_train_dict, y_train, P_test_dict, y_test,
                          model_names, fold_k, n_init=3):
    """
    Optimiza pesos per-modelo via Nelder-Mead en train_eval,
    evalúa en test, retorna (w_opt, auc_test, preds_test, auc_train).

    Parameters
    ----------
    P_train_dict : dict
        Dict de arrays (n_train,) con predicciones por modelo
    y_train : (n_train,) array
        Labels de train
    P_test_dict : dict
        Dict de arrays (n_test,) con predicciones por modelo
    y_test : (n_test,) array
        Labels de test
    model_names : list
        Nombres de modelos (mismo orden en dicts)
    fold_k : int
        Fold index (para logging)
    n_init : int
        Número de inicializaciones aleatorias a probar

    Returns
    -------
    w_opt : (n_models,) array
        Pesos óptimos (sumados a 1)
    auc_test : float
        AUC en test con pesos óptimos
    preds_test : (n_test,) array
        Predicciones test combinadas
    auc_train : float
        AUC en train_eval con pesos óptimos (para diagnosticar overfitting)
    """

    def objective(w_raw):
        """Función objetivo: -AUC en train_eval."""
        # Softmax para garantizar [0,1] y Σ=1
        w = np.exp(w_raw) / np.exp(w_raw).sum()

        # Predicción combinada
        P_cand = np.zeros_like(y_train, dtype=float)
        for model, w_i in zip(model_names, w):
            P_cand += w_i * P_train_dict[model]

        # AUC en train_eval
        auc = roc_auc_score(y_train, P_cand)
        return -auc  # minimizar -AUC

    # Probar múltiples inicializaciones
    best_result = None
    best_obj = np.inf

    for init_idx in range(n_init):
        if init_idx == 0:
            # Inicialización 1: pesos iguales
            w_init = np.log(np.ones(len(model_names)) / len(model_names))
        else:
            # Inicializaciones aleatorias
            w_rand = np.random.dirichlet(np.ones(len(model_names)))
            w_init = np.log(w_rand)

        # Optimizar
        result = minimize(
            objective, w_init, method='Nelder-Mead',
            options={'maxiter': 1000, 'xatol': 1e-5, 'fatol': 1e-5}
        )

        if result.fun < best_obj:
            best_obj = result.fun
            best_result = result

    # Pesos óptimos
    w_opt = np.exp(best_result.x) / np.exp(best_result.x).sum()

    # Evaluar en train_eval
    P_train_final = np.zeros_like(y_train, dtype=float)
    for model, w_i in zip(model_names, w_opt):
        P_train_final += w_i * P_train_dict[model]
    auc_train = roc_auc_score(y_train, P_train_final)

    # Evaluar en test
    P_test_final = np.zeros_like(y_test, dtype=float)
    for model, w_i in zip(model_names, w_opt):
        P_test_final += w_i * P_test_dict[model]
    auc_test = roc_auc_score(y_test, P_test_final)

    return w_opt, auc_test, P_test_final, auc_train


def main(work_dir, train_source, tissue_patching, task_name,
         foundational_models, fold_start=0, fold_end=4, max_folds=None):
    """
    Entrena meta-learner Nelder-Mead en late fusion para todos los folds.
    """

    # Load split file
    split_file = f"{work_dir}/{train_source}/{task_name}/k=all.tsv"
    df = pd.read_csv(split_file, sep="\t")
    label_col = task_name
    df = df.dropna(subset=[label_col]).reset_index(drop=True)

    # Detectar número de folds
    fold_cols = [c for c in df.columns if c.startswith("fold_")]
    n_folds = len(fold_cols)
    if max_folds:
        n_folds = min(n_folds, max_folds)

    print(f"\n{'='*80}")
    print(f"Opción A: Late Fusion + Nelder-Mead Weight Optimization")
    print(f"{'='*80}")
    print(f"Dataset: {train_source} / Task: {task_name}")
    print(f"Models: {foundational_models}")
    print(f"Folds: {fold_start}..{fold_end} (requested), {n_folds} available")
    print(f"{'='*80}\n")

    # Directorio de resultados
    results_subdir = f"{work_dir}/{train_source}/{task_name}/abmil/ensemble4_nelder_mead_{'_'.join(foundational_models)}"
    Path(results_subdir).mkdir(parents=True, exist_ok=True)

    # Iterar sobre folds
    all_fold_metrics = []
    all_weights = []

    for fold_k in range(fold_start, min(fold_end + 1, n_folds)):
        print(f"\n{'='*60}")
        print(f"Fold {fold_k}")
        print(f"{'='*60}")

        # Cargar predicciones in-sample (train_eval) para cada modelo
        P_train_dict = {}
        for model in foundational_models:
            P_train, y_train = load_fold_predictions(
                work_dir, train_source, tissue_patching, task_name,
                model, fold_k, split="val"
            )
            if P_train is None:
                print(f"  ⚠ No in-sample predictions for {model}")
                P_train_dict = None
                break
            P_train_dict[model] = P_train

        if P_train_dict is None:
            print(f"  ⚠ Skipping fold {fold_k}: missing in-sample data")
            continue

        # Cargar predicciones test para cada modelo
        P_test_dict = {}
        for model in foundational_models:
            P_test, y_test = load_fold_predictions(
                work_dir, train_source, tissue_patching, task_name,
                model, fold_k, split="test"
            )
            if P_test is None:
                print(f"  ⚠ No test predictions for {model}")
                P_test_dict = None
                break
            P_test_dict[model] = P_test

        if P_test_dict is None:
            print(f"  ⚠ Skipping fold {fold_k}: missing test data")
            continue

        print(f"  Train (in-sample): {len(y_train)} samples")
        print(f"  Test: {len(y_test)} samples")

        # Optimizar pesos
        w_opt, auc_test, preds_test, auc_train = optimize_fold_weights(
            P_train_dict, y_train, P_test_dict, y_test,
            foundational_models, fold_k, n_init=3
        )

        # Métricas adicionales
        acc = accuracy_score(y_test, (preds_test > 0.5).astype(int))
        prec = precision_score(y_test, (preds_test > 0.5).astype(int), zero_division=0)
        rec = recall_score(y_test, (preds_test > 0.5).astype(int), zero_division=0)
        f1 = f1_score(y_test, (preds_test > 0.5).astype(int), zero_division=0)

        print(f"  Weights: {', '.join(f'{m}={w:.4f}' for m, w in zip(foundational_models, w_opt))}")
        print(f"  Train AUC (in-sample): {auc_train:.4f}")
        print(f"  Test AUC: {auc_test:.4f}, Acc: {acc:.4f}, Prec: {prec:.4f}, Rec: {rec:.4f}, F1: {f1:.4f}")

        # Guardar predicciones y métricas
        fold_results_dir = Path(results_subdir) / "test_metrics" / f"fold_{fold_k}"
        fold_results_dir.mkdir(parents=True, exist_ok=True)

        np.save(fold_results_dir / "preds.npy", preds_test)
        np.save(fold_results_dir / "labels.npy", y_test)

        metrics = {
            "overall": {
                "macro-ovr-auc": float(auc_test),
                "train_auc": float(auc_train),
                "accuracy": float(acc),
                "precision": float(prec),
                "recall": float(rec),
                "f1": float(f1),
            },
            "weights": {model: float(w) for model, w in zip(foundational_models, w_opt)}
        }

        with open(fold_results_dir / "metrics.json", "w") as f:
            json.dump(metrics, f, indent=2)

        all_fold_metrics.append({
            "fold": fold_k,
            "test_auc": auc_test,
            "train_auc": auc_train,
            "acc": acc,
            "prec": prec,
            "rec": rec,
            "f1": f1,
        })

        all_weights.append({
            "fold": fold_k,
            **{model: float(w) for model, w in zip(foundational_models, w_opt)}
        })

    print(f"\n{'='*80}")
    print(f"✓ Nelder-Mead optimization complete ({len(all_fold_metrics)} folds)")
    print(f"Results saved to: {results_subdir}")
    print(f"{'='*80}\n")

    # Summary statistics
    if all_fold_metrics:
        test_aucs = [m["test_auc"] for m in all_fold_metrics]
        train_aucs = [m["train_auc"] for m in all_fold_metrics]

        print("Summary (Test):")
        print(f"  Mean AUC: {np.mean(test_aucs):.4f} ± {np.std(test_aucs):.4f}")
        print(f"  Min AUC: {np.min(test_aucs):.4f}")
        print(f"  Max AUC: {np.max(test_aucs):.4f}")

        print("\nSummary (Train in-sample):")
        print(f"  Mean AUC: {np.mean(train_aucs):.4f} ± {np.std(train_aucs):.4f}")
        print(f"  Mean overfitting: {(np.mean(train_aucs) - np.mean(test_aucs)):.4f}")

        # Estadísticas de pesos
        print("\nWeight Statistics:")
        for model in foundational_models:
            weights = [w[model] for w in all_weights]
            print(f"  {model}: {np.mean(weights):.4f} ± {np.std(weights):.4f} (range: [{np.min(weights):.4f}, {np.max(weights):.4f}])")

        # Guardar summary
        with open(Path(results_subdir) / "summary.json", "w") as f:
            json.dump({
                "n_folds": len(all_fold_metrics),
                "test_auc_mean": float(np.mean(test_aucs)),
                "test_auc_std": float(np.std(test_aucs)),
                "train_auc_mean": float(np.mean(train_aucs)),
                "train_auc_std": float(np.std(train_aucs)),
                "mean_overfitting": float(np.mean(train_aucs) - np.mean(test_aucs)),
            }, f, indent=2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--work_dir", required=True)
    parser.add_argument("--train_source", required=True)
    parser.add_argument("--tissue_patching", required=True)
    parser.add_argument("--task_name", required=True)
    parser.add_argument("--foundational_models", nargs="+", required=True)
    parser.add_argument("--fold_start", type=int, default=0)
    parser.add_argument("--fold_end", type=int, default=4)
    parser.add_argument("--max_folds", type=int, default=None)

    args = parser.parse_args()
    main(**vars(args))
