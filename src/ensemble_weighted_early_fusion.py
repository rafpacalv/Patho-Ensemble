#!/usr/bin/env python3
"""
Opción D: Weighted Early Fusion + Nelder-Mead.

Flujo:
1. Para cada fold, optimiza pesos (w_ct, w_vir, w_conch) vía Nelder-Mead
   en predicciones in-sample
2. Aplica pesos a embeddings brutos: embedding_ponderado = w * embedding
3. Concatena embeddings ponderados: [w_ct*ct ⊕ w_vir*vir ⊕ w_conch*conch]
4. Entrena ABMIL único sobre embeddings concatenados ponderados
5. Evalúa en test

Comparación: vs Opción A (Late Fusion + Nelder-Mead)
Hipótesis: Pesos aplicados a embeddings (no predicciones finales) permitirán
a ABMIL aprender mejor cómo combinar la información multimodal.
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
import h5py
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))
from train_abmil import (
    load_features_cached, create_fold_split, save_fold_outputs
)
from utils import get_features_dir, resolve_latent_dim
import abmil_engine


def load_embeddings_for_fold(work_dir, train_source, model_name, fold_k,
                             split_df, label_col, is_train=True):
    """
    Carga embeddings para un fold específico (train o test).

    Returns:
        embeddings_dict: {slide_id: (n_patches, latent_dim)}
        labels: (n_samples,) array con labels
        slide_ids: lista de slide_ids en orden
    """
    feats_dir = get_features_dir(work_dir, train_source, model_name)

    if is_train:
        # Cargar splits
        tr_df, va_df, te_df = create_fold_split(split_df, fold_k, label_col)
        slide_ids = tr_df['slide_id'].tolist()
        labels = tr_df[label_col].values
    else:
        # Test split
        tr_df, va_df, te_df = create_fold_split(split_df, fold_k, label_col)
        slide_ids = te_df['slide_id'].tolist()
        labels = te_df[label_col].values

    # Cargar embeddings cached
    embeddings_dict = load_features_cached(feats_dir, slide_ids)

    return embeddings_dict, labels, slide_ids


def optimize_fold_weights(P_train_dict, y_train, model_names, fold_k, n_init=3):
    """
    Optimiza pesos per-modelo via Nelder-Mead en train_eval.

    Parameters
    ----------
    P_train_dict : dict
        Predicciones in-sample por modelo
    y_train : array
        Labels de train
    model_names : list
        Nombres de modelos
    fold_k : int
        Fold index
    n_init : int
        Número de inicializaciones aleatorias

    Returns
    -------
    w_opt : (n_models,) array
        Pesos óptimos normalizados con softmax
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

    return w_opt


def main(work_dir, train_source, tissue_patching, task_name,
         foundational_models, fold_start=0, fold_end=4, epochs=100, max_folds=None):
    """
    Entrena Weighted Early Fusion con Nelder-Mead para optimizar pesos en embeddings.
    """

    device = "cuda" if torch.cuda.is_available() else "cpu"

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
    print(f"Opción D: Weighted Early Fusion + Nelder-Mead")
    print(f"{'='*80}")
    print(f"Dataset: {train_source} / Task: {task_name}")
    print(f"Models: {foundational_models}")
    print(f"Pesos: Optimizados via Nelder-Mead, aplicados a embeddings brutos")
    print(f"Folds: {fold_start}..{fold_end} (requested), {n_folds} available")
    print(f"{'='*80}\n")

    # Resolver dimensiones latentes
    latent_dims = {
        model: resolve_latent_dim(work_dir, train_source, model)
        for model in foundational_models
    }
    fused_latent_dim = sum(latent_dims.values())
    print(f"Latent dimensions: {latent_dims}")
    print(f"Fused dimension: {fused_latent_dim}\n")

    # Directorio de resultados
    results_subdir = f"{work_dir}/{train_source}/{task_name}/abmil/weighted_early_fusion_{'_'.join(foundational_models)}"
    Path(results_subdir).mkdir(parents=True, exist_ok=True)

    # Iterar sobre folds
    all_fold_metrics = []
    all_weights = []

    for fold_k in range(fold_start, min(fold_end + 1, n_folds)):
        print(f"\n{'='*60}")
        print(f"Fold {fold_k}")
        print(f"{'='*60}")

        # Crear splits
        tr_df, va_df, te_df = create_fold_split(df, fold_k, label_col)

        if tr_df is None or te_df is None or te_df.empty:
            print(f"Fold {fold_k}: no data")
            continue

        tr_stems = tr_df['slide_id'].tolist()
        tr_y = tr_df[label_col].values
        tr_groups = tr_df['case_id'].values

        te_stems = te_df['slide_id'].tolist()
        te_y = te_df[label_col].values

        va_stems = va_df['slide_id'].tolist() if va_df is not None and not va_df.empty else []
        va_y = va_df[label_col].values if va_df is not None and not va_df.empty else np.array([])

        # ==== FASE 1: Optimizar pesos en predicciones in-sample ====
        print(f"  FASE 1: Optimizando pesos en predicciones in-sample...")

        P_train_dict = {}
        for model in foundational_models:
            # Cargar embeddings de train
            tr_feats = load_features_cached(work_dir, train_source, model, tr_stems, latent_dims[model])

            # Entrenar ABMIL temporal en train para obtener predicciones
            model_obj = abmil_engine.train_abmil(
                feats=tr_feats,
                stems=tr_stems,
                y=tr_y,
                groups=tr_groups,
                in_dim=latent_dims[model],
                n_classes=2,
                device=device,
                seed=42 + fold_k,
                max_epochs=20,  # Rápido, solo para obtener predicciones
                patience=3,
                proj_dim=512,
                dropout=0.25,
                wd=1e-4,
                val_frac=0.2,
                graphs=None,
                model_factory=None,
                multi_head_attention=False
            )

            # Predicciones in-sample
            P_train = abmil_engine.abmil_predict(model_obj, tr_feats, tr_stems, device)

            # Normalizar a 1D (probabilidades clase 1)
            if P_train.ndim == 2:
                P_train = P_train[:, 1]

            P_train_dict[model] = P_train

        # Optimizar pesos
        w_opt = optimize_fold_weights(P_train_dict, tr_y, foundational_models, fold_k, n_init=3)
        print(f"  Pesos óptimos: {', '.join(f'{m}={w:.4f}' for m, w in zip(foundational_models, w_opt))}")

        # ==== FASE 2: Aplicar pesos a embeddings brutos, entrenar ABMIL fusionado ====
        print(f"  FASE 2: Aplicando pesos a embeddings, entrenando ABMIL fusionado...")

        # Cargar embeddings de train y test
        feats_dict = {}
        for model in foundational_models:
            feats_dict[model] = load_features_cached(work_dir, train_source, model, tr_stems + te_stems, latent_dims[model])

        # Aplicar pesos y concatenar embeddings de train
        tr_feats_weighted = {}
        for slide_id in tr_stems:
            weighted_embeddings = []
            for model, w_i in zip(foundational_models, w_opt):
                emb = feats_dict[model][slide_id]  # (n_patches, latent_dim)
                weighted_emb = w_i * emb  # Aplicar peso
                weighted_embeddings.append(weighted_emb)

            # Concatenar: (n_patches, fused_dim)
            tr_feats_weighted[slide_id] = np.concatenate(weighted_embeddings, axis=1)

        # Convertir a tensores torch
        tr_feats_torch = {}
        for slide_id, emb in tr_feats_weighted.items():
            tr_feats_torch[slide_id] = torch.from_numpy(emb).float()

        # Entrenar ABMIL sobre embeddings ponderados concatenados
        model_fused = abmil_engine.train_abmil(
            feats=tr_feats_torch,
            stems=tr_stems,
            y=tr_y,
            groups=tr_groups,
            in_dim=fused_latent_dim,
            n_classes=2,
            device=device,
            seed=42 + fold_k,
            max_epochs=epochs,
            patience=8,
            proj_dim=512,
            dropout=0.25,
            wd=1e-4,
            val_frac=0.2,
            graphs=None,
            model_factory=None,
            multi_head_attention=False
        )

        print(f"  Modelo fusionado entrenado (val_auc={model_fused._val_auc:.4f})")

        # ==== FASE 3: Evaluar en test con pesos óptimos ====
        print(f"  FASE 3: Evaluando en test...")

        # Aplicar pesos y concatenar embeddings de test
        te_feats_weighted = {}
        for slide_id in te_stems:
            weighted_embeddings = []
            for model, w_i in zip(foundational_models, w_opt):
                emb = feats_dict[model][slide_id]
                weighted_emb = w_i * emb
                weighted_embeddings.append(weighted_emb)
            te_feats_weighted[slide_id] = np.concatenate(weighted_embeddings, axis=1)

        # Convertir a tensores
        te_feats_torch = {}
        for slide_id, emb in te_feats_weighted.items():
            te_feats_torch[slide_id] = torch.from_numpy(emb).float()

        # Predicciones en test
        te_preds = abmil_engine.abmil_predict(model_fused, te_feats_torch, te_stems, device)

        # Normalizar a 1D (probabilidades clase 1)
        if te_preds.ndim == 2:
            te_preds = te_preds[:, 1]

        # Métricas
        auc_test = roc_auc_score(te_y, te_preds)
        acc = accuracy_score(te_y, (te_preds > 0.5).astype(int))
        prec = precision_score(te_y, (te_preds > 0.5).astype(int), zero_division=0)
        rec = recall_score(te_y, (te_preds > 0.5).astype(int), zero_division=0)
        f1 = f1_score(te_y, (te_preds > 0.5).astype(int), zero_division=0)

        print(f"  Test AUC: {auc_test:.4f}, Acc: {acc:.4f}, Prec: {prec:.4f}, Rec: {rec:.4f}, F1: {f1:.4f}")

        # Guardar resultados
        fold_results_dir = Path(results_subdir) / "test_metrics" / f"fold_{fold_k}"
        fold_results_dir.mkdir(parents=True, exist_ok=True)

        np.save(fold_results_dir / "preds.npy", te_preds)
        np.save(fold_results_dir / "labels.npy", te_y)

        metrics = {
            "overall": {
                "macro-ovr-auc": float(auc_test),
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
            "auc": auc_test,
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
    print(f"✓ Weighted Early Fusion complete ({len(all_fold_metrics)} folds)")
    print(f"Results saved to: {results_subdir}")
    print(f"{'='*80}\n")

    # Summary
    if all_fold_metrics:
        aucs = [m["auc"] for m in all_fold_metrics]

        print("Summary (Test):")
        print(f"  Mean AUC: {np.mean(aucs):.4f} ± {np.std(aucs):.4f}")
        print(f"  Min AUC: {np.min(aucs):.4f}")
        print(f"  Max AUC: {np.max(aucs):.4f}")

        print("\nWeight Statistics:")
        for model in foundational_models:
            weights = [w[model] for w in all_weights]
            print(f"  {model}: {np.mean(weights):.4f} ± {np.std(weights):.4f} (range: [{np.min(weights):.4f}, {np.max(weights):.4f}])")

        # Guardar summary
        with open(Path(results_subdir) / "summary.json", "w") as f:
            json.dump({
                "n_folds": len(all_fold_metrics),
                "test_auc_mean": float(np.mean(aucs)),
                "test_auc_std": float(np.std(aucs)),
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
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--max_folds", type=int, default=None)

    args = parser.parse_args()
    main(**vars(args))
