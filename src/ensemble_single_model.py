#!/usr/bin/env python3
"""
Train a simple meta-learner (LogisticRegression) on a single model's
in-sample meta-features. Used for early fusion hybrid (single fused model
+ meta-learner) evaluation.

Usage:
  python ensemble_single_model.py \\
      --work_dir WORK_DIR \\
      --train_source DATASET \\
      --tissue_patching PATCHING \\
      --task_name TASK \\
      --foundational_model MODEL_NAME \\
      --meta_model logreg
"""

import argparse
import json
import os
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
import torch

sys.path.insert(0, str(Path(__file__).parent))
from utils import Metrics


def load_meta_features(model_dir, fold_k, split="val"):
    """Load meta-features from fold_k."""
    meta_file = f"{model_dir}/fold_{fold_k}/preds.npy"
    labels_file = f"{model_dir}/fold_{fold_k}/labels.npy"

    if not os.path.exists(meta_file) or not os.path.exists(labels_file):
        return None, None

    meta = np.load(meta_file)  # shape (n_samples, 2) or (n_samples,)
    labels = np.load(labels_file)  # shape (n_samples,)

    # Ensure 2D for compatibility
    if meta.ndim == 1:
        meta = meta.reshape(-1, 1)

    return meta, labels


def main(work_dir, train_source, tissue_patching, task_name, foundational_model,
         meta_model="logreg", max_folds=None):
    """
    Train meta-learner on a single foundational model's in-sample predictions.
    """

    # Load split file
    split_file = f"{work_dir}/{train_source}/{task_name}/k=all.tsv"
    df = pd.read_csv(split_file, sep="\t")
    label_col = task_name
    df = df.dropna(subset=[label_col]).reset_index(drop=True)

    # Detect number of folds
    fold_cols = [c for c in df.columns if c.startswith("fold_")]
    n_folds = len(fold_cols)
    if max_folds:
        n_folds = min(n_folds, max_folds)

    print(f"\n{'='*80}")
    print(f"Training meta-learner on: {foundational_model}")
    print(f"Dataset: {train_source} / Task: {task_name}")
    print(f"Meta-model: {meta_model}")
    print(f"Folds: {n_folds}")
    print(f"{'='*80}\n")

    # Directory structure
    model_dir = f"{work_dir}/{train_source}/{task_name}/abmil/{foundational_model}_{tissue_patching}_train_eval/val_outputs"

    if not os.path.isdir(model_dir):
        print(f"ERROR: Meta-features directory not found: {model_dir}")
        sys.exit(1)

    results_subdir = f"{work_dir}/{train_source}/{task_name}/abmil/ensemble4_{foundational_model}_logreg"
    Path(results_subdir).mkdir(parents=True, exist_ok=True)

    print(f"Meta-features dir: {model_dir}")
    print(f"Results subdir: {results_subdir}\n")

    # Iterate over folds
    all_fold_metrics = []

    for fold_k in range(n_folds):
        print(f"\nProcessing fold {fold_k}...")

        # Load in-sample (train) predictions for meta-training
        train_meta, train_labels = load_meta_features(model_dir, fold_k, split="val")

        if train_meta is None:
            print(f"  ⚠ No meta-features found for fold {fold_k}")
            continue

        # Load test predictions
        test_dir = f"{work_dir}/{train_source}/{task_name}/abmil/{foundational_model}_{tissue_patching}/test_outputs"
        test_meta, test_labels = load_meta_features(test_dir, fold_k, split="test")

        if test_meta is None:
            print(f"  ⚠ No test predictions found for fold {fold_k}")
            continue

        print(f"  Train meta-features: {train_meta.shape}, labels: {train_labels.shape}")
        print(f"  Test predictions: {test_meta.shape}, labels: {test_labels.shape}")

        # Train LogisticRegression
        logreg = LogisticRegression(max_iter=1000, random_state=42)
        logreg.fit(train_meta, train_labels)

        # Predict on test
        test_preds = logreg.predict_proba(test_meta)[:, 1]  # Class 1 probability

        # Compute metrics
        from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, f1_score

        auc = roc_auc_score(test_labels, test_preds)
        acc = accuracy_score(test_labels, (test_preds > 0.5).astype(int))
        prec = precision_score(test_labels, (test_preds > 0.5).astype(int), zero_division=0)
        rec = recall_score(test_labels, (test_preds > 0.5).astype(int), zero_division=0)
        f1 = f1_score(test_labels, (test_preds > 0.5).astype(int), zero_division=0)

        print(f"  AUC: {auc:.4f}, Acc: {acc:.4f}, Prec: {prec:.4f}, Rec: {rec:.4f}, F1: {f1:.4f}")

        # Save predictions and metrics
        fold_results_dir = Path(results_subdir) / f"test_metrics" / f"fold_{fold_k}"
        fold_results_dir.mkdir(parents=True, exist_ok=True)

        # Save predictions (using preds.npy for consistency with project standards)
        np.save(fold_results_dir / "preds.npy", test_preds)
        np.save(fold_results_dir / "labels.npy", test_labels)

        # Save metrics
        metrics = {
            "overall": {
                "macro-ovr-auc": float(auc),
                "accuracy": float(acc),
                "precision": float(prec),
                "recall": float(rec),
                "f1": float(f1),
            }
        }

        with open(fold_results_dir / "metrics.json", "w") as f:
            json.dump(metrics, f, indent=2)

        all_fold_metrics.append({
            "fold": fold_k,
            "auc": auc,
            "acc": acc,
            "prec": prec,
            "rec": rec,
            "f1": f1,
        })

    print(f"\n{'='*80}")
    print(f"✓ Meta-learner training complete ({len(all_fold_metrics)} folds)")
    print(f"Results saved to: {results_subdir}")
    print(f"{'='*80}\n")

    # Summary statistics
    if all_fold_metrics:
        aucs = [m["auc"] for m in all_fold_metrics]
        print(f"Summary:")
        print(f"  Mean AUC: {np.mean(aucs):.4f} ± {np.std(aucs):.4f}")
        print(f"  Min AUC: {np.min(aucs):.4f}")
        print(f"  Max AUC: {np.max(aucs):.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--work_dir", required=True)
    parser.add_argument("--train_source", required=True)
    parser.add_argument("--tissue_patching", required=True)
    parser.add_argument("--task_name", required=True)
    parser.add_argument("--foundational_model", required=True)
    parser.add_argument("--meta_model", default="logreg")
    parser.add_argument("--max_folds", type=int, default=None)

    args = parser.parse_args()
    main(**vars(args))
