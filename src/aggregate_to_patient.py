#!/usr/bin/env python3
"""
Opción B: Per-Patient Aggregation

Convert per-slide predictions to per-patient predictions by grouping
slides that belong to the same patient (case_id) and averaging predictions.

This is a post-processing step that doesn't require retraining ABMIL models.
It works on existing test_outputs and generates patient_outputs.

Hypothesis: Aggregating multiple slides per patient recovers information loss
from per-model ABMIL collapse, potentially improving ensemble performance.
"""

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))
from utils import Metrics


def aggregate_to_patient(split_df, preds, labels, slide_ids):
    """
    Aggregate per-slide predictions to per-patient level.

    Parameters
    ----------
    split_df : pd.DataFrame
        DataFrame with columns: case_id, slide_id, {label_col}
    preds : np.ndarray
        Shape (n_slides,), per-slide predictions (probabilities)
    labels : np.ndarray
        Shape (n_slides,), per-slide labels (binary)
    slide_ids : list
        List of slide_ids in order corresponding to preds/labels

    Returns
    -------
    patient_preds : np.ndarray
        Shape (n_patients,), aggregated predictions (mean pooling)
    patient_labels : np.ndarray
        Shape (n_patients,), patient-level labels
    patient_ids : list
        List of case_ids (patient IDs)
    """

    # Create mapping slide_id → index
    slide_to_idx = {sid: i for i, sid in enumerate(slide_ids)}

    patient_preds = []
    patient_labels = []
    patient_ids = []

    # Group by patient
    for case_id, group in split_df.groupby('case_id', sort=False):
        slides_in_patient = group['slide_id'].tolist()

        # Get indices of slides for this patient that are in this fold
        indices = [slide_to_idx[sid] for sid in slides_in_patient
                   if sid in slide_to_idx]

        if not indices:
            # Patient has no slides in this fold
            continue

        # Aggregate: mean pooling of predictions
        patient_pred = preds[indices].mean()

        # Patient label: mode of slide labels (all should be same)
        patient_label = labels[indices].astype(int).mode()[0]

        patient_preds.append(patient_pred)
        patient_labels.append(patient_label)
        patient_ids.append(case_id)

    return np.array(patient_preds), np.array(patient_labels), patient_ids


def main(work_dir, train_source, tissue_patching, task_name,
         foundational_models, fold_start=0, fold_end=49, max_folds=None):
    """
    Aggregate per-slide predictions to per-patient for all folds.
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
    print(f"Opción B: Per-Patient Aggregation")
    print(f"{'='*80}")
    print(f"Dataset: {train_source} / Task: {task_name}")
    print(f"Models: {foundational_models}")
    print(f"Folds: {fold_start}..{fold_end} (requested), {n_folds} available")
    print(f"{'='*80}\n")

    # Create output directory
    output_base = f"{work_dir}/{train_source}/{task_name}/abmil"
    Path(output_base).mkdir(parents=True, exist_ok=True)

    # Iterate over folds and models
    all_stats = {}

    for fold_k in tqdm(range(fold_start, min(fold_end + 1, n_folds)),
                       desc="Processing folds"):

        # Create fold split
        fold_col = f"fold_{fold_k}"
        tr_df = df[df[fold_col] == 'train'].copy()
        te_df = df[df[fold_col] == 'test'].copy()

        if te_df.empty:
            print(f"  ⚠️  Fold {fold_k}: no test data, skipping")
            continue

        # For each model, aggregate test predictions
        for model in foundational_models:
            test_pred_dir = f"{output_base}/{model}_{tissue_patching}/test_outputs/fold_{fold_k}"
            patient_output_dir = f"{output_base}/{model}_{tissue_patching}/patient_outputs/fold_{fold_k}"

            # Check if source exists
            if not os.path.exists(f"{test_pred_dir}/preds.npy"):
                print(f"  ⚠️  {model} fold {fold_k}: no predictions found")
                continue

            # Load per-slide predictions
            preds = np.load(f"{test_pred_dir}/preds.npy")
            labels = np.load(f"{test_pred_dir}/labels.npy")

            # Normalize predictions to 1D if needed (handle 2D output)
            if preds.ndim == 2:
                preds = preds[:, 1]

            # Get slide IDs from test set (in order they appear in predictions)
            slide_ids = te_df['slide_id'].tolist()

            # Aggregate to patient level
            # Note: labels is numpy array, convert to pandas for mode() method
            labels_series = pd.Series(labels)
            patient_preds, patient_labels, patient_ids = aggregate_to_patient(
                te_df, preds, labels_series, slide_ids
            )

            # Create output directory
            Path(patient_output_dir).mkdir(parents=True, exist_ok=True)

            # Save aggregated predictions
            np.save(f"{patient_output_dir}/preds.npy", patient_preds)
            np.save(f"{patient_output_dir}/labels.npy", patient_labels)

            # Save patient IDs for reference
            with open(f"{patient_output_dir}/patient_ids.txt", "w") as f:
                f.write("\n".join(patient_ids))

            # Statistics
            n_slides = len(slide_ids)
            n_patients = len(patient_ids)
            avg_slides_per_patient = n_slides / n_patients if n_patients > 0 else 0

            key = f"{model}_fold{fold_k}"
            all_stats[key] = {
                "n_slides": int(n_slides),
                "n_patients": int(n_patients),
                "avg_slides_per_patient": float(avg_slides_per_patient),
                "accuracy": float(np.mean((preds > 0.5).astype(int) == labels)),  # Rough accuracy
            }

    print(f"\n{'='*80}")
    print(f"✓ Per-patient aggregation complete")
    print(f"Results saved to: {output_base}/<model>_{tissue_patching}/patient_outputs/")
    print(f"{'='*80}\n")

    # Summary statistics
    if all_stats:
        total_slides = sum(s["n_slides"] for s in all_stats.values())
        total_patients = sum(s["n_patients"] for s in all_stats.values())
        avg_slides = total_slides / total_patients if total_patients > 0 else 0

        print("Summary:")
        print(f"  Total slides processed: {total_slides}")
        print(f"  Total patients processed: {total_patients}")
        print(f"  Average slides per patient: {avg_slides:.2f}")

        # Save summary
        with open(f"{output_base}/patient_aggregation_summary.json", "w") as f:
            json.dump(all_stats, f, indent=2)

        print(f"  Statistics saved to: {output_base}/patient_aggregation_summary.json")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Aggregate per-slide predictions to per-patient level"
    )
    parser.add_argument("--work_dir", required=True,
                       help="Root working directory (must end in /patches)")
    parser.add_argument("--train_source", required=True,
                       help="Dataset name (e.g., cptac_brca)")
    parser.add_argument("--tissue_patching", required=True,
                       help="Patching strategy (e.g., 20x_224px_0px_overlap)")
    parser.add_argument("--task_name", required=True,
                       help="Task name (e.g., TP53_mutation)")
    parser.add_argument("--foundational_models", nargs="+", required=True,
                       help="List of foundational models")
    parser.add_argument("--fold_start", type=int, default=0,
                       help="Starting fold (default: 0)")
    parser.add_argument("--fold_end", type=int, default=49,
                       help="Ending fold (default: 49)")
    parser.add_argument("--max_folds", type=int, default=None,
                       help="Maximum number of folds to process")

    args = parser.parse_args()
    main(**vars(args))
