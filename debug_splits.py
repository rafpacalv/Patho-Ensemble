#!/usr/bin/env python3
"""
Diagnostic script to check split files for empty folds and missing data.

Usage:
    python debug_splits.py \
        --work_dir /shared/home/PARADIS/datos \
        --train_source cptac_brca \
        --task_name TP53_mutation
"""
import argparse
import os
import sys
import pandas as pd
from pathlib import Path
import h5py

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))
from utils import get_features_dir


def check_splits(work_dir, train_source, task_name, foundational_model=None):
    """Check if all folds have data and feature files exist."""
    split_file = f"{work_dir}/{train_source}/{task_name}/k=all.tsv"

    if not Path(split_file).exists():
        print(f"❌ Split file not found: {split_file}")
        return

    df = pd.read_csv(split_file, sep="\t")
    fold_cols = [c for c in df.columns if c.startswith("fold_")]

    print(f"\n📋 Split File Analysis: {split_file}")
    print(f"Total columns: {len(df.columns)}")
    print(f"Total rows: {len(df)}")
    print(f"Number of folds: {len(fold_cols)}\n")

    # Check each fold
    empty_folds = []
    for fold_k in range(len(fold_cols)):
        fold_col = f"fold_{fold_k}"

        # Count samples per split
        n_train = len(df[df[fold_col] == "train"])
        n_val = len(df[df[fold_col] == "val"])
        n_test = len(df[df[fold_col] == "test"])

        status = "✅" if n_test > 0 else "❌"
        print(f"{status} Fold {fold_k:2d}: train={n_train:3d}, val={n_val:3d}, test={n_test:3d}")

        if n_test == 0:
            empty_folds.append(fold_k)

    if empty_folds:
        print(f"\n⚠️  Empty test folds: {empty_folds}")
    else:
        print(f"\n✅ All folds have test data")

    # Check if feature files exist (if model specified)
    if foundational_model:
        check_features(work_dir, train_source, task_name, foundational_model, df)


def check_features(work_dir, train_source, task_name, foundational_model, df):
    """Check if feature files exist for all slides."""
    print(f"\n🔍 Checking features for {foundational_model}:")

    try:
        feats_dir = get_features_dir(work_dir, train_source, foundational_model)
    except FileNotFoundError as e:
        print(f"❌ Feature directory not found!\n{e}")
        return

    print(f"Feature directory: {feats_dir}")

    # Count existing feature files
    h5_files = set(f.replace('.h5', '') for f in os.listdir(feats_dir) if f.endswith('.h5'))
    print(f"Found {len(h5_files)} feature files")

    # Check if all slides have features
    missing_slides = []
    for slide_id in df['slide_id'].unique():
        if slide_id not in h5_files:
            missing_slides.append(slide_id)

    if missing_slides:
        print(f"❌ Missing {len(missing_slides)} feature files:")
        for sid in missing_slides[:5]:
            print(f"   - {sid}.h5")
        if len(missing_slides) > 5:
            print(f"   ... and {len(missing_slides) - 5} more")
    else:
        print(f"✅ All {len(df['slide_id'].unique())} slides have feature files")

    # Check feature dimensions (sample one file)
    if h5_files:
        sample_file = list(h5_files)[0]
        sample_path = os.path.join(feats_dir, f"{sample_file}.h5")
        try:
            with h5py.File(sample_path, 'r') as f:
                shape = f['features'].shape
                print(f"✅ Sample feature shape: {shape}")
        except Exception as e:
            print(f"❌ Error reading sample file: {e}")


def main():
    parser = argparse.ArgumentParser(description="Check split files and feature data")
    parser.add_argument("--work_dir", type=str, required=True)
    parser.add_argument("--train_source", type=str, required=True)
    parser.add_argument("--task_name", type=str, required=True)
    parser.add_argument("--foundational_model", type=str, default=None,
                       help="Optional: check features for specific model")

    args = parser.parse_args()

    print("=" * 60)
    print("PARADIS Data Validation")
    print("=" * 60)

    check_splits(args.work_dir, args.train_source, args.task_name, args.foundational_model)

    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()
