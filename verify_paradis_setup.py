#!/usr/bin/env python3
"""
Verification script for PARADIS data compatibility with Patho-Ensemble.
Checks if all required data and configurations are in place.
"""

import os
import sys
from pathlib import Path
import h5py
import pandas as pd

PARADIS_PATH = "/shared/home/PARADIS/datos"

class Colors:
    GREEN = '\033[92m'
    RED = '\033[91m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    END = '\033[0m'

def print_success(msg):
    print(f"{Colors.GREEN}✓{Colors.END} {msg}")

def print_error(msg):
    print(f"{Colors.RED}✗{Colors.END} {msg}")

def print_warning(msg):
    print(f"{Colors.YELLOW}⚠{Colors.END} {msg}")

def print_info(msg):
    print(f"{Colors.BLUE}ℹ{Colors.END} {msg}")

def check_paradis_exists():
    """Check if PARADIS directory exists."""
    if os.path.exists(PARADIS_PATH):
        print_success(f"PARADIS directory found: {PARADIS_PATH}")
        return True
    else:
        print_error(f"PARADIS directory NOT found: {PARADIS_PATH}")
        return False

def check_features_available():
    """Check available foundational models."""
    features_path = os.path.join(PARADIS_PATH, "features")
    if not os.path.exists(features_path):
        print_error(f"Features directory not found: {features_path}")
        return False

    print_info("Scanning available foundational models...")
    models = {}

    for dataset in os.listdir(features_path):
        dataset_path = os.path.join(features_path, dataset)
        if not os.path.isdir(dataset_path):
            continue

        for model_dir in os.listdir(dataset_path):
            if model_dir.startswith("features_"):
                model_name = model_dir.replace("features_", "").replace("_monai", "")
                if model_name not in models:
                    models[model_name] = []
                if dataset not in models[model_name]:
                    models[model_name].append(dataset)

    if models:
        print_success(f"Found {len(models)} foundational models:")
        for model in sorted(models.keys()):
            datasets = models[model]
            print(f"  • {model}: {len(datasets)} datasets")
        return True
    else:
        print_error("No foundational models found")
        return False

def check_datasets_available():
    """Check available datasets and tasks."""
    patches_path = os.path.join(PARADIS_PATH, "patches")
    if not os.path.exists(patches_path):
        print_error(f"Patches directory not found: {patches_path}")
        return False

    print_info("Scanning available datasets...")
    datasets = {}

    for dataset in os.listdir(patches_path):
        dataset_path = os.path.join(patches_path, dataset)
        if not os.path.isdir(dataset_path):
            continue

        tasks = []
        for item in os.listdir(dataset_path):
            item_path = os.path.join(dataset_path, item)
            if os.path.isdir(item_path):
                k_all = os.path.join(item_path, "k=all.tsv")
                if os.path.exists(k_all):
                    tasks.append(item)

        if tasks:
            datasets[dataset] = tasks

    if datasets:
        print_success(f"Found {len(datasets)} datasets:")
        for dataset in sorted(datasets.keys()):
            tasks = datasets[dataset]
            print(f"  • {dataset}: {len(tasks)} tasks")
        return True
    else:
        print_error("No datasets found")
        return False

def check_sample_split():
    """Check a sample split file format."""
    sample_split = os.path.join(
        PARADIS_PATH,
        "patches/cptac_brca/TP53_mutation/k=all.tsv"
    )

    if not os.path.exists(sample_split):
        print_warning(f"Sample split not found: {sample_split}")
        return False

    try:
        df = pd.read_csv(sample_split, sep='\t')
        n_rows = len(df)
        n_folds = sum(1 for col in df.columns if col.startswith('fold_'))

        print_success(f"Sample split file valid:")
        print(f"  • {n_rows} samples (slides)")
        print(f"  • {n_folds} folds")
        print(f"  • Columns: {', '.join(df.columns[:4])}...")

        # Check for val splits
        fold_col = 'fold_0' if 'fold_0' in df.columns else None
        if fold_col:
            splits = set(df[fold_col].unique())
            has_val = 'val' in splits
            if has_val:
                print_success("  • Validation splits already present (val/train/test)")
            else:
                print_warning("  • Only train/test splits found (no val)")

        return True
    except Exception as e:
        print_error(f"Error reading split file: {e}")
        return False

def check_sample_embeddings():
    """Check sample embedding files."""
    sample_features = os.path.join(
        PARADIS_PATH,
        "features/cptac_brca/features_ctranspath_monai"
    )

    if not os.path.exists(sample_features):
        print_warning(f"Sample features not found: {sample_features}")
        return False

    h5_files = [f for f in os.listdir(sample_features) if f.endswith('.h5')]

    if not h5_files:
        print_error("No HDF5 files found in features directory")
        return False

    try:
        sample_file = os.path.join(sample_features, h5_files[0])
        with h5py.File(sample_file, 'r') as f:
            if 'features' not in f:
                print_error(f"'features' dataset not found in {h5_files[0]}")
                return False

            shape = f['features'].shape
            dtype = f['features'].dtype

            print_success(f"Sample embedding file valid:")
            print(f"  • File: {h5_files[0]}")
            print(f"  • Shape: {shape} (samples × dimensions)")
            print(f"  • Data type: {dtype}")

            if 'coords' in f:
                coords_shape = f['coords'].shape
                print(f"  • Coordinates: {coords_shape}")

            return True
    except Exception as e:
        print_error(f"Error reading embedding file: {e}")
        return False

def check_patching_strategies():
    """Check available patching strategies."""
    sample_task = os.path.join(
        PARADIS_PATH,
        "patches/cptac_brca/TP53_mutation"
    )

    if not os.path.exists(sample_task):
        print_warning(f"Sample task directory not found: {sample_task}")
        return False

    strategies = []
    for item in os.listdir(sample_task):
        item_path = os.path.join(sample_task, item)
        if os.path.isdir(item_path) and item.startswith('20x'):
            strategies.append(item)

    if strategies:
        print_success(f"Found {len(strategies)} patching strategies:")
        for strategy in sorted(strategies):
            print(f"  • {strategy}")
        return True
    else:
        print_warning("No patching strategies found")
        return False

def main():
    """Run all checks."""
    print(f"\n{Colors.BLUE}{'='*60}")
    print("PARADIS Data Compatibility Check")
    print(f"{'='*60}{Colors.END}\n")

    checks = [
        ("PARADIS Directory", check_paradis_exists),
        ("Foundational Models", check_features_available),
        ("Datasets", check_datasets_available),
        ("Sample Split File", check_sample_split),
        ("Sample Embeddings", check_sample_embeddings),
        ("Patching Strategies", check_patching_strategies),
    ]

    results = {}
    for check_name, check_func in checks:
        print(f"\n{Colors.BLUE}Checking {check_name}...{Colors.END}")
        try:
            results[check_name] = check_func()
        except Exception as e:
            print_error(f"Unexpected error: {e}")
            results[check_name] = False

    # Summary
    print(f"\n{Colors.BLUE}{'='*60}")
    print("Summary")
    print(f"{'='*60}{Colors.END}\n")

    passed = sum(1 for v in results.values() if v)
    total = len(results)

    for check_name, result in results.items():
        status = "PASS" if result else "FAIL"
        color = Colors.GREEN if result else Colors.RED
        print(f"{color}{status}{Colors.END} - {check_name}")

    print(f"\n{Colors.BLUE}Total: {passed}/{total} checks passed{Colors.END}\n")

    if passed == total:
        print_success("✓ All checks passed! Ready to use Patho-Ensemble with PARADIS data.")
        print_info("\nQuick start command:")
        print("  python src/train_abmil.py \\")
        print("    --foundational_model ctranspath \\")
        print("    --latent_dim 768 \\")
        print("    --work_dir /shared/home/PARADIS/datos \\")
        print("    --train_source cptac_brca \\")
        print("    --tissue_patching 20x_224px_0px_overlap \\")
        print("    --task_name TP53_mutation \\")
        print("    --epochs 100 \\")
        print("    --fold all\n")
        return 0
    else:
        print_error(f"✗ {total - passed} check(s) failed. Review errors above.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
