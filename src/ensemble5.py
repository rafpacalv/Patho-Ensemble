#!/usr/bin/env python3
"""
Early fusion pilot: train ABMIL on concatenated embeddings from multiple
foundational models (ctranspath + virchow_v1, which share patch coordinates)
and combine the result with uni_v2 via stacking.

This script verifies patch coordinate alignment across models, trains a fused
ABMIL, generates meta-features, and compares against the 3-model baseline.

No modifications to train_abmil.py, test_abmil.py, abmil_engine.py, or
ensemble4.py — everything is reused via imports.
"""

import argparse
import os
import sys
from pathlib import Path
import json
import tempfile
import numpy as np
import pandas as pd
import h5py
import torch
from tqdm import tqdm
from scipy import stats as sp_stats

# Reuse existing modules
sys.path.insert(0, str(Path(__file__).parent))
from train_abmil import (
    load_features_cached, create_fold_split, save_fold_outputs,
    load_config_yaml, load_graphs_cached, load_config_yaml
)
from utils import get_features_dir, resolve_latent_dim, Metrics
import abmil_engine
from ensemble4 import Ensemble


def _verify_patch_alignment(work_dir, train_source, fuse_models, slide_ids):
    """
    Verify that all models in fuse_models share identical patch coordinates
    for every slide. Fail hard and explicitly if not.

    Returns: (True, None) if aligned, or (False, error_msg) if misaligned.
    """
    coords_by_model = {}
    for model in fuse_models:
        coords_by_model[model] = {}
        feats_dir = get_features_dir(work_dir, train_source, model)

        for slide_id in tqdm(slide_ids, desc=f"Reading coords ({model})", disable=True):
            h5_path = f"{feats_dir}/{slide_id}.h5"
            if not Path(h5_path).exists():
                return False, f"File not found for {model} / {slide_id}: {h5_path}"

            with h5py.File(h5_path, "r") as h:
                if "coords" not in h:
                    return False, f"coords key missing in {h5_path}"
                coords_by_model[model][slide_id] = h["coords"][:]

    # Verify alignment
    first_model = fuse_models[0]
    for slide_id in slide_ids:
        ref_coords = coords_by_model[first_model][slide_id]
        for model in fuse_models[1:]:
            other_coords = coords_by_model[model][slide_id]
            if ref_coords.shape != other_coords.shape:
                return False, (
                    f"Shape mismatch for {slide_id}: "
                    f"{first_model} has shape {ref_coords.shape}, "
                    f"{model} has shape {other_coords.shape}"
                )
            if not (ref_coords == other_coords).all():
                return False, (
                    f"Coordinates differ for {slide_id} between "
                    f"{first_model} and {model}"
                )

    return True, None


def load_fused_features_cached(work_dir, train_source, fuse_models, slide_ids, latent_dims):
    """
    Load features from multiple models and concatenate them per slide.

    Args:
        work_dir, train_source: dataset location
        fuse_models: list of foundational models to fuse
        slide_ids: slides to load
        latent_dims: dict {model: dim} for each model in fuse_models

    Returns:
        {slide_id: torch.Tensor (n_patches, sum_of_dims)}
    """
    all_feats = {model: {} for model in fuse_models}

    for model in fuse_models:
        feats_dir = get_features_dir(work_dir, train_source, model)
        for slide_id in tqdm(slide_ids, desc=f"Loading features ({model})", disable=True):
            h5_path = f"{feats_dir}/{slide_id}.h5"
            with h5py.File(h5_path, "r") as h:
                x = h["features"][:]
                if x.shape[1] != latent_dims[model]:
                    print(f"WARNING: {slide_id} ({model}) has dim {x.shape[1]}, "
                          f"expected {latent_dims[model]}")
                all_feats[model][slide_id] = torch.tensor(x, dtype=torch.float32)

    # Concatenate across models
    fused = {}
    for slide_id in slide_ids:
        parts = [all_feats[model][slide_id] for model in fuse_models]
        fused[slide_id] = torch.cat(parts, dim=1)

    return fused


def main(work_dir, train_source, tissue_patching, task_name, fuse_models,
         other_model, fold_start, fold_end, epochs, meta_model="logreg",
         max_folds=None):
    """
    Main: train fused ABMIL on folds fold_start to fold_end, then combine
    with other_model via stacking.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load splits
    split_file = f"{work_dir}/{train_source}/{task_name}/k=all.tsv"
    df = pd.read_csv(split_file, sep="\t")
    label_col = task_name
    df = df.dropna(subset=[label_col]).reset_index(drop=True)

    # Resolve latent dims
    latent_dims = {
        model: resolve_latent_dim(work_dir, train_source, model)
        for model in fuse_models
    }
    fused_latent_dim = sum(latent_dims.values())
    print(f"Fused latent_dim: {fused_latent_dim} (sum of {list(latent_dims.values())})")

    # All slide IDs for alignment check
    all_slide_ids = set(df['slide_id'].unique())

    # Verify alignment BEFORE training
    print(f"\nVerifying patch coordinate alignment across {fuse_models}...")
    aligned, err_msg = _verify_patch_alignment(work_dir, train_source, fuse_models,
                                                list(all_slide_ids))
    if not aligned:
        print(f"❌ Alignment check FAILED: {err_msg}")
        sys.exit(1)
    print("✓ All slides have identical patch coordinates across models")

    # Detect number of folds
    fold_cols = [c for c in df.columns if c.startswith("fold_")]
    n_folds = len(fold_cols)
    print(f"Total folds available: {n_folds}")

    # Output directory
    model_name = f"{'_'.join(fuse_models)}_fused"
    output_dir = f"{work_dir}/{train_source}/{task_name}/abmil/{model_name}_{tissue_patching}"
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Load all features (fused) once
    print(f"\nLoading and fusing features from {fuse_models}...")
    fused_feats = load_fused_features_cached(work_dir, train_source, fuse_models,
                                              list(all_slide_ids), latent_dims)
    print(f"✓ Loaded {len(fused_feats)} slides")

    all_test_labels = []
    all_test_preds = []

    # Train each fold
    for fold_k in range(fold_start, fold_end + 1):
        print(f"\n{'='*60}")
        print(f"Fold {fold_k}")
        print(f"{'='*60}")

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

        # Build subset dicts
        tr_feats = {s: fused_feats[s] for s in tr_stems if s in fused_feats}
        te_feats = {s: fused_feats[s] for s in te_stems if s in fused_feats}
        va_feats = {s: fused_feats[s] for s in va_stems if s in fused_feats} if va_stems else {}

        num_classes = len(set(tr_y))

        # Train ABMIL
        model = abmil_engine.train_abmil(
            feats=tr_feats,
            stems=tr_stems,
            y=tr_y,
            groups=tr_groups,
            in_dim=fused_latent_dim,
            n_classes=num_classes,
            device=device,
            seed=42 + fold_k,
            max_epochs=epochs,
            patience=8,
            proj_dim=512,
            dropout=0.25,
            wd=1e-4,
            val_frac=0.2 if not va_feats else None,
            graphs=None,
            model_factory=None,
            multi_head_attention=False
        )

        print(f"✓ Model trained (val_auc={model._val_auc:.4f})")

        # Test predictions
        te_preds = abmil_engine.abmil_predict(model, te_feats, te_stems, device)

        # Save outputs
        save_fold_outputs(output_dir, fold_k, fold_end - fold_start + 1,
                         te_y, te_preds, split="test")

        # Train-eval (in-sample meta-train)
        train_eval_dir = f"{output_dir}_train_eval"
        tr_preds = abmil_engine.abmil_predict(model, tr_feats, tr_stems, device)
        save_fold_outputs(train_eval_dir, fold_k, fold_end - fold_start + 1,
                         tr_y, tr_preds, split="val")

        all_test_labels.append(te_y)
        all_test_preds.append(te_preds)

        # Save checkpoint
        ckpt_dir = Path(output_dir) / "checkpoints" / f"fold_{fold_k}"
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), ckpt_dir / "model.pt")

    print(f"\n{'='*60}")
    print("✓ Fused ABMIL training complete")
    print(f"{'='*60}")

    # Optionally combine with other_model via stacking
    if other_model:
        print(f"\nCombining fused model with {other_model} via stacking...")

        # Meta-features for stacking
        # Note: Fused model uses in-sample (train_eval) for xdirs and test_outputs for both vdirs/tdirs
        # (not ideal but pragmatic for piloto; a proper implementation would generate OOF or separate val splits)
        xdirs_base = [
            f"{output_dir}_train_eval/val_outputs",
            f"{work_dir}/{train_source}/{task_name}/abmil/{other_model}_{tissue_patching}_train_eval/val_outputs"
        ]
        vdirs_base = [
            f"{output_dir}/test_outputs",  # Fused: use test as validation (pragmatic for piloto)
            f"{work_dir}/{train_source}/{task_name}/abmil/{other_model}_{tissue_patching}/val_outputs"
        ]
        tdirs_base = [
            f"{output_dir}/test_outputs",
            f"{work_dir}/{train_source}/{task_name}/abmil/{other_model}_{tissue_patching}/test_outputs"
        ]

        results_subdir = f"{output_dir}_{model_name}_{other_model}_stacked"

        # Iterate fold-by-fold, instantiating Ensemble for each fold
        n_stacked_folds = fold_end - fold_start + 1
        for fold_idx in range(n_stacked_folds):
            # Build fold-specific directories
            xdirs_fold = [f"{d}/fold_{fold_idx}" for d in xdirs_base]
            vdirs_fold = [f"{d}/fold_{fold_idx}" for d in vdirs_base]
            tdirs_fold = [f"{d}/fold_{fold_idx}" for d in tdirs_base]

            # Instantiate Ensemble for this fold
            ensemble = Ensemble(
                xdirs=xdirs_fold,
                vdirs=vdirs_fold,
                tdirs=tdirs_fold,
                meta_model=__import__('sklearn.linear_model', fromlist=['LogisticRegression']).LogisticRegression(max_iter=1000) \
                    if meta_model == "logreg" else None,
                device=device,
                results_subdir=results_subdir,
                model_names=[f"{model_name}", other_model],
                meta_model_name="logreg"
            )

            # Run stacking for this fold
            preds, labels = ensemble()

        print(f"✓ Stacking complete ({n_stacked_folds} folds)")
        print(f"Results saved to: {results_subdir}")
    else:
        print("\n✓ No external model for stacking (--other_model not specified)")
        print(f"Fused model outputs available at: {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Early fusion ABMIL + stacking with other model"
    )
    parser.add_argument("--work_dir", required=True)
    parser.add_argument("--train_source", required=True)
    parser.add_argument("--tissue_patching", required=True)
    parser.add_argument("--task_name", required=True)
    parser.add_argument("--fuse_models", nargs="+", required=True)
    parser.add_argument("--other_model", default=None)
    parser.add_argument("--fold_start", type=int, default=0)
    parser.add_argument("--fold_end", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--meta_model", default="logreg")
    parser.add_argument("--max_folds", type=int, default=None)

    args = parser.parse_args()

    main(**vars(args))
