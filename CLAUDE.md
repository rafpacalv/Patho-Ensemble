# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**Patho-Ensemble** is a machine learning pipeline for training and evaluating foundational model ensembles for pathology image classification. It uses Attention-Based Multiple Instance Learning (ABMIL) with patch embeddings from various foundational models (ViT, CTransPath, Virchow, UNI, etc.) and can combine them using weighted averaging or meta-learning approaches.

The project is designed to work with datasets from the PARADIS database (`/shared/home/PARADIS/datos/`), which provides pre-computed patch embeddings and pre-split datasets.

## Architecture & Key Concepts

### Core Workflow Stages

1. **Data Preparation**: Uses `patho_bench.SplitFactory` to discover splits (TSV files with fold assignments) and embeddings
2. **Single Model Training**: `train_abmil.py` trains individual ABMIL models on patch embeddings from a specific foundational model
3. **Evaluation**: `test_abmil.py` evaluates on training splits (for meta-learner features); both scripts call `patho_bench.ExperimentFactory` which handles PyTorch training loops
4. **Ensemble Combination**: 
   - `ensemble.py` uses weighted averaging (weights learned from validation metrics)
   - `ensemble4.py` uses a meta-learner (Logistic Regression on concatenated base model outputs)
5. **Metrics Computation**: `utils.Metrics` computes per-fold and aggregate stats with bootstrapped 95% confidence intervals

### Key Dependencies

- **patho_bench**: Custom framework that abstracts dataset loading, experiment orchestration, and metrics (installed from GitHub)
- **PyTorch** (2.7.0): Model training and inference
- **scikit-learn**: Metrics computation (F1, AUC-ROC, etc.), Logistic Regression for ensemble4
- **h5py**: Reading pre-computed patch embeddings
- **optuna**: Hyperparameter optimization (imported but not actively used in current scripts)

### Data Structure

All data lives under `work_dir/{dataset_name}/{task_name}/`:
- `k=all.tsv`: Master split file with fold assignments (train/val/test across 50 folds)
- `abmil/{model}_{patching_strategy}/`: Model outputs (checkpoints, val_outputs, test_outputs, metrics)
- `ensemble/`, `ensemble4/`: Ensemble results

Patch embeddings are stored in `work_dir/../features/{model}_monai/*.h5` files.

## Common Development Tasks

### Running Training

**Single fold (quick test)**:
```bash
python src/train_abmil.py \
    --foundational_model ctranspath \
    --latent_dim 768 \
    --work_dir /shared/home/PARADIS/datos \
    --train_source cptac_brca \
    --tissue_patching 20x_224px_0px_overlap \
    --task_name TP53_mutation \
    --epochs 100 \
    --fold 0
```

**All 50 folds**:
```bash
python src/train_abmil.py \
    --foundational_model ctranspath \
    --latent_dim 768 \
    --work_dir /shared/home/PARADIS/datos \
    --train_source cptac_brca \
    --tissue_patching 20x_224px_0px_overlap \
    --task_name TP53_mutation \
    --epochs 100 \
    --fold all
```

**Create validation splits** (one-time, only for custom datasets):
```bash
python src/create_val_splits.py \
    -i data/k=all.tsv \
    -o data/k=all_with_val.tsv \
    --n-folds 50 \
    --val-frac 0.15
```

### Running Ensembles

**Weighted averaging ensemble**:
```bash
python src/ensemble.py \
    --foundational_models ctranspath uni_v2 virchow_v1 \
    --work_dir /shared/home/PARADIS/datos \
    --train_source cptac_brca \
    --tissue_patching 20x_224px_0px_overlap \
    --task_name TP53_mutation \
    --weights_type auc_roc
```

**Meta-learner ensemble** (requires running `test_abmil.py` first):
```bash
# Step 1: Get training predictions for each model
for model in ctranspath uni_v2 virchow_v1; do
  python src/test_abmil.py \
      --foundational_model $model \
      --latent_dim 768 \
      --work_dir /shared/home/PARADIS/datos \
      --train_source cptac_brca \
      --tissue_patching 20x_224px_0px_overlap \
      --task_name TP53_mutation \
      --epochs 100
done

# Step 2: Train meta-learner
python src/ensemble4.py \
    --foundational_models ctranspath uni_v2 virchow_v1 \
    --work_dir /shared/home/PARADIS/datos \
    --train_source cptac_brca \
    --tissue_patching 20x_224px_0px_overlap \
    --task_name TP53_mutation
```

### Environment Setup

```bash
# Create conda environment from environment.yml
conda env create -f environment.yml
conda activate patho-ensemble

# Verify PARADIS data access
python verify_paradis_setup.py
```

The `environment.yml` pins specific versions of all dependencies including PyTorch 2.7.0 with CUDA 12 support.

## File Organization

```
src/
├── train_abmil.py          # Main training script for single models
├── test_abmil.py           # Evaluation on training set (for ensemble4)
├── ensemble.py             # Weighted averaging ensemble
├── ensemble4.py            # Meta-learner ensemble
├── create_val_splits.py    # Create validation splits for custom datasets
├── utils.py                # Metrics and weighting utilities
└── run_trident.py          # Batch WSI processing (wrapper script)
```

## Important Implementation Details

### ABMIL Model Configuration

The model architecture is fixed in `train_abmil.py` (lines 42-48):
- Input feature dimension: `--latent_dim` (e.g., 768 for most models, 2560 for Virchow)
- Attention heads: 1
- Head dimension: 512
- Dropout: 0.25
- Gated: False
- Bag size: 2048

To modify architecture, edit these parameters in `train_abmil.py` before training.

### Fold Structure

The codebase supports 50-fold cross-validation with a specific split structure:
- Each `k=*.tsv` file has columns: `case_id`, `slide_id`, `label`, `fold_0` through `fold_49`
- Each fold column contains 'train', 'val', or 'test'
- Patient-level stratification ensures all slides from the same patient stay in the same fold

When using PARADIS data, fold columns already include validation splits. The `--create_val` flag should only be used for custom datasets.

### Metrics Computation

The `Metrics` class in `utils.py` computes:
- **Per-fold metrics**: Saved in `{split}_metrics/fold_X/` directories
- **Aggregate metrics**: ROC curves, confusion matrices, precision-recall curves
- **Bootstrap CIs**: 95% confidence intervals computed via bootstrap resampling (default: 1000 iterations)
- **Output format**: JSON files saved in results directory

### Ensemble Weighting

`MetricDistance` in `utils.py` computes per-fold weights for `ensemble.py`:
1. Reads validation metrics (e.g., AUC-ROC) from individual model outputs
2. Uses metric values to weight models (higher performance = higher weight)
3. Applies softmax normalization
4. Saves weights for reproducibility

## Key Assumptions & Constraints

1. **Patch embeddings must be HDF5 files** in `work_dir/../features/{model}_monai/` with consistent naming
2. **All models in an ensemble must be trained on the same splits** (same `k=*.tsv` file)
3. **No need to create validation splits for PARADIS data** — they're already included in split files
4. **Latent dimension must match the actual embedding dimension** — check with `verify_paradis_setup.py`
5. **Patching strategy must exactly match directory names** (e.g., `20x_224px_0px_overlap`)
6. **Training is single-GPU or CPU** — uses `device="cpu"` by default in ensemble modules

## Debugging & Troubleshooting

### Missing data errors

Run verification script to check PARADIS setup:
```bash
python verify_paradis_setup.py
```

Check exact paths:
```bash
ls /shared/home/PARADIS/datos/patches/cptac_brca/TP53_mutation/
ls /shared/home/PARADIS/datos/features/cptac_brca/features_ctranspath_monai/
```

### Dimension mismatches

Verify embedding dimensions in HDF5 files:
```python
import h5py, os
path = '/shared/home/PARADIS/datos/features/cptac_brca/features_virchow_v1_monai'
h5_file = [f for f in os.listdir(path) if f.endswith('.h5')][0]
with h5py.File(os.path.join(path, h5_file), 'r') as f:
    print('Actual dimension:', f['features'].shape[1])
```

### Memory issues

Train single fold instead of all 50:
```bash
--fold 0  # instead of --fold all
```

Or reduce batch size by editing bag_size in `train_abmil.py` (line 33, default: 2048).

## References

- **README.md**: Complete reference documentation with detailed parameter descriptions
- **QUICK_START.md**: Practical walkthrough with 4 example scenarios
- **PARADIS_COMPATIBILITY.md**: Technical details about PARADIS data structure and available models/datasets
- **examples_paradis.sh**: Interactive script with more command examples
- **CHANGES_SUMMARY.md**: Recent changes and why they were made
