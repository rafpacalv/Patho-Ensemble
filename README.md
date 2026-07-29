# Patho-Ensemble

A collection of scripts for training and evaluating foundational model ensembles for pathology image classification using Patho-Bench. The necessary dependencies using conda are listed in `environment.yml`.

## Overview

This project implements a complete pipeline for:
1. **Creating validation splits** at the patient level
2. **Training individual ABMIL models** with different foundational models
3. **Evaluating ensemble predictions** using weighted averaging or meta-learning approaches
4. **Computing comprehensive classification metrics** with bootstrapping and confidence intervals

## Scripts

### Core Training & Inference

#### `train_abmil.py`
Trains an Attention-Based Multiple Instance Learning (ABMIL) model using patch embeddings from foundational models.

**Key Features:**
- Supports multiple foundational models (e.g., ViT, CTransPath)
- Optionally creates stratified validation splits (--create_val flag)
- Trains on specific fold or all folds
- Saves best model checkpoint based on validation loss
- Evaluates on test set

**Arguments:**
- `--foundational_model`: Model name (e.g., "vit", "ctranspath")
- `--latent_dim`: Feature dimension of the foundational model
- `--work_dir`: Root working directory
- `--train_source`: Source of training data
- `--tissue_patching`: Patching strategy identifier
- `--task_name`: Task/dataset name
- `--epochs`: Number of training epochs
- `--mtr`: (placeholder argument)
- `--create_val`: If set, creates validation splits (15% of training data)
- `--n_folds`: Number of cross-validation folds (default: 50)
- `--fold`: Specific fold to train ("all" for all folds)

**Example:**
```bash
python src/train_abmil.py \
    --foundational_model vit \
    --latent_dim 768 \
    --work_dir /path/to/data \
    --train_source tcga \
    --tissue_patching 256x256 \
    --task_name tumor_classification \
    --epochs 100 \
    --fold all \
    --create_val
```

#### `test_abmil.py`
Re-evaluates ABMIL models on the training set (useful for computing meta-learner features in ensemble4).

**Key Features:**
- Evaluates training split predictions (for ensemble4 meta-learner training)
- Creates a separate evaluation directory (`*_train_eval`)
- Uses validation loss criterion

**Arguments:** Same as `train_abmil.py` (without --create_val, --n_folds, --fold)

**Example:**
```bash
python src/test_abmil.py \
    --foundational_model vit \
    --latent_dim 768 \
    --work_dir /path/to/data \
    --train_source tcga \
    --tissue_patching 256x256 \
    --task_name tumor_classification \
    --epochs 100
```

### Ensemble & Evaluation

#### `ensemble.py`
Combines predictions from multiple ABMIL models using learned weights based on a validation metric.

**Key Features:**
- Computes per-fold weights based on model performance metric
- Normalizes weights using softmax
- Performs weighted averaging of predictions
- Computes comprehensive classification metrics with bootstrapping
- Saves ensemble weights for reproducibility

**Arguments:**
- `--foundational_models`: List of model names (space-separated)
- `--work_dir`: Root working directory
- `--train_source`: Source of training data
- `--tissue_patching`: Patching strategy identifier
- `--task_name`: Task/dataset name
- `--weights_type`: Metric used for weighting (e.g., "auc_roc", "f1_score")

**Example:**
```bash
python src/ensemble.py \
    --foundational_models vit ctranspath dino \
    --work_dir /path/to/data \
    --train_source tcga \
    --tissue_patching 256x256 \
    --task_name tumor_classification \
    --weights_type auc_roc
```

#### `ensemble4.py`
Advanced ensemble using a meta-learner (Logistic Regression) trained on base model outputs.

**Key Features:**
- Trains logistic regression on concatenated training split predictions
- Uses training predictions to fit meta-learner
- Evaluates on validation split for intermediate assessment
- Makes final predictions on test split
- Saves meta-learner coefficients and odds ratios for interpretability
- Computes metrics with bootstrapping

**Arguments:** Same as `ensemble.py` (no weights_type argument)

**Example:**
```bash
python src/ensemble4.py \
    --foundational_models vit ctranspath dino \
    --work_dir /path/to/data \
    --train_source tcga \
    --tissue_patching 256x256 \
    --task_name tumor_classification
```

**Note:** Requires running `test_abmil.py` first to generate training split predictions for meta-learner fitting.

### Data Preparation

#### `create_val_splits.py`
Creates stratified validation splits at the patient/case level across multiple folds.

**Key Features:**
- Stratified shuffle split ensuring label balance
- Case-level stratification (all slides from same patient kept together)
- 15% of training data converted to validation (default)
- Supports custom fold prefixes and fold counts
- Reproducible with random seed (default: 42)

**Arguments:**
- `-i, --input-csv`: Input split file (TSV or CSV)
- `-o, --output-csv`: Output split file with validation columns
- `--fold-prefix`: Prefix for fold columns (default: "fold_")
- `--n-folds`: Number of folds to process (default: 50)
- `--val-frac`: Fraction of training to convert to validation (default: 0.15)
- `--seed`: Random seed for reproducibility (default: 42)

**Example:**
```bash
python src/create_val_splits.py \
    --input-csv data/splits.tsv \
    --output-csv data/splits_with_val.tsv \
    --n-folds 50 \
    --val-frac 0.15
```

### Utilities

#### `utils.py`
Provides helper classes for metrics computation and ensemble weighting.

**Key Classes:**

**`Metrics`**
- Computes per-fold and aggregate classification metrics
- Generates ROC curves, confusion matrices, precision-recall curves
- Performs bootstrap resampling for 95% confidence intervals
- Saves results in structured JSON format

**`MetricDistance`**
- Calculates per-fold weights for ensemble.py
- Reads validation metrics from individual model outputs
- Normalizes metric values into probability weights

#### `run_trident.py`
Wrapper script for batch processing of whole-slide images (WSI).

**Purpose:** Processes a batch of WSI files through TRIDENT pipeline.

**Arguments:**
- `--task`: Task identifier
- `--wsi_dir`: Directory containing WSI files
- `--job_dir`: Output directory for results
- `--patch_encoder`: Encoder model for patch extraction
- `--patch_size`: Size of patches to extract
- `--mag`: Magnification level

## Workflow

### Single Model Training
```bash
# Step 1: Create validation splits (one-time)
python src/create_val_splits.py \
    -i data/k=all.tsv \
    -o data/k=split.tsv

# Step 2: Train ABMIL model
python src/train_abmil.py \
    --foundational_model vit \
    --latent_dim 768 \
    --work_dir /data \
    --train_source tcga \
    --tissue_patching 256x256 \
    --task_name tumor_classification \
    --epochs 100 \
    --fold all \
    --create_val
```

### Ensemble with Weighted Averaging
```bash
# Step 1: Train multiple models (repeat for each model)
python src/train_abmil.py --foundational_model vit ...
python src/train_abmil.py --foundational_model ctranspath ...
python src/train_abmil.py --foundational_model dino ...

# Step 2: Evaluate ensemble
python src/ensemble.py \
    --foundational_models vit ctranspath dino \
    --work_dir /data \
    --train_source tcga \
    --tissue_patching 256x256 \
    --task_name tumor_classification \
    --weights_type auc_roc
```

### Ensemble with Meta-Learner
```bash
# Step 1: Train multiple models (repeat for each model)
python src/train_abmil.py --foundational_model vit ...
python src/train_abmil.py --foundational_model ctranspath ...
python src/train_abmil.py --foundational_model dino ...

# Step 2: Evaluate on training split (generates meta-learner features)
python src/test_abmil.py \
    --foundational_model vit \
    --work_dir /data ...
# Repeat for each foundational model

# Step 3: Train meta-learner on ensemble
python src/ensemble4.py \
    --foundational_models vit ctranspath dino \
    --work_dir /data \
    --train_source tcga \
    --tissue_patching 256x256 \
    --task_name tumor_classification
```

## Output Structure

```
work_dir/
├── train_source/
│   └── task_name/
│       ├── abmil/
│       │   ├── model1_tissue_patching/
│       │   │   ├── val_outputs/     # Validation predictions per fold
│       │   │   ├── test_outputs/    # Test predictions per fold
│       │   │   ├── val_metrics/     # Per-fold validation metrics
│       │   │   └── checkpoints/     # Saved model checkpoints
│       │   ├── model2_tissue_patching/
│       │   ├── ensemble/            # ensemble.py results
│       │   └── ensemble4/           # ensemble4.py results
│       └── config.yaml              # Task configuration
```

## Using PARADIS Data

The scripts are fully compatible with datasets and features available in `/shared/home/PARADIS/datos`. This section explains how to use the existing data.

### Available Datasets

Multiple datasets with pre-computed patch embeddings are available:

| Dataset | Tasks | # Samples | Features Available |
|---------|-------|-----------|-------------------|
| **cptac_brca** | TP53_mutation, PIK3CA_mutation, Immune_class | 113 | ✓ |
| **cptac_coad** | TP53_mutation, KRAS_mutation, APC_mutation, Immune_class, MSI_H + more | 327 | ✓ |
| **cptac_gbm** | TP53_mutation, EGFR_mutation, Immune_class | TBD | ✓ |
| **cptac_hnsc** | Immune_class, Histologic_Grade, CASP8_mutation | TBD | ✓ |
| **cptac_lscc** | Immune_class, Histologic_Grade, ARID1A_mutation | TBD | ✓ |
| **cptac_luad** | TP53_mutation, KRAS_mutation, EGFR_mutation, Immune_class, OS, STK11_mutation | TBD | ✓ |
| **cptac_ccrcc** | BAP1_mutation, VHL_mutation, PBRM1_mutation, Immune_class | TBD | ✓ |
| **crc_outcomes** | Multiple grading & morphology tasks | TBD | ✓ |
| **hancook** | Multiple grading & morphology tasks | TBD | ✓ |
| **bc_therapy** | ER_status, HER2_status, Grade, Residual_cancer_burden | TBD | ✓ |
| **cervical_subtype** | Subtype classification | TBD | ✓ |
| **imp** | Grade classification | TBD | ✓ |
| **cptac_lung** | Subtype classification | TBD | ✓ |

### Available Foundational Models

The following foundational models have pre-computed embeddings:

- **UNI models**: uni_v1, uni_v2 (with 224px and 256px variants)
- **Virchow models**: virchow_v1, virchow2 (vision transformer-based)
- **CTransPath**: ctranspath (convolutional transformer)
- **Conch models**: conch_v1_5, conch512
- **Gigapath**: gigapath, gigapath_256px (foundation model)
- **Hoptimus models**: hoptimus0, hoptimus1
- **Phikon**: phikon_v2
- **ResNet**: resnet50 (baseline CNN)

**Feature Dimensions:**
- Most models: 768D
- Virchow models: 2560D
- Some variants: custom dimensions

**Patching Strategies:**
- 20x magnification, 224px patches, 0px overlap
- 20x magnification, 256px patches, 0px overlap
- Additional variants available

### Data Structure in PARADIS

```
/shared/home/PARADIS/datos/
├── features/
│   ├── cptac_brca/
│   │   ├── features_ctranspath_monai/     # embeddings (.h5)
│   │   ├── features_uni_v2_monai/
│   │   ├── features_virchow_v1_monai/
│   │   └── ... (other models)
│   ├── cptac_coad/
│   └── ... (other datasets)
├── patches/
│   ├── cptac_brca/
│   │   ├── TP53_mutation/
│   │   │   ├── k=all.tsv                  # master split file
│   │   │   ├── 20x_224px_0px_overlap/     # patches per strategy
│   │   │   └── 20x_256px_0px_overlap/
│   │   ├── PIK3CA_mutation/
│   │   └── Immune_class/
│   ├── cptac_coad/
│   └── ... (other datasets)
└── wsis/                                   # original whole-slide images
```

### Split Files Format

Each task has a `k=all.tsv` file with the following structure:

```
case_id    slide_id                           label  fold_0  fold_1  fold_2  ...  fold_49
20BR007    20BR007-547f521a-5851-426c-...     1      train   train   train   ...  train
06BR006    06BR006-3799eeb5-c966-4059-...     0      train   val     train   ...  train
...
```

**Columns:**
- `case_id`: Patient identifier
- `slide_id`: Unique slide identifier  
- `label`: Target label (0/1 for binary, multi-class for others)
- `fold_0` to `fold_49`: 50-fold cross-validation splits (train/val/test)

These files **already include validation splits** created using stratified patient-level splits, so you can use them directly without running `create_val_splits.py`.

### Quick Start with PARADIS Data

#### Example 1: Train single ABMIL model on TP53 mutation (cptac_brca)

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

**Expected output:**
```
/shared/home/PARADIS/datos/cptac_brca/TP53_mutation/abmil/
└── ctranspath_20x_224px_0px_overlap/
    ├── val_outputs/fold_0...fold_49/
    ├── test_outputs/fold_0...fold_49/
    ├── val_metrics_summary.json
    └── checkpoints/
```

#### Example 2: Compare multiple foundational models on same task

```bash
# Train each model
for model in ctranspath uni_v2 virchow_v1; do
  python src/train_abmil.py \
      --foundational_model $model \
      --latent_dim $([ "$model" = "virchow_v1" ] && echo 2560 || echo 768) \
      --work_dir /shared/home/PARADIS/datos \
      --train_source cptac_brca \
      --tissue_patching 20x_224px_0px_overlap \
      --task_name TP53_mutation \
      --epochs 100 \
      --fold all
done

# Ensemble with weighted averaging
python src/ensemble.py \
    --foundational_models ctranspath uni_v2 virchow_v1 \
    --work_dir /shared/home/PARADIS/datos \
    --train_source cptac_brca \
    --tissue_patching 20x_224px_0px_overlap \
    --task_name TP53_mutation \
    --weights_type auc_roc
```

#### Example 3: Multiple tasks on cptac_brca

```bash
for task in TP53_mutation PIK3CA_mutation Immune_class; do
  python src/train_abmil.py \
      --foundational_model ctranspath \
      --latent_dim 768 \
      --work_dir /shared/home/PARADIS/datos \
      --train_source cptac_brca \
      --tissue_patching 20x_224px_0px_overlap \
      --task_name $task \
      --epochs 100 \
      --fold all
done
```

#### Example 4: Meta-learner ensemble on larger dataset (cptac_coad)

```bash
# Train base models
for model in ctranspath uni_v2 virchow_v1 phikon_v2; do
  python src/train_abmil.py \
      --foundational_model $model \
      --latent_dim $([ "$model" = "virchow_v1" ] && echo 2560 || echo 768) \
      --work_dir /shared/home/PARADIS/datos \
      --train_source cptac_coad \
      --tissue_patching 20x_224px_0px_overlap \
      --task_name TP53_mutation \
      --epochs 100 \
      --fold all
done

# Evaluate on training split
for model in ctranspath uni_v2 virchow_v1 phikon_v2; do
  python src/test_abmil.py \
      --foundational_model $model \
      --latent_dim $([ "$model" = "virchow_v1" ] && echo 2560 || echo 768) \
      --work_dir /shared/home/PARADIS/datos \
      --train_source cptac_coad \
      --tissue_patching 20x_224px_0px_overlap \
      --task_name TP53_mutation \
      --epochs 100
done

# Train meta-learner
python src/ensemble4.py \
    --foundational_models ctranspath uni_v2 virchow_v1 phikon_v2 \
    --work_dir /shared/home/PARADIS/datos \
    --train_source cptac_coad \
    --tissue_patching 20x_224px_0px_overlap \
    --task_name TP53_mutation
```

### Important Notes for PARADIS Data

1. **No need to create validation splits**: The `k=all.tsv` files already contain stratified train/val/test splits across 50 folds. Do NOT use `--create_val` flag.

2. **Feature dimensions matter**: Different models have different embedding dimensions. The `--latent_dim` parameter must match:
   - Most models (ctranspath, uni, conch, phikon, hoptimus): 768D
   - Virchow models: 2560D
   - Check actual dimensions if using custom/newer models

3. **Patching strategy must exist**: The `--tissue_patching` parameter must match exactly with directory names in:
   ```
   /shared/home/PARADIS/datos/patches/{dataset}/{task}/{patching_strategy}/
   ```
   Common options: `20x_224px_0px_overlap`, `20x_256px_0px_overlap`

4. **Work directory must be PARADIS**: Set `--work_dir /shared/home/PARADIS/datos` to use pre-computed embeddings and splits.

5. **Patho-Bench Integration**: These scripts rely on `patho_bench.SplitFactory` and `patho_bench.ExperimentFactory` which automatically discover splits and features in the specified structure.

## Dependencies

Install dependencies using conda:
```bash
conda env create -f environment.yml
conda activate patho-ensemble
```

Main dependencies:
- PyTorch
- patho-bench (custom framework)
- scikit-learn
- pandas
- numpy
- optuna (for hyperparameter optimization)
- tqdm (progress bars)
- h5py (for HDF5 feature files)
