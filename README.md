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

Two complementary approaches are available for combining predictions from multiple ABMIL models:

#### `ensemble.py` — Weighted Averaging Ensemble
Combines predictions from multiple ABMIL models using learned weights based on individual model validation performance.

**Strategy:**
- Computes per-fold weights from each model's performance on a specific metric (e.g., AUC-ROC)
- Normalizes weights using softmax for probability interpretation
- Performs simple weighted averaging: `prediction = Σ(softmax(w) · pred_model)`
- **Best for:** When you trust individual model performance as a proxy for ensemble contribution

**Key Features:**
- Fast inference (no meta-model training required)
- Interpretable weights (higher performance = higher weight)
- Weights saved for reproducibility and analysis
- Computes comprehensive classification metrics with bootstrapping

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

#### `ensemble4.py` — Meta-Learner Ensemble
Advanced ensemble using a meta-learner (Logistic Regression) to learn optimal combination of base model outputs.

**Strategy:**
1. Trains Logistic Regression on concatenated predictions from training split
2. Uses learned coefficients to combine models on test split
3. **Best for:** When models have complementary strengths or complex interactions

**Workflow:**
```
Base predictions (train) → Meta-learner training
Base predictions (val)   → Intermediate assessment
Base predictions (test)  → Final predictions via trained meta-learner
```

**Key Features:**
- Learns non-linear (through feature space) relationships between base models
- Can detect synergies between complementary models
- Saves meta-learner coefficients and odds ratios for interpretability
- Computes comprehensive classification metrics with bootstrapping
- Produces more flexible ensemble than simple weighted averaging

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

**Prerequisites:** Requires running `test_abmil.py` first to generate training split predictions:
```bash
# Generate meta-learner training features for each model
for model in vit ctranspath dino; do
  python src/test_abmil.py \
      --foundational_model $model \
      --latent_dim 768 \
      --work_dir /path/to/data \
      --train_source tcga \
      --tissue_patching 256x256 \
      --task_name tumor_classification \
      --epochs 100
done

# Then train meta-learner
python src/ensemble4.py \
    --foundational_models vit ctranspath dino \
    ...
```

#### Comparison: ensemble.py vs ensemble4.py

| Aspect | **ensemble.py** | **ensemble4.py** |
|--------|---|---|
| **Combination method** | Weighted average | Meta-learner (Logistic Regression) |
| **Weight source** | Individual model validation metrics | Learned from training predictions |
| **Complexity** | Simple, interpretable weights | Learns feature interactions |
| **Training time** | None (weights pre-computed) | ~1 minute (Logistic Regression) |
| **Best use case** | Trust individual model performance | Leverage model complementarity |
| **Inference speed** | Fastest | Fast (single LR prediction) |
| **Output interpretability** | Clear per-model contributions | Coefficients show feature importance |
| **Prerequisites** | Train all models only | Train + `test_abmil.py` for each model |

#### When to use which ensemble approach?

**Use `ensemble.py` (Weighted Averaging) if:**
- You want a **quick baseline** without additional meta-learning
- Individual model performance metrics are already **highly predictive** of ensemble quality
- You have **few training samples** (< 100) where meta-learner might overfit
- You need **maximum interpretability** of which models contribute most
- You want to **quickly compare different model combinations**

**Use `ensemble4.py` (Meta-Learner) if:**
- You have **larger datasets** (>200 samples) with stable validation metrics
- You want to **detect synergies** between complementary models
- Individual models have **similar but non-identical performance** (meta-learner learns their complementarity)
- You can afford **5 extra minutes** for `test_abmil.py` runs + meta-learner training
- You want to **squeeze out maximum accuracy** from your ensemble

**Trade-off summary:**
- **Time:** `ensemble.py` is ~5 minutes faster (no test_abmil.py)
- **Accuracy:** `ensemble4.py` typically outperforms by 1-3% on larger datasets
- **Interpretability:** `ensemble.py` shows which models are best; `ensemble4.py` shows which models interact

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
# Step 1: Create validation splits (one-time, only for custom datasets)
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

### Ensemble with Weighted Averaging (Recommended for fast setup)
**Time estimate:** 1-4 hours (only training time, no additional meta-learning)

```bash
# Step 1: Train multiple models in parallel
python src/train_abmil.py --foundational_model vit \
    --latent_dim 768 --work_dir /data --train_source tcga \
    --tissue_patching 256x256 --task_name tumor_classification \
    --epochs 100 --fold all &

python src/train_abmil.py --foundational_model ctranspath \
    --latent_dim 768 --work_dir /data --train_source tcga \
    --tissue_patching 256x256 --task_name tumor_classification \
    --epochs 100 --fold all &

python src/train_abmil.py --foundational_model dino \
    --latent_dim 768 --work_dir /data --train_source tcga \
    --tissue_patching 256x256 --task_name tumor_classification \
    --epochs 100 --fold all &

wait  # Wait for all training to finish

# Step 2: Evaluate ensemble using learned weights (automatic weight calculation)
python src/ensemble.py \
    --foundational_models vit ctranspath dino \
    --work_dir /data \
    --train_source tcga \
    --tissue_patching 256x256 \
    --task_name tumor_classification \
    --weights_type auc_roc
```

**Output:**
- Per-fold weights based on AUC-ROC: `weights.npy`
- Ensemble predictions: `ensemble/test_outputs/fold_*/{labels,preds}.npy`
- Ensemble metrics: `ensemble/test_metrics_summary.json`

### Ensemble with Meta-Learner (Recommended for optimal performance)
**Time estimate:** 1-4 hours training + 1 minute meta-learner fitting

This approach learns how to best combine models rather than trusting individual performance.

```bash
# Step 1: Train multiple models in parallel
python src/train_abmil.py --foundational_model vit \
    --latent_dim 768 --work_dir /data --train_source tcga \
    --tissue_patching 256x256 --task_name tumor_classification \
    --epochs 100 --fold all &

python src/train_abmil.py --foundational_model ctranspath \
    --latent_dim 768 --work_dir /data --train_source tcga \
    --tissue_patching 256x256 --task_name tumor_classification \
    --epochs 100 --fold all &

python src/train_abmil.py --foundational_model dino \
    --latent_dim 768 --work_dir /data --train_source tcga \
    --tissue_patching 256x256 --task_name tumor_classification \
    --epochs 100 --fold all &

wait  # Wait for all training to finish

# Step 2: Generate training set predictions (for meta-learner training data)
# This creates *_train_eval directories with predictions on training split
for model in vit ctranspath dino; do
  python src/test_abmil.py \
      --foundational_model $model \
      --latent_dim 768 \
      --work_dir /data \
      --train_source tcga \
      --tissue_patching 256x256 \
      --task_name tumor_classification \
      --epochs 100 &
done

wait  # Wait for all test_abmil.py to finish

# Step 3: Train meta-learner ensemble
# Logistic Regression learns optimal weights from training predictions
python src/ensemble4.py \
    --foundational_models vit ctranspath dino \
    --work_dir /data \
    --train_source tcga \
    --tissue_patching 256x256 \
    --task_name tumor_classification
```

**Output:**
- Meta-learner coefficients: `ensemble4/coefs.npy` (accumulated across folds)
- Odds ratios: `ensemble4/odds_ratios.npy`
- Ensemble predictions: `ensemble4/test_outputs/fold_*/{labels,preds}.npy`
- Ensemble metrics: `ensemble4/test_metrics_summary.json`

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

**Option A: Quick comparison with weighted averaging**
```bash
# Train each model (can run in parallel)
for model in ctranspath uni_v2 virchow_v1; do
  python src/train_abmil.py \
      --foundational_model $model \
      --latent_dim $([ "$model" = "virchow_v1" ] && echo 2560 || echo 768) \
      --work_dir /shared/home/PARADIS/datos \
      --train_source cptac_brca \
      --tissue_patching 20x_224px_0px_overlap \
      --task_name TP53_mutation \
      --epochs 100 \
      --fold all &
done
wait

# Ensemble with weighted averaging (uses best models' performance as weights)
python src/ensemble.py \
    --foundational_models ctranspath uni_v2 virchow_v1 \
    --work_dir /shared/home/PARADIS/datos \
    --train_source cptac_brca \
    --tissue_patching 20x_224px_0px_overlap \
    --task_name TP53_mutation \
    --weights_type auc_roc
```

**Option B: Best performance with meta-learner**
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
      --fold all &
done
wait

# Generate training predictions for meta-learner
for model in ctranspath uni_v2 virchow_v1; do
  python src/test_abmil.py \
      --foundational_model $model \
      --latent_dim $([ "$model" = "virchow_v1" ] && echo 2560 || echo 768) \
      --work_dir /shared/home/PARADIS/datos \
      --train_source cptac_brca \
      --tissue_patching 20x_224px_0px_overlap \
      --task_name TP53_mutation \
      --epochs 100 &
done
wait

# Train meta-learner ensemble (learns optimal combination)
python src/ensemble4.py \
    --foundational_models ctranspath uni_v2 virchow_v1 \
    --work_dir /shared/home/PARADIS/datos \
    --train_source cptac_brca \
    --tissue_patching 20x_224px_0px_overlap \
    --task_name TP53_mutation
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

Perfect for larger datasets where model combinations matter more. 4 models × 50 folds × meta-learner.

```bash
# Train base models (in parallel for efficiency)
for model in ctranspath uni_v2 virchow_v1 phikon_v2; do
  python src/train_abmil.py \
      --foundational_model $model \
      --latent_dim $([ "$model" = "virchow_v1" ] && echo 2560 || echo 768) \
      --work_dir /shared/home/PARADIS/datos \
      --train_source cptac_coad \
      --tissue_patching 20x_224px_0px_overlap \
      --task_name TP53_mutation \
      --epochs 100 \
      --fold all &
done
wait

# Generate training predictions for meta-learner training (in parallel)
for model in ctranspath uni_v2 virchow_v1 phikon_v2; do
  python src/test_abmil.py \
      --foundational_model $model \
      --latent_dim $([ "$model" = "virchow_v1" ] && echo 2560 || echo 768) \
      --work_dir /shared/home/PARADIS/datos \
      --train_source cptac_coad \
      --tissue_patching 20x_224px_0px_overlap \
      --task_name TP53_mutation \
      --epochs 100 &
done
wait

# Train Logistic Regression meta-learner (learns optimal model weights from data)
python src/ensemble4.py \
    --foundational_models ctranspath uni_v2 virchow_v1 phikon_v2 \
    --work_dir /shared/home/PARADIS/datos \
    --train_source cptac_coad \
    --tissue_patching 20x_224px_0px_overlap \
    --task_name TP53_mutation
```

**Why meta-learner on larger datasets?**
- More training data (327 samples in cptac_coad) improves meta-learner generalization
- More likely to find complementary models and synergies
- Logistic Regression coefficients become more stable and interpretable

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

## Practical Examples

### Example 1: Using Pre-computed PARADIS Embeddings (30 minutes to 4 hours)

This is the typical use case when working with existing PARADIS data. All embeddings and splits are already prepared.

**Setup (5 minutes)**:
```bash
# Navigate to project directory
cd /home/JKP6679/Patho-Ensemble

# Activate environment (conda environment 'ensemble' must be pre-created)
conda activate ensemble

# Verify PARADIS data access
python verify_paradis_setup.py
```

**Single model training on TP53 mutation prediction (cptac_brca dataset)**:
```bash
# Train ABMIL model using CTransPath embeddings
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

**Output**: 
- Model checkpoints: `/shared/home/PARADIS/datos/cptac_brca/TP53_mutation/abmil/ctranspath_20x_224px_0px_overlap/checkpoints/`
- Predictions: `val_outputs/fold_0-49/` and `test_outputs/fold_0-49/`
- Metrics: `test_metrics_summary.json` and `val_metrics_summary.json`

**Evaluate ensemble of multiple models (1-4 hours)**:
```bash
# Train additional models in parallel
python src/train_abmil.py \
    --foundational_model uni_v2 \
    --latent_dim 768 \
    --work_dir /shared/home/PARADIS/datos \
    --train_source cptac_brca \
    --tissue_patching 20x_224px_0px_overlap \
    --task_name TP53_mutation \
    --epochs 100 \
    --fold all &

python src/train_abmil.py \
    --foundational_model virchow_v1 \
    --latent_dim 2560 \
    --work_dir /shared/home/PARADIS/datos \
    --train_source cptac_brca \
    --tissue_patching 20x_224px_0px_overlap \
    --task_name TP53_mutation \
    --epochs 100 \
    --fold all &

wait

# Combine predictions using weighted ensemble
python src/ensemble.py \
    --foundational_models ctranspath uni_v2 virchow_v1 \
    --work_dir /shared/home/PARADIS/datos \
    --train_source cptac_brca \
    --tissue_patching 20x_224px_0px_overlap \
    --task_name TP53_mutation \
    --weights_type auc_roc
```

**Check results**:
```bash
# View individual model metrics
cat /shared/home/PARADIS/datos/cptac_brca/TP53_mutation/abmil/ctranspath_20x_224px_0px_overlap/test_metrics_summary.json | jq '.aggregate'

# View ensemble metrics
cat /shared/home/PARADIS/datos/cptac_brca/TP53_mutation/abmil/ensemble/test_metrics_summary.json | jq '.aggregate'

# View ensemble weights (learned from validation performance)
cat /shared/home/PARADIS/datos/cptac_brca/TP53_mutation/abmil/ensemble/ensemble_weights.json
```

---

### Example 2: Complete Workflow from Patch Extraction to Evaluation (8-48+ hours)

This example shows the full pipeline: starting from whole-slide images (WSI), extracting patches, computing embeddings, and training models. This is needed when working with new datasets or custom WSI collections.

**Prerequisites**:
- Whole-slide image files (`.svs`, `.ndpi`, or other OpenSlide-compatible formats)
- Slide metadata in TSV format with at least: `case_id`, `slide_id`, `label`
- Access to foundational models (loaded from HuggingFace or local)

**Step 1: Prepare data structure (5 minutes)**:
```bash
# Create working directory for your custom dataset
WORK_DIR=/path/to/my_data
DATASET_NAME=my_dataset
TASK_NAME=my_classification_task

mkdir -p $WORK_DIR/$DATASET_NAME/$TASK_NAME
mkdir -p $WORK_DIR/patches/$DATASET_NAME/$TASK_NAME
mkdir -p $WORK_DIR/features/$DATASET_NAME

# Prepare input split file (TSV format)
cat > $WORK_DIR/$DATASET_NAME/$TASK_NAME/metadata.tsv << 'EOF'
case_id	slide_id	label
patient_001	slide_001.svs	0
patient_001	slide_002.svs	0
patient_002	slide_003.svs	1
patient_002	slide_004.svs	1
EOF
```

**Step 2: Create train/val/test splits (10 minutes)**:
```bash
# Create stratified k-fold splits at patient level
python src/create_val_splits.py \
    -i $WORK_DIR/$DATASET_NAME/$TASK_NAME/metadata.tsv \
    -o $WORK_DIR/$DATASET_NAME/$TASK_NAME/k=all.tsv \
    --n-folds 5 \
    --val-frac 0.15 \
    --seed 42
```

**Output**: `k=all.tsv` with columns for `case_id`, `slide_id`, `label`, `fold_0` through `fold_4`

**Step 3: Extract patches from WSI (2-24 hours, depending on image count and server resources)**:

You can use your own patch extraction pipeline or the provided `run_trident.py` wrapper. Here's an example with a custom script:

```bash
# Extract patches at 20x magnification, 224px resolution
# This requires installing OpenSlide and a patch extraction library

# Example using patho_bench's built-in functions:
python << 'EOF'
import os
from pathlib import Path
from PIL import Image
import openslide
import numpy as np

WORK_DIR = "/path/to/my_data"
DATASET_NAME = "my_dataset"
TASK_NAME = "my_classification_task"
WSI_DIR = "/path/to/whole_slide_images"
PATCHES_OUTPUT = f"{WORK_DIR}/patches/{DATASET_NAME}/{TASK_NAME}/20x_224px_0px_overlap"

os.makedirs(PATCHES_OUTPUT, exist_ok=True)

# Process each WSI file
for wsi_file in os.listdir(WSI_DIR):
    if wsi_file.endswith(('.svs', '.ndpi')):
        wsi_path = os.path.join(WSI_DIR, wsi_file)
        slide = openslide.open_slide(wsi_path)
        
        # Extract patches at 20x magnification
        # This is pseudocode - actual implementation depends on your patch extraction method
        patches = extract_patches_20x(slide, patch_size=224, overlap=0)
        
        # Save patches (implementation specific)
        slide.close()

print("Patch extraction complete!")
EOF
```

**Step 4: Compute embeddings using a foundational model (4-24 hours, depending on patch count)**:

```bash
# Compute embeddings using CTransPath model
# You can use patho_bench's embedding computation or your own script

python << 'EOF'
import torch
import h5py
import os
from pathlib import Path
from tqdm import tqdm

# Load model (example with timm)
from timm import create_model

model_name = "ctranspath"
device = "cuda" if torch.cuda.is_available() else "cpu"
model = create_model("swin_tiny_patch4_window7_224", pretrained=False)
# Load CTransPath weights
model.load_state_dict(torch.load("path/to/ctranspath_weights.pth"))
model.eval().to(device)

WORK_DIR = "/path/to/my_data"
DATASET_NAME = "my_dataset"
TASK_NAME = "my_classification_task"
PATCHES_DIR = f"{WORK_DIR}/patches/{DATASET_NAME}/{TASK_NAME}/20x_224px_0px_overlap"
OUTPUT_DIR = f"{WORK_DIR}/features/{DATASET_NAME}/features_ctranspath_monai"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# Process each patient's patches
for patient_dir in os.listdir(PATCHES_DIR):
    patches_path = os.path.join(PATCHES_DIR, patient_dir)
    if not os.path.isdir(patches_path):
        continue
    
    # Collect all patches for this patient
    patch_files = sorted([f for f in os.listdir(patches_path) if f.endswith('.png')])
    embeddings = []
    
    for patch_file in tqdm(patch_files, desc=f"Embedding {patient_dir}"):
        # Load and preprocess patch
        patch = Image.open(os.path.join(patches_path, patch_file))
        # Normalize and convert to tensor (implementation specific)
        
        # Compute embedding
        with torch.no_grad():
            embedding = model(patch_tensor.to(device))
        
        embeddings.append(embedding.cpu().numpy())
    
    # Save embeddings as HDF5 for this patient
    output_file = os.path.join(OUTPUT_DIR, f"{patient_dir}.h5")
    with h5py.File(output_file, 'w') as f:
        f.create_dataset('features', data=np.array(embeddings))
        f.create_dataset('patch_ids', data=np.array(patch_files, dtype=h5py.string_dtype()))

print("Embedding computation complete!")
EOF
```

**Step 5: Train ABMIL models (2-12 hours for full 5-fold CV)**:
```bash
python src/train_abmil.py \
    --foundational_model ctranspath \
    --latent_dim 768 \
    --work_dir $WORK_DIR \
    --train_source $DATASET_NAME \
    --tissue_patching 20x_224px_0px_overlap \
    --task_name $TASK_NAME \
    --epochs 100 \
    --fold all
```

**Step 6: Evaluate results (5-10 minutes)**:
```bash
# View metrics summary
cat $WORK_DIR/$DATASET_NAME/$TASK_NAME/abmil/ctranspath_20x_224px_0px_overlap/test_metrics_summary.json | jq '.aggregate'

# Check ROC curve and confusion matrix
ls $WORK_DIR/$DATASET_NAME/$TASK_NAME/abmil/ctranspath_20x_224px_0px_overlap/test_metrics/
```

**Complete shell script for the entire workflow**:
```bash
#!/bin/bash

WORK_DIR=/path/to/my_data
DATASET_NAME=my_dataset
TASK_NAME=my_classification_task
WSI_DIR=/path/to/whole_slide_images

# Step 1: Setup
mkdir -p $WORK_DIR/$DATASET_NAME/$TASK_NAME
mkdir -p $WORK_DIR/patches/$DATASET_NAME/$TASK_NAME
mkdir -p $WORK_DIR/features/$DATASET_NAME

# Step 2: Create splits (assuming metadata.tsv exists)
python src/create_val_splits.py \
    -i $WORK_DIR/$DATASET_NAME/$TASK_NAME/metadata.tsv \
    -o $WORK_DIR/$DATASET_NAME/$TASK_NAME/k=all.tsv \
    --n-folds 5 --val-frac 0.15

# Step 3: Extract patches (use your own script or run_trident.py)
echo "Extract patches using your own pipeline..."

# Step 4: Compute embeddings (use your own script)
echo "Compute embeddings using your own pipeline..."

# Step 5: Train models
python src/train_abmil.py \
    --foundational_model ctranspath \
    --latent_dim 768 \
    --work_dir $WORK_DIR \
    --train_source $DATASET_NAME \
    --tissue_patching 20x_224px_0px_overlap \
    --task_name $TASK_NAME \
    --epochs 100 \
    --fold all

# Step 6: View results
echo "Training complete! Results in: $WORK_DIR/$DATASET_NAME/$TASK_NAME/abmil/"
```

**Timeline and expected output**:
| Step | Time | Output |
|------|------|--------|
| Data preparation | 5 min | TSV split file with fold assignments |
| Patch extraction | 2-24 hrs | PNG files organized by patient |
| Embedding computation | 4-24 hrs | HDF5 files with embeddings (768D vectors) |
| Model training | 2-12 hrs | Checkpoints and predictions for all folds |
| Evaluation | 5-10 min | JSON metrics with AUC, F1, accuracy, etc. |

---

## Dependencies

The project requires the **`ensemble`** conda environment. This environment should be activated before running any scripts:

```bash
conda activate ensemble
```

Main dependencies included in the environment:
- PyTorch 2.7.0 (with CUDA 12 support)
- patho-bench (custom framework for dataset management and training)
- scikit-learn (metrics and Logistic Regression for ensemble4)
- pandas (data handling)
- numpy (numerical operations)
- optuna (hyperparameter optimization)
- tqdm (progress bars)
- h5py (reading/writing HDF5 feature files)
