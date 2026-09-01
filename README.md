# Patho-Ensemble

A comprehensive machine learning pipeline for training and evaluating foundational model ensembles for pathology image classification using Patho-Bench and ABMIL (Attention-Based Multiple Instance Learning).

**Latest improvements**: experiment matrix driven by [`experiments.yaml`](experiments.yaml)
(any dataset × task without editing scripts), automatic `--latent_dim` detection from
the `.h5` files, Fast Geometric Ensembles for the ABMIL base models, and TabPFN as a
meta-classifier.

---

## 📋 Table of Contents

1. [Overview](#overview)
2. [Quick Start](#quick-start)
3. [Scripts & Components](#scripts--components)
4. [Ensemble Approaches](#ensemble-approaches)
5. [Running Multiple Experiments](#running-multiple-experiments)
6. [Data Preparation](#data-preparation)
7. [PARADIS Data Usage](#paradis-data-usage)
7. [Experiments & Meta-Learners](#experiments--meta-learners)
8. [Advanced Optimization](#advanced-optimization)
9. [Cluster Configuration](#cluster-configuration)
10. [Troubleshooting](#troubleshooting)
11. [Output Structure](#output-structure)
12. [Dependencies](#dependencies)

---

## Overview

This project implements a complete pipeline for:

1. **Data Preparation**: Automatic validation split generation with stratification at patient level
2. **Single Model Training**: ABMIL models with different foundational models (ViT, CTransPath, Virchow, UNI, etc.)
3. **Ensemble Evaluation**: Multiple ensemble strategies (weighted averaging, meta-learner, advanced optimization)
4. **Meta-Learner Experiments**: Logistic Regression, MLP single-cycle, MLP with Snapshot Ensembles
5. **Comprehensive Metrics**: Per-fold and aggregate statistics with 95% bootstrapped confidence intervals

### Key Features

- ✅ **Automatic validation split creation** across 50 folds with case-level stratification
- ✅ **Multiple ensemble strategies**: Weighted averaging, meta-learner (Logistic Regression, MLP, Snapshot Ensemble)
- ✅ **Sensitivity analysis**: Detect optimization potential before spending compute
- ✅ **Robust error handling**: Handles edge cases (empty folds, missing validation data)
- ✅ **PARADIS integration**: Works seamlessly with pre-computed embeddings and pre-split datasets
- ✅ **SLURM support**: Automated pipeline orchestration via sbatch scripts
- ✅ **Reproducible**: Full neural network training with consistent seeds and checkpointing

---

## Quick Start

### Installation

```bash
cd /shared/home/JKP6679/Patho-Ensemble

# Create and activate conda environment
conda env create -f environment.yml
conda activate ensemble

# Verify PARADIS data access (optional)
python verify_paradis_setup.py
```

### Sanity check (do this first)

The single cheapest check — it catches path and dimension problems before you burn
GPU hours:

```bash
cd src && python3 -c "
from utils import get_features_dir, detect_latent_dim
W = '/home/JKP6679/Patho-Ensemble/PARADIS/datos/patches'
for m in ['ctranspath', 'uni_v2', 'virchow_v1']:
    d = get_features_dir(W, 'cptac_brca', m)
    print(f'{m:12s} dim={detect_latent_dim(d):5d}  {d}')"
```

Expected — note the dimensions differ per model, and every path resolves under
`datos/features/`:

```
ctranspath   dim=  768  .../PARADIS/datos/features/cptac_brca/features_ctranspath_monai
uni_v2       dim= 1536  .../PARADIS/datos/features/cptac_brca/features_uni_v2
virchow_v1   dim= 2560  .../PARADIS/datos/features/cptac_brca/features_virchow_v1_monai
```

### Run the pipeline

Experiments are declared in [`experiments.yaml`](experiments.yaml). Check what is
active, then launch:

```bash
python src/experiment_config.py --list      # show active experiments

./run_full_pipeline.sbatch                  # run the enabled ones
```

`run_full_pipeline.sbatch` queues three dependent SLURM jobs
(`--dependency=afterok`) and exits — it holds no GPU while waiting, and a failed
stage cancels the ones after it:

| Stage | Script | Duration |
|---|---|---|
| 1. Base ABMIL + train predictions | `train_base_models.sbatch` | ~30 min |
| 2. Snapshot cycles (FGE / SE) | `train_base_models_fge_v2.sbatch` | ~1 h per FGE variant, ~3–4 h per SE variant |
| 3. Meta-learners + comparison table | `run_3_experiments_v2_improved.sbatch` | ~15 min |

Measured on cptac_brca (112 slides): ctranspath completes 50 folds in ~6 min;
`uni_v2` (1536D) and `virchow_v1` (2560D) scale up from there. The `--time` caps in
the SBATCH headers are deliberate slack, not estimates.

**Optional stage, run separately:** `build_oof_features.sbatch` (~39 min)
generates out-of-fold meta-features. It is not part of the chain because it is a
one-time prerequisite per dataset, and only the configurations that ask for
`--meta_features oof` need it. Run it after stage 1 and before stage 3 if you
plan to use anything more flexible than `logreg` as the meta-learner — see
[The meta-features](#the-meta-features-in-sample-vs-out-of-fold) for why it
matters.

Stage 2 runs every variant in `fge_variants` sequentially. To fan them out as
parallel SLURM jobs instead, submit one per variant with `FGE_ONLY=<name>`.

Other ways to launch:

```bash
# One specific dataset/task, even if disabled in experiments.yaml
DATASET=cptac_coad TASK=MSI_H ./run_full_pipeline.sbatch

# Every experiment marked enabled: true
ALL_ACTIVE=1 ./run_full_pipeline.sbatch

# A single stage on its own
DATASET=cptac_brca TASK=TP53_mutation sbatch train_base_models.sbatch
```

Stages already completed are detected and skipped, so re-running after a failure
resumes rather than restarting.

### Monitor progress

```bash
squeue -u $USER
tail -f logs/cptac_brca_TP53_mutation/*.log
```

### Verify success

```bash
W=PARADIS/datos/patches/cptac_brca/TP53_mutation/abmil

# Real checkpoints — count model.pt, not directories: some folds hold
# legacy epoch_9.pt files that the current pipeline cannot load
ls $W/ctranspath_20x_224px_0px_overlap/checkpoints/fold_*/model.pt | wc -l   # expect 50

cat $W/ensemble4/test_metrics_summary.json
```

---

## Scripts & Components

### Core Training & Inference

#### `train_abmil.py` — Single Model Training

Trains an Attention-Based Multiple Instance Learning (ABMIL) model using patch embeddings from foundational models.

**Key Features:**
- Supports multiple foundational models (ctranspath, uni_v2, virchow_v1, conch512, etc.)
- Automatic validation split handling
- Trains on specific fold or all 50 folds
- Saves best model checkpoint based on validation loss
- Generates val_outputs and test_outputs

**Arguments:**
```
--foundational_model    Model name (ctranspath, uni_v2, virchow_v1, etc.)
--latent_dim            Feature dimension (768 for most, 2560 for Virchow, 1536 for UNI v2)
--work_dir              Root working directory
--train_source          Dataset name (cptac_brca, cptac_coad, etc.)
--tissue_patching       Patching strategy (20x_224px_0px_overlap, etc.)
--task_name             Task name (TP53_mutation, PIK3CA_mutation, etc.)
--epochs                Training epochs (default: 100)
--fold                  Fold to train: "all" or 0-49 (default: all)
```

**Example:**
```bash
python src/train_abmil.py \
    --foundational_model ctranspath \
    --latent_dim 768 \
    --work_dir /home/JKP6679/Patho-Ensemble/PARADIS/datos/patches \
    --train_source cptac_brca \
    --tissue_patching 20x_224px_0px_overlap \
    --task_name TP53_mutation \
    --epochs 100 \
    --fold all
```

#### `test_abmil.py` — Training Set Evaluation

Re-evaluates ABMIL models on the training set for meta-learner ensemble features.

**Usage**: Same arguments as `train_abmil.py` (without fold selection)

> These predictions are **in-sample** — each model predicts on the slides it
> trained on, reaching AUC 0.94 against 0.77 at test. That is fine for
> `logreg` and actively harmful for flexible meta-learners; use
> `build_oof_features.py` below and see
> [The meta-features](#the-meta-features-in-sample-vs-out-of-fold).

#### `build_oof_features.py` — Out-of-Fold Meta-Features

Nested, patient-grouped cross-validation inside each outer fold's train split,
so every slide gets a prediction from a model that never saw it.

```bash
python src/build_oof_features.py \
    --foundational_model ctranspath \
    --work_dir /home/JKP6679/Patho-Ensemble/PARADIS/datos/patches \
    --train_source cptac_brca \
    --tissue_patching 20x_224px_0px_overlap \
    --task_name TP53_mutation \
    --n_inner 3
```

Writes `{model}_{strategy}_train_eval_oof/val_outputs_oof/fold_*/`, consumed by
`ensemble4.py --meta_features oof`. Cost is `n_inner` × the base training
(~39 min for 3 models × 50 folds on `cptac_brca` with `n_inner=3`).

Use `build_oof_features.sbatch` rather than calling it per model: the sbatch
also runs the leak check that gates everything downstream.

### Ensemble & Evaluation

#### `ensemble.py` — Weighted Averaging Ensemble

Combines predictions using per-fold weights based on validation metric performance (e.g., AUC-ROC).

**Strategy:**
- Computes weights from each model's validation performance
- Uses softmax normalization for probability interpretation
- Fast inference (no meta-model training required)
- Highly interpretable weights

**Example:**
```bash
python src/ensemble.py \
    --foundational_models ctranspath uni_v2 virchow_v1 conch512 \
    --work_dir /home/JKP6679/Patho-Ensemble/PARADIS/datos/patches \
    --train_source cptac_brca \
    --tissue_patching 20x_224px_0px_overlap \
    --task_name TP53_mutation \
    --weights_type auc_roc
```

#### `ensemble4.py` — Meta-Learner Ensemble

Stacking: learns how to combine the base models from their predictions. Nine
meta-learners are available (see
[Available meta-learners](#available-meta-learners)), selected with
`--meta_model`.

**Baseline**
```bash
python src/ensemble4.py \
    --foundational_models ctranspath uni_v2 virchow_v1 \
    --work_dir /home/JKP6679/Patho-Ensemble/PARADIS/datos/patches \
    --train_source cptac_brca \
    --tissue_patching 20x_224px_0px_overlap \
    --task_name TP53_mutation \
    --meta_model logreg
```

**With out-of-fold meta-features** — recommended for anything more flexible than
`logreg`, see [the section above](#the-meta-features-in-sample-vs-out-of-fold):
```bash
python src/ensemble4.py \
    ... --meta_model tabpfn --meta_features oof
```

**Minimal-capacity combiner** (4 parameters: one weight per base model plus a
temperature):
```bash
python src/ensemble4.py \
    ... --meta_model logit_avg --meta_features oof
```

**MLP with snapshot averaging, and its control:**
```bash
python src/ensemble4.py ... --meta_model mlp_snapshot --meta_epochs 120 --n_cycles 6 --restart_lr 5e-3
python src/ensemble4.py ... --meta_model mlp_deepens  --n_members 6   # independent seeds
```

**Feature-space options** (see
[Feature-space flags](#feature-space-flags)):
```bash
python src/ensemble4.py ... --feature_space logit --drop_redundant_class
```

**Prerequisites:**
- `test_abmil.py` for all models — always.
- `build_oof_features.py` as well, if using `--meta_features oof`.
- `train_abmil_fge.py` with the matching `--fge_tag`, if using
  `--base_source <variant>`.

`--meta_features oof` is only defined for `--base_source standard`: no
out-of-fold features are generated for the snapshot variants, and the
combination raises instead of silently reading the wrong tree.

#### `ensemble_nm_advanced.py` — Advanced Nelder-Mead Optimization

Optimizes ensemble weights using multi-start Nelder-Mead, Differential Evolution, or regularized approaches.

**Advantages over simple Nelder-Mead:**
- **Multi-start**: Explores from multiple initializations, avoids local minima
- **Sensitivity analysis**: Reports whether optimization space is "flat" or sensitive
- **Global methods**: Supports Differential Evolution for irregular search spaces
- **Regularization**: Optional penalty on extreme weights to prevent overfitting

**Example:**
```bash
python src/ensemble_nm_advanced.py \
    --foundational_models ctranspath uni_v2 virchow_v1 conch512 \
    --work_dir /home/JKP6679/Patho-Ensemble/PARADIS/datos/patches \
    --train_source cptac_brca \
    --tissue_patching 20x_224px_0px_overlap \
    --task_name TP53_mutation \
    --method nelder-mead-multistart \
    --n_starts 10 \
    --lambda_reg 0.01
```

#### `meta_models.py` — Meta-Learners

Sklearn-compatible implementations, ordered by capacity. They drop into
`ensemble4.py` without a wrapper: `_fit_meta_model` inspects the `fit` signature
to decide whether to pass a validation set.

- **`LogitAveragingMetaClassifier`**: one weight per base model over log-odds,
  plus temperature. 4 parameters; weights positive and summing to 1 by
  construction.
- **`_TinyMLP`**: shared architecture (Linear → ReLU → Dropout → Linear).
- **`MLPMetaClassifier`**: single cosine cycle + early stopping.
- **`SnapshotMLPMetaClassifier`**: cyclic LR with warm restarts, snapshot
  averaging (Huang et al., ICLR 2017).
- **`FGEMLPMetaClassifier`**: warm-up then short piecewise-linear cycles
  (Garipov et al., ICLR 2018).
- **`DeepEnsembleMLPMetaClassifier`**: k MLPs with independent seeds — the
  reference that tells apart "SE works" from "ensembling works".
- **`GatingMLPMetaClassifier`**: per-sample convex combination of the base
  models (mixture-of-experts).

Shared helpers: `_auc_from_probs` (binary or macro-OVR, returns −1.0 when
undefined so early stopping treats it as no improvement) and `_as_log_probs`
(reshapes `[N, n_models·C]` into per-model log-probabilities).

### Data Preparation

#### `create_val_splits.py` — Automatic Validation Split Generation

Creates stratified validation splits (15% of training data) at patient level across all folds.

**Example:**
```bash
python src/create_val_splits.py \
    -i /home/JKP6679/Patho-Ensemble/PARADIS/datos/patches/cptac_brca/TP53_mutation/k=all.tsv \
    -o /home/JKP6679/Patho-Ensemble/PARADIS/datos/patches/cptac_brca/TP53_mutation/k=all_with_val.tsv \
    --n-folds 50 \
    --val-frac 0.15
```

### Utilities

#### `utils.py` — Metrics & Weighting

**`Metrics` class:**
- Computes per-fold and aggregate classification metrics
- ROC curves, confusion matrices, precision-recall curves
- 95% bootstrapped confidence intervals (1000 iterations by default)
- Saves results in structured JSON format

**`MetricDistance` class:**
- Calculates per-fold weights for ensemble.py
- Intelligent metric name mapping (auc_roc, macro-ovr-auc, etc.)
- Fallback to test metrics if validation metrics missing
- Equal weight assignment for problematic folds

---

## Ensemble Approaches

### Comparison Matrix

| Aspect | Weighted Avg | Meta-Learner | Advanced NM |
|--------|---|---|---|
| **Speed** | ⚡ Fast (~1 min) | ⚡ Fast (~5-15 min) | ⚡ Fast (~5-10 min) |
| **Optimization** | Heuristic | Learned | Advanced search |
| **Complexity** | Simple | Medium-High | High |
| **Prerequisites** | Train all models | Train + test_abmil | Train + validation splits |
| **Expected improvement** | Baseline | +1-3% | +0.5-2% |
| **Best for** | Quick baseline | Larger datasets | Fine-tuning |

### When to Use Each

**Use Weighted Averaging (`ensemble.py`) if:**
- You want quick setup without meta-learning
- Individual model performance already predictive of ensemble quality
- Few training samples (< 100) where meta-learner might overfit
- Maximum interpretability is critical

**Use Meta-Learner (`ensemble4.py`) if:**
- You have larger datasets (> 200 samples)
- You want to detect model synergies
- Individual models have similar but non-identical performance
- You can spend 5-15 extra minutes on test_abmil.py + meta-learner training

**Use Advanced Nelder-Mead (`ensemble_nm_advanced.py`) if:**
- Weighted averaging doesn't match meta-learner performance
- You want sensitivity analysis before committing to optimization
- Space may be irregular (use Differential Evolution)
- You need fine-grained weight tuning with regularization

---

## Running Multiple Experiments

Experiments are declared in [`experiments.yaml`](experiments.yaml). Adding a
dataset/task combination is a one-line change — no script edits.

```yaml
work_dir: /home/JKP6679/Patho-Ensemble/PARADIS/datos/patches

defaults:
  models: [ctranspath, uni_v2, virchow_v1]
  tissue_patching: 20x_224px_0px_overlap
  epochs: 100

  # Snapshot variants. `schedule` picks the LR shape (fge | se); each variant
  # writes to its own tree, so they coexist.
  fge_variants:
    - {name: fge_orig,  lr_1: 2e-4, lr_2: 2e-5, cycle_length: 4,  n_cycles: 6, cycle_patience: 2}
    - {name: fge_wbase, lr_1: 2e-4, lr_2: 2e-5, cycle_length: 4,  n_cycles: 6, cycle_patience: 2, include_base: true}
    - {name: se_orig,   schedule: se, lr_1: 1e-3, lr_2: 1e-6, cycle_length: 15, n_cycles: 6, cycle_patience: 2}

  meta_configs:
    - {name: m0_logreg_std,  meta_model: logreg, base_source: standard}
    - {name: m2_logreg_fge,  meta_model: logreg, base_source: fge_orig}
    - {name: f1_logreg_oof,  meta_model: logreg, meta_features: oof}
    - {name: f15_tabpfn_oof, meta_model: tabpfn, meta_features: oof}

experiments:
  - {dataset: cptac_brca, task: TP53_mutation, enabled: true, max_folds: 50}
  - {dataset: cervical_subtype, task: subtype, enabled: false, max_folds: 10}
```

**Per-variant fields** (`fge_variants`), defaults in `FGE_DEFAULTS`:
`schedule` (`fge`|`se`), `lr_1`, `lr_2`, `cycle_length`, `n_cycles`,
`cycle_patience`, `include_base`.

**Per-config fields** (`meta_configs`), defaults in `META_DEFAULTS`:
`base_source` (`standard` or a variant name), `meta_features`
(`insample`|`oof`), `feature_space` (`prob`|`logit`), `drop_redundant_class`.

Any row may override any `defaults` key — useful for datasets with fewer
foundational models available:

```yaml
  - {dataset: cptac_luad, task: EGFR_mutation, enabled: true,
     models: [ctranspath, uni_v2]}
```

> **`max_folds` belongs on the experiment row, not in `defaults`.** Fold counts
> vary by dataset (`cptac_brca` 50, `cervical_subtype` 5, `imp` 1), and a global
> `max_folds` silently caps every dataset that has more. The pipeline uses
> `min(max_folds, actual fold_* columns)`, so setting it too high is harmless —
> setting it in `defaults` is not.

### Commands

```bash
python src/experiment_config.py --list          # readable summary of active runs
python src/experiment_config.py --list-active   # machine-readable "dataset<TAB>task"

./run_full_pipeline.sbatch                      # first active experiment
ALL_ACTIVE=1 ./run_full_pipeline.sbatch         # every enabled: true row

# A combination not in the file (uses defaults, warns on stderr)
DATASET=cptac_gbm TASK=EGFR_mutation ./run_full_pipeline.sbatch
```

Results and logs are namespaced per experiment, so concurrent runs on different
tasks never collide:

```
PARADIS/datos/patches/{dataset}/{task}/abmil/ensemble4*/
logs/{dataset}_{task}/
```

### Meta-learner output directories

The directory name is built by appending, in order: the meta-model's base name,
then `_{base_source}` if not `standard`, then `_{meta_features}` if not
`insample`, then `_{feature_space}` if not `prob`, then `_dropc`.

| `--meta_model` | `--base_source` | `--meta_features` | Output |
|---|---|---|---|
| `logreg` | `standard` | `insample` | `ensemble4/` |
| `tabpfn` | `standard` | `insample` | `ensemble4_tabpfn/` |
| `logreg` | `fge_orig` | `insample` | `ensemble4_fge_orig/` |
| `tabpfn` | `standard` | `oof` | `ensemble4_tabpfn_oof/` |
| `logit_avg` | `standard` | `oof` | `ensemble4_logit_avg_oof/` |

> Every variant lands in its own directory on purpose: comparing regimes
> requires keeping both, and an overwritten result is indistinguishable from a
> current one. The same mapping is duplicated in
> `run_3_experiments_v2_improved.sbatch`; **if you register a meta-model in
> `ensemble4.py`, add it there too.** A mismatch does not raise — it reads the
> wrong directory and prints a plausible, wrong table.

Stage 3 prints the paired comparison described under
[Analysis protocol](#analysis-protocol).

---

## Data Preparation

### Automatic Validation Split Workflow (SBATCH)

The pipeline launcher (`run_full_pipeline.sbatch`) automates these stages:

```
[PASO 0] Create/replace validation splits (train 85% / val 15%)
         └─ Runs create_val_splits.py automatically
         └─ Marks <2 case folds with minimal validation
         └─ Verifies split counts

[PASO 1-2] Verification
         └─ Verify that get_features_dir() works
         └─ Verify that test_abmil.py imports correctly

[PASO 3] Smoke test (ctranspath)
         └─ Generate predictions (50 folds)
         └─ Verify everything works

[PASO 4] Training (uni_v2)  ⏱️ ~45-60 minutes
         └─ Train 50 folds
         └─ Generate checkpoints and metrics

[PASO 5] Predictions (uni_v2 + virchow_v1)
         └─ Generate training predictions (2 models × 50 folds)
         └─ Data for meta-learner

[PASO 6] Ensemble (3 meta-learner experiments)
         └─ Logistic Regression (baseline)
         └─ MLP single-cycle
         └─ MLP snapshot ensemble
         └─ Generate comparison report

[RESULTADO] Final outputs in:
         └─ PARADIS/datos/patches/cptac_brca/TP53_mutation/abmil/
```

---

## PARADIS Data Usage

> Full reference: [PARADIS_DATA_REFERENCE.md](PARADIS_DATA_REFERENCE.md)

### `--work_dir` must end in `/patches`

Embeddings and splits do **not** share a parent directory:

| Data | Path |
|---|---|
| Embeddings | `PARADIS/datos/features/{dataset}/` |
| Splits & results | `PARADIS/datos/patches/{dataset}/{task}/` |

So the correct value is:

```
--work_dir /home/JKP6679/Patho-Ensemble/PARADIS/datos/patches
```

`get_features_dir` resolves embeddings by also trying `{work_dir}/../features/`,
which lands on `datos/features/` as intended.

> **Do not use `--work_dir .../datos`.** It resolves splits to `datos/{dataset}/`,
> a leftover test tree with its own splits and checkpoints. Results written there
> are not comparable with the real ones.

### Available datasets & tasks

All figures below were read off the data, not copied from older notes. Earlier
versions of this file overstated several dataset sizes — `cptac_coad` was listed
as 327 samples when it is in fact the **smallest** dataset available.

| Dataset | Tasks | Slides | Patients | Folds | Feature dirs |
|---|---|---:|---:|---:|---:|
| **imp** | grade | 5333 | 5333 | **1** ⚠️ | 12 |
| **cervical_subtype** | subtype (4 classes) | 599 | 599 | **5** | 12 |
| **hancook** | 7 tasks (grading, invasion…) | 387 | 383 | **5** | 1 ⚠️ |
| **cptac_luad** | EGFR, KRAS, STK11, TP53, OS, Immune_class | 324 | 108 | 50 | 1 ⚠️ |
| **cptac_lscc** | ARID1A, KEAP1, Histologic_Grade, Immune_class | 304 | 108 | 50 | 8 |
| **cptac_hnsc** | CASP8_mutation, Histologic_Grade, Immune_class | 256 | 107 | 50 | 8 |
| **cptac_ccrcc** | VHL, PBRM1, BAP1, Immune_class | 245 | 103 | 50 | 8 |
| **cptac_gbm** | EGFR_mutation, TP53_mutation, Immune_class | 243 | 99 | 50 | 15 |
| **bc_therapy** | er_status, her2_status, grade, RCB | 166 | 166 | 50 | 12 |
| **crc_outcomes** | 5 tasks (braf_*) | 136 | 134 | 50 | 1 ⚠️ |
| **cptac_brca** | TP53_mutation, PIK3CA_mutation, Immune_class | 112 | 103 | 50 | 12 |
| **cptac_coad** | TP53, KRAS, APC, ARID1A, ACVR2A, PIK3CA, SETD1B, MSI_H, Immune_class | 98 | 94 | 50 | 12 |

⚠️ One feature directory is not enough for the 3-model default ensemble — override
`models` in that experiment's row in `experiments.yaml`.

**Three columns that matter more than they look:**

- **Patients, not slides**, is the effective sample size: splits are stratified at
  patient level, so the `cptac_*` datasets have far less independent data than
  their slide counts suggest (`cptac_luad`: 324 slides but only 108 patients).
- **Folds** drives the width of the confidence intervals — see below.
- ⚠️ **`imp` has a single fold**, which sends `utils.Metrics` down a different code
  path (bootstrap over pooled predictions, emitting `lower`/`upper`) instead of the
  multi-fold path (`mean`/`se`). Comparison tooling must handle both key sets.

### How the confidence intervals are computed

For multi-fold runs `utils.Metrics` reports **`se = std(per-fold scores) / √n_folds`**.
The bootstrap path is only taken when `num_folds == 1`. Two consequences that are
easy to get wrong:

- **The number of patients does not enter the formula.** Picking a dataset with
  more patients does not by itself narrow the interval.
- **What matters is the number of folds and the stability of each fold's score.**
  These pull in opposite directions when the sample is fixed: more folds means
  smaller test sets, hence noisier per-fold scores.

Measured on this data:

| | std across folds | folds | SE | CI width |
|---|---:|---:|---:|---:|
| cptac_brca (test=22/fold) | 0.092 | 50 | 0.013 | 0.051 |
| cervical_subtype (test=120/fold) | 0.020 | 5 | 0.009 | **0.035** |

cervical_subtype ends up with the *narrower* interval despite having a tenth of the
folds, because each fold's AUC is estimated from 120 test samples instead of 22.
Use this table's shape — not patient count alone — when choosing a dataset for a
comparison that needs to resolve small differences.

### Embedding dimensions — no longer your problem

`--latent_dim` is now **optional**. Omit it and the dimension is read from the
`.h5` itself, which removes this failure mode entirely:

```
RuntimeError: size mismatch for projection.0.weight:
  copying a param with shape torch.Size([512, 1536]) from checkpoint,
  the shape in current model is torch.Size([512, 768])
```

Verified dimensions in `cptac_brca` — note that `uni_v2` has **no** `_monai`
suffix, so the directory name is not a reliable guide:

| Model | Directory | Dimension |
|---|---|---|
| ctranspath | `features_ctranspath_monai` | 768 |
| uni_v2 | `features_uni_v2` | **1536** |
| virchow_v1 | `features_virchow_v1_monai` | **2560** |
| uni_v1 | `features_uni_v1` | 1024 |
| conch512 | `features_conch512_monai` | 768 |
| phikon_v2 | `features_phikon_v2` | 1024 |
| virchow2 | `features_virchow2` | 2560 |

To check one by hand:

```bash
cd src && python3 -c "
from utils import get_features_dir, detect_latent_dim
d = get_features_dir('/home/JKP6679/Patho-Ensemble/PARADIS/datos/patches', 'cptac_brca', 'uni_v2')
print(d, detect_latent_dim(d))"
```

Passing `--latent_dim` explicitly still works and takes precedence, which is useful
for reproducing older runs.

### Data structure

```
PARADIS/datos/
│
├── features/                          # Embeddings (HDF5)
│   └── {dataset}/
│       ├── features_{model}_monai/    # with suffix
│       └── features_{model}/          # without suffix (e.g. uni_v2)
│           └── *.h5                   # coords [N,2] int64, features [N,D] float32
│
├── patches/                           # ← work_dir root
│   └── {dataset}/
│       ├── {strategy}/                # e.g. 20x_224px_0px_overlap
│       │   ├── patches/               #   patch coordinates
│       │   └── visualization/         #   JPG thumbnails
│       │
│       └── {task}/                    # e.g. TP53_mutation
│           ├── k=all.tsv              #   master splits (50 folds)
│           ├── config.yaml            #   task metadata
│           └── abmil/
│               ├── {model}_{strategy}/
│               │   ├── checkpoints/fold_*/model.pt
│               │   ├── val_outputs/fold_*/
│               │   ├── test_outputs/fold_*/
│               │   ├── checkpoints_fge/fold_*/snapshot_*.pt
│               │   ├── val_outputs_fge/fold_*/
│               │   └── test_outputs_fge/fold_*/
│               ├── {model}_{strategy}_train_eval/
│               │   └── val_outputs/fold_*/        ← meta-learner features
│               ├── {model}_{strategy}_train_eval_fge/
│               │   └── val_outputs_fge/fold_*/    ← same, FGE variant
│               └── ensemble4*/                     ← ensemble results
│
└── wsis/                              # Original whole-slide images
```

Note the asymmetry in the FGE tree: `_train_eval` holds `val_outputs/`, while
`_train_eval_fge` holds `val_outputs_fge/`.

### Split file

`patches/{dataset}/{task}/k=all.tsv` — tab-separated:

| case_id | slide_id | TP53_mutation | fold_0 | … | fold_49 |
|---|---|---|---|---|---|
| 01BR001 | 01BR001-0684a407 | 1 | train | … | test |
| 01BR002 | 01BR002-a1c2d4e5 | 0 | val | … | train |

Typical split (cptac_brca / TP53_mutation — 112 slides, 103 patients):
train 75 · val 15 · test 22. Stratification is at patient level.

### Quick examples

```bash
# Single model, single fold — latent_dim auto-detected
python src/train_abmil.py \
    --foundational_model ctranspath \
    --work_dir /home/JKP6679/Patho-Ensemble/PARADIS/datos/patches \
    --train_source cptac_brca \
    --tissue_patching 20x_224px_0px_overlap \
    --task_name TP53_mutation \
    --epochs 100 \
    --fold 0
```

```bash
# Three base models, all 50 folds, via the parametrized launcher
DATASET=cptac_brca TASK=TP53_mutation sbatch train_base_models.sbatch
```

```bash
# Weighted-averaging ensemble
python src/ensemble.py \
    --foundational_models ctranspath uni_v2 virchow_v1 \
    --work_dir /home/JKP6679/Patho-Ensemble/PARADIS/datos/patches \
    --train_source cptac_brca \
    --tissue_patching 20x_224px_0px_overlap \
    --task_name TP53_mutation \
    --weights_type auc_roc
```

### Important notes

1. **`--work_dir` ends in `/patches`.** Anything else writes to the test tree.
2. **`--latent_dim` is optional** — auto-detected from the `.h5`.
3. **Splits already include validation.** Do not pass `--create_val` for PARADIS data.
4. **`--tissue_patching` must match the directory name** exactly.
5. **Count `model.pt`, not directories,** when checking whether a model is trained:
   some folds contain legacy `epoch_9.pt` files (old `patho_bench` format, different
   architecture) that the current pipeline cannot load.

---

## Experiments & Meta-Learners

### The meta-features: in-sample vs out-of-fold

**Read this before interpreting any meta-learner result.** It is the single
biggest factor in how these models rank against each other.

The meta-learner is trained on the base models' predictions. By default those
come from `test_abmil.py`, which takes the fold's ABMIL and predicts on **the
same slides it trained on**. Measured on `cptac_brca/TP53_mutation`:

| Split | What it is | Base-model AUC |
|---|---|---:|
| `_train_eval/val_outputs` (**meta-train**) | predictions on their *own* train split | **0.934–0.944** |
| `val_outputs` (meta-val, early stopping) | real validation, unseen | 0.795–0.859 |
| `test_outputs` (meta-test) | test | 0.764–0.795 |

So the meta-learner learns to combine signals worth AUC 0.94 and is then applied
where they are worth 0.77. This is the classic stacking pitfall that Wolpert
(1992) and Breiman (1996) address by requiring **out-of-fold** predictions, and
it penalises flexible meta-models most: the better they fit a misspecified
training distribution, the worse they transfer. It explains why a plain MLP
collapses (macro-F1 falls from 0.672 to 0.39–0.53 in `grid_search_results.json`),
why TabPFN underperforms, and why Snapshot Ensembles appear to "fix" the MLP —
they regularise it back towards logistic regression rather than beyond it
(AUC 0.7445 vs the 0.744 baseline).

`src/build_oof_features.py` generates the corrected features via **nested,
patient-grouped cross-validation**: within each outer fold's train split it runs
an inner K-fold, so every slide gets a prediction from a model that never saw
it.

```bash
DATASET=cptac_brca TASK=TP53_mutation sbatch build_oof_features.sbatch
```

Measured on `cptac_brca`: 39 min for 3 models × 50 folds with `n_inner=3`, and
base-model AUC on the meta-features drops from 0.93–0.94 to **0.68–0.71**. The
sbatch runs that check itself and **fails the job** if the drop does not happen,
because a leak here silently invalidates every downstream comparison.

Note the residual bias: out-of-fold AUC lands *below* test AUC (0.70 vs 0.77),
because inner models train on 50 slides instead of 75. The mismatch does not
vanish — it flips from **+0.17 optimistic to −0.08 pessimistic**, roughly halved.
The sign matters more than the size: a meta-learner trained on signals *worse*
than it will see learns to combine conservatively, while one trained on *better*
signals learns to over-trust. Raise `--n_inner 5` to shrink it further, at ~1 h
extra GPU.

Select the regime with `--meta_features {insample,oof}`. Results are written to
separate directories (`ensemble4_oof/`, …), so both regimes can be compared
rather than one overwriting the other.

### Available meta-learners

Ordered by capacity — which is the axis that matters here, given ~75 meta-train
samples and a feature matrix of rank 4:

| `--meta_model` | Parameters (3 models, binary) | Notes |
|---|---|---|
| `logit_avg` | 4 | One weight per base model over log-odds, plus temperature. Weights positive and sum to 1 by construction. The learned-weight counterpart of `ensemble.py`. |
| `logreg` | 7 | Baseline. Everything is measured against it. |
| `mlp` | ~130 | `_TinyMLP`, single cosine cycle + early stopping. |
| `mlp_snapshot` | ~130 × m | Snapshot Ensembles (Huang et al., ICLR 2017) over the meta-learner. |
| `mlp_fge` | ~130 × m | FGE (Garipov et al., ICLR 2018) over the meta-learner. |
| `mlp_deepens` | ~130 × k | k MLPs with independent seeds. **The control that makes `mlp_snapshot` interpretable** — SE is sold as a cheap approximation to training separate models, so without this reference you cannot tell "SE works" from "ensembling works, and SE is a worse way to do it". |
| `gating` | ~110 | Per-sample weights over base models (mixture-of-experts); prediction stays inside the convex hull of the base predictions. Only meaningful with `--meta_features oof`. |
| `tabpfn` | — | Prior-Data Fitted Network, in-context learning. |
| `lightgbm` | — | Tabular reference. Not SE/FGE-compatible (already an ensemble). |

### Feature-space flags

Two structural problems with the raw features, both measured on `cptac_brca`:

- **Rank deficiency.** Each softmax vector sums to 1, so the 6 feature columns
  span only **4 dimensions**. `--drop_redundant_class` removes one class column
  per base model.
- **Wrong space.** Probabilities are combined directly; the natural space for
  combining classifiers is log-odds. `--feature_space logit` applies
  `log(p/(1−p))` with clipping, identically to train/val/test.

`--drop_redundant_class` is rejected for `logit_avg` and `gating`, which need
each base model's full class distribution.

### Architecture Details

#### _TinyMLP (shared)
```
Input (in_dim ~ 6–10)
  ↓
Linear(in_dim → hidden_dim=16)
  ↓
ReLU()
  ↓
Dropout(p=0.1)
  ↓
Linear(hidden_dim → num_classes)
  ↓
Output logits [N, num_classes]
```

#### MLPMetaClassifier (single-cycle)
- **Optimizer**: AdamW(lr=1e-3, wd=1e-4)
- **LR Schedule**: CosineAnnealingLR(T_max=epochs)
- **Training**: Standard PyTorch loop, 1 epoch of data per epoch
- **Early stopping**: On validation AUC, patience=8, keeps best state in memory
- **Interface**: `.fit(X_train, y_train, X_val=None, y_val=None)`, `.predict_proba(X)`

#### SnapshotMLPMetaClassifier (cyclic + snapshots)
- **Optimizer**: AdamW(lr=restart_lr=5e-3, wd=1e-4)
- **LR Schedule**: CosineAnnealingWarmRestarts(T_0=cycle_len, T_mult=1, eta_min=0.005×1e-3)
- **Training**: M cycles of `epochs/M` each; forces traversal of multiple minima
- **Snapshots**: Saves state_dict at end of each cycle
- **Prediction**: Averages softmax of last m snapshots (h_Ensemble = (1/m)∑h_i(x))
- **Interface**: `.fit(X_train, y_train, ...)`, `.predict_proba(X)` (ignores X_val/y_val)

---

## Snapshot-based ensembles on the base models (FGE & SE)

Both methods diversify a *single* ABMIL by cycling its learning rate and keeping
a snapshot per cycle. They differ only in the shape of the cycle, so
`src/abmil_fge.py` implements both and `--schedule` picks between them — the
rest of the loop (snapshots, inter-cycle early stopping, `--include_base`) is
shared, which is what makes the comparison attributable to the method.

| | `--schedule fge` | `--schedule se` |
|---|---|---|
| Paper | Garipov et al., ICLR 2018 | Huang et al., ICLR 2017 |
| LR shape | piecewise-linear ramp, small amplitude | restart high, cosine down to ~0 |
| Cycle length | short (3–4 epochs) | long (15+ epochs) |
| Starting point | an already-converged minimum | pushes *out* of the current minimum |
| `--lr_1` means | cycle ceiling | restart LR (α₀) |
| Cost on `cptac_brca` | ~1 h (3 models × 50 folds) | ~3–4 h (longer cycles) |

### Measured results — a bounded null

Run on `cptac_brca/TP53_mutation` with 50 folds (jobs 69938–69943), FGE does
**not** improve the ensemble. None of the 9 configurations separates from the
baseline on any metric, and the paired CIs bound the effect on AUC below
**+0.017**, with a minimum detectable effect of ≈0.016 — a null result *with a
bound*, not an absence of evidence.

The mechanism is measured, not guessed: **the diversity budget is already spent
on the multi-model axis.**

| Pairwise correlation of predictions | r |
|---|---:|
| Snapshots of the same model (FGE variants) | 0.931–0.977 |
| Different foundation models | **0.797** |

Snapshots of one model are near-duplicates of each other compared to the spread
already present between `ctranspath`, `uni_v2` and `virchow_v1`. Worse for the
diversity hypothesis: across the four FGE variants, **variance reduction and AUC
do not correlate** — `fge_wbase` is the most diverse (5.0 % reduction) and the
worst of the four, `fge_lowlr` the least diverse (1.3 %) and the best. SE is
being evaluated precisely because it moves further along that axis; the
pre-registered prediction is that it lands at r ≈ 0.88–0.93, still far above the
0.797 floor, with AUC inside ±0.017 of baseline.

Full write-up: [`INFORME_ABLATION_FGE_cervical_subtype.md`](INFORME_ABLATION_FGE_cervical_subtype.md).

### How to run

Variants are declared in `experiments.yaml` and each writes to its own tree, so
they coexist:

```bash
# All variants declared for the active experiment
DATASET=cptac_brca TASK=TP53_mutation sbatch train_base_models_fge_v2.sbatch

# A single variant (useful to fan out one SLURM job per variant)
FGE_ONLY=se_orig DATASET=cptac_brca TASK=TP53_mutation \
    sbatch train_base_models_fge_v2.sbatch
```

**Output structure** — `<tag>` is the variant name (`fge_orig`, `se_orig`, …):

```
{model}_{strategy}/
├── checkpoints_<tag>/fold_*/
│   ├── base_model.pt                 # warm-start checkpoint
│   └── snapshot_0.pt … snapshot_n.pt
├── val_outputs_<tag>/fold_*/         # labels, averaged preds, preds_per_snapshot
└── test_outputs_<tag>/fold_*/
{model}_{strategy}_train_eval_<tag>/
└── val_outputs_<tag>/fold_*/         # meta-learner features from these bases
```

`preds_per_snapshot.npy` is what makes the diversity analysis above possible —
keep it.

**Scripts:**
- `src/train_abmil_fge.py` — orchestrator (convergence + cycles), `--schedule`, `--fge_tag`
- `src/abmil_fge.py` — cycle training loop, shared by both schedules
- `src/test_abmil_fge.py` — meta-learner features from the snapshots (reads by tag; needs no change per variant)
- `src/fge_utils.py` — `fge_cycle_lr`, `se_cycle_lr`, and the `cycle_lr` dispatcher

### A caveat on early stopping

Measured across both datasets, `fge_orig` stops at ≈2.6 snapshots/fold despite
`n_cycles: 6` — the inter-cycle patience cuts long before the budget is spent
(2.64 on brca, 2.60 on cervical). `fge_pat3` raises patience and does produce
more snapshots (3.77), so the mechanism works; it just does not help.

---

## Analysis protocol

All configurations are evaluated on **the same folds**, so the correct test is
paired. `run_3_experiments_v2_improved.sbatch` reports, per metric:

- **Paired Δ with its 95 % CI** against the baseline configuration.
- **G/E/P** — folds won / tied / lost. Ties are not noise: with 22 test slides
  per fold, kappa took the same value in up to 22 of 50 folds, and reading
  "10/50" as "10 wins, 40 losses" badly overstates the case.
- **MDE** — minimum detectable effect at 80 % power, so a non-significant
  result is reported as *bounded* rather than indeterminate.
- **Holm-corrected p** per metric family. With ~19 configurations, some p < 0.05
  is expected by chance; an uncorrected one already cost us a false positive on
  `cervical_subtype` that failed to replicate.

Comparing independent confidence intervals — the previous behaviour — is a much
weaker test and misled us in both directions: it hid a systematic effect on
cervical and would have hidden the bound on brca.

The comparison table only reads results from configurations that **completed in
the current run**. A stale `test_metrics_summary.json` from an earlier
configuration otherwise gets reported as if it were current, which is how a
failed baseline once turned into a plausible and entirely false table.

---

## Advanced Optimization

### Advanced Nelder-Mead vs. Standard Nelder-Mead

**Problem**: Standard Nelder-Mead may not improve weighted averaging when:
- Individual models have similar performance (flat optimization space)
- Validation metrics don't correlate perfectly with test performance
- Initialization starts at equal weights (already near-optimal)

**Solution**: Multi-start and global optimization methods

### Optimization Methods

#### 1. Multi-Start Nelder-Mead (Recommended) ⭐

```bash
sbatch --export=ALL,NM_METHOD=nelder-mead-multistart,NM_N_STARTS=10 slurm/train4_ensemble_nm_tp53.sbatch
```

- Runs Nelder-Mead from 10 random initializations
- Selects best result
- Fast (< 5 minutes for 50 folds)
- Expected improvement: 0.5-2%

#### 2. Differential Evolution (Global Optimization)

```bash
sbatch --export=ALL,NM_METHOD=differential-evolution slurm/train4_ensemble_nm_tp53.sbatch
```

- True global optimization across weight space
- Better for irregular/multimodal spaces
- Slower (10-15 minutes for 50 folds)
- Expected improvement: 1-3%

#### 3. With Regularization

```bash
sbatch --export=ALL,NM_LAMBDA_REG=0.02 slurm/train4_ensemble_nm_tp53.sbatch
```

- Penalizes extreme weights
- Prevents overfitting to validation noise
- Better generalization to test set
- Recommended when validation data is small

### Sensitivity Analysis

Each optimization automatically reports:

```
📊 Sensibilidad promedio: 0.000456
   ✓ Espacio sensible: Nelder-Mead puede mejorar
```

**Interpretation:**
- **> 0.001**: Optimization space sensitive, methods should find improvements
- **0.0001-0.001**: Moderately flat space, try different methods or regularization
- **< 0.0001**: Very flat space, improvements will be minimal regardless of method

---

## Cluster Configuration

### Your Cluster Setup

- **Default partition**: `main`
- **Available GPUs**: A30, A40, A100
- **Default Slurm queue**: Uses constraints (`--constraint=gpu`)

### Available GPU Nodes

| Node | GPU | Count | Memory | Status | CPUs |
|------|-----|-------|--------|--------|------|
| gpu01-gpu02 | A30 | 2× | 32 GB | idle | 32 |
| gpu03, gpu05, gpu06 | A40 | 3× | 32 GB | idle | 32 |
| gpu04 | A40 | 1× | 32 GB | mixed | 32 |
| gpu07 | A100 | 1× | 32 GB | mixed | 32 |
| gpu08 | A100 | 1× | 32 GB | mixed | 32 |

### Sbatch Configuration

#### Default (recommended)
```bash
#SBATCH --partition=main
#SBATCH --constraint=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=4:00:00
```

#### Request specific GPU type

**For A40 GPU:**
```bash
#SBATCH --constraint=gpu,a40
```

**For A100 GPU:**
```bash
#SBATCH --constraint=gpu,a100
```

### Common Commands

```bash
# Queue the full chain (base -> FGE -> meta-learners)
./run_full_pipeline.sbatch

# Monitor in real-time
tail -f logs/cptac_brca_TP53_mutation/*.log

# Check job status
squeue -u $USER

# Cancel job
scancel <JOB_ID>

# Check GPU usage on specific node
ssh gpu05 nvidia-smi
```

---

## Troubleshooting

### Issue 1: Missing Validation Data

**Error:**
```
FileNotFoundError: [...]/val_outputs/fold_0/preds.npy
```

**Solution**: SBATCH script automatically handles this with PASO 0 (validation split creation)

### Issue 2: Empty Folds in Metrics

**Error:**
```
ValueError: Found array with 0 sample(s) while a minimum of 1 is required
```

**Solution**: ✅ Fixed in current version - empty folds are filtered before metrics computation

### Issue 3: Flat Optimization Space

**Symptom**: No improvement even with advanced methods

**Diagnosis:**
```bash
# Check optimization_stats.json
python -c "
import json
with open('[...]/abmil/optimization_stats.json') as f:
    stats = json.load(f)
    sensitivities = [s.get('avg_sensitivity', 0) for s in stats]
    print(f'Average sensitivity: {sum(sensitivities)/len(sensitivities):.6f}')
"
```

**Solutions:**
1. Increase regularization: `--lambda_reg 0.05`
2. Try Differential Evolution: `--method differential-evolution`
3. Check if all models have similar performance (expected behavior)

### Issue 4: Metric Name Mismatches

**Error:**
```
KeyError: 'auc_roc'
```

**Solution**: ✅ Fixed - `MetricDistance` now maps metric names intelligently

### Issue 5: Missing Features or Embeddings

**Error:**
```
FileNotFoundError: features_uni_v2 not found
```

**Solution**: Verify that embedding directories exist:
```bash
ls /home/JKP6679/Patho-Ensemble/PARADIS/datos/patches/features/cptac_brca/ | grep features_
```

### Issue 6: Dimension Mismatches

**Error:**
```
RuntimeError: expected scalar type Float but found Double
```

**Cause**: `--latent_dim` doesn't match actual feature dimension

**Solution**: Check HDF5 file structure
```bash
python3 -c "
import h5py
with h5py.File('/home/JKP6679/Patho-Ensemble/PARADIS/datos/patches/features/cptac_brca/features_virchow_v1_monai/sample.h5', 'r') as f:
    print('Actual dimension:', f['features'].shape[1])
"
```

---

## Output Structure

```
work_dir/train_source/task_name/abmil/

├── {model1}_{tissue_patching}/
│   ├── checkpoints/fold_*/*.pt          # Best model checkpoints
│   ├── val_outputs/fold_*/{labels,preds}.npy
│   ├── test_outputs/fold_*/{labels,preds}.npy
│   ├── val_metrics/fold_*/metrics.json
│   ├── test_metrics/fold_*/metrics.json
│   ├── val_metrics_summary.json         # Aggregate validation metrics
│   └── test_metrics_summary.json        # Aggregate test metrics
│
├── {model2}_{tissue_patching}/
│   └── ... (same structure)
│
├── {model}_train_eval/                   # Training set predictions (for meta-learner)
│   └── val_outputs/fold_*/{labels,preds}.npy
│
├── ensemble/                             # Weighted averaging results
│   ├── test_outputs/fold_*/{labels,preds}.npy
│   ├── test_metrics_summary.json
│   └── weights.npy                      # Per-fold weights
│
├── ensemble4/                            # Logistic Regression meta-learner
│   ├── test_outputs/fold_*/{labels,preds}.npy
│   ├── test_metrics_summary.json
│   └── coefs.npy / odds_ratios.npy      # Model coefficients
│
├── ensemble4_mlp/                        # MLP single-cycle meta-learner
│   ├── test_outputs/fold_*/{labels,preds}.npy
│   └── test_metrics_summary.json
│
├── ensemble4_mlp_snapshot/               # MLP snapshot ensemble meta-learner
│   ├── test_outputs/fold_*/{labels,preds}.npy
│   └── test_metrics_summary.json
│
├── ensemble_nm_adv/                      # Advanced Nelder-Mead results
│   ├── test_outputs/fold_*/{labels,preds}.npy
│   ├── test_metrics_summary.json
│   ├── comparison_report.txt            # Ensemble comparison
│   └── optimization_stats.json          # Per-fold optimization details
│
├── weights.npy                          # Weighted average ensemble weights
├── weights_nm_adv.npy                   # Nelder-Mead optimized weights
└── optimization_stats.json              # Optimization analysis
```

---

## Dependencies

Main dependencies (pinned in `environment.yml`):

- **PyTorch** 2.7.0 (with CUDA 12 support)
- **patho-bench** (custom framework for dataset management)
- **scikit-learn** (metrics, Logistic Regression)
- **pandas** (data handling)
- **numpy** (numerical operations)
- **scipy** (optimization algorithms)
- **h5py** (HDF5 file I/O)
- **tqdm** (progress bars)
- **torch** (deep learning)

### Setup

```bash
# Create environment from file
conda env create -f environment.yml
conda activate ensemble

# Or manual installation
conda create -n ensemble python=3.10
conda activate ensemble
pip install -r requirements.txt
```

---

## Recent Experimental Campaign (2026-08-21)

### Pipeline Alternatives Exploration

Three major hypotheses tested on **cptac_brca / TP53_mutation** (50 folds):

| Approach | ΔAUCμ | 95% CI | Status |
|----------|---------|-------|--------|
| **Opción C** — Early fusion (concat embeddings) | -0.0056 | [-0.0250, +0.0124] | ❌ No improvement |
| **Opción A** — Late fusion + Nelder-Mead weights | +0.0052 | [+0.0007, +0.0106] | ⚠️ Marginal, p=0.0581 |
| **Opción D** — Weighted early fusion | -0.0182 | [-0.0421, +0.0061] | ❌ Degrades performance |

**Key finding**: Current LogReg meta-learner (late fusion) remains near-optimal. Patch-level early fusion doesn't recover model collapse, and weight optimization yields marginal (borderline-insignificant) gains.

📄 **Full report**: [EXPERIMENTAL_SESSION_SUMMARY_20260821.md](EXPERIMENTAL_SESSION_SUMMARY_20260821.md)

---

## Recent Improvements

### v3.1: Pipeline Alternatives Explored (Session 2026-08-21)

- ✅ Implemented Opción A: Late Fusion + Nelder-Mead weight optimization
- ✅ Implemented Opción C: Early Fusion via patch-level concatenation (ctranspath ⊕ virchow_v1)
- ✅ Implemented Opción D: Weighted Early Fusion (weights on embeddings before ABMIL)
- ✅ Comprehensive statistical comparison: paired t-tests, 95% bootstrap CIs, Holm correction
- ✅ **Conclusion**: LogReg + late fusion is approximately optimal given fixed models

### v3.0: Fast Geometric Ensembles & TabPFN Meta-Classifier

- ✅ FGE for ABMIL base models (piecewise-linear LR cycles, snapshot averaging)
- ✅ FGE for MLP meta-learner (`FGEMLPMetaClassifier`)
- ✅ TabPFN support as meta-classifier (Prior-Data Fitting for small data)
- ✅ Flag `--base_source {standard, fge}` for choosing base model type
- ✅ Flag `--meta_model {logreg, mlp, mlp_snapshot, mlp_fge, tabpfn}`
- ✅ Fixed critical leakage bug in `ensemble4.py` (val vs test outputs)
- ✅ Improved SBATCH templates with prerequisite validation & detailed reporting

### v2.3: Meta-Learner Neural Networks

- ✅ `ensemble4.py`: Support for MLP single-cycle and MLP snapshot ensembles
- ✅ `meta_models.py`: Sklearn-compatible neural meta-learners
- ✅ Snapshot Ensemble implementation (Huang et al., ICLR 2017)
- ✅ Backward-compatible CLI with new meta-learner options
- ✅ Comprehensive testing via `test_ensemble4_neural.py`

### v2.2: Advanced Optimization

- ✅ `ensemble_nm_advanced.py`: Multi-start, Differential Evolution, regularization
- ✅ Sensitivity analysis to detect optimization potential
- ✅ Automatic fallback to equal weights when validation missing
- ✅ Comprehensive optimization statistics logging

### v2.1: Automatic Validation Splits

- ✅ `create_val_splits.py`: Automatic 85/15 train/validation splits
- ✅ Case-level stratification across 50 folds
- ✅ Integrated into SBATCH for one-command execution
- ✅ Handles edge cases (< 2 cases per fold)

### v2.0: Robust Error Handling

- ✅ Empty fold detection and graceful skipping
- ✅ Metric name mapping for different JSON formats
- ✅ Missing validation data fallback
- ✅ Detailed error messages and diagnostics

---

## Citation

If you use Patho-Ensemble in your research, please cite:

```bibtex
@software{patho_ensemble_2024,
  title={Patho-Ensemble: Foundational Model Ensembles for Pathology Classification},
  author={Your Name},
  year={2024},
  url={https://github.com/[path]}
}
```

---

## Documentation Index

**Developer & Architecture**:
- [CLAUDE.md](CLAUDE.md) — guidance for Claude Code, architecture overview, key assumptions
- [PARADIS_DATA_REFERENCE.md](PARADIS_DATA_REFERENCE.md) — data structure, datasets, fold counts

**Experiments & Validation**:
- [EXPERIMENTAL_SESSION_SUMMARY_20260821.md](EXPERIMENTAL_SESSION_SUMMARY_20260821.md) — 
  pipeline alternatives tested (early fusion, Nelder-Mead, weighted embeddings)
- [WHAT_DIDNT_WORK.md](WHAT_DIDNT_WORK.md) — failed experiments with analysis, 
  to avoid re-running them
- [INFORME_ABLATION_FGE_cervical_subtype.md](INFORME_ABLATION_FGE_cervical_subtype.md) — 
  FGE ablation and 50-fold replication results

**Configuration**:
- [experiments.yaml](experiments.yaml) — experiment matrix; comments explain each variant

---

## Support

For issues or questions:
1. Check the [Troubleshooting](#troubleshooting) section
2. Review SBATCH script comments (`slurm/train4_ensemble_nm_tp53.sbatch`)
3. Check `optimization_stats.json` for detailed per-fold analysis
4. Review example scripts in the repository
5. Consult [CLAUDE.md](CLAUDE.md) for architecture details and key assumptions
