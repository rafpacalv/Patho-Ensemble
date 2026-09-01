# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**Patho-Ensemble** is a machine learning pipeline for training and evaluating foundational model ensembles for pathology image classification. It uses Attention-Based Multiple Instance Learning (ABMIL) with patch embeddings from various foundational models (ViT, CTransPath, Virchow, UNI, etc.) and can combine them using weighted averaging or meta-learning approaches.

## Data location — read this first

The working tree is **`/home/JKP6679/Patho-Ensemble/PARADIS/datos/`**.

`/shared/home/PARADIS/datos/` is a **different user's test copy**. Do not read
from or write to it; results produced there are not comparable with this
project's, because they were trained on that tree's splits.

Within the working tree, embeddings and splits hang off **different parents**:

| Data | Path |
|---|---|
| Embeddings | `PARADIS/datos/features/{dataset}/features_{model}[_monai]/*.h5` |
| Splits, checkpoints, results | `PARADIS/datos/patches/{dataset}/{task}/` |

`--work_dir` is therefore `.../datos/patches`, and `utils.get_features_dir`
resolves the features by also trying `{work_dir}/../features/`. It tries the
`_monai` suffix first and then no suffix — both exist in practice (`uni_v2` has
no suffix).

## Architecture & Key Concepts

### Core Workflow Stages

1. **Base training** — `train_abmil.py` trains one ABMIL per foundational model
   and fold. The training engine is `abmil_engine.py` (self-contained: gated
   attention, early stopping, patient-grouped internal validation split).
   `patho_bench` is **no longer** in the training path.
2. **Meta-features** — the meta-learner needs base-model predictions:
   - `test_abmil.py` predicts on each fold's *own train split* (**in-sample**).
   - `build_oof_features.py` produces **out-of-fold** predictions via nested,
     patient-grouped CV. See "Meta-features" below — this distinction dominates
     meta-learner results.
3. **Early fusion (optional)** — `ensemble5.py` trains a single ABMIL on
   concatenated embeddings from multiple foundational models *at the patch level*.
   **It does not work**: measured Δ AUC = −0.0056 on cptac_brca over 50 folds.
   See "Early Fusion Mechanism" below. Requires models that share identical
   patch coordinates — see "Patch coordinate groups" for which ones do.
4. **Snapshot diversification (optional)** — `train_abmil_fge.py` +
   `abmil_fge.py` add snapshot ensembles on top of converged base models, with
   `--schedule fge` (Garipov et al., ICLR 2018) or `--schedule se`
   (Huang et al., ICLR 2017). `test_abmil_fge.py` turns snapshots into
   meta-features.
5. **Ensemble combination**:
   - `ensemble.py` — weighted averaging, weights from validation metrics.
   - `ensemble4.py` — stacking with any of nine meta-learners
     (`src/meta_models.py`).
5. **Metrics** — `utils.Metrics` computes per-fold and aggregate stats.

### Meta-features: in-sample vs out-of-fold

Measured on `cptac_brca/TP53_mutation`, base-model AUC on each split:

| Split | Role | AUC |
|---|---|---:|
| `_train_eval/val_outputs` | meta-train (in-sample) | **0.934–0.944** |
| `val_outputs` | meta-val (early stopping) | 0.795–0.859 |
| `test_outputs` | meta-test | 0.764–0.795 |

The meta-learner trains on signals worth 0.94 and is applied where they are
worth 0.77 — the stacking pitfall Wolpert (1992) and Breiman (1996) solve with
out-of-fold predictions. It penalises flexible meta-models most, which is why a
plain MLP collapses, TabPFN underperforms, and Snapshot Ensembles appear to
"fix" the MLP (they regularise it back to logistic regression, not past it).

`build_oof_features.py` drops that to 0.68–0.71. Select with
`ensemble4.py --meta_features {insample,oof}`.

### Base model selection dominates combination rule

Measured 2026-08-21 on `cptac_brca/TP53_mutation`, 50 folds, with
`ensemble4.py --meta_model logreg` (defaults), all in one pass against
current base predictions:

| Config | AUC | bacc | kappa |
|---|---:|---:|---:|
| `ctranspath+uni_v2+virchow_v1` (the historical trio) | 0.7616 | 0.6702 | 0.3502 |
| **`ctranspath+uni_v2+conch_v1_5`** | **0.7986** | **0.7302** | **0.4629** |
| all four (adding conch, keeping virchow) | 0.7986 | 0.7291 | 0.4606 |

Paired Δ for the swap: AUC **+0.0370** [+0.0230, +0.0514], W/T/L 34/4/12;
kappa **+0.1127** [+0.0657, +0.1601]. All four metrics significant after Holm
(p ≈ 1.4e-4) and 1.7–1.8× the MDE. Results in
`results_model_selection_sweep.json`.

Two consequences:

- **`virchow_v1` is dispensable, not merely weaker** (AUC alone 0.6823).
  Adding conch without removing it ties with the swap, and dropping it with no
  replacement (`ctranspath+uni_v2` = 0.7650) already beats the trio.
- **The gain is the base model, not the combiner.** conch_v1_5 alone scores
  0.7902; stacking lifts it to 0.7986. The meta-learner is worth +0.008 over
  the best single model, while changing the base model is worth +0.037 —
  against +0.0052 for Nelder-Mead, −0.001 for the feature ladder and −0.0056
  for early fusion. **Check which base models go in before tuning how they are
  combined.**

### Patch coordinate groups

Models with byte-identical `coords` can be fused, compared per patch, or share
attention masks; models whose coords differ cannot without spatial registration.

**Coordinate grouping is a property of the DATASET's patching, not of the
models.** Verified on 30 slides per dataset:

| Dataset | `ctranspath` == `conch_v1_5` | All three of the trio equal |
|---|---|---|
| `cptac_brca` | 30/30 | **0/30** |
| `cptac_gbm` | 30/30 | **30/30** |
| `bc_therapy` | 30/30 | **30/30** |
| `cervical_subtype` | 30/30 | **30/30** |

So in three of the four datasets the whole trio can be fused at patch level.
The limitation that kept `ensemble5.py` to two of three models is **specific to
`cptac_brca`**, and three-way early fusion has never been tried anywhere else.

An earlier version of this file listed fixed groups by model (4068 patches for
hoptimus1/phikon_v2/uni_v1/uni_v2/virchow2, 2100 for conch_v1_5/ctranspath/
virchow_v1). Both the grouping-by-model and the patch counts are **false in
general** — they described `cptac_brca` only. Always recompute a `coords` hash
per model *per dataset*; never assume.

### Key Dependencies

- **PyTorch** (2.8.0): training and inference
- **scikit-learn**: metrics, LogisticRegression, `StratifiedGroupKFold`
- **h5py**: reading patch embeddings
- **scipy**: paired tests in the comparison table
- **tabpfn** (8.1.0), **lightgbm** (4.7.0): optional meta-learners, imported
  lazily so their absence does not break the rest
- **patho_bench**: still supplies `Metrics` via
  `experiments/utils/ClassificationMixin.py`; no longer used for training

### Data Structure

Under `work_dir/{dataset}/{task}/abmil/`:

```
{model}_{patching}/
├── checkpoints/fold_*/model.pt          # base models
├── checkpoints_{tag}/fold_*/            # snapshot variants (fge_orig, se_orig, …)
├── val_outputs/, test_outputs/          # base predictions
└── val_outputs_{tag}/, test_outputs_{tag}/
{model}_{patching}_train_eval/val_outputs/        # meta-features, in-sample
{model}_{patching}_train_eval_oof/val_outputs_oof/ # meta-features, out-of-fold
{model}_{patching}_train_eval_{tag}/val_outputs_{tag}/
ensemble4*/                              # one directory per meta-config
```

## Common Development Tasks

### Running Training

**Single fold (quick test)**:
```bash
python src/train_abmil.py \
    --foundational_model ctranspath \
    --work_dir /home/JKP6679/Patho-Ensemble/PARADIS/datos/patches \
    --train_source cptac_brca \
    --tissue_patching 20x_224px_0px_overlap \
    --task_name TP53_mutation \
    --epochs 100 \
    --fold 0
```

**All folds** (`min(max_folds, fold_* columns)`):
```bash
python src/train_abmil.py \
    --foundational_model ctranspath \
    --work_dir /home/JKP6679/Patho-Ensemble/PARADIS/datos/patches \
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
    --work_dir /home/JKP6679/Patho-Ensemble/PARADIS/datos/patches \
    --train_source cptac_brca \
    --tissue_patching 20x_224px_0px_overlap \
    --task_name TP53_mutation \
    --weights_type auc_roc
```

**Early fusion ensemble** (concatenate embeddings at patch level — **measured
Δ AUC = −0.0056; kept for reference, not recommended**):
```bash
# Single command: trains fused ABMIL (ctranspath ⊕ virchow_v1) + stacks with uni_v2
python src/ensemble5.py \
    --work_dir /home/JKP6679/Patho-Ensemble/PARADIS/datos/patches \
    --train_source cptac_brca \
    --tissue_patching 20x_224px_0px_overlap \
    --task_name TP53_mutation \
    --fuse_models ctranspath virchow_v1 \
    --other_model uni_v2 \
    --fold_start 0 \
    --fold_end 49 \
    --epochs 100 \
    --meta_model logreg
```
**Note**: Models in `--fuse_models` must share identical patch coordinates 
(verified automatically; fails hard if mismatch). `--other_model` combines via 
late fusion stacking afterward.

**Meta-learner ensemble** (requires running `test_abmil.py` first):
```bash
# Step 1: Get training predictions for each model
for model in ctranspath uni_v2 virchow_v1; do
  python src/test_abmil.py \
      --foundational_model $model \
      --work_dir /home/JKP6679/Patho-Ensemble/PARADIS/datos/patches \
      --train_source cptac_brca \
      --tissue_patching 20x_224px_0px_overlap \
      --task_name TP53_mutation \
      --epochs 100
done

# Step 2: Train meta-learner
python src/ensemble4.py \
    --foundational_models ctranspath uni_v2 virchow_v1 \
    --work_dir /home/JKP6679/Patho-Ensemble/PARADIS/datos/patches \
    --train_source cptac_brca \
    --tissue_patching 20x_224px_0px_overlap \
    --task_name TP53_mutation
```

### Environment Setup

```bash
# Create conda environment from environment.yml
conda env create -f environment.yml
conda activate ensemble

# Verify PARADIS data access
python verify_paradis_setup.py
```

The `environment.yml` pins specific versions of all dependencies including PyTorch (2.7.0 pinned; 2.8.0 in the live env) with CUDA 12 support.

## File Organization

```
src/
├── abmil_engine.py         # ABMIL architecture + training loop (the engine)
├── train_abmil.py          # Base training, all folds
├── test_abmil.py           # Meta-features, in-sample
├── build_oof_features.py   # Meta-features, out-of-fold (nested CV)
│
├── abmil_fge.py            # Snapshot cycles on a converged model (FGE and SE)
├── train_abmil_fge.py      # Orchestrator: convergence + cycles, --schedule/--fge_tag
├── test_abmil_fge.py       # Meta-features from snapshots (reads by tag)
├── fge_utils.py            # fge_cycle_lr / se_cycle_lr / cycle_lr dispatcher
│
├── ensemble.py             # Weighted averaging
├── ensemble4.py            # Stacking; 9 meta-learners
├── ensemble5.py            # Early fusion at patch level; measured −0.0056 AUC (does not work)
├── meta_models.py          # The meta-learner implementations
├── ensemble_nm.py, ensemble_nm_advanced.py   # Nelder-Mead weight optimisation
│
├── experiment_config.py    # Reads experiments.yaml, emits shell vars
├── utils.py                # Metrics, path/dimension resolution, weighting
├── grid_search_mlp.py, ablation_study_mlp.py, format_metrics.py
└── create_val_splits.py, run_trident.py
```

Launchers in the repo root:

```
run_full_pipeline.sbatch              # submitter; chains the stages with afterok
train_base_models.sbatch              # stage 1: base ABMIL + in-sample meta-features
                                      #   MODELS_OVERRIDE=".." to train a different model set
build_oof_features.sbatch             # optional: out-of-fold meta-features + leak check
train_base_models_fge_v2.sbatch       # stage 2: snapshot variants (FGE_ONLY=<name> to select)
run_3_experiments_v2_improved.sbatch  # stage 3: meta-learners + paired comparison
experiments.yaml                      # the experiment matrix
```

> `src/ensemble4 copy.py` and `src/train_abmil copy.py` are stale duplicates,
> and `run_3_experiments.sbatch`, `run_3_experiments_v2.sbatch` and
> `train_base_models_fge.sbatch` are superseded launchers that still point at
> the test tree. Do not use them as references.

## Important Implementation Details

### ABMIL Model Configuration

The model architecture is fixed in `train_abmil.py` (lines 42-48):
- Input feature dimension: `--latent_dim` (e.g., 768 for most models, 2560 for Virchow)
- Attention heads: 1
- Head dimension: 512
- Dropout: 0.25
- Gated: False
- Bag size: **the full bag by default** (`--bag_size` is opt-in)

To modify architecture, edit these parameters in `train_abmil.py` before training.

> An earlier version of this file claimed "Bag size: 2048". That was **false**:
> `_apply` passed *every* patch on each forward and bag sampling did not exist
> until it was implemented for the sweep. It is now available via `--bag_size`,
> acts only inside the training loop (eval, prediction and meta-features always
> use the full bag), and **measured null** — 12 comparisons (256/512/1024 ×
> 4 metrics), none significant. See §N10 of the consolidated report.

### Fold Structure

- Each `k=*.tsv` has columns `case_id`, `slide_id`, `<task_col>`, and one
  `fold_*` column per fold, each holding `train`, `val` or `test`.
- Stratification is at **patient** level: all slides of a `case_id` stay
  together.
- **The number of folds varies by dataset** — `cptac_brca` has 50,
  `cervical_subtype` and `hancook` have 5, `imp` has 1. Never assume 50:
  assuming it once marked a fully successful training as failed and cancelled
  an `afterok` chain. The pipeline computes
  `min(max_folds, actual fold_* columns)`.
- `max_folds` belongs on the **experiment row** in `experiments.yaml`, not in
  `defaults`, where it silently caps every dataset that has more.

PARADIS fold columns already include validation splits; `--create_val` is only
for custom datasets.

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

### Analysis protocol

All configurations are evaluated on the **same folds**, so comparisons must be
**paired**. `run_3_experiments_v2_improved.sbatch` reports, per metric: paired Δ
with 95 % CI, wins/ties/losses, minimum detectable effect at 80 % power, and
Holm-corrected p-values.

Two failure modes this exists to prevent, both already encountered:

- Comparing **independent** CIs hid a systematic effect on `cervical_subtype`
  and would have hidden the bound on `cptac_brca`.
- With 5 folds, an uncorrected "5 out of 5 folds, p = 0.024" looked convincing
  and **failed to replicate** at 50 folds, with the sign flipping. Report ties
  explicitly: with 22 test slides per fold, kappa tied in up to 22 of 50 folds.

A non-significant result should be reported **with its MDE**, so it reads as
bounded rather than indeterminate.

#### Stored `ensemble4_*` results go stale when base models are re-predicted

A saved results directory is only meaningful against the base predictions it
was computed from. On `cptac_brca/TP53_mutation` the base predictions of
ctranspath, virchow_v1 and conch_v1_5 were regenerated on **2026-08-21 between
19:26 and 19:28**; every `ensemble4_*` directory older than that was computed
against predictions that no longer exist on disk:

- `ensemble4_fea_e0..e3` (19-ago 21:15–21:17) — the feature ladder
- `ensemble4_idea3_e0..e2`, `ensemble4_idea3_attn_e0..e2` (20-ago 12:10–12:43)
- `ensemble4_{svm,knn,nb}_*` (19-ago 13:03–14:23)

**`E0 = 0.7918` in `results_ladder_cptac_brca_TP53_mutation.json` does not
reproduce. The current value for that same trio and the same code is 0.7616.**
Confirmed by refitting: `ensemble4_fea_e0/coefs.npy` matches a refit on the
current `preds.npy` in none of the 50 folds.

Comparisons *within* one of those groups are still valid — same base
predictions, same moment — so the ladder's relative conclusion (extra features
add nothing) stands. What is not valid is quoting the absolute numbers or
comparing across groups.

For any new contrast, **recompute the baseline in the same pass**; never reuse
an old `ensemble4_*` directory. To check one, compare `stat -c %y` on its
`coefs.npy` against `{model}_{patching}_train_eval/val_outputs/fold_0/preds.npy`.

### Early Fusion Mechanism — Empirical Results (Updated 2026-08-21)

**⚠️ IMPORTANT**: The early fusion hypothesis was tested empirically in August 2026
and **failed**. This section documents the failed hypothesis for historical
reference. See [EXPERIMENTAL_SESSION_SUMMARY_20260821.md](EXPERIMENTAL_SESSION_SUMMARY_20260821.md)
for full results.

**Original hypothesis** (before empirical testing):
```
Late fusion (traditional pipeline):
ctranspath  → ABMIL → collapsed[p₁]
uni_v2      → ABMIL → collapsed[p₂]
virchow_v1  → ABMIL → collapsed[p₃]
→ Meta-learner sees only final opinions, loses patch-level disagreement

Early fusion (hypothesized in ensemble5.py):
[ctranspath ⊕ virchow_v1] (concatenated at patch level)
→ Single ABMIL with in_dim = 768 + 2560 = 3328
→ Attention learns to weight patch-level agreement/disagreement
→ Retains local information before collapsing to slide-level prediction
→ Then stacks with uni_v2 for final combination
```

**Empirical results** (cptac_brca, 50 folds, Session 2026-08-21):

| Configuration | Δ AUC | CI95% | Status |
|---|---|---|---|
| Early fusion (3 models concatenated) | **-0.0056** | [-0.0250, +0.0124] | ❌ No improvement |
| Weighted early fusion (optimized weights on embeddings) | **-0.0182** | [-0.0421, +0.0061] | ❌ Degrades performance |
| Late fusion + Nelder-Mead (baseline) | **+0.0052** | [+0.0007, +0.0106] | ⚠️ Marginal improvement |

**Key finding**: Concatenating embeddings at patch level **does not recover the
information loss from ABMIL's per-model collapse**. The 4096D fused input also
overwhelms ABMIL's attention mechanism, which was designed for 768–2560D single-model
inputs. Scaling embeddings before concatenation (weighted early fusion) makes
performance significantly worse.

**Conclusion**: The pipeline's late-fusion structure (independent ABMIL → LogReg
meta-learner) is **approximately optimal** for the fixed 3-model ensemble, given
the data and architecture constraints.

## Key Assumptions & Constraints

1. **Patch embeddings are HDF5** under `PARADIS/datos/features/{dataset}/`,
   with or without the `_monai` suffix — `get_features_dir` tries both.
2. **All models in an ensemble share the same splits** (same `k=*.tsv`).
3. **`--latent_dim` is auto-detected** from the `.h5`; do not pass it. Hardcoded
   values caused a silent 768/1536/2560 mismatch.
4. **Patching strategy must match directory names exactly**
   (`20x_224px_0px_overlap`).
5. **Training is single-GPU or CPU**; ensemble modules default to `device="cpu"`.
6. **Snapshot variants never share a directory** — `--fge_tag` namespaces
   checkpoints and outputs, so variants coexist.
7. **If you register a meta-model in `ensemble4.py`, add it to the `SUBDIR` map
   in `run_3_experiments_v2_improved.sbatch` too.** A mismatch does not raise:
   it reads the wrong directory and prints a plausible, wrong table.
8. **Late fusion is superior to early fusion** (empirically verified on cptac_brca).
   Patch-level concatenation does not recover per-model collapse information, and
   4096D fused embeddings overwhelm ABMIL's attention. Weighted early fusion
   (applying optimized weights to embeddings before ABMIL) is worse still
   (Δ AUC = -0.0182). See "Early Fusion Mechanism" above.
9. **The set of base models is a variable, not a given.** `defaults.models` in
   `experiments.yaml` is a historical choice, not a validated one: swapping one
   member was worth +0.037 AUC, roughly 7× the best combination-rule result
   ever measured here. Sweep base-model subsets before investing in meta-learner
   variants. `MODELS_OVERRIDE=".." sbatch train_base_models.sbatch` trains a
   different set without touching the experiment matrix.
10. **All eight base models are now comparable.** `hoptimus1`, `phikon_v2`,
    `uni_v1` and `virchow2` were retrained on 2026-08-21 and share the current
    split and labels exactly (75 meta-train / 15 val / 22 test per fold,
    verified row by row). An earlier version of this file said they were **not
    comparable** because they came from a sept-2025 `patho_bench`
    `TrainableSlideEncoder` campaign evaluated at patient level (103 rows); that
    restriction is **obsolete** — the retraining resolved it, and all 255-subset
    sweeps depend on the eight being comparable.
    Still live: the `_train_eval/test_outputs` from dec-2025 (20 rows) are
    *not* usable, and the archived `epoch_N.pt` under
    `abmil/_legacy_patho_bench_sept2025/` are not loadable by `abmil_engine`.

## Debugging & Troubleshooting

### Missing data errors

```bash
ls /home/JKP6679/Patho-Ensemble/PARADIS/datos/patches/cptac_brca/TP53_mutation/
ls /home/JKP6679/Patho-Ensemble/PARADIS/datos/features/cptac_brca/
```

### Dimension mismatches

Should not occur — `resolve_latent_dim` reads it from the `.h5`. To inspect:

```python
from utils import get_features_dir, detect_latent_dim
W = "/home/JKP6679/Patho-Ensemble/PARADIS/datos/patches"
d = get_features_dir(W, "cptac_brca", "virchow_v1")
print(d, detect_latent_dim(d))
```

### Bash pitfalls in the sbatch scripts

They run under `set -euo pipefail`, where two patterns bite:

- A non-matching glob is passed literally to `ls`, which exits 2; `pipefail`
  propagates it and `set -e` kills the script **with an empty stderr**. Use the
  `count_files()` helper.
- An unset variable aborts the script. Initialise accumulators before the loop.

### Memory issues

```bash
--fold 0        # instead of --fold all
--max_folds 5   # or cap the sweep
```

## References

- **INFORME_CONSOLIDADO_TESIS_20260901.md** (and `.html`) — **the single source
  of truth for results.** Consolidates the whole 11-aug → 1-sep campaign: the
  metric definitions and their degenerate baselines, the statistical protocol
  (MDE, reproducibility floor, the fact that 50 folds are *not* 50 independent
  experiments), the two uses of entropy, all 6 positive / 18 negative /
  2 incomplete experiments each with dataset, base models, meta-learner and
  extras, and the open research directions. **It carries only current figures**
  — every superseded or invalidated number was dropped. Prefer it over any
  individual `INFORME_*` / `ESTADO_*` file, which are kept only as the long-form
  record of a single session and may quote baselines that no longer reproduce.
- **README.md** — full reference documentation
- **PARADIS_DATA_REFERENCE.md** — data tree, datasets, tasks, fold counts
- **`results_*.json`** — the raw paired statistics behind every figure in the
  consolidated report; its final section maps each result block to its file
- **experiments.yaml** — the experiment matrix; comments there record why each
  variant exists
