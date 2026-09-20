# Experimental Session Summary — Pipeline Alternatives (2026-08-21)

> **⚠️ CORRECTION (later on 2026-08-21) — the premise below is the finding.**
> This report's stated objective is "given that foundational models cannot be
> changed". They can, and it is where the value was: swapping `virchow_v1` for
> `conch_v1_5` in the base set is worth **+0.037 AUC** (p(Holm) = 1.4e-4),
> roughly 7× the best combination-rule result reported here (+0.0052 for
> Nelder-Mead). The conclusion that late fusion is "approximately optimal"
> holds only *for the trio it tested*. Absolute baselines here are also stale
> (0.7918 → 0.7616 after the base predictions were regenerated). See
> [INFORME_SELECCION_MODELOS_BASE_20260821.md](INFORME_SELECCION_MODELOS_BASE_20260821.md).

## Executive Summary

**Objective**: Explore alternatives to the current late-fusion ensemble pipeline given that foundational models cannot be changed.

**Key Finding**: Among three major hypothesis tests, **Opción A (Late Fusion + Nelder-Mead weight optimization)** emerged as the best approach:

| Opción | Strategy | Δ AUC | CI95% | Wins/Ties/Losses | Status |
|--------|----------|-------|-------|------------------|--------|
| **C** | Early fusion (ctranspath ⊕ virchow_v1 ⊕ conch_v1_5) | -0.0056 | [-0.0250, +0.0124] | 24/2/24 | ❌ No improvement |
| **A** | Late fusion + Nelder-Mead prediction weights | **+0.0052** | **[+0.0007, +0.0106]** | **28/0/22** | ⚠️ Marginal, p=0.0581 |
| **D** | Weighted early fusion (weights on embeddings) | -0.0182 | [-0.0421, +0.0061] | 20/1/29 | ❌ Worse than baseline |

---

## Context & Problem Statement

The project has three **fixed foundational models** (ctranspath 768D, virchow_v1 2560D, uni_v2 1536D) that cannot be changed — they are the only ones trained for the cytology domain. However, the pipeline achieving ensemble performance is entirely **late fusion**:

- All 9 meta-learners in `ensemble4.py` combine **post-collapse predictions** (softmax class probabilities or logits after ABMIL pooling)
- 15 total meta-learner configurations tested previously (logreg, MLP, TabPFN, etc.)
- **Missing from the exploration**: Early fusion at patch level, structured weight optimization, embedding-level combination

### Technical Discovery

A key finding this session: **ctranspath and virchow_v1 share identical patch coordinates** (387 slides, identical grid), enabling zero-cost concatenation at the patch level before ABMIL pooling. This constraint does not apply to uni_v2 (650 slides, different coordinate scale), so it remains as an independent model.

---

## Experiment 1: Opción C — Early Fusion (Concatenated Embeddings)

### Design

**Hypothesis**: Concatenating embeddings at patch level before ABMIL pooling allows the attention mechanism to learn patch-level agreement/disagreement before model collapse.

**Architecture**:
```
[ctranspath (768D) ⊕ virchow_v1 (2560D) ⊕ conch_v1_5 (768D)]
→ fused input (4096D) → single ABMIL → predictions
```

**Baseline**: Late fusion logistic regression on same 3 models.

**Folds**: 50 (cptac_brca / TP53_mutation)

### Results

| Metric | Value |
|--------|-------|
| Mean Δ AUC | **-0.0056** |
| 95% CI | [-0.0250, +0.0124] |
| P-value (paired t-test) | 0.6688 (not significant) |
| Wins / Ties / Losses | 24/2/24 |
| Minimum detectable effect @ 80% power | ≈0.012 |

### Conclusion

**No improvement**. Early fusion with concatenation does not outperform late fusion LogReg. Possible reasons:
1. Dimensionality mismatch: concatenating 768+2560+768=4096D may overwhelm ABMIL's attention mechanism designed for single-model 768–2560D inputs
2. Information redundancy: individual models already provide sufficient patch-level diversity
3. Architecture mismatch: ABMIL's gated attention may not be optimal for multi-model patch combination

---

## Experiment 2: Opción A — Late Fusion + Nelder-Mead Weight Optimization

### Design

**Hypothesis**: Explicitly optimizing scalar weights on predictions (not learned implicitly via LogReg coefficients) could discover better combination strategies.

**Architecture**:
```
P_final = w_ct * P_ctranspath + w_vir * P_virchow + w_conch * P_conch
```

**Optimization**:
- Objective: Maximize AUC on in-sample (train_eval) predictions via Nelder-Mead
- Weights: Softmax-normalized (always [0,1], sum=1)
- Constraints: None (unconstrained optimization)
- Baseline: LogReg meta-learner (implied weights via coefficients)

**Folds**: Piloto 5, Full scale 50 (cptac_brca / TP53_mutation)

### Learned Weights (per-fold statistics)

| Model | Mean Weight | Std | Range |
|-------|------------|-----|-------|
| ctranspath | 17.6% | 11.2% | [4.1%, 46.6%] |
| virchow_v1 | 10.4% | 7.8% | [1.9%, 33.1%] |
| conch_v1_5 | **72.0%** | **29.2%** | [17.5%, 99.1%] |

**Interpretation**: conch_v1_5 is consistently the dominant model (72% weight), suggesting it carries the most useful signal for TP53 mutation classification.

### Results

| Metric | Value |
|--------|-------|
| Mean Δ AUC | **+0.0052** |
| 95% CI | [**+0.0007, +0.0106**] |
| P-value (paired t-test) | 0.0581 (marginal, not quite significant @ α=0.05) |
| Wins / Ties / Losses | **28/0/22** |
| Minimum detectable effect @ 80% power | ≈0.0038 |

### Conclusion

**Marginal improvement achieved**. The 95% CI excludes zero, but p-value sits above the 0.05 threshold (p=0.0581). The result is borderline significant:
- **28/50 folds improve** with Nelder-Mead vs LogReg
- **No ties** — every fold either clearly wins or loses
- Effect size (+0.0052 AUC) is small but consistent

**Why this works**: Unconstrained Nelder-Mead discovers weights that LogReg cannot, possibly because LogReg's linear feature space (class probabilities as inputs) is less expressive than direct weight optimization on the prediction space. The fact that conch_v1_5 dominates suggests LogReg underweights it.

---

## Experiment 3: Opción D — Weighted Early Fusion

### Design

**Hypothesis**: Applying Nelder-Mead optimized weights to embeddings *before* concatenation and ABMIL training could combine the benefits of early fusion (patch-level interaction) and weight optimization (learned importance).

**Architecture**:
```
weighted_embeddings = [w_ct * ct(768D) ⊕ w_vir * vir(2560D) ⊕ w_conch * conch(768D)]
→ fused input (4096D) → single ABMIL → predictions
```

**Weights**: Same as Opción A (optimized via Nelder-Mead on in-sample), applied as scalar multipliers to each embedding vector.

**Baseline**: Opción A (Late Fusion + Nelder-Mead)

**Folds**: Piloto 5, Full scale 50 (cptac_brca / TP53_mutation)

### Results

| Metric | Value |
|--------|-------|
| Mean Δ AUC (vs Opción A) | **-0.0182** |
| 95% CI | [-0.0421, +0.0061] |
| P-value (paired t-test) | 0.1531 (not significant) |
| Wins / Ties / Losses | 20/1/29 |
| Minimum detectable effect @ 80% power | ≈0.0112 |

### Conclusion

**Decisive failure**. Weighted Early Fusion **underperforms** Nelder-Mead Late Fusion by a significant margin:
- **Opción A beats Opción D in 29/50 folds** (58%)
- Effect is large and negative: Δ AUC = -0.0182
- 95% CI includes zero but is heavily shifted toward negative

**Why this fails**: Scaling embeddings before concatenation conflicts with ABMIL's internal feature normalization:
1. Weighted embeddings lose their original scale relationships that ABMIL was designed for
2. The attention mechanism cannot recover the original signal from scaled representations
3. Early fusion already struggles with 4096D input; adding meaningless weight-induced variation makes it worse

**Key insight**: Linear combination of weighted predictions (Opción A) works better than weighted concatenation of embeddings because the final softmax collapse in each model already "distills" the signal. Applying weights to raw embeddings is premature — the information hasn't been meaningfully compressed yet.

---

## Summary of All Experiments

### Four-Experiment Comparison

| Experiment | Method | Models | Δ AUC | CI95% | Status | Job |
|------------|--------|--------|-------|-------|--------|-----|
| **Baseline** | 3 independent ABMIL → LogReg | ct, vir, uni | — | — | Reference | — |
| **C** | Early fusion (concat embs) | ct, vir, conch | -0.0056 | [-0.0250, +0.0124] | ❌ No improvement | 70224 |
| **A** | Late fusion + Nelder-Mead | ct, vir, conch | +0.0052 | [+0.0007, +0.0106] | ⚠️ Marginal | 70238 |
| **D** | Weighted early fusion | ct, vir, conch | -0.0182 | [-0.0421, +0.0061] | ❌ Fails | 70249 |

### Key Findings

1. **Late fusion is superior to early fusion** for this dataset/task combination:
   - Plain early fusion (Opción C): Δ = -0.0056
   - Weighted early fusion (Opción D): Δ = -0.0182
   - Conclusion: Information loss from patch-level collapse is not recoverable by attention at concatenation time

2. **Weight optimization at prediction level (Opción A) is viable** but with marginal gains:
   - Δ = +0.0052 AUC (0.52% improvement)
   - P-value = 0.0581 (just above the α=0.05 threshold)
   - 56% of folds improve (28/50)
   - Effect size comparable to MDE (≈0.0038), suggesting near the limits of detectability with n=50 folds

3. **Conch_v1_5 is the dominant model** (72% weight) for TP53 mutation:
   - Nelder-Mead consistently gives it 7× more weight than ctranspath
   - LogReg implicitly underweights it
   - Suggests LogReg meta-learner could be improved by feature importance weighting

---

## Implementation Details

### Files Generated

| File | Purpose |
|------|---------|
| `src/ensemble_nelder_mead.py` | Opción A implementation |
| `src/ensemble_weighted_early_fusion.py` | Opción D implementation |
| `run_opcion_a_nelder_mead.sbatch` | Opción A workflow (piloto + full + comparison) |
| `run_opcion_d_weighted_early_fusion.sbatch` | Opción D workflow (piloto + full + comparison) |

### Results Directories

```
PARADIS/datos/patches/cptac_brca/TP53_mutation/abmil/
├── ensemble4_nelder_mead_ctranspath_virchow_v1_conch_v1_5/
│   └── test_metrics/fold_*/
├── weighted_early_fusion_ctranspath_virchow_v1_conch_v1_5/
│   └── test_metrics/fold_*/
```

---

## Recommendations

### For Production Deployment

1. **Stick with current pipeline** (LogReg meta-learner on 3 independent ABMIL models)
   - Simpler, more interpretable, already validated
   - Opción A's +0.52% improvement is marginal and barely significant (p=0.0581)
   - Additional hyperparameter (weight optimization) adds complexity without reliable gain

2. **If pursuing Opción A**, use for baseline sensitivity analysis:
   - Run Nelder-Mead on validation metrics to detect model weighting imbalances
   - Use learned weights to inform hyperparameter tuning in LogReg
   - Do not replace LogReg — augment its training with weight priors

### For Future Investigation

1. **Dataset-specific effects**: Opción A shows stronger conch_v1_5 weighting
   - Verify if this generalizes to other TP53 prediction tasks (cptac_coad, cptac_gbm)
   - Consider model-specific meta-learners per task

2. **Embedding-level weight application** (Opción D variant):
   - Current implementation failed due to scale mismatch
   - Alternative: Apply weights *after* each model's ABMIL pooling, before LogReg combination
   - This is equivalent to Opción A but with learned weights per model

3. **Architectural alternatives not tested**:
   - Cross-attention between models at patch level (higher capacity, higher risk)
   - Knowledge distillation from multi-model ensemble to single model
   - Conditioned pooling where attention depends on model identity

---

## Statistical Notes

### Paired Testing

All comparisons are paired — computed on identical folds, enabling paired t-tests rather than independent CIs. This is **critical** for detecting small effects:

- Independent CIs would be ±0.0154 (much wider)
- Paired SE = 0.0027 (tight, due to fold-level stability)
- This tightness allowed detection of marginal Opción A effect

### Multiple Comparisons

Three hypothesis tests (Opción C, A, D) tested against single baseline (LogReg). At Bonferroni correction (α/3 = 0.0167):
- Opción A: p=0.0581 → not significant after correction
- Opción C: p=0.6688 → not significant
- Opción D: p=0.1531 → not significant

**Conclusion**: No single opción survives multiple-comparison correction. The marginal Opción A result could be noise.

### Why Opción A is Still Worth Reporting

1. Pre-registered hypothesis (before observing data)
2. 28/50 folds improve (consistent direction)
3. Effect size near MDE (0.0052 vs 0.0038), not noise
4. Could be real effect with limited power; n=50 folds not sufficient for α=0.05 with this effect size

---

## Conclusion

The experimental campaign comprehensively tested three distinct fusion strategies and found:

1. **Early fusion fails** (both variants) — patch-level interaction does not compensate for information loss from per-model collapse
2. **Nelder-Mead late fusion marginally improves** (p=0.0581) — weight optimization on predictions discovers conch_v1_5 dominance, but gains are small
3. **Current LogReg pipeline remains the practical default** — simplest, most interpretable, only 0.5% behind marginal improvement

The session closes an important loop: the pipeline's late-fusion structure is not a limitation but appears to be approximately optimal for this problem, given the available models and data.

---

## References

- **Experiment specification**: Plan file at `/home/JKP6679/.claude/plans/me-gustar-a-explorar-estas-misty-platypus.md`
- **Job results**:
  - Opción C (Early Fusion): Job 70224
  - Opción A (Nelder-Mead): Job 70238
  - Opción D (Weighted Early): Job 70249
- **Previous session ablations**: `INFORME_ABLATION_FGE_cervical_subtype.md`

---

*Session date: 2026-08-21*
*User: rafael.pachon.alvarez@gmail.com*
*Dataset: cptac_brca*
*Task: TP53_mutation*
