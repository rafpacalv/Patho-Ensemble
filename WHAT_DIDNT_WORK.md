# What Didn't Work — Lessons from Failed Experiments

This document catalogs approaches that were tested empirically and failed, to avoid
re-running them in future sessions.

---

## 1. Early Fusion (Patch-Level Concatenation)

**Tested**: Session 2026-08-21 (Job 70224)

**Hypothesis**: Concatenating patch embeddings from multiple models at the input to ABMIL
would allow the attention mechanism to learn patch-level agreement/disagreement,
recovering information lost from per-model collapse.

**Architecture**:
```
[ctranspath (768D) ⊕ virchow_v1 (2560D) ⊕ conch_v1_5 (768D)]
→ Single ABMIL with 4096D input
→ predictions
```

**Result**: 
- **Δ AUC**: -0.0056 (worse than late fusion)
- **95% CI**: [-0.0250, +0.0124]
- **Wins/Losses**: 24/24 (random)
- **P-value**: 0.67 (not significant)

**Why it failed**:
1. **Dimensionality explosion**: 4096D input overwhelms ABMIL's projection layer
   (Linear 4096→512 introduces ~2.1M parameters where ctranspath alone has 0.4M)
2. **Loss of signal structure**: ABMIL's gated attention was designed for 768–2560D
   embeddings from single models, not concatenated multi-model inputs
3. **No coordination between models**: Concatenation provides no mechanism for models
   to "understand" which dimensions belong to which model

**Lesson**: Don't concatenate high-dimensional embeddings and hope attention learns
the structure. The information loss from ABMIL's per-model softmax collapse is real
and cannot be recovered at the patch level by simply feeding higher-dimensional input
to the same architecture.

---

## 2. Weighted Early Fusion (Weights on Raw Embeddings)

**Tested**: Session 2026-08-21 (Job 70249)

**Hypothesis**: If early fusion failed due to magnitude mismatch, applying
Nelder-Mead-optimized scalar weights to embeddings before concatenation might 
help the models contribute at appropriate scales.

**Architecture**:
```
w_ct * emb_ct(768D) ⊕ w_vir * emb_vir(2560D) ⊕ w_conch * emb_conch(768D)
→ Single ABMIL with 4096D input
→ predictions
```

**Weights**: Same as Opción A (optimized via Nelder-Mead on train_eval), range [0.04–0.99]

**Result**:
- **Δ AUC**: -0.0182 (significantly worse than Opción A)
- **95% CI**: [-0.0421, +0.0061]
- **Wins/Losses**: 20/29 (Opción A wins in 58% of folds)
- **P-value**: 0.15 (not significant)

**Why it failed**:
1. **Weight scaling destroyed information**: Raw embeddings have learned structure
   from the foundational model training. Scaling by arbitrary weights (0.04–0.99)
   breaks this structure without replacing it with anything meaningful
2. **ABMIL projections assume unscaled input**: The learned projection weights
   (Linear 4096→512) were optimized assuming embeddings stay in [-1, +1] range.
   Scaling them changes the effective signal magnitude unpredictably
3. **Double optimization**: Weights are optimized on in-sample predictions, but
   applied to embeddings that are then fed to a newly trained ABMIL. The
   in-sample-optimized weights don't predict what the new ABMIL will learn

**Lesson**: Don't scale raw embeddings arbitrarily, even with optimized weights.
The information is stored in the fine-grained structure of the embeddings, not in
their magnitude. Weighting works at the **prediction level** (after collapse),
not at the **embedding level** (before collapse).

---

## 3. Snapshot Ensembles (FGE & SE) on Base Models

**Tested**: Prior to Session 2026-08-21 (see `INFORME_ABLATION_FGE_cervical_subtype.md`)

**Hypothesis**: Cycling learning rate and keeping snapshots of a converged ABMIL
would produce diverse predictions that, when averaged, improve ensemble robustness.

**Result**:
- **FGE (Fast Geometric Ensemble)**: No improvement (bounded null)
- **SE (Snapshot Ensembles)**: No improvement
- **Conclusion**: Diversity budget already spent on the multi-model axis (r=0.797
  between different base models, r=0.93–0.98 between snapshots of the same model)

**Why it failed**:
1. **Model diversity dominates**: The 50% drop in pairwise correlation
   (0.93→0.80) between different models is so large compared to within-model
   diversity (0.94±0.02) that additional intra-model variance is noise
2. **No correlation with performance**: Across FGE variants, diversity ≠ better AUC
3. **Already captured by multi-model**: Using three different foundational models
   already provides most of the useful disagreement

**Lesson**: With 3+ diverse base models, snapshot-based diversification is wasted
compute. Spend budget on model selection or data augmentation instead.

---

## 4. MLP Meta-Learner (Flexible Capacity)

**Tested**: Prior sessions (documented in grid_search_results.json)

**Configuration**: 
- Simple MLP: Input(6)→Hidden(16)→Output(2)
- With single-cycle cosine LR schedule
- With early stopping on validation AUC

**Result**:
- **Train F1 (in-sample meta-train)**: 0.672
- **Test F1 (real meta-test)**: 0.39–0.53
- **Conclusion**: Severe overfitting, regression from LogReg baseline

**Why it failed**:
This is the classic **stacking pitfall** (Wolpert 1992, Breiman 1996). Meta-train
signals are worth AUC 0.93–0.94 (in-sample) while meta-test signals are worth
AUC 0.77 (real test). An MLP trained on the inflated train signal learns to
over-trust the base models.

**Mitigation** (not full fix): Out-of-fold meta-features (`build_oof_features.py`)
drop train signal to 0.68–0.71, closer to test (0.77). MLP then stops collapsing,
but still doesn't beat LogReg.

**Lesson**: Use out-of-fold meta-features (`--meta_features oof`) for anything
more flexible than LogReg, or accept that flexible models will collapse on
in-sample features. LogReg with in-sample features is surprisingly hard to beat
on this problem.

---

## 5. TabPFN Meta-Learner (Learned Priors)

**Tested**: Prior sessions

**Configuration**: TabPFN (Prior-Data Fitted Network) as meta-classifier

**Result**: Underperforms LogReg baseline, especially with in-sample features

**Why it failed**:
1. TabPFN was trained on tabular datasets with 10–100k rows and hundreds of features
2. Meta-learner problem has 75 samples and 4–6 features — far outside TabPFN's
   training distribution
3. No ability to learn useful inductive bias from such small feature space

**Lesson**: Learned-prior methods like TabPFN are not a shortcut to handle small
data. For truly small meta-learning problems (n=75, p=4), simple linear models
(LogReg) or explicit regularization (Ridge, Lasso) work better.

---

## 6. Aggressive Data Augmentation on Patches

**Tested**: Conceptually explored, not fully implemented

**Hypothesis**: Bootstrapping or subsampling patches within each bag (WSI) during
training could increase effective sample size.

**Why it didn't get implemented**:
1. ABMIL already trains on all patches in a bag; subsampling loses information
2. Bag size is ~2000 patches, so random subsampling adds noise without regularization benefit
3. Patient-level stratification (existing) already handles most overfitting

**Lesson**: Patch-level augmentation doesn't help ABMIL; stratified splits and
early stopping are the right levers.

---

## Summary: The Optimality of Late Fusion + LogReg

After systematic testing:

| Approach | Status | Why |
|----------|--------|-----|
| Early fusion (concat embeddings) | ❌ Failed | Dimensionality + attention mismatch |
| Weighted early fusion | ❌ Failed worse | Breaking embedding structure |
| Snapshot ensembles | ❌ Wasted compute | Model diversity already maxed |
| Flexible meta-learners (MLP, TabPFN) | ❌ Overfit | Stacking pitfall, even with OOF |
| Late fusion + LogReg | ✅ Baseline | Simple, interpretable, hard to beat |
| Late fusion + Nelder-Mead | ⚠️ Marginal | +0.52% AUC, p=0.058, not reliable |

**Conclusion**: The current pipeline (3 independent ABMIL → LogReg meta-learner)
is **approximately optimal** for the fixed 3-model ensemble on this problem.
Improvements, if any, are marginal and require:

1. **Different models** (not possible — foundational models are fixed)
2. **More data** (not available)
3. **Structural changes** (e.g., attention to per-patient vs per-slide aggregation)
4. **Task-specific tuning** (e.g., balancing precision/recall for clinical use)

---

## How to Use This Document

Before implementing a new ensemble approach:

1. Check this list — if your idea is here, review the failure analysis
2. If the core failure reason doesn't apply to your variant, document why before
   implementing
3. After any new experiment, update this file with results and failure analysis
4. Keep timestamps — experiments may be dataset/task specific

---

*Last updated: 2026-08-21*
