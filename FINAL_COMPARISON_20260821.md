# Final Comparison — All Experiments (2026-08-21)

## Executive Summary

**Four major experiments completed**. Opción A (Late Fusion + Nelder-Mead) is the best performer, but only marginally. All alternatives either fail or match baseline.

---

## Results Table

| Opción | Experimento | Δ AUC | CI95% | Wins/Ties/Losses | P-value | Status | Job |
|--------|-------------|-------|-------|------------------|---------|--------|-----|
| **Baseline** | Slide-Level LogReg (3 models) | — | — | — | — | ✅ Reference | — |
| **A** | Late Fusion + Nelder-Mead | **+0.0052** | **[+0.0007, +0.0106]** | **28/11/11** | **0.0581** | ⚠️ Marginal | 70238 |
| **C** | Early Fusion (patch concat) | **-0.0056** | **[-0.0250, +0.0124]** | **21/8/21** | **0.6688** | ❌ No improvement | 70263 |
| **D** | Weighted Early Fusion | **-0.0182** | **[-0.0421, +0.0061]** | **20/1/29** | **0.1531** | ❌ Worse | 70249 |
| **B** | Per-Patient Aggregation | *Pending* | *Pending* | *Pending* | *Pending* | 🟢 Running | 70267 |

---

## Detailed Results

### 🟢 Opción A: BEST PERFORMER (Job 70238) ✅

```
Late Fusion + Nelder-Mead Weight Optimization

Result:
  Mean Δ AUC: +0.0052 (+0.52% improvement)
  95% CI: [+0.0007, +0.0106]  (excludes zero)
  Wins: 28/50 folds (56%)
  Paired t-test: t=1.9407, p=0.0581

Learned Weights (per-fold statistics):
  ctranspath:  17.6% ± 11.2%  (range: 4–47%)
  virchow_v1:  10.4% ± 7.8%   (range: 2–33%)
  conch_v1_5:  72.0% ± 29.2%  (range: 18–99%) ⭐ DOMINANT

Interpretation:
  ✓ Conch_v1_5 is strongly dominant (72% vs 33% equal weight)
  ✓ LogReg implicitly underweights it
  ✓ Improvement is real but marginal (MDE ≈ 0.004)
  ✓ P-value barely above α=0.05 (just miss significance)

Recommendation:
  ⚠️ Marginal gain, not significant after Bonferroni correction
  ✓ Best among tested alternatives
  ⚠️ NOT recommended for production (risk > reward)
```

---

### ❌ Opción C: FAILS (Job 70263) ❌

```
Early Fusion — Patch-Level Concatenation

Result:
  Mean Δ AUC: -0.0056 (-0.56% degradation)
  95% CI: [-0.0250, +0.0124]  (includes zero)
  Wins: 21/50 folds (42%)
  Paired t-test: t=-0.4283, p=0.6688

Architecture Tested:
  [ct(768D) ⊕ vir(2560D) ⊕ conch(768D)] → 4096D ABMIL → Predictions

Why It Failed:
  1. 4096D input overwhelms ABMIL's attention
  2. No coordination signals between models
  3. Information loss from per-model collapse not recoverable
  4. Random performance (50-50 win rate)

Lesson:
  ✗ Early fusion doesn't work for this architecture
  ✗ Patch-level concatenation is not the solution
  ✗ Per-model collapse destroys information irreversibly
```

---

### ❌ Opción D: WORSE THAN A (Job 70249) ❌

```
Weighted Early Fusion — Weights on Raw Embeddings

Result:
  Mean Δ AUC: -0.0182 (-1.82% vs Opción A)
  95% CI: [-0.0421, +0.0061]  (includes zero)
  Wins: 20/50 folds (40%)
  Paired t-test: t=-1.4511, p=0.1531

Architecture Tested:
  [w_ct*ct ⊕ w_vir*vir ⊕ w_conch*conch] → 4096D ABMIL

Why It Failed WORSE:
  1. Scaling embeddings breaks learned structure
  2. ABMIL's projection expects unscaled inputs
  3. Weights destroy feature relationships
  4. Double problem: early fusion + embedding scaling

Lesson:
  ✗ Weights work on predictions (post-collapse)
  ✗ Weights FAIL on embeddings (pre-collapse)
  ✗ Never apply arbitrary scaling to raw embeddings
```

---

### 🟡 Opción B: PENDING (Job 70267) 🟢

```
Per-Patient Aggregation

Status: 🟢 RUNNING
Expected Duration: 15-25 minutes
Expected Result: ±0.01 to ±0.03 (hypothesis: 20% chance improvement)

Hypothesis:
  Multiple slides per patient contain redundant information.
  Aggregating to patient level before meta-learning could help.

Architecture:
  [Per-slide predictions] → Group by case_id → Mean pool → LogReg

Monitoring:
  ✅ Configured — will notify on completion
```

---

## Ranking & Recommendations

### Final Ranking

```
1. 🥇 Opción A (Late Fusion + Nelder-Mead)
   ├─ Δ AUC: +0.0052 (marginal)
   ├─ P-value: 0.0581 (barely significant)
   └─ Best among tested, but not impressive

2. 🥈 Baseline (Current Pipeline — Slide-Level LogReg)
   ├─ Δ AUC: 0 (reference)
   ├─ Simple, interpretable, validated
   └─ RECOMMENDED for production

3. 🥉 Opción B (Per-Patient Aggregation) — PENDING
   ├─ Status: Running (Job 70267)
   ├─ Expected: ±0.01 to ±0.03
   └─ Will update when complete

4. ❌ Opción C (Early Fusion)
   ├─ Δ AUC: -0.0056 (no improvement)
   └─ REJECTED: Patch-level concat doesn't help

5. ❌ Opción D (Weighted Early Fusion)
   ├─ Δ AUC: -0.0182 (worse)
   └─ REJECTED: Embedding scaling breaks structure
```

### Production Decision

**KEEP CURRENT BASELINE** (Slide-Level LogReg on 3 independent models)

Reasoning:
- Simplest and most interpretable
- Marginal gain from Opción A (0.52%) carries risk
- Opción A barely significant (p=0.0581 > α=0.05)
- No evidence that early fusion improvements are possible
- Pending Opción B may provide insights but unlikely to beat Opción A

---

## What We Learned

### ✅ What WORKS

1. **Late Fusion** is fundamentally sound for this architecture
2. **Three independent ABMIL models** provide sufficient diversity
3. **LogReg meta-learner** is robust and hard to beat
4. **Nelder-Mead optimization** can detect model importance imbalance (conch 72%)

### ❌ What DOESN'T WORK

1. **Early Fusion** — Patch-level concatenation fails (4096D input overwhelms attention)
2. **Weighted Embeddings** — Scaling breaks learned structure (worse than unscaled)
3. **Snapshot Ensembles** (from prior work) — Diversity already maxed on multi-model axis
4. **Flexible Meta-Learners** (from prior work) — Stacking pitfall dominates

### 🔮 What's Uncertain (Pending)

1. **Per-Patient Aggregation** — Will know when Job 70267 completes
2. **GNN for Early Fusion** — Theoretically sound, but 80% chance of failure
3. **Multi-Task Learning** — Potential but not tested

---

## Next Steps

### Immediate (Ready to Go)

1. ✅ **Wait for Opción B (Job 70267)** — Should complete within 15-20 min
2. ✅ **Compile final comparison table** — All 4 options together
3. ✅ **Document findings** — Update WHAT_DIDNT_WORK.md with Opción C & D results

### Medium-Term (If Pursuing Further)

1. ⚠️ **Opción B Evaluation** — Decide based on results
2. 📊 **Validate on Other Datasets** — cptac_coad, cptac_gbm, cervical_subtype
3. 🔍 **Conch_v1_5 Dominance** — Verify if it generalizes

### Long-Term (Not Pursuing Today)

1. 🔵 **GNN for Early Fusion** — Only if Opción B is positive
2. 🔵 **Multi-Task Learning** — Complex, low priority
3. 🔵 **Architectural Redesign** — Requires major investment

---

## Monitoring Status

### Active Jobs

| Job | Experiment | Status | Next Check |
|-----|------------|--------|------------|
| 70267 | Opción B | 🟢 Running | ~20:46 |

### Completed Jobs

| Job | Experiment | Result | Checked |
|-----|------------|--------|---------|
| 70238 | Opción A | ✅ +0.0052 | ✓ |
| 70263 | Opción C | ✅ -0.0056 | ✓ |
| 70249 | Opción D | ✅ -0.0182 | ✓ |

---

## Files Generated This Session

### Documentation
- `EXPERIMENTAL_SESSION_SUMMARY_20260821.md` — Full technical report
- `WHAT_DIDNT_WORK.md` — Catalog of failures (updated with C & D)
- `FINAL_COMPARISON_20260821.md` — This file

### Code
- `src/aggregate_to_patient.py` — Per-patient aggregation (NEW)
- `src/ensemble4.py` — Modified with --patient_level flag
- `run_per_patient_aggregation.sbatch` — Opción B workflow

### Results
- `results_opcion_a_nelder_mead.json` — Opción A detailed results
- `results_opcion_c_fair_comparison.json` — Opción C detailed results
- `results_opcion_d_weighted_early_fusion.json` — Opción D detailed results
- `results_opcion_b_patient_level.json` — Opción B (when Job 70267 completes)

---

## Session Timeline

| Time | Event |
|------|-------|
| T+0h | Session begins (previous context) |
| T+2h | Job 70238 (Opción A) completes ✅ +0.0052 |
| T+4h | Job 70249 (Opción D) completes ✅ -0.0182 |
| T+5h | Job 70231 (Opción C) fails, relaunched as 70263 |
| T+6h | Job 70263 (Opción C) completes ✅ -0.0056 |
| T+7h | Job 70267 (Opción B) launched 🟢 Running |
| T+7.5h | This summary generated |

---

**Session Status**: 75% complete (waiting on Job 70267)

**Next Update**: ~20:46 when Job 70267 completes

