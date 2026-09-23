# Experimental Session Summary — Feature/Model Selection for Stacking (2026-09-22 → 2026-09-23)

## Executive Summary

Literature review on feature/model selection for late-fusion ensembles, followed
by two derived experiments on `cptac_brca / TP53_mutation` (50 folds), run on
the same two base-model trios used throughout this thesis: **historical**
(`ctranspath+uni_v2+virchow_v1`, AUC 0.7616) and **winning**
(`ctranspath+uni_v2+conch_v1_5`, AUC 0.7986 — see CLAUDE.md, "Base model
selection dominates combination rule").

| Experiment | Result | Status |
|---|---|---|
| Feature ladder, re-measured + extended (jobs 72573–72575) | Null confirmed on **both** trios; no step significant | ❌ No improvement (expected null) |
| L1 / elastic-net view selection on the meta-learner (jobs 72576, 72582) | No AUC gain at any non-degenerate C; but L1 sparsity **independently reconstructs** the base-model ranking already known from the subset sweep | ❌ No AUC gain — ✅ mechanistic confirmation |

Two real bugs were found and fixed in `src/analyze_feature_engineering_ladder_v2.py`
during this session (below). No new experiment code was executed as part of
writing this summary — everything here reads results already on disk.

---

## A. Literature review

No file was produced from the review itself (web research, not saved). Four
families of literature were surveyed, motivating experiments B and C below:

1. **Classical ensemble selection/pruning** — the accuracy–diversity trade-off
   in choosing which ensemble members to keep.
2. **Feature/view selection specific to stacking** — Sylvain et al., "View
   selection in multi-view stacking: choosing the meta-learner": non-negative
   lasso / elastic-net over the meta-learner's inputs to select *views*
   (here: base models) rather than arbitrary features. This is the direct
   basis for experiment C.
3. **Dynamic Ensemble Selection (DES)** — META-DES, the `deslib` library:
   per-instance competence estimation instead of a single global combiner.
   Reviewed but not yet run; flagged by the user as the next experiment to
   launch after this summary.
4. **2025–2026 pathology foundation-model fusion papers**: ELF (arXiv
   2508.16085); "Information-Driven Fusion of Pathology Foundation Models"
   (arXiv 2512.11104 — prunes by inter-FM embedding correlation before
   fusion, strong results on kidney/prostate/rectal); FuseCPath
   (arXiv 2510.27237).

---

## B. Feature engineering ladder — re-measured + extended (jobs 72573, 72574, 72575)

Re-ran the E0→E1→E2→E3 ladder (raw probabilities → `+disagreement` →
`+margin_avg` → `+entropy_per_model`, on top of `logreg`) on **both** trios in
one pass, after the "stale `ensemble4_*` baseline" fix already documented in
CLAUDE.md. Launcher: `run_feature_engineering_ladder_v2.sbatch`.

| Trio | E0 | E1 | E2 | E3 | Any step significant (Holm)? |
|---|---:|---:|---:|---:|---|
| historical | 0.7616 | 0.7618 | 0.7611 | 0.7602 | No — E3 Δ=−0.0015, adj. p=0.059 (misses α=0.05) |
| winning | 0.7986 | 0.7988 | 0.7991 | 0.7978 | No — all adj. p ≥ 0.17, MDE ≈ ±0.016 |

Source: `results_ladder_cptac_brca_TP53_mutation.json` (historical),
`results_ladder_conch_cptac_brca_TP53_mutation.json` (winning). Verified
directly against both files: E0/E1/E2/E3 means and the per-step Δ, CI, p and
adjusted-p all match the table above to 4 decimals.

**Bug found and fixed** in `src/analyze_feature_engineering_ladder_v2.py`
(jobs 72574, 72575 regenerated the result JSONs after the fix): the
significance summary conflated the marginal band (0.05 ≤ adj. p < 0.10, `*`)
with the declared α=0.05 (`**`), and printed "mejora significativa" for any
significant change regardless of the sign of Δ — so E3's significant
*degradation* on the historical trio was being reported as an improvement.
Fixed: `significant` now means `adj. p < 0.05` only, a separate `marginal`
field carries the 0.05–0.10 band, and the printed line states
"mejora"/"empeora" per `sign(Δ)`. This fix is already reflected in CLAUDE.md's
"Feature engineering ladder" section.

**Conclusion**: the null generalises — it is not an artefact of the
historical trio being weak. Adding derived meta-features to the combiner's
input does not help on either trio; both put E3 (all three extras) at the
worst point estimate.

---

## C. View selection for the meta-learner via L1 / elastic net (jobs 72576, 72582)

New code: `logreg_en` (elastic net, `penalty='elasticnet'`, `solver='saga'`)
added to `src/ensemble4.py` alongside the pre-existing but never cleanly
evaluated `logreg_l1`. Both run with `--feature_space logit
--drop_redundant_class`, which in this binary task leaves **exactly one
log-odds column per base model** — so L1/elastic-net sparsity here is
model-level selection, not arbitrary-feature selection. Launchers:
`run_meta_view_selection.sbatch` (generation, both trios, 8 configs × 2 = 16
runs of 50 folds) + `run_reanalyze_meta_view_selection.sbatch` (reanalysis).

Note the baseline in this experiment (`logreg` under `feature_space=logit`)
is a *different* number from the ladder's E0 (which used `feature_space=prob`):
0.7655/0.7980 here vs 0.7616/0.7986 in the ladder — both verified directly
from `results_meta_view_selection_{hist,win}_cptac_brca_TP53_mutation.json`.
Each experiment is internally paired against its own same-feature-space
baseline, so this does not affect either experiment's own conclusions.

### AUC results

| Config | historical AUC | Δ vs baseline (Holm) | winning AUC | Δ vs baseline (Holm) |
|---|---:|---|---:|---|
| Baseline (logit, dropc) | 0.7655 | — | 0.7980 | — |
| L1 C=0.01 | 0.5000 | **−0.2655, adj. p=9.3e-23** | 0.5289 | **−0.2692, adj. p=7.8e-19** |
| L1 C=0.1 | 0.7574 | −0.0081, adj. p=1.0 | 0.7911 | −0.0069, adj. p=0.96 |
| L1 C=1.0 | 0.7679 | +0.0024, adj. p=1.0 | 0.7975 | −0.0005, adj. p=1.0 |
| L1 C=10.0 | 0.7718 | +0.0063, adj. p=0.65 | 0.7932 | −0.0048, adj. p=0.70 |
| EN l1_ratio=0.2 (C=1.0) | 0.7667 | +0.0012, adj. p=1.0 | 0.7980 | +0.00001, adj. p=0.99 |
| EN l1_ratio=0.5 (C=1.0) | 0.7666 | +0.0011, adj. p=0.48 | 0.7984 | +0.0004, adj. p=1.0 |
| EN l1_ratio=0.8 (C=1.0) | 0.7671 | +0.0016, adj. p=0.91 | 0.7978 | −0.0002, adj. p=1.0 |

Verified against both `results_meta_view_selection_*.json` files: all means,
deltas, CIs, p-values and adjusted p-values match to the precision shown.

- **L1 C=0.01 collapses to a trivial classifier** (AUC 0.5000 historical,
  0.5289 winning) — the only significant result in either trio, and it is a
  significant *degradation* (Holm p < 0.001 both).
- **No config at C ∈ {0.1, 1.0, 10.0} or elastic-net l1_ratio ∈ {0.2, 0.5, 0.8}
  at C=1.0 differs significantly from baseline in either trio** — all
  adjusted p ≥ 0.1.

**Second bug found and fixed**, also in `src/analyze_feature_engineering_ladder_v2.py`
(reused for this reanalysis; job 72582 regenerated both result JSONs after
the fix): labels containing their own `=` (e.g. `"L1 C=0.01"`) collided under
`--subdirs LABEL=SUBDIR` parsing — `label.split("=", 1)` truncated the label
and silently overwrote the previous entry in the `escalones` dict, dropping 3
of 4 L1 configs from the analysis with no error. Fixed: the script now aborts
with an explicit error if a label repeats, and the launcher was changed to
use `:` instead of `=` inside labels (`"L1 C:0.01"` etc.) to avoid the
collision going forward.

### Mechanistic finding: L1 sparsity reconstructs the known base-model ranking

Not an AUC result — this is about which base-model coefficients L1 zeroes
out, measured directly from each config's `coefs.npy` (shape `(50 folds, 3
models)`, one row per fold) as the fraction of the 50 folds in which that
model's coefficient is exactly 0:

| Trio | C | ctranspath off | uni_v2 off | 3rd model off |
|---|---|---:|---:|---:|
| historical | 0.1 | 88% | 10% | **virchow_v1: 100%** |
| winning | 1.0 | 70% | 66% | **conch_v1_5: 8%** |

(Verified directly by loading
`.../abmil/ensemble4_logreg_l1_logit_dropc_c01_hist/coefs.npy` and
`.../ensemble4_logreg_l1_logit_dropc_c1_win/coefs.npy` and computing the
per-column zero-fraction: `[0.88, 0.10, 1.00]` and `[0.70, 0.66, 0.08]`
respectively, matching the table exactly.)

On the historical trio, L1 turns `virchow_v1` off in **every** fold while
almost never touching `uni_v2` — independently rediscovering that
`virchow_v1` is dispensable, already known from the base-model subset sweep
(`virchow_v1` alone: AUC 0.6823, CLAUDE.md). On the winning trio, L1 almost
never turns `conch_v1_5` off while suppressing the other two most of the
time — rediscovering `conch_v1_5` as the strongest single member (alone: AUC
0.7902, CLAUDE.md).

This is **triangulation**: two independent methods — the base-model subset
sweep (which trains and tests each model combination directly) and L1
sparsity on the meta-learner's logit inputs (which never sees AUC, only the
logistic loss under an L1 penalty) — converge on the same base-model ranking
in both trios. This holds even though L1 never *beats* the unpenalized
baseline in AUC at any of the C values tested; the ranking signal is present
in the coefficients before it ever shows up as an AUC gain.

---

## D. Open item — CLAUDE.md not yet updated for C

CLAUDE.md was updated for section B (the feature ladder re-measurement) but
**not** for section C (L1/elastic-net view selection). The user was asked
whether to add it and, instead of answering, requested this summary plus a
second experiment (DES) — so that CLAUDE.md update remains an open decision,
not something done in this session. Do not assume it is written; it is not.

---

## What this implies for the thesis narrative

This session adds a second, independent confirmation of the thesis's central
claim — **"the gain is the base model, not the combiner"** — this time from
the input side rather than the meta-model-family side (already covered by
"no metaclassifier beats logreg"). The feature ladder (section B) shows that
what the combiner *sees* cannot be enriched profitably; the L1 experiment
(section C) shows that even when it is allowed to *select* among what it
sees, the ceiling doesn't move (only C=0.01 changes anything, and it breaks
the classifier). What L1 sparsity is actually good for here is not lifting
AUC but **confirming, via a second and unrelated method, which base model to
drop or keep** — the same ranking the subset sweep already found. That is a
useful corroboration to cite, but it does not open a new lever: base-model
selection remains the dominant lever, and feature/view engineering around
the combiner remains a documented null.

---

*Session dates: 2026-09-22 → 2026-09-23*
*User: rafael.pachon.alvarez@gmail.com*
*Dataset: cptac_brca*
*Task: TP53_mutation*
*Jobs: 72573, 72574, 72575 (feature ladder); 72576, 72582 (view selection)*
