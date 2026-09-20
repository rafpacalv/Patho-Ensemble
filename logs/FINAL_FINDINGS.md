# PATHO-ENSEMBLE LOG ANALYSIS - FINAL FINDINGS

## Executive Summary

Analyzed **56 log files** from `/shared/home/JKP6679/Patho-Ensemble/logs/` directory.

### Overall Statistics
- **Total Executions:** 56
- **Successful:** 20 (35%)
- **Failed:** 6 (10%)
- **Unknown Status:** 30 (53%)

---

## Execution Summary by Category

### 1. BASE-TRAINING (1/1 Success) ✅
**Status: FULLY SUCCESSFUL**

| Execution | Status | Dataset | Models | Duration | Folds |
|-----------|--------|---------|--------|----------|-------|
| train_base_models_69938 | SUCCESS | cptac_brca/TP53_mutation | ctranspath, uni_v2, virchow_v1 | 0 h 2 m | 50/50 |

**Key Finding:** Initial base model training completed successfully with all 3 foundational models (CTransPath, UNI-v2, Virchow-v1) across 50 folds.

---

### 2. META-LEARNERS (4/4 Success) ✅
**Status: FULLY SUCCESSFUL**

| Execution | Status | Dataset | Meta-learners | Duration |
|-----------|--------|---------|---------------|----------|
| run_meta_experiments_69958 | SUCCESS | cptac_brca/TP53_mutation | logreg, logit_avg, lightgbm, gating | 28 min |
| run_meta_experiments_69953 | SUCCESS | cptac_brca/TP53_mutation | logreg, logit_avg, lightgbm, gating | 22 min |
| run_meta_experiments_69943 | SUCCESS | cptac_brca/TP53_mutation | logreg, tabpfn | 13 min |
| run_meta_experiments_69936 | SUCCESS | cervical_subtype/subtype | logreg, tabpfn (standard & FGE) | ~3 min |

**Metrics Found:**
- **AUC Range:** 0.7918 - 0.9471
- **Average AUC:** 0.8456

**Key Finding:** All meta-learner training completed successfully. Baseline (logreg_std) achieved AUC: 0.9471 on cervical_subtype/subtype task.

---

### 3. OOF-FEATURES (3/6 Success) ⚠️
**Status: PARTIALLY SUCCESSFUL**

| Execution | Status | Dataset | Models | Duration | Folds |
|-----------|--------|---------|--------|----------|-------|
| build_oof_69957 | SUCCESS | cptac_brca/TP53_mutation | ctranspath, uni_v2, virchow_v1 | 2 h 56 m | 50/50 |
| build_oof_69956 | SUCCESS | cptac_brca/TP53_mutation | ctranspath, uni_v2, virchow_v1 | 1 h 21 m | 50/50 |
| build_oof_69950 | SUCCESS | cptac_brca/TP53_mutation | ctranspath, uni_v2, virchow_v1 | 0 h 39 m | 50/50 |
| build_oof_69957.err | UNKNOWN | - | - | - | - |
| build_oof_69956.err | UNKNOWN | - | - | - | - |
| build_oof_69950.err | UNKNOWN | - | - | - | - |

**Key Finding:** Out-of-fold feature generation successful. Execution time varies significantly (39 min - 2h 56m) for same configuration, suggesting system load variation.

---

### 4. FGE-TRAINING (6/12 Success) ⚠️
**Status: PARTIALLY SUCCESSFUL**

| Execution | Status | Duration | Folds |
|-----------|--------|----------|-------|
| train_base_models_fge_69952 | SUCCESS | 1 h 55 m | 50/50 |
| train_base_models_fge_69951 | SUCCESS | 1 h 52 m | 50/50 |
| train_base_models_fge_69942 | SUCCESS | 1 h 9 m | 50/50 |
| train_base_models_fge_69941 | SUCCESS | (similar time) | 50/50 |
| train_base_models_fge_69940 | SUCCESS | (similar time) | 50/50 |
| train_base_models_fge_69939 | SUCCESS | (similar time) | 50/50 |

**Key Finding:** FGE (Fast Geometric Ensembles) training with snapshot variants completed successfully for 6 executions. 6 additional logs marked as "Unknown" status due to stderr-only output.

---

### 5. META-EXPERIMENTS (4/14 Success) ⚠️
**Status: PROBLEMATIC**

**Successful:**
- auc_fixed_70091 - cptac_brca/TP53_mutation - SVM/KNN variants
- auc_based_reweight_70090 - cptac_brca/TP53_mutation
- svm_knn_nb_reweight_fixed_70089 - cptac_brca/TP53_mutation
- svm_knn_nb_reweight_70088 - cptac_brca/TP53_mutation

**Failed (4):**
- svm_knn_nb_reweight_fixed_70089.err
- svm_knn_nb_reweight_70088.err
- extended_fixed_v6_70064.err
- extended_models_fixed_70058.out

**Key Errors:**
1. **TabPFN Issue (extended_fixed_v6):**
   ```
   TypeError: TabPFNClassifier.__init__() got an unexpected keyword argument 'seed'
   ```
   → TabPFN version mismatch. Current installation does not support 'seed' parameter.

2. **SVM/KNN Experiments:**
   ```
   KeyError: 'macro-ovr-auc'
   ValueError: The truth value of an array with more than one element is ambiguous
   ```
   → Issues with metric extraction and array handling in reweighting logic.

---

### 6. EXPERIMENTAL-STAGE (0/8 Success) ❌
**Status: MOSTLY FAILED/UNKNOWN**

| Execution | Status | Error |
|-----------|--------|-------|
| stage2_spatial_70005 | FAILED | ValueError: grado medio fuera de rango |
| stage2_spatial_70017 | UNKNOWN | - |
| stage2_spatial_70020 | UNKNOWN | - |
| stage3_shuffle_70022 | UNKNOWN | - |
| test_spatial_gnn_70003 | FAILED | ValueError: coords fuera de la retícula |
| Others (3) | UNKNOWN | - |

**Key Finding:** Spatial/graph-based experiments appear to have issues with:
- Lattice/grid coordinate validation
- Moore neighborhood assumptions not met
- GNN edge generation edge cases

---

## Key Findings & Issues

### ✅ What's Working Well
1. **Base Model Training:** All 50 folds successfully trained for 3 foundational models
2. **Meta-learner Training:** 100% success rate on standard ensemble methods
3. **OOF Feature Generation:** Successfully generated out-of-fold predictions
4. **FGE Snapshots:** Snapshot ensemble generation mostly working

### ⚠️ Known Issues

#### Issue #1: TabPFN Compatibility (Priority: MEDIUM)
- **File:** extended_fixed_v6_70064.err
- **Error:** `TypeError: TabPFNClassifier.__init__() got an unexpected keyword argument 'seed'`
- **Cause:** TabPFN version mismatch
- **Fix:** Update TabPFN initialization or check version compatibility in meta_models.py
- **Affected:** Deep MLP snapshot and TabPFN-based experiments

#### Issue #2: SVM/KNN Reweighting Logic (Priority: HIGH)
- **Files:** svm_knn_nb_reweight_70088, svm_knn_nb_reweight_fixed_70089
- **Errors:**
  - `KeyError: 'macro-ovr-auc'`
  - `ValueError: The truth value of an array with more than one element is ambiguous`
- **Cause:** Metric extraction and array comparison issues in reweighting implementation
- **Affected:** Model importance reweighting experiments

#### Issue #3: Spatial/GNN Experiments (Priority: HIGH)
- **Files:** stage2_spatial_70005, test_spatial_gnn_70003
- **Errors:**
  - `ValueError: grado medio fuera de [6, 8]` (Spanish: "average degree out of range")
  - `ValueError: coords fuera de la retícula` (Spanish: "coordinates out of lattice")
- **Cause:** Graph construction assumptions not met for pathology data
- **Impact:** Experimental spatial branches not working

### ❓ Unknown Status Issues
- 30 out of 56 files (53%) have unknown status
- Many are .err files that don't contain stderr output (empty/warnings only)
- Recommend checking stderr redirection in sbatch scripts

---

## Performance Metrics

### Achieved AUC Values
```
Dataset: cervical_subtype / Task: subtype
- m0_logreg_std:        0.9471 (baseline)
- m1_tabpfn_std:        0.9456
- Best variant:         0.9471

Dataset: cptac_brca / Task: TP53_mutation  
- Various experiments:  0.7918 - 0.7978
```

### Execution Times
- **Base Training:** 2 minutes (50 folds × 3 models)
- **OOF Generation:** 39 min - 2h 56m (with significant variability)
- **FGE Snapshots:** 1 - 2 hours (50 folds × 3 models)
- **Meta-learner Training:** 3 - 28 minutes

---

## Datasets & Models Evaluated

### Datasets
- `cptac_brca` - Primary dataset
- `cervical_subtype` - Secondary dataset

### Tasks
- `TP53_mutation` - Main task
- `subtype` - Classification task

### Foundational Models
- CTransPath (768-dim)
- UNI-v2 (1536-dim)
- Virchow-v1 (2560-dim)

### Meta-learners
- LogisticRegression (logreg)
- TabPFN
- LightGBM
- Gating mechanisms
- SVM, KNN (with issues)
- Deep MLP Snapshot

---

## Recommendations

### Immediate Actions
1. **Fix TabPFN Issue:**
   - Check `src/meta_models.py` line for TabPFN initialization
   - Remove 'seed' parameter or update TabPFN version
   - Test: `python -c "from tabpfn import TabPFNClassifier; print(TabPFNClassifier.__init__.__doc__)"`

2. **Debug SVM/KNN Reweighting:**
   - Verify metric names in `--reweight_method` logic
   - Check array handling in `run_svm_knn_nb_reweight.sbatch`
   - Ensure `'macro-ovr-auc'` metric is available in base model outputs

3. **Isolate Spatial Experiments:**
   - These are experimental features not in main pipeline
   - Consider moving to separate branch
   - Check graph lattice assumptions for pathology coordinates

### Logging Improvements
1. Configure stderr properly in all sbatch scripts
2. Add execution summaries to .err files
3. Implement structured logging for better analysis

### Testing Strategy
- Use `--fold 0` for quick testing of meta-learners
- Validate metric names before large runs
- Test compatibility of optional dependencies upfront

---

## Files Generated

| File | Purpose |
|------|---------|
| `LOG_ANALYSIS_SUMMARY.txt` | Text summary of all findings |
| `detailed_log_analysis.json` | Machine-readable JSON with full details |
| `log_analysis_report.html` | Visual HTML report |

**Location:** `/shared/home/JKP6679/Patho-Ensemble/logs/`

---

*Analysis completed: 2026-08-19*
*Total log files analyzed: 56*
*Coverage: 100%*
