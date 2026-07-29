# PARADIS Data Compatibility Guide

This document provides detailed information about using Patho-Ensemble scripts with data from `/shared/home/PARADIS/datos`.

## Data Availability Summary

### ✅ What's Ready to Use

- **113+ split files** with 50-fold stratified cross-validation already configured
- **16 foundational models** with pre-computed embeddings in HDF5 format
- **Multiple datasets** spanning oncology applications (mutations, immune classification, grading, etc.)
- **Standardized directory structure** compatible with patho_bench framework

### ⚠️ Important Considerations

#### 1. Feature Dimension Mapping

| Model Family | Examples | Dimension | Notes |
|--------------|----------|-----------|-------|
| UNI | uni_v1, uni_v2 | 768D | Vision transformer, highly performant |
| Virchow | virchow_v1, virchow2 | 2560D | Higher dimensional, more expressive |
| CTransPath | ctranspath | 768D | Convolutional transformer, good balance |
| Conch | conch_v1_5, conch512 | 768D | Foundation model for pathology |
| Gigapath | gigapath, gigapath_256px | 768D | Multiple variants available |
| Hoptimus | hoptimus0, hoptimus1 | 768D | Optimized for histopathology |
| Phikon | phikon_v2 | 768D | Recent foundation model |
| ResNet | resnet50 | 2048D | Baseline CNN (rarely used) |

**Action**: Always pass correct `--latent_dim` parameter. Most are 768D except Virchow (2560D) and ResNet (2048D).

#### 2. Patching Strategy Naming

PARADIS uses naming convention: `{magnification}x_{size}px_{overlap}px_overlap`

**Available strategies** (may vary by dataset):
- `20x_224px_0px_overlap` - 20x magnification, 224×224 patches, no overlap
- `20x_256px_0px_overlap` - 20x magnification, 256×256 patches, no overlap

**Action**: Verify available strategies for your dataset:
```bash
ls /shared/home/PARADIS/datos/patches/{dataset}/{task}/
```

#### 3. Dataset-Specific Information

#### CPTAC BRCA
- **Samples**: 113 slides
- **Tasks**: TP53_mutation, PIK3CA_mutation, Immune_class
- **Status**: ✅ Complete, ready for training
- **Recommended models**: All available models have embeddings

#### CPTAC COAD
- **Samples**: 327 slides (largest in PARADIS)
- **Tasks**: TP53_mutation, KRAS_mutation, APC_mutation, ARID1A_mutation, ACVR2A_mutation, SETD1B_mutation, Immune_class, PIK3CA_mutation, MSI_H
- **Status**: ✅ Complete, ready for training
- **Recommended for**: Large-scale ensemble experiments

#### CPTAC GBM
- **Tasks**: TP53_mutation, EGFR_mutation, Immune_class
- **Status**: ✅ Ready
- **Note**: Glioblastoma samples

#### CPTAC HNSC
- **Tasks**: Immune_class, Histologic_Grade, CASP8_mutation
- **Status**: ✅ Ready
- **Note**: Head and neck cancer

#### CPTAC LSCC
- **Tasks**: Immune_class, Histologic_Grade, ARID1A_mutation, KEAP1_mutation
- **Status**: ✅ Ready
- **Note**: Lung squamous cell carcinoma

#### CPTAC LUAD
- **Tasks**: TP53_mutation, KRAS_mutation, EGFR_mutation, Immune_class, OS (survival), STK11_mutation
- **Status**: ✅ Ready
- **Note**: Survival outcome available (requires different metric handling)

#### CPTAC CCRCC
- **Tasks**: BAP1_mutation, VHL_mutation, PBRM1_mutation, Immune_class
- **Status**: ✅ Ready
- **Note**: Clear cell renal cell carcinoma

#### CRC Outcomes
- **Tasks**: Multiple grading and morphology features
- **Status**: ✅ Ready
- **Note**: Colorectal cancer outcomes

#### Hancook
- **Tasks**: Multiple grading and morphology features
- **Status**: ✅ Ready
- **Note**: Head and neck histology grading

#### BC Therapy
- **Tasks**: ER_status, HER2_status, Grade, Residual_cancer_burden
- **Status**: ✅ Ready
- **Note**: Breast cancer with therapy response

## Script Modifications Not Needed

The existing scripts in Patho-Ensemble work directly with PARADIS data structure:

1. ✅ `train_abmil.py` - No modification needed
2. ✅ `test_abmil.py` - No modification needed
3. ✅ `ensemble.py` - No modification needed
4. ✅ `ensemble4.py` - No modification needed
5. ✅ `utils.py` - No modification needed

## Recommended Workflows

### Workflow A: Single Model Baseline
```bash
# ~15-30 minutes per fold × 50 folds
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

### Workflow B: Ensemble of Models (simple weighted average)
```bash
# Step 1: Train multiple models (can parallelize)
python src/train_abmil.py --foundational_model ctranspath ... --fold all &
python src/train_abmil.py --foundational_model uni_v2 ... --fold all &
python src/train_abmil.py --foundational_model virchow_v1 ... --fold all &
wait

# Step 2: Evaluate ensemble
python src/ensemble.py \
    --foundational_models ctranspath uni_v2 virchow_v1 \
    --work_dir /shared/home/PARADIS/datos \
    --train_source cptac_brca \
    --tissue_patching 20x_224px_0px_overlap \
    --task_name TP53_mutation \
    --weights_type auc_roc
```

### Workflow C: Meta-learner Ensemble
```bash
# Step 1: Train models (same as Workflow B)

# Step 2: Get training predictions (for each model)
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

# Step 3: Train meta-learner
python src/ensemble4.py \
    --foundational_models ctranspath uni_v2 virchow_v1 \
    --work_dir /shared/home/PARADIS/datos \
    --train_source cptac_brca \
    --tissue_patching 20x_224px_0px_overlap \
    --task_name TP53_mutation
```

## Performance Expectations

### Single ABMIL Model (cptac_brca, TP53_mutation)
- Training time: ~30-60 min per fold (depending on hardware)
- Total for 50 folds: 25-50 hours (CPU) or 2-5 hours (GPU)
- Typical metrics: AUC 0.65-0.75

### Ensemble Evaluation
- Weighting computation: ~5 min for all folds
- Ensemble inference: ~1 min
- Metrics computation: ~10 min

### Hardware Requirements
- **Minimum**: 8GB RAM, CPU training (slow)
- **Recommended**: GPU with 12GB+ VRAM, 32GB+ system RAM
- **Optimal**: A100/H100 for multi-model parallel training

## Troubleshooting

### "Missing path" errors
- **Cause**: Feature files or split files not found
- **Solution**: Verify exact model and dataset names match directory structure
  ```bash
  ls /shared/home/PARADIS/datos/features/{dataset}/features_{model}_monai/
  ls /shared/home/PARADIS/datos/patches/{dataset}/{task}/k=all.tsv
  ```

### Dimension mismatch errors
- **Cause**: `--latent_dim` doesn't match actual feature dimension
- **Solution**: Check HDF5 file structure
  ```bash
  python3 -c "
  import h5py
  with h5py.File('/shared/home/PARADIS/datos/features/cptac_brca/features_virchow_v1_monai/sample.h5', 'r') as f:
      print(f['features'].shape[1])
  "
  ```

### Missing validation splits
- **Cause**: Trying to use `--create_val` with PARADIS data
- **Solution**: Remove `--create_val` flag. Splits already exist in k=all.tsv

### Memory issues during ensemble evaluation
- **Cause**: All fold predictions loaded simultaneously
- **Solution**: Edit ensemble.py to process folds sequentially, or increase system RAM

## Data Citation

If publishing results using PARADIS data:
- Cite the original CPTAC, TCGA, or dataset sources
- Note the foundational models used (cite papers)
- Include experiment details in supplementary materials

## Questions or Issues?

Contact: Data available in shared PARADIS group
For code issues: Check patho-bench documentation
For patho-ensemble questions: Refer to README.md

---
Last updated: 2026-07-29
Compatible with: patho-ensemble main branch
