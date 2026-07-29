# Quick Start Guide - Patho-Ensemble with PARADIS Data

⏱️ **Estimated reading time: 5 minutes**

## Prerequisites Checklist

- [ ] Access to `/shared/home/PARADIS/datos` 
- [ ] Python 3.7+
- [ ] Conda installed
- [ ] Git cloned or in `/home/JKP6679/Patho-Ensemble`

## Step 1: Setup Environment (5 min)

```bash
cd /home/JKP6679/Patho-Ensemble

# Create conda environment
conda env create -f environment.yml
conda activate patho-ensemble

# Verify PARADIS data access
python verify_paradis_setup.py
```

**Expected output**: 5/6 checks pass ✅

## Step 2: Choose Your Path

### Path A: Quick Test (15 min, single fold)
Run a single fold to verify everything works:

```bash
python src/train_abmil.py \
    --foundational_model ctranspath \
    --latent_dim 768 \
    --work_dir /shared/home/PARADIS/datos \
    --train_source cptac_brca \
    --tissue_patching 20x_224px_0px_overlap \
    --task_name TP53_mutation \
    --epochs 10 \
    --fold 0
```

✅ **Success**: Look for `val_outputs/fold_0/` and `test_outputs/fold_0/`

---

### Path B: Full Training (4-12 hours, all 50 folds)
Train complete model across all folds:

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

Output: `/shared/home/PARADIS/datos/cptac_brca/TP53_mutation/abmil/ctranspath_20x_224px_0px_overlap/`

---

### Path C: Ensemble Comparison (12-24 hours)

#### Step C1: Train multiple models
```bash
for model in ctranspath uni_v2 virchow_v1; do
  latim=768
  [ "$model" = "virchow_v1" ] && latim=2560
  
  python src/train_abmil.py \
    --foundational_model $model \
    --latent_dim $latim \
    --work_dir /shared/home/PARADIS/datos \
    --train_source cptac_brca \
    --tissue_patching 20x_224px_0px_overlap \
    --task_name TP53_mutation \
    --epochs 100 \
    --fold all &
done
wait
```

#### Step C2: Evaluate ensemble
```bash
python src/ensemble.py \
    --foundational_models ctranspath uni_v2 virchow_v1 \
    --work_dir /shared/home/PARADIS/datos \
    --train_source cptac_brca \
    --tissue_patching 20x_224px_0px_overlap \
    --task_name TP53_mutation \
    --weights_type auc_roc
```

Output: `/shared/home/PARADIS/datos/cptac_brca/TP53_mutation/abmil/ensemble/`

---

### Path D: Meta-Learner Ensemble (16-36 hours)
Most sophisticated approach using Logistic Regression meta-learner:

```bash
# Step 1: Train base models (see Path C1)

# Step 2: Get training predictions for meta-learner
for model in ctranspath uni_v2 virchow_v1; do
  latim=768
  [ "$model" = "virchow_v1" ] && latim=2560
  
  python src/test_abmil.py \
    --foundational_model $model \
    --latent_dim $latim \
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

Output: `/shared/home/PARADIS/datos/cptac_brca/TP53_mutation/abmil/ensemble4/`

---

## Step 3: Monitor Progress

### Check training progress
```bash
# Watch for new checkpoint directories
ls -la /shared/home/PARADIS/datos/cptac_brca/TP53_mutation/abmil/ctranspath_20x_224px_0px_overlap/

# Check fold outputs
ls /shared/home/PARADIS/datos/cptac_brca/TP53_mutation/abmil/ctranspath_20x_224px_0px_overlap/test_outputs/
```

### View results
```bash
# Metrics for single model
cat /shared/home/PARADIS/datos/cptac_brca/TP53_mutation/abmil/ctranspath_20x_224px_0px_overlap/test_metrics_summary.json

# Metrics for ensemble
cat /shared/home/PARADIS/datos/cptac_brca/TP53_mutation/abmil/ensemble/test_metrics_summary.json
```

## Available Datasets to Try

| Dataset | Size | Use Case |
|---------|------|----------|
| cptac_brca | 113 | Quick testing |
| cptac_coad | 327 | Large-scale experiments |
| cptac_luad | ? | Survival analysis |
| bc_therapy | ? | Treatment response |

## Available Models to Compare

| Model | Dim | Notes |
|-------|-----|-------|
| ctranspath | 768 | Good baseline |
| uni_v2 | 768 | State-of-the-art |
| virchow_v1 | 2560 | Higher dimensional |
| conch_v1_5 | 768 | Pathology-specific |
| phikon_v2 | 768 | Recent foundation model |
| hoptimus1 | 768 | Optimized for histology |

## Troubleshooting

### Error: "Missing path"
```bash
# Verify data exists
ls /shared/home/PARADIS/datos/patches/cptac_brca/TP53_mutation/k=all.tsv
ls /shared/home/PARADIS/datos/features/cptac_brca/features_ctranspath_monai/*.h5
```

### Error: "Dimension mismatch"
```bash
# Check actual embedding dimension
python3 -c "
import h5py, os
path = '/shared/home/PARADIS/datos/features/cptac_brca/features_virchow_v1_monai'
h5_file = [f for f in os.listdir(path) if f.endswith('.h5')][0]
with h5py.File(os.path.join(path, h5_file), 'r') as f:
    print('Actual dimension:', f['features'].shape[1])
"
# Use returned value for --latent_dim
```

### Error: "Out of memory"
```bash
# Train single fold instead of all
--fold 0  # instead of --fold all
```

---

## What to Read Next

| Document | Purpose |
|----------|---------|
| **README.md** | Complete reference guide |
| **PARADIS_COMPATIBILITY.md** | Technical details about data |
| **examples_paradis.sh** | More command examples |
| **CHANGES_SUMMARY.md** | What changed and why |

## Need Help?

1. **Quick verification**: `python verify_paradis_setup.py`
2. **See examples**: `./examples_paradis.sh` (interactive menu)
3. **Check logs**: Look in model output directories for `*_logs_*.txt`
4. **Read docs**: Start with README.md section "Troubleshooting"

---

## Typical Timeline

| Phase | Time | What's Happening |
|-------|------|------------------|
| Environment setup | 5 min | Installing dependencies |
| Data verification | 2 min | Checking PARADIS access |
| Quick test | 5-15 min | Testing single fold |
| Full training | 2-50 hrs | Training 50-fold CV (hardware dependent) |
| Ensemble training | 2-24 hrs | Training additional models |
| Evaluation | 10-30 min | Computing metrics and ensemble predictions |

---

## Success Criteria

✅ **Setup is working if**:
- `verify_paradis_setup.py` passes 5/6 checks
- You can see `val_outputs/fold_X/` directories appear

✅ **Training is working if**:
- Logs show training loss decreasing
- New files appear in `val_outputs/` and `test_outputs/`
- No errors in console output

✅ **Ensemble is working if**:
- Metrics files appear in `ensemble/` directory
- JSON files contain AUC, F1, accuracy scores

---

**Ready to start?** Go to Step 1 above! 🚀

---

*Last updated: 2026-07-29*
*Status: ✅ Production Ready*
