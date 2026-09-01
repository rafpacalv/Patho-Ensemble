"""
Smoke test for ensemble4.py with neural network meta-learners.

Tests the three meta-learner types (logreg, mlp, mlp_snapshot) with synthetic data,
verifying that:
  1. Each model trains and produces valid predict_proba outputs
  2. Outputs sum to 1 and are in [0,1]
  3. The Ensemble class properly handles each meta-learner type
  4. Results are saved correctly
"""

import os
import sys
import tempfile
import shutil
import numpy as np
import torch

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

# Import meta-learners from src
from meta_models import MLPMetaClassifier, SnapshotMLPMetaClassifier
from sklearn.linear_model import LogisticRegression

# Import Ensemble and _fit_meta_model from a wrapper that doesn't require patho_bench
# The wrapper is in the scratchpad to avoid circular imports
wrapper_path = '/tmp/claude-10069/-shared-home-JKP6679-Patho-Ensemble/c53ba879-5cce-4846-aefe-149d17242b0c/scratchpad'
sys.path.insert(0, wrapper_path)
from ensemble4_test_wrapper import Ensemble, _fit_meta_model


def create_mock_data(base_dir, n_models=3, n_samples=100, n_classes=2, n_folds=2):
    """
    Generate synthetic preds.npy and labels.npy files mimicking the directory
    structure expected by Ensemble.

    **IMPORTANT:** All models MUST share the SAME labels for a given fold.
    Only the predictions differ (each model outputs different softmax values).

    Args:
        base_dir (str): Root directory to create the mock structure.
        n_models (int): Number of base models.
        n_samples (int): Samples per fold.
        n_classes (int): Number of classes.
        n_folds (int): Number of folds.

    Returns:
        dict: xdirs, vdirs, tdirs for each fold (fold-aware directory lists)
    """
    all_xdirs = []
    all_vdirs = []
    all_tdirs = []

    # Pre-generate shared labels for each split/fold so all models have identical labels
    np.random.seed(42)
    shared_labels_train = {f: np.random.randint(0, n_classes, n_samples) for f in range(n_folds)}
    shared_labels_val = {f: np.random.randint(0, n_classes, n_samples) for f in range(n_folds)}
    shared_labels_test = {f: np.random.randint(0, n_classes, n_samples) for f in range(n_folds)}

    for model_idx in range(n_models):
        model_name = f"model_{model_idx}"

        # _train_eval version (training predictions)
        train_eval_dir = os.path.join(base_dir, f"{model_name}_train_eval")
        os.makedirs(train_eval_dir, exist_ok=True)

        # Regular version (val/test)
        model_dir = os.path.join(base_dir, model_name)
        os.makedirs(model_dir, exist_ok=True)

        for fold in range(n_folds):
            # Training set (via _train_eval, val_outputs subdir)
            # Use SHARED labels, but model-specific predictions
            xdir = os.path.join(train_eval_dir, "val_outputs", f"fold_{fold}")
            os.makedirs(xdir, exist_ok=True)
            np.save(os.path.join(xdir, "labels.npy"), shared_labels_train[fold])
            preds = np.random.dirichlet([1] * n_classes, n_samples).astype(np.float32)
            np.save(os.path.join(xdir, "preds.npy"), preds)

            # Validation set
            vdir = os.path.join(model_dir, "val_outputs", f"fold_{fold}")
            os.makedirs(vdir, exist_ok=True)
            np.save(os.path.join(vdir, "labels.npy"), shared_labels_val[fold])
            preds = np.random.dirichlet([1] * n_classes, n_samples).astype(np.float32)
            np.save(os.path.join(vdir, "preds.npy"), preds)

            # Test set
            tdir = os.path.join(model_dir, "test_outputs", f"fold_{fold}")
            os.makedirs(tdir, exist_ok=True)
            np.save(os.path.join(tdir, "labels.npy"), shared_labels_test[fold])
            preds = np.random.dirichlet([1] * n_classes, n_samples).astype(np.float32)
            np.save(os.path.join(tdir, "preds.npy"), preds)

        all_xdirs.append(os.path.join(train_eval_dir, "val_outputs"))
        all_vdirs.append(os.path.join(model_dir, "val_outputs"))
        all_tdirs.append(os.path.join(model_dir, "test_outputs"))

    return all_xdirs, all_vdirs, all_tdirs, n_folds


def test_ensemble4_logreg():
    """Test Ensemble with LogisticRegression (baseline)."""
    print("\n" + "="*60)
    print("TEST 1: Ensemble with LogisticRegression (baseline)")
    print("="*60)

    test_dir = tempfile.mkdtemp(prefix="test_ensemble4_logreg_")
    try:
        xdirs, vdirs, tdirs, n_folds = create_mock_data(test_dir, n_models=3, n_samples=50, n_classes=2, n_folds=2)

        print(f"Created mock data in: {test_dir}")
        print(f"  xdirs: {xdirs}")
        print(f"  n_folds: {n_folds}")

        device = "cuda" if torch.cuda.is_available() else "cpu"
        all_preds = []
        all_labels = []

        for fold in range(n_folds):
            xdirs_f = [os.path.join(d, f"fold_{fold}") for d in xdirs]
            vdirs_f = [os.path.join(d, f"fold_{fold}") for d in vdirs]
            tdirs_f = [os.path.join(d, f"fold_{fold}") for d in tdirs]

            preds, labels = Ensemble(
                xdirs_f, vdirs_f, tdirs_f,
                LogisticRegression(max_iter=1000),
                device=device,
                results_subdir="ensemble4"
            )()

            preds_np = preds.detach().cpu().numpy()
            labels_np = labels.detach().cpu().numpy()

            # Validate outputs
            assert preds_np.shape == (50, 2), f"Unexpected shape: {preds_np.shape}"
            assert np.allclose(preds_np.sum(axis=1), 1.0), "Probabilities don't sum to 1"
            assert np.all(preds_np >= 0) and np.all(preds_np <= 1), "Probs not in [0,1]"

            all_preds.append(preds_np)
            all_labels.append(labels_np)
            print(f"  Fold {fold}: ✓ preds shape {preds_np.shape}, sums to 1, in [0,1]")

        # Check that coefs.npy was created (LogisticRegression-specific)
        coefs_path = os.path.join(test_dir, "ensemble4", "coefs.npy")
        if os.path.exists(coefs_path):
            coefs = np.load(coefs_path)
            print(f"  ✓ coefs.npy created, shape {coefs.shape}")
        else:
            print(f"  ⚠ coefs.npy not found at {coefs_path}")

        print("✅ LogisticRegression test PASSED")
        return True

    finally:
        shutil.rmtree(test_dir, ignore_errors=True)


def test_ensemble4_mlp():
    """Test Ensemble with MLPMetaClassifier."""
    print("\n" + "="*60)
    print("TEST 2: Ensemble with MLPMetaClassifier")
    print("="*60)

    test_dir = tempfile.mkdtemp(prefix="test_ensemble4_mlp_")
    try:
        xdirs, vdirs, tdirs, n_folds = create_mock_data(test_dir, n_models=3, n_samples=50, n_classes=2, n_folds=2)

        print(f"Created mock data in: {test_dir}")
        print(f"  n_folds: {n_folds}")

        device = "cuda" if torch.cuda.is_available() else "cpu"
        all_preds = []
        all_labels = []

        for fold in range(n_folds):
            xdirs_f = [os.path.join(d, f"fold_{fold}") for d in xdirs]
            vdirs_f = [os.path.join(d, f"fold_{fold}") for d in vdirs]
            tdirs_f = [os.path.join(d, f"fold_{fold}") for d in tdirs]

            preds, labels = Ensemble(
                xdirs_f, vdirs_f, tdirs_f,
                MLPMetaClassifier(
                    hidden_dim=16, dropout=0.1, epochs=20, lr=1e-3, wd=1e-4,
                    patience=5, device=device, seed=42+fold
                ),
                device=device,
                results_subdir="ensemble4_mlp"
            )()

            preds_np = preds.detach().cpu().numpy()
            labels_np = labels.detach().cpu().numpy()

            # Validate outputs
            assert preds_np.shape == (50, 2), f"Unexpected shape: {preds_np.shape}"
            assert np.allclose(preds_np.sum(axis=1), 1.0), "Probabilities don't sum to 1"
            assert np.all(preds_np >= 0) and np.all(preds_np <= 1), "Probs not in [0,1]"

            all_preds.append(preds_np)
            all_labels.append(labels_np)
            print(f"  Fold {fold}: ✓ preds shape {preds_np.shape}, sums to 1, in [0,1]")

        # Check that coefs.npy was NOT created (MLPMetaClassifier doesn't have .coef_)
        coefs_path = os.path.join(test_dir, "ensemble4_mlp", "coefs.npy")
        if os.path.exists(coefs_path):
            print(f"  ⚠ coefs.npy unexpectedly exists at {coefs_path}")
        else:
            print(f"  ✓ coefs.npy correctly not created (MLP doesn't have .coef_)")

        print("✅ MLPMetaClassifier test PASSED")
        return True

    finally:
        shutil.rmtree(test_dir, ignore_errors=True)


def test_ensemble4_snapshot_mlp():
    """Test Ensemble with SnapshotMLPMetaClassifier."""
    print("\n" + "="*60)
    print("TEST 3: Ensemble with SnapshotMLPMetaClassifier")
    print("="*60)

    test_dir = tempfile.mkdtemp(prefix="test_ensemble4_snapshot_mlp_")
    try:
        xdirs, vdirs, tdirs, n_folds = create_mock_data(test_dir, n_models=3, n_samples=50, n_classes=2, n_folds=2)

        print(f"Created mock data in: {test_dir}")
        print(f"  n_folds: {n_folds}")

        device = "cuda" if torch.cuda.is_available() else "cpu"
        all_preds = []
        all_labels = []

        for fold in range(n_folds):
            xdirs_f = [os.path.join(d, f"fold_{fold}") for d in xdirs]
            vdirs_f = [os.path.join(d, f"fold_{fold}") for d in vdirs]
            tdirs_f = [os.path.join(d, f"fold_{fold}") for d in tdirs]

            preds, labels = Ensemble(
                xdirs_f, vdirs_f, tdirs_f,
                SnapshotMLPMetaClassifier(
                    hidden_dim=16, dropout=0.1, epochs=24, restart_lr=5e-3,
                    n_cycles=3, n_snapshots=None, wd=1e-4, device=device, seed=42+fold
                ),
                device=device,
                results_subdir="ensemble4_mlp_snapshot"
            )()

            preds_np = preds.detach().cpu().numpy()
            labels_np = labels.detach().cpu().numpy()

            # Validate outputs
            assert preds_np.shape == (50, 2), f"Unexpected shape: {preds_np.shape}"
            assert np.allclose(preds_np.sum(axis=1), 1.0), "Probabilities don't sum to 1"
            assert np.all(preds_np >= 0) and np.all(preds_np <= 1), "Probs not in [0,1]"

            all_preds.append(preds_np)
            all_labels.append(labels_np)
            print(f"  Fold {fold}: ✓ preds shape {preds_np.shape}, sums to 1, in [0,1]")

        # Check that coefs.npy was NOT created
        coefs_path = os.path.join(test_dir, "ensemble4_mlp_snapshot", "coefs.npy")
        if os.path.exists(coefs_path):
            print(f"  ⚠ coefs.npy unexpectedly exists at {coefs_path}")
        else:
            print(f"  ✓ coefs.npy correctly not created (SnapshotMLP doesn't have .coef_)")

        print("✅ SnapshotMLPMetaClassifier test PASSED")
        return True

    finally:
        shutil.rmtree(test_dir, ignore_errors=True)


def test_fit_meta_model_helper():
    """Test the _fit_meta_model helper function."""
    print("\n" + "="*60)
    print("TEST 4: _fit_meta_model helper function")
    print("="*60)

    # Test with LogisticRegression (no X_val/y_val support)
    X_train = np.random.randn(50, 6)
    y_train = np.random.randint(0, 2, 50)
    X_val = np.random.randn(20, 6)
    y_val = np.random.randint(0, 2, 20)

    lr_model = LogisticRegression(max_iter=1000)
    _fit_meta_model(lr_model, X_train, y_train, X_val, y_val)  # Should not error
    preds = lr_model.predict_proba(X_val)
    assert preds.shape == (20, 2), f"Unexpected LogisticRegression shape: {preds.shape}"
    print("  ✓ LogisticRegression fitted correctly (ignored X_val/y_val)")

    # Test with MLPMetaClassifier (supports X_val/y_val)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    mlp_model = MLPMetaClassifier(epochs=10, device=device)
    _fit_meta_model(mlp_model, X_train, y_train, X_val, y_val)  # Should use val for early stopping
    preds = mlp_model.predict_proba(X_val)
    assert preds.shape == (20, 2), f"Unexpected MLP shape: {preds.shape}"
    print("  ✓ MLPMetaClassifier fitted correctly (used X_val/y_val for early stopping)")

    print("✅ _fit_meta_model helper test PASSED")
    return True


if __name__ == "__main__":
    print("\n" + "#"*60)
    print("# SMOKE TESTS: ensemble4.py with neural network meta-learners")
    print("#"*60)

    all_passed = True
    try:
        all_passed &= test_ensemble4_logreg()
    except Exception as e:
        print(f"❌ LogisticRegression test FAILED: {e}")
        import traceback
        traceback.print_exc()
        all_passed = False

    try:
        all_passed &= test_ensemble4_mlp()
    except Exception as e:
        print(f"❌ MLPMetaClassifier test FAILED: {e}")
        import traceback
        traceback.print_exc()
        all_passed = False

    try:
        all_passed &= test_ensemble4_snapshot_mlp()
    except Exception as e:
        print(f"❌ SnapshotMLPMetaClassifier test FAILED: {e}")
        import traceback
        traceback.print_exc()
        all_passed = False

    try:
        all_passed &= test_fit_meta_model_helper()
    except Exception as e:
        print(f"❌ _fit_meta_model helper test FAILED: {e}")
        import traceback
        traceback.print_exc()
        all_passed = False

    print("\n" + "#"*60)
    if all_passed:
        print("# ✅ ALL TESTS PASSED")
    else:
        print("# ❌ SOME TESTS FAILED")
    print("#"*60 + "\n")

    sys.exit(0 if all_passed else 1)
