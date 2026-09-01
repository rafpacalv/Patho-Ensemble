#!/usr/bin/env python3
"""
Test script para verificar que TabPFN devuelve (n,2) siempre.
"""
import numpy as np
import sys
sys.path.insert(0, '/shared/home/JKP6679/Patho-Ensemble/src')

from meta_models import TabPFNMetaClassifier, SnapshotTabPFNMetaClassifier

print("=" * 80)
print("TEST: TabPFN predict_proba() shape")
print("=" * 80)

# Crear datos de prueba
np.random.seed(42)
X_train = np.random.randn(100, 10).astype(np.float32)
y_train = np.random.randint(0, 2, 100)

X_test_single = np.random.randn(1, 10).astype(np.float32)
X_test_multi = np.random.randn(10, 10).astype(np.float32)

# Test 1: TabPFNMetaClassifier
print("\n1️⃣ Testing TabPFNMetaClassifier...")
try:
    model = TabPFNMetaClassifier(device="cpu")
    model.fit(X_train, y_train)

    probs_single = model.predict_proba(X_test_single)
    probs_multi = model.predict_proba(X_test_multi)

    print(f"   Single sample probs shape: {probs_single.shape}")
    print(f"   Multi sample probs shape: {probs_multi.shape}")
    print(f"   Single probs: {probs_single}")
    print(f"   Multi probs[0]: {probs_multi[0]}")

    # Verificar que sea (n, 2)
    assert probs_single.shape == (1, 2), f"❌ Expected (1, 2), got {probs_single.shape}"
    assert probs_multi.shape == (10, 2), f"❌ Expected (10, 2), got {probs_multi.shape}"

    # Verificar que suma a 1
    assert np.allclose(probs_single.sum(axis=1), 1.0), "❌ Probabilities don't sum to 1"
    assert np.allclose(probs_multi.sum(axis=1), 1.0), "❌ Probabilities don't sum to 1"

    print("   ✅ TabPFNMetaClassifier: OK")
except Exception as e:
    print(f"   ❌ Error: {e}")
    import traceback
    traceback.print_exc()

# Test 2: SnapshotTabPFNMetaClassifier
print("\n2️⃣ Testing SnapshotTabPFNMetaClassifier...")
try:
    model = SnapshotTabPFNMetaClassifier(device="cpu", n_snapshots=2)
    model.fit(X_train, y_train)

    probs_single = model.predict_proba(X_test_single)
    probs_multi = model.predict_proba(X_test_multi)

    print(f"   Single sample probs shape: {probs_single.shape}")
    print(f"   Multi sample probs shape: {probs_multi.shape}")
    print(f"   Single probs: {probs_single}")
    print(f"   Multi probs[0]: {probs_multi[0]}")

    # Verificar que sea (n, 2)
    assert probs_single.shape == (1, 2), f"❌ Expected (1, 2), got {probs_single.shape}"
    assert probs_multi.shape == (10, 2), f"❌ Expected (10, 2), got {probs_multi.shape}"

    # Verificar que suma a 1
    assert np.allclose(probs_single.sum(axis=1), 1.0), "❌ Probabilities don't sum to 1"
    assert np.allclose(probs_multi.sum(axis=1), 1.0), "❌ Probabilities don't sum to 1"

    print("   ✅ SnapshotTabPFNMetaClassifier: OK")
except Exception as e:
    print(f"   ❌ Error: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "=" * 80)
print("✅ Todos los tests pasaron")
print("=" * 80)
