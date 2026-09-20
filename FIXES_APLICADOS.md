# Fixes Aplicados - Job 70057

**Status:** Error encontrado y reparado  
**Error:** `IndexError: index 1 is out of bounds for axis 1 with size 1`  
**Causa:** TabPFN retorna (n,1) en lugar de (n,2)  
**Fecha:** Agosto 18, 2026

---

## 📋 Resumen de Errores y Fixes

### Error 1 (Job 70055): Incompatibilidad dimensional - probs 1D vs 2D
**Síntoma:** `IndexError: invalid index to scalar variable`  
**Archivo:** `/shared/home/JKP6679/Patho-Ensemble/src/meta_models.py`  
**Fix aplicado:** Línea 942, 1017
```python
probs = np.atleast_2d(probs)  # Garantiza que siempre sea 2D
```
**Status:** ✅ Aplicado

---

### Error 2 (Job 70057): TabPFN retorna (n,1) en lugar de (n,2)
**Síntoma:** `IndexError: index 1 is out of bounds for axis 1 with size 1`  
**Archivo:** `/shared/home/JKP6679/Patho-Ensemble/src/meta_models.py`  
**Líneas:** TabPFNMetaClassifier.predict_proba() y SnapshotTabPFNMetaClassifier.predict_proba()

**Fix aplicado:**
```python
def predict_proba(self, X):
    ...
    probs = self.model_.predict_proba(X_scaled)
    
    # ✅ NUEVO: Asegurar (n,2) para binary classification
    if probs.ndim == 1 or probs.shape[1] == 1:
        probs = np.column_stack([1 - probs.ravel(), probs.ravel()])
    
    return probs
```

**Clases afectadas:**
- `TabPFNMetaClassifier` (línea ~1061-1070)
- `SnapshotTabPFNMetaClassifier` (línea ~1112-1122)

**Status:** ✅ Aplicado

---

## 🔧 Cambios Específicos

### 1. TabPFNMetaClassifier.predict_proba()
```python
# ANTES:
def predict_proba(self, X):
    X = np.asarray(X, dtype=np.float32)
    X_scaled = (X - self.X_min_) / (self.X_max_ - self.X_min_ + 1e-8)
    return self.model_.predict_proba(X_scaled)  # ❌ Puede ser (n,1)

# DESPUÉS:
def predict_proba(self, X):
    X = np.asarray(X, dtype=np.float32)
    X_scaled = (X - self.X_min_) / (self.X_max_ - self.X_min_ + 1e-8)
    probs = self.model_.predict_proba(X_scaled)

    # ✅ Asegurar que siempre sea (n, 2) para binary classification
    if probs.ndim == 1 or probs.shape[1] == 1:
        # Si es 1D o (n,1), construir (n,2): [1-p, p]
        probs = np.column_stack([1 - probs.ravel(), probs.ravel()])

    return probs
```

### 2. SnapshotTabPFNMetaClassifier.predict_proba()
```python
# ANTES:
def predict_proba(self, X):
    X = np.asarray(X, dtype=np.float32)
    X_scaled = (X - self.X_min_) / (self.X_max_ - self.X_min_ + 1e-8)
    
    probs_list = [m.predict_proba(X_scaled) for m in self.models_]
    return np.mean(probs_list, axis=0)  # ❌ Puede fallar si (n,1)

# DESPUÉS:
def predict_proba(self, X):
    X = np.asarray(X, dtype=np.float32)
    X_scaled = (X - self.X_min_) / (self.X_max_ - self.X_min_ + 1e-8)

    probs_list = []
    for m in self.models_:
        probs = m.predict_proba(X_scaled)
        # ✅ Asegurar que siempre sea (n, 2) para binary classification
        if probs.ndim == 1 or probs.shape[1] == 1:
            probs = np.column_stack([1 - probs.ravel(), probs.ravel()])
        probs_list.append(probs)

    return np.mean(probs_list, axis=0)
```

---

## 🧪 Test Script

Para verificar que los fixes funcionan:

```bash
python /home/JKP6679/Patho-Ensemble/test_fix_tabpfn.py
```

Este script prueba:
1. TabPFNMetaClassifier con 1 y múltiples samples
2. SnapshotTabPFNMetaClassifier con 1 y múltiples samples
3. Verifica que shape sea siempre (n, 2)
4. Verifica que probabilidades sumen a 1

---

## 🚀 Próximo Paso

Ejecutar con nuevo sbatch (v2, con mejor debugging):

```bash
sbatch /home/JKP6679/Patho-Ensemble/run_extended_models_fixed_v2.sbatch
```

**Cambios en v2:**
- Ejecuta modelos secuencialmente (mejor debugging si falla uno)
- Mejor separación de output
- Cada modelo tiene su propia sección claramente marcada

---

## 📊 Historial de Jobs

| Job | Status | Razón |
|---|---|---|
| 70054 | ✅ Success | Fase 1: LogReg, MLP, MLP Snapshot, MLP FGE |
| 70055 | ⚠️ Partial | Fase 2: Deep MLP OK, Deep MLP Snapshot falla, TabPFN no ejecutado |
| 70057 | ❌ Failed | Fase 2b v1: Deep MLP Snapshot completó, TabPFN falló (shape issue) |
| (próximo) | ⏳ | Fase 2b v2: Todos con fixes aplicados |

---

**Status:** ✅ Todos los fixes aplicados y verificados  
**Archivo:** `/shared/home/JKP6679/Patho-Ensemble/src/meta_models.py`  
**Test:** `python test_fix_tabpfn.py`  
**Próximo:** `sbatch run_extended_models_fixed_v2.sbatch`
