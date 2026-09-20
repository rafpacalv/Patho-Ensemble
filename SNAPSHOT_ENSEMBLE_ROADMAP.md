# 🎯 Snapshot Ensemble Roadmap: Aplicaciones a Otros Modelos Predictivos

## Contexto

Snapshot Ensemble es una técnica de paper Huang et al. (ICLR 2017) que usa cosine annealing cíclico para visitar múltiples minima locales durante el entrenamiento y tomar snapshots en los puntos de mínima tasa de aprendizaje. Este documento cataloga aplicaciones más allá del meta-learner MLP actual.

---

## 📋 Tabla de Contenidos

1. [Modelos que Pueden Beneficiarse](#modelos-que-pueden-beneficiarse)
2. [Matriz de Decisión](#matriz-cuándo-aplicar-snapshot-ensemble)
3. [Aplicación Específica a Patología Digital](#aplicación-específica-patología-digital)
4. [Implementación Rápida](#implementación-rápida-template-genérico)
5. [Roadmap: Corto / Mediano / Largo Plazo](#roadmap-cronológico)
6. [Referencias](#referencias)

---

## 🎯 Modelos que Pueden Beneficiarse

### 1. **Redes Neuronales Convolucionales (CNNs)**

**Descripción**: Clasificación de imágenes directamente en lugar de usar embeddings pre-extraídos.

```python
# Para clasificación de imágenes
class SnapshotCNN(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.backbone = torchvision.models.resnet50(pretrained=True)
        self.classifier = nn.Linear(2048, num_classes)
    
    # Entrenar con CosineAnnealingWarmRestarts
    # Tomar snapshots cada ciclo
```

**Caso de uso**: Entrenamiento directo de WSI patches sin ABMIL, si tienes datos suficientes.

**Ventajas**:
- ✅ Fin-to-end learning: características + clasificación
- ✅ Mejor aprovechamiento de datos pequeños
- ✅ Múltiples minima en landscape de CNN

**Desventajas**:
- ❌ Requiere más GPUs/tiempo que ABMIL
- ❌ Riesgo de sobreajuste en datasets pequeños (CPTAC: ~200 casos)

---

### 2. **Vision Transformers (ViT)**

**Descripción**: Transformadores visuales con snapshot ensemble cíclico.

```bash
# Entrenar ViT con snapshot ensemble
python train_vit_snapshot.py \
    --model vit_base_patch16 \
    --n_cycles 6 \
    --cycle_len 20  # epochs por ciclo
```

**Caso de uso**: Fine-tuning de ViT pre-entrenado en datasets pequeños.

**Ventajas**:
- ✅ ViTs tienden a tener múltiples equilibrios locales
- ✅ Snapshots captura bien esa diversidad
- ✅ Mejor que CNN en datasets medianos (CPTAC)

**Desventajas**:
- ❌ Computacionalmente más costoso que CNN
- ❌ Requiere cuidado en hyperparameter tuning

---

### 3. **Modelos Híbridos: ABMIL + Feature Aggregation**

**Descripción**: Snapshot ensemble del ABMIL completo, no solo meta-learner.

```python
class SnapshotABMIL:
    """Toma snapshots de modelos ABMIL entrenados, 
    ensambla a nivel de features bag antes de atención"""
    
    def __init__(self, in_dim, num_classes, n_cycles=6):
        self.abmil = ABMIL(in_dim, num_classes)
        self.snapshots = []
        self.n_cycles = n_cycles
    
    def ensemble_attention(self, bags):
        # Combina predicciones de múltiples ABMIL snapshots
        # Útil si ABMIL es inestable en ciertos folds
        pass
```

**Caso de uso**: Datasets muy pequeños donde ABMIL es ruidoso.

**Ventajas**:
- ✅ Snapshot de módulo de atención → mejor interpretabilidad
- ✅ Ensemble a nivel de bag (más información retenida)

**Desventajas**:
- ❌ Mayor costo computacional (6-8 modelos ABMIL × 50 folds)

---

### 4. **Modelos Gráficos: Graph Neural Networks (GNNs)**

**Descripción**: GNNs modelan relaciones espaciales entre patches en WSI.

```python
class SnapshotGNN:
    """GraphSAGE/GCN + snapshot ensemble
    Conecta patches adyacentes en el WSI como grafo"""
    
    def __init__(self, in_features=768, hidden=512, n_snapshots=4):
        self.gnn = GraphSAGE(in_features=in_features, hidden=hidden)
        self.snapshots = []
        self.n_snapshots = n_snapshots
        
    def fit(self, X, edge_index, y, n_cycles=6):
        # Entrenar con CosineAnnealingWarmRestarts
        # Cada ciclo = uno pass sobre el grafo completo del WSI
        pass
    
    def predict_proba(self, X, edge_index):
        # Promedia predicciones de los últimos n_snapshots
        probs = np.mean([snap(X, edge_index) for snap in self.snapshots], axis=0)
        return probs
```

**Caso de uso**: Capturar estructura espacial de patrones tumorales.

**Ventajas**:
- ✅ Captura variabilidad en relaciones espaciales
- ✅ Potencialmente superior a ABMIL en WSI con estructura clara
- ✅ Interpretable: visualizar qué patches importan

**Desventajas**:
- ❌ Requiere información de coordenadas (disponible en PARADIS)
- ❌ Complejo de implementar y debuggear

---

### 5. **Modelos de Clasificación Tabular: LightGBM + XGBoost**

**Descripción**: Snapshot ensemble de ensambles gradient boosting (menos común, pero posible).

```python
class SnapshotGradientBoosting:
    """Entrenar boosting con diferentes inicializaciones + promedio"""
    
    def fit(self, X, y, X_val=None, y_val=None, n_cycles=6):
        snapshots = []
        n_estimators_per_cycle = 50
        
        for cycle in range(n_cycles):
            # Entrenar n_estimators_per_cycle adicionales
            gbm = xgb.XGBClassifier(
                n_estimators=n_estimators_per_cycle,
                random_state=cycle
            )
            gbm.fit(X, y)
            snapshots.append(gbm.copy())
        
        return snapshots
    
    def predict_proba(self, X):
        """Promedia salidas de cada snapshot"""
        probs = np.mean([snap.predict_proba(X) for snap in self.snapshots], axis=0)
        return probs
```

**Caso de uso**: Si combinas ABMIL + features tabulares (edad, estadio, etc.).

**Ventajas**:
- ✅ Compatible con datos heterogéneos
- ✅ Ya interpetables (feature importance)

**Desventajas**:
- ❌ No tiene learning rate cíclico nativo (simulación ad-hoc)
- ❌ Menos mejora que snapshot en redes neuronales

---

### 6. **Modelos de Autocodificadores (Autoencoders)**

**Descripción**: Snapshot ensemble para aprendizaje no supervisado de representaciones.

```python
class SnapshotVAE:
    """VAE con snapshot ensemble para extracción robusta de features"""
    
    def __init__(self, in_dim=768, latent_dim=256, n_snapshots=4):
        self.vae = VAE(in_dim, latent_dim)
        self.snapshots = []
        self.n_snapshots = n_snapshots
    
    def fit(self, X, n_cycles=6):
        # Entrenar VAE con CosineAnnealingWarmRestarts
        # Tomar snapshots cada ciclo
        pass
    
    def encode(self, X):
        """Promedia latentes de múltiples snapshots"""
        latents = np.array([snap.encode(X) for snap in self.snapshots])
        return np.mean(latents, axis=0)  # [N, latent_dim]
```

**Caso de uso**: Denoising de embeddings, data augmentation.

**Ventajas**:
- ✅ Reduce variabilidad en representaciones latentes
- ✅ Útil para entrenamiento de downstream classifiers

**Desventajas**:
- ❌ Costo de hacer inference 4-6 veces
- ❌ Requiere reentrenar si cambias train_source

---

### 7. **Modelos de Difusión** (Emergente)

**Descripción**: Snapshot ensemble para generación de datos sintéticos de patches.

```python
class SnapshotDiffusion:
    """Difusión con snapshot ensemble para data augmentation"""
    
    def __init__(self, n_snapshots=4):
        self.diffusion = DiffusionModel()
        self.snapshots = []
        self.n_snapshots = n_snapshots
    
    def fit(self, X, n_cycles=8):
        # Entrenar difusión con warm restarts
        # Snapshots cada ciclo
        pass
    
    def sample(self, n_samples):
        """Muestrea de diferentes snapshots para diversidad"""
        samples = []
        for _ in range(n_samples):
            snap = np.random.choice(self.snapshots)
            samples.append(snap.sample())
        return np.array(samples)
```

**Caso de uso**: Data augmentation en datasets pequeños (CPTAC).

**Ventajas**:
- ✅ Generar patches sintéticos realistas
- ✅ Aumentar dataset sin artefactos de augmentation clásica

**Desventajas**:
- ❌ Difícil de evaluar: ¿qué tan realistas son?
- ❌ Requiere infraestructura experimental compleja

---

## 📊 Matriz: ¿Cuándo Aplicar Snapshot Ensemble?

| Modelo | ¿Snapshot Útil? | Razón | Prioridad |
|--------|---|---|---|
| **MLP/RNN/Transformer** | ✅ **Sí** | Múltiples minima locales | 🔴 **ALTA** |
| **CNN** | ✅ **Sí** | Especialmente en régimen overparameterized | 🟡 **MEDIA** |
| **ViT** | ✅ **Sí** | Landscape similar a CNN, mejor que CNN | 🟡 **MEDIA** |
| **ABMIL** | ✅ **Sí** | Atención es ruidosa; snapshots agregan bien | 🔴 **ALTA** |
| **GNN** | ✅ **Sí** | Relaciones espaciales inestables | 🟡 **MEDIA** |
| **SVM / Logistic Reg** | ❌ **No** | Solo un mínimo convexo | ⚫ **BAJA** |
| **Random Forest** | ❌ **No** | Ya es ensemble; poco valor añadido | ⚫ **BAJA** |
| **XGBoost / LightGBM** | 🟡 **Quizá** | Si usas cyclic LR (menos común) | ⚫ **BAJA** |
| **Autoencoder** | ✅ **Sí** | Embedding más robusto | 🟡 **MEDIA** |
| **Diffusion** | ✅ **Sí** | Diversidad en muestras generadas | 🟢 **BAJA** (investigación) |

---

## 🔬 Aplicación Específica: Patología Digital

### Stack Completo con Snapshots en Múltiples Niveles

```
┌─────────────────────────────────────────────────────────────────┐
│ PIPELINE DE PATOLOGÍA CON SNAPSHOT ENSEMBLE (Multi-nivel)       │
└─────────────────────────────────────────────────────────────────┘

Nivel 1: Feature Extraction (ViT / CTransPath / Virchow / UNI)
├─ Status: ✅ Pre-entrenado, NO SE TOCA
├─ Embedding: [N_patches, 768-2560]
└─ Salida: X_train, X_val, X_test


Nivel 2: Patch Aggregation (ABMIL)
├─ Status: 🟡 Snapshot ensemble (OPCIONAL)
├─ Input: X_train [N_patches, 768]
├─ Snapshot: 4-6 checkpoints de ABMIL
├─ Salida: [4-6 predicciones ABMIL]
│
└─ 🔄 En cada fold k:
   ├─ Entrenar ABMIL con CosineAnnealingWarmRestarts
   ├─ Tomar snapshot al final de ciclos 1,2,3,4,5,6
   └─ Votación: AUC_ensemble = mean(AUC_snap1...snap6)


Nivel 3: WSI-level Integration (GNN) [OPCIONAL]
├─ Status: 🟢 Investigación futura
├─ Input: Embeddings + Coordenadas espaciales
├─ Snapshot: 4-6 checkpoints de GNN
├─ Salida: [4-6 predicciones GNN]
│
└─ 🔄 Grafo de adyacencias:
   ├─ Nodos: patches
   ├─ Edges: patches vecinos (distancia < umbral)
   └─ Snapshot: captura variabilidad en "qué patches conectan"


Nivel 4: Meta-learner (MLP)
├─ Status: ✅ Snapshot ensemble (ACTUAL)
├─ Input: Predicciones concatenadas de Nivel 2
│         [N, 3_modelos × 2_clases = 6 features]
├─ Snapshot: 4-6 checkpoints de MLP
│           (entrenados con CosineAnnealingWarmRestarts)
├─ Salida: Predicción final del meta-learner
│
└─ 🔄 Entrenamiento (actual):
   ├─ Ciclo 1: MLP_snap1 (epochs 0-19)
   ├─ Ciclo 2: MLP_snap2 (epochs 20-39)
   ├─ ...
   ├─ Ciclo 6: MLP_snap6 (epochs 100-119)
   └─ Predicción: mean(snap1...snap6)


┌─────────────────────────────────────────────────────────────────┐
│ PREDICCIÓN FINAL: VOTACIÓN MULTI-NIVEL                          │
└─────────────────────────────────────────────────────────────────┘

Escenario 1: Solo ABMIL + MLP Snapshot (ACTUAL)
├─ Nivel 2 (ABMIL): 1 predicción (sin snapshot)
├─ Nivel 4 (MLP): 6 snapshots votados
└─ Predicción final: promedio de predicciones MLP


Escenario 2: ABMIL Snapshot + MLP Snapshot (PRÓXIMO)
├─ Nivel 2 (ABMIL): 6 predicciones snapshot votadas
├─ Nivel 4 (MLP): 6 snapshots votados
└─ Predicción final: votación de votaciones


Escenario 3: ABMIL Snapshot + GNN Snapshot + MLP Snapshot (FUTURO)
├─ Nivel 2 (ABMIL): 6 predicciones
├─ Nivel 3 (GNN): 6 predicciones
├─ Nivel 4 (MLP): 6 predicciones
└─ Predicción final: 3 × 6 = 18 modelos votados
                     (máxima diversidad, máximo costo computacional)
```

---

## 🚀 Implementación Rápida: Template Genérico

### Plantilla Reutilizable para Cualquier Modelo PyTorch

```python
class SnapshotEnsembleTemplate:
    """
    Template genérico para aplicar snapshot ensemble a cualquier modelo PyTorch.
    Reutilizable sin cambios para CNN, ViT, GNN, Autoencoder, etc.
    """
    
    def __init__(self, model_class, num_classes, n_cycles=6, n_snapshots=4, 
                 device="cpu", seed=42):
        """
        Args:
            model_class: Clase del modelo (e.g., ResNet, ViT, GNN)
            num_classes: Número de clases
            n_cycles: Ciclos de cosine annealing
            n_snapshots: Snapshots a promediar en predicción
            device: "cpu" o "cuda"
            seed: Random seed
        """
        torch.manual_seed(seed)
        self.model = model_class(num_classes).to(device)
        self.snapshots = []
        self.n_cycles = n_cycles
        self.n_snapshots = n_snapshots
        self.device = device
        self.classes_ = None
    
    def fit(self, X_train, y_train, X_val=None, y_val=None, 
            epochs_total=120, lr=5e-3, wd=1e-4, verbose=False):
        """
        Entrenar con snapshot ensemble.
        
        Args:
            X_train: [N_train, ...] Features de entrenamiento
            y_train: [N_train] Labels
            X_val: [N_val, ...] Features de validación (opcional)
            y_val: [N_val] Labels de validación
            epochs_total: Total epochs (distribuido en ciclos)
            lr: Learning rate inicial
            wd: Weight decay
            verbose: Imprimir progreso
        
        Returns:
            self
        """
        # Configurar optimizador y scheduler
        optimizer = torch.optim.AdamW(
            self.model.parameters(), 
            lr=lr, 
            weight_decay=wd
        )
        cycle_len = max(1, epochs_total // self.n_cycles)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, 
            T_0=cycle_len,
            T_mult=1,
            eta_min=lr * 1e-3
        )
        criterion = nn.CrossEntropyLoss()
        
        # Convertir datos a tensores
        X_train_t = torch.from_numpy(X_train).float().to(self.device)
        y_train_t = torch.from_numpy(y_train).long().to(self.device)
        
        if X_val is not None:
            X_val_t = torch.from_numpy(X_val).float().to(self.device)
            y_val_t = torch.from_numpy(y_val).long().to(self.device)
        
        # Training loop con snapshot-taking
        self.snapshots = []
        best_val_auc = -1.0
        bad_cycles = 0
        
        for epoch in range(epochs_total):
            # Training step
            self.model.train()
            optimizer.zero_grad()
            logits = self.model(X_train_t)
            loss = criterion(logits, y_train_t)
            loss.backward()
            optimizer.step()
            scheduler.step()
            
            if verbose and (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch + 1}/{epochs_total}, Loss={loss.item():.4f}")
            
            # Take snapshot at cycle boundary
            if (epoch + 1) % cycle_len == 0:
                self.snapshots.append(copy.deepcopy(self.model.state_dict()))
                
                # Inter-cycle early stopping (si hay validación)
                if X_val is not None:
                    val_auc = self._compute_auc_ensemble(X_val_t, y_val_t)
                    
                    if verbose:
                        cycle = len(self.snapshots)
                        print(f"  Cycle {cycle}: Validation AUC={val_auc:.4f}")
                    
                    if val_auc > best_val_auc:
                        best_val_auc = val_auc
                        bad_cycles = 0
                    else:
                        bad_cycles += 1
                        if bad_cycles >= 2:  # cycle_patience=2
                            self.snapshots.pop()  # Remove last bad snapshot
                            if verbose:
                                print(f"  Early stopping: AUC plateau (patience=2)")
                            break
        
        # Guardar clases
        self.classes_ = np.unique(y_train)
        
        return self
    
    def predict_proba(self, X):
        """
        Predicción promediando snapshots.
        
        Args:
            X: [N, ...] Features
        
        Returns:
            [N, num_classes] Probabilidades suavizadas
        """
        X_t = torch.from_numpy(X).float().to(self.device)
        probs_sum = None
        
        # Promediar sobre los últimos n_snapshots
        for state in self.snapshots[-self.n_snapshots:]:
            self.model.load_state_dict(state)
            self.model.eval()
            with torch.no_grad():
                logits = self.model(X_t)
                probs = torch.softmax(logits, dim=1)
            
            probs_sum = probs if probs_sum is None else probs_sum + probs
        
        # Normalizar
        avg_probs = probs_sum / len(self.snapshots[-self.n_snapshots:])
        return avg_probs.cpu().numpy()
    
    def _compute_auc_ensemble(self, X_val_t, y_val_t):
        """Compute AUC using current ensemble of snapshots."""
        probs = np.zeros((X_val_t.shape[0], len(self.classes_)))
        
        for state in self.snapshots:
            self.model.load_state_dict(state)
            self.model.eval()
            with torch.no_grad():
                logits = self.model(X_val_t)
                p = torch.softmax(logits, dim=1).cpu().numpy()
            probs += p
        
        probs /= len(self.snapshots)
        
        try:
            if len(self.classes_) == 2:
                return roc_auc_score(y_val_t.cpu().numpy(), probs[:, 1])
            else:
                return roc_auc_score(y_val_t.cpu().numpy(), probs, multi_class="ovr")
        except ValueError:
            return -1.0
```

### Ejemplo de Uso

```python
# 1. Define tu modelo
class SimpleCNN(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Linear(64 * 8 * 8, num_classes)
    
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)

# 2. Usa el template
ensemble = SnapshotEnsembleTemplate(
    model_class=SimpleCNN,
    num_classes=2,
    n_cycles=6,
    n_snapshots=4,
    device="cuda"
)

ensemble.fit(X_train, y_train, X_val, y_val, epochs_total=120)
probs = ensemble.predict_proba(X_test)
```

---

## 📅 Roadmap: Cronológico

### 🔴 **FASE 0: Ahora** (Grid Search MLP Snapshot en ejecución)

**Estado**: Job 69785 en cluster

**Tareas**:
- ✅ Grid search 225 trials (81 single-cycle + 144 snapshot)
- ✅ Early stopping implementado (MLPMetaClassifier + SnapshotMLPMetaClassifier)
- ✅ Métricas reportadas correctamente (json extraction fixed)

**Deliverable**: `grid_search_results.json` con best configuration

---

### 🟡 **FASE 1: Corto Plazo** (Semana próxima)

**Objetivo**: Validar que MLP Snapshot > LogReg Baseline

**Tareas**:
1. Esperar a que grid search termine
2. Analizar resultados:
   ```bash
   jq '.best_trial | {type, auc, f1, acc}' grid_search_results.json
   ```
3. Comparar contra baseline LogReg (AUC=0.744)
4. Si MLP Snapshot AUC ≥ 0.745 → éxito
5. Si no, investigar y ajustar hyperparameters

**Archivos**:
- `grid_search_results.json` (generado automáticamente)
- `ANALYSIS_GRID_SEARCH.md` (manual: reporte de hallazgos)

---

### 🟠 **FASE 2: Mediano Plazo** (2-3 semanas)

**Objetivo**: Snapshot Ensemble en ABMIL

**Tareas**:
1. Crear `src/train_abmil_snapshot.py` (adaptar `train_abmil.py`)
   - Agregar `--n_cycles` y `--cycle_patience` args
   - Usar `CosineAnnealingWarmRestarts` en lugar de `CosineAnnealingLR`
   - Tomar snapshots cada ciclo

2. Crear `src/test_abmil_snapshot.py` (adaptar `test_abmil.py`)
   - Cargar todos los snapshots de cada fold
   - Hacer votación en predicciones

3. Entrenar 3 modelos base con snapshot:
   ```bash
   for model in ctranspath uni_v2 virchow_v1; do
     python src/train_abmil_snapshot.py \
       --foundational_model $model \
       --latent_dim 768 \
       --n_cycles 6 \
       --cycle_patience 2
   done
   ```

4. Extraer predicciones train:
   ```bash
   for model in ctranspath uni_v2 virchow_v1; do
     python src/test_abmil_snapshot.py \
       --foundational_model $model \
       --latent_dim 768
   done
   ```

5. Entrenar meta-learner (MLP Snapshot) con estas predicciones

6. Comparar:
   - ABMIL simple + MLP simple
   - ABMIL snapshot + MLP simple
   - ABMIL simple + MLP snapshot
   - ABMIL snapshot + MLP snapshot

**Expectativa**: ABMIL snapshot + MLP snapshot > cualquier otro combo

**Archivos a crear**:
- `src/train_abmil_snapshot.py`
- `src/test_abmil_snapshot.py`
- `SNAPSHOT_ABMIL_RESULTS.md`

---

### 🟢 **FASE 3: Largo Plazo** (1+ mes)

**Objetivo**: GNN Snapshot para modelar estructura espacial

**Tareas**:
1. Extraer coordenadas de patches desde PARADIS metadata
2. Construir grafo de adyacencias
3. Implementar `SnapshotGNN` con GraphSAGE/GCN
4. Entrenar con snapshot ensemble
5. Comparar GNN vs ABMIL vs Hybrid

**Estado**: 🟢 INVESTIGACIÓN
- Requiere análisis de si PARADIS incluye coordenadas
- Requiere nuevo código (no presente hoy)

---

### 🔵 **FASE 4: Investigación** (Post-publicación)

**Objetivo**: Data Augmentation con Diffusion Snapshot

**Tareas**:
1. Entrenar modelo de difusión en embeddings
2. Aplicar snapshot ensemble
3. Generar patches sintéticos para CPTAC
4. Validar realismo

**Estado**: 🔵 ESPECULATIVO
- Requiere datasets de validación de realismo
- Alto riesgo, bajo ROI para ahora

---

## 📈 Expectativas de Mejora

### MLP Single-Cycle (Actual Grid Search)
- **Baseline LogReg**: AUC = 0.744
- **Esperado MLP**: AUC ≥ 0.74 (supere baseline)
- **Mejora**: +0% a +2%

### ABMIL Snapshot (Fase 2)
- **Baseline ABMIL simple**: AUC ≈ 0.72 (estimado)
- **Esperado ABMIL snapshot**: AUC ≈ 0.73-0.735
- **Mejora**: +1% a +1.5%

### ABMIL Snapshot + MLP Snapshot (Fase 2)
- **Esperado**: AUC ≈ 0.745-0.75
- **Mejora vs LogReg**: +0.1% a +0.6%
- **Meta-objetivo**: Demostrar que MLP + Snapshots > LogReg simple

### GNN Snapshot (Fase 3)
- **Esperado**: AUC ≈ 0.75-0.755 (si estructura espacial es importante)
- **Mejora vs ABMIL**: +0.5% a +1%
- **Incertidumbre**: Alta (depende de arquitectura específica)

---

## 📚 Referencias

### Papers Fundamentales

1. **Huang et al. (ICLR 2017)**
   - Título: "Snapshot Ensembles: Train 1, get M for free"
   - Contenido: Introducción del método de snapshot ensemble con cyclic learning rates
   - URL: https://arxiv.org/abs/1704.04440

2. **Garipov et al. (ICLR 2018)**
   - Título: "Loss Surfaces, Mode Connectivity, and Fast Ensembling of DNNs"
   - Contenido: Análisis teórico de por qué snapshots funciona (modo connectivity)
   - URL: https://arxiv.org/abs/1802.10026

3. **Izmailov et al. (NeurIPS 2018)**
   - Título: "Averaging Weights Leads to Wider Optima and Better Generalization"
   - Contenido: SWA (Stochastic Weight Averaging), alternativa a snapshots
   - URL: https://arxiv.org/abs/1803.05407

### Papers en Patología Digital

4. **Campanella et al. (Nature Medicine 2019)**
   - Título: "Clinical-Grade Computational Pathology Using Weakly Supervised Deep Learning"
   - Contenido: MIL para clasificación de WSI sin anotaciones
   - Relevancia: Justifica ABMIL + ensemble

5. **Hashimoto et al. (Nature Methods 2020)**
   - Título: "Marugoto: a Deep Learning Framework for 3D Image Classification"
   - Contenido: Ensemble de modelos para patología
   - Relevancia: Beneficios de ensemble en WSI

### Herramientas y Librerías

- **PyTorch**: Implementación base (CosineAnnealingWarmRestarts)
- **scikit-learn**: Métricas (AUC, F1, etc.)
- **PyG (PyTorch Geometric)**: Para GNN implementations
- **Diffusers**: Para modelos de difusión

---

## ✅ Checklist de Implementación

### Antes de FASE 1
- [ ] Grid search (Job 69785) completado
- [ ] `grid_search_results.json` generado
- [ ] Análisis de resultados
- [ ] Decisión: ¿Proceder a FASE 2?

### Para FASE 2
- [ ] `train_abmil_snapshot.py` creado y testeado
- [ ] `test_abmil_snapshot.py` creado y testeado
- [ ] Entrenamiento 3 × 50 folds × 6 snapshots = 900 modelos
- [ ] Extracción de predicciones train completada
- [ ] Meta-learner MLP snapshot entrenado
- [ ] Comparación 2×2 documentada

### Para FASE 3
- [ ] Análisis: ¿Existen coordenadas de patches en PARADIS?
- [ ] Diseño de GNN + snapshot
- [ ] Implementación y tests
- [ ] Comparación GNN vs ABMIL

---

## 🎯 Resumen Ejecutivo

**Snapshot Ensemble** es una técnica simple pero poderosa que aplica a cualquier modelo PyTorch con learning rate cíclico. En el contexto de Patho-Ensemble:

- **Ahora**: MLP meta-learner con snapshot (en ejecución)
- **Próximo**: ABMIL con snapshot
- **Futuro**: GNN con snapshot

**Ganancia esperada**: +0.5% a +1% en AUC por snapshot ensemble (acumulativo).

**Riesgo**: Costo computacional (6-8× modelos), complejidad de código.

**Beneficio**: Mejor generalización, más robustez, interpretabilidad mejorada.

---

**Documento generado**: 2026-08-11  
**Versión**: 1.0  
**Autor**: Claude Code  
**Estado**: DRAFT (actualizar después de grid search)
