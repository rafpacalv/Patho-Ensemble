# Opción A: Optimización de Pesos con Nelder-Mead

> **⚠️ CORRECCIÓN (2026-08-21)** — el baseline de referencia (0.7913/0.7918) es
> anterior a la regeneración de las predicciones base; el vigente para el mismo
> trío es **0.7616**. Más de fondo: el +0.0052 que consigue Nelder-Mead
> optimizando pesos queda muy por debajo del **+0.037** que da cambiar
> `virchow_v1` por `conch_v1_5` en el conjunto base. Ver
> [INFORME_SELECCION_MODELOS_BASE_20260821.md](INFORME_SELECCION_MODELOS_BASE_20260821.md).

**Fecha**: 21 de agosto de 2026  
**Objetivo**: Comparar pesos optimizados (Nelder-Mead) vs pesos fijos (LogReg) en late fusion  
**Configuración**: ctranspath, virchow_v1, conch_v1_5 (3 modelos, 50 folds)

---

## Motivación

En Opción C, vimos que early fusion NO mejora sobre late fusion con 3 modelos idénticos (Δ AUC = -0.0056, no significativo). 

Hipótesis alternativa: El problema no es el **mecanismo** (late vs early), sino los **pesos de combinación**. 
- **Opción C baseline**: LogReg aprende implícitamente pesos logísticos (acoplados a features del modelo)
- **Opción A propuesta**: Nelder-Mead optimiza explícitamente pesos de combinación lineal, sin asumir features

**Pregunta**: ¿Pesos óptimos vía Nelder-Mead superan LogReg en late fusion?

---

## Diseño Experimental

### 1. Estrategia de Optimización

**Función objetivo** (por fold):
```
minimize: -AUC_val(w_ct, w_vir, w_conch)
s.t.
  w_i ∈ [0, 1] para cada modelo
  Σw_i = 1 (normalización)
```

**Predicción combinada**:
```
P_ensemble = w_ct * P_ctranspath + w_vir * P_virchow + w_conch * P_conch
```

**Optimizador**: scipy.optimize.minimize con método `Nelder-Mead`
- Simplex de 3 vértices (uno por peso)
- No requiere gradientes (robusta a ruido numérico)
- Manejo de restricciones vía penalty (convertir a problema sin restricciones)

### 2. Flujo de Ejecución

**Fase 1: Entrenamiento (fold-by-fold)**
```
Para cada fold k = 0..49:
  1. Carga predicciones in-sample (train_eval) → P_train_ct, P_train_vir, P_train_conch
  2. Carga labels de train → y_train
  3. Ejecuta Nelder-Mead:
     - Predicción candidata: P_cand = w_ct*P_ct + ... (con w normalizado)
     - Calcula AUC en train_eval
     - Retorna -AUC (minimización)
  4. Guarda pesos óptimos w*_k
```

**Fase 2: Evaluación (fold-by-fold)**
```
Para cada fold k:
  1. Carga w*_k (pesos óptimos del fold k)
  2. Carga predicciones test → P_test_ct, P_test_vir, P_test_conch
  3. Carga labels test → y_test
  4. Calcula P_final = w*_k · P_test (combinación)
  5. Calcula AUC_test, guarda resultados
```

### 3. Comparación contra Baseline

**Baseline (Opción C)**: Late Fusion + LogReg
- AUC medio: 0.7913 (del log anterior)
- 21/50 folds mejoran, 26/50 empeoran vs early fusion

**Propuesta (Opción A)**: Late Fusion + Nelder-Mead  
- AUC medio: ¿?
- Hipótesis: Nelder-Mead puede ser más robusto en folds con desacuerdo entre modelos

### 4. Métricas de Éxito

Opción A es **exitosa** si:
- [ ] Mean AUC > 0.7913 (supera LogReg baseline)
- [ ] CI95% no contiene 0 en comparación pareada
- [ ] p-value < 0.05 (significancia)

Opción A es **inconcluyente** si:
- [ ] Mean AUC ≈ 0.7913 ± 0.01 (similar a LogReg)
- [ ] CI95% contiene 0
- [ ] p-value > 0.05

---

## Implementación

### Archivo Principal: `src/ensemble_nelder_mead.py`

```python
def optimize_fold_weights(P_train_dict, y_train, P_test_dict, y_test, 
                          model_names, fold_k):
    """
    Optimiza pesos per-modelo via Nelder-Mead en train_eval,
    evalúa en test, retorna (w_opt, auc_test, preds_test).
    
    P_train_dict: dict of (n_train,) arrays, predicciones por modelo
    y_train: (n_train,) labels
    P_test_dict: dict of (n_test,) arrays
    y_test: (n_test,) labels
    model_names: ['ctranspath', 'virchow_v1', 'conch_v1_5']
    """
    
    def objective(w_raw):
        # Aplicar softmax para garantizar [0,1] y Σ=1
        w = np.exp(w_raw) / np.exp(w_raw).sum()
        
        # Predicción combinada
        P_cand = np.zeros_like(y_train)
        for model, w_i in zip(model_names, w):
            P_cand += w_i * P_train_dict[model]
        
        # AUC en train_eval
        auc = roc_auc_score(y_train, P_cand)
        return -auc  # minimizar -AUC
    
    # Inicializar con pesos iguales
    w_init = np.log(np.array([1.0, 1.0, 1.0]) / 3.0)
    
    # Optimizar
    result = minimize(objective, w_init, method='Nelder-Mead',
                     options={'maxiter': 1000, 'xatol': 1e-4, 'fatol': 1e-4})
    
    w_opt = np.exp(result.x) / np.exp(result.x).sum()
    
    # Evaluar en test
    P_final = np.zeros_like(y_test)
    for model, w_i in zip(model_names, w_opt):
        P_final += w_i * P_test_dict[model]
    
    auc_test = roc_auc_score(y_test, P_final)
    
    return w_opt, auc_test, P_final
```

### CLI

```bash
python src/ensemble_nelder_mead.py \
    --work_dir /home/JKP6679/Patho-Ensemble/PARADIS/datos/patches \
    --train_source cptac_brca \
    --tissue_patching 20x_224px_0px_overlap \
    --task_name TP53_mutation \
    --foundational_models ctranspath virchow_v1 conch_v1_5 \
    --fold_start 0 \
    --fold_end 49
```

---

## Resultados Esperados

- **Pesos medios** por fold (descubrimiento)
- **AUC por fold** (50 folds)
- **Comparación pareada** vs LogReg baseline
- **Fold-by-fold breakdown** (dónde Nelder-Mead gana/pierde)

---

## Timeline

- **Piloto (5 folds)**: Verificar convergencia, pesos razonables
- **Escala completa (50 folds)**: ~2-3 horas
- **Comparación pareada + reporte**: ~30 min

**Estimado total**: ~4-5 horas

---

## Limitaciones

1. **Overfitting local**: Nelder-Mead optimiza sobre train_eval (mismo datos que LogReg entrena). La comparación es a igualdad de "conocimiento" del fold.
2. **Inicialización**: Usando pesos iguales; si converge a mínimo local subóptimo, podría penalizar el método.
3. **Hipótesis de linealidad**: Asume que combinación lineal ponderada es suficiente (vs LogReg que aprende no-linealidad).

Mitiga: Robustecer con múltiples inicializaciones aleatorias en el piloto.

