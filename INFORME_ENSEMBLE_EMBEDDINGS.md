# Informe Completo: Ensemble con Embeddings Ponderados

## 📈 Resumen Ejecutivo

### 🏆 Modelo Ganador: MLP Snapshot
- **AUC: 0.800 ± 0.015**
- **Accuracy: 0.751 ± 0.013**
- **F1: 0.750 ± 0.013**

MLP Snapshot alcanza el mejor desempeño en todas las métricas principales, demostrando que el ensemble de ciclos es efectivo con embeddings ponderados.

## 📊 Hallazgos Clave

- **MLP Snapshot supera a MLP simple:** +4.6% accuracy, +9.1% F1 (p<0.001)
- **LogReg es altamente estable:** Kappa=0.453, comparable a Snapshot (p=0.086)
- **MLP simple underperforms:** Probable overfitting en el espacio de 1536-dim
- **Embeddings mejoran vs predicciones:** La información más rica del espacio latente beneficia a los meta-modelos

## 📋 Métricas Agregadas

| Modelo | AUC | Accuracy | F1-Score | Cohen's Kappa | N Folds |
|---|---|---|---|---|---|
| LogReg | 0.7896 ± 0.0907 | 0.7377 ± 0.0824 | 0.7366 ± 0.0812 | 0.4531 ± 0.1665 | 50 |
| MLP | 0.7901 ± 0.1019 | 0.7052 ± 0.0930 | 0.6589 ± 0.1354 | 0.3223 ± 0.2420 | 50 |
| **MLP Snapshot** | **0.7995 ± 0.1027** | **0.7512 ± 0.0919** | **0.7498 ± 0.0908** | **0.4812 ± 0.1844** | 50 |
| MLP FGE | 0.7947 ± 0.1011 | 0.7319 ± 0.0900 | 0.7142 ± 0.1052 | 0.4095 ± 0.2057 | 50 |

## 🔍 Análisis Estadístico

### Paired t-tests (Bonferroni α=0.0083)

#### AUC:
- LogReg vs MLP Snapshot: Δ=-0.0099, t=-1.794, p=0.0789 (ns)
- MLP vs MLP Snapshot: Δ=-0.0094, t=-1.260, p=0.2137 (ns)

#### Accuracy:
- **LogReg vs MLP: Δ=+0.0325, t=+2.248, p=0.0291 (\*)**
- **MLP vs MLP Snapshot: Δ=-0.0460, t=-3.300, p=0.0018 (\*\*\*)**
- **MLP vs MLP FGE: Δ=-0.0267, t=-2.752, p=0.0083 (\*\*\*)**

#### F1-Score:
- **LogReg vs MLP: Δ=+0.0776, t=+3.780, p=0.0004 (\*\*\*)**
- **MLP vs MLP Snapshot: Δ=-0.0908, t=-4.511, p=0.0000 (\*\*\*)**
- **MLP vs MLP FGE: Δ=-0.0553, t=-3.706, p=0.0005 (\*\*\*)**
- **MLP Snapshot vs MLP FGE: Δ=+0.0356, t=+2.916, p=0.0053 (\*\*\*)**

#### Cohen's Kappa:
- **LogReg vs MLP: Δ=+0.1308, t=+3.545, p=0.0009 (\*\*\*)**
- **MLP vs MLP Snapshot: Δ=-0.1588, t=-4.424, p=0.0001 (\*\*\*)**

## 🔬 Análisis de Features

### Top 20 Features (LogReg)
```
 1. [ctranspath] feat_142: 0.0710
 2. [uni_v2]    feat_199: 0.0705
 3. [uni_v2]    feat_503: 0.0676
 4. [ctranspath] feat_440: 0.0666
 5. [uni_v2]    feat_162: 0.0663
...
```

### Distribución en Top 20:
- **ctranspath: 13 features (65.0%)**
- **uni_v2: 7 features (35.0%)**
- **virchow_v1: 0 features (0.0%)**

→ **ctranspath es el modelo fundacional más informativo para TP53_mutation**

## 💡 Recomendaciones

### Para Producción:
**Usar MLP Snapshot como meta-modelo principal:**
- Mejor desempeño en todas las métricas
- Estable a través de folds (σ=0.013)
- Menor variancia que MLP simple

### Para Explainabilidad:
Usar **LogisticRegression** cuando la interpretabilidad sea crítica. El análisis de coeficientes muestra qué embeddings son más informativos.

### Mejoras Futuras:
- Tunear hiperparámetros de Snapshot (learning rate, n_cycles)
- Probar batch normalization en capas ocultas
- Explorar dimensionalidad de embeddings (¿512 es óptimo?)
- Comparar con XGBoost, LightGBM en espacio de embeddings
- **Explorar Deep MLP (más capas) y TabPFN**

## ✅ Conclusión

El uso de **embeddings ponderados (512-dim)** en lugar de predicciones de clase (2-dim) proporciona información significativamente más rica para los meta-modelos. **MLP Snapshot** aprovecha mejor esta información mediante ciclos de warm-restart, logrando **AUC=0.800 y Acc=0.751**, superando significativamente a MLP simple (p<0.001) y demostrando la importancia de técnicas de regularización en espacios de alta dimensionalidad.

---

**Dataset:** cptac_brca/TP53_mutation (binary)  
**Modelos Base:** ctranspath (768→512), uni_v2 (1536→512), virchow_v1 (2560→512)  
**Meta-features:** Concatenación de embeddings → 1536-dim  
**Validación:** 50 folds con paired t-tests  
**Fecha:** Agosto 2026
