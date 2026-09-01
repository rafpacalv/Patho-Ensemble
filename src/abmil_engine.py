"""
Motor ABMIL autocontenido — adaptación de monai_qupath/src/handoff/msi_pipeline.py
Solo la funcionalidad core necesaria para train + predict:
  - GatedAttention + ABMIL (gated attention, proj_dim=512)
  - train_abmil: entrena con early-stopping basado en validación
  - abmil_predict: predice sobre stems
  - _grouped_val_split: genera split train/val estratificado y agrupado por case_id
"""
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import roc_auc_score


# Receta de entrenamiento: el usuario honra --epochs (no lo sobreescribimos con 40 fijo)
_RECIPES = {
    "ours": dict(lr=2e-4, wd=1e-4, epochs=40, patience=8, smooth=0.1),
    "titan": dict(lr=1e-4, wd=1e-5, epochs=20, patience=99, smooth=0.0)
}
ABMIL_RECIPE = "ours"


# ───────────────────────── ABMIL (gated attention) ─────────────────────────
class GatedAttention(nn.Module):
    def __init__(self, in_dim, hidden_dim=256):
        super().__init__()
        self.V = nn.Linear(in_dim, hidden_dim, bias=False)
        self.U = nn.Linear(in_dim, hidden_dim, bias=False)
        self.w = nn.Linear(hidden_dim, 1, bias=False)

    def forward(self, h):
        # Gated attention: rama V (tanh) * rama U (sigmoid)
        a = torch.tanh(self.V(h)) * torch.sigmoid(self.U(h))
        return F.softmax(self.w(a), dim=0)  # (N,1), suma 1


class MultiHeadGatedAttention(nn.Module):
    """Multi-head gated attention con combinación mean-pool.

    Cada cabeza es una rama gated attention independiente:
        a_k = softmax(w_k(tanh(V_k(h)) ⊙ sigmoid(U_k(h))))

    La combinación es el promedio de pesos entre cabezas: a = mean([a_1, ..., a_n_heads])
    """
    def __init__(self, in_dim, hidden_dim=256, n_heads=4):
        super().__init__()
        self.n_heads = n_heads
        self.hidden_dim = hidden_dim

        # Parámetros por cabeza (compartidos)
        self.V = nn.Linear(in_dim, hidden_dim, bias=False)
        self.U = nn.Linear(in_dim, hidden_dim, bias=False)
        # Proyecciones de salida independientes por cabeza
        self.w_heads = nn.Linear(hidden_dim, n_heads, bias=False)

    def forward(self, h):
        # Gated attention: rama V (tanh) * rama U (sigmoid)
        a_gated = torch.tanh(self.V(h)) * torch.sigmoid(self.U(h))  # (N, hidden_dim)

        # Calcular logits para cada cabeza
        logits = self.w_heads(a_gated)  # (N, n_heads)

        # Softmax por cabeza (independiente)
        a_heads = F.softmax(logits, dim=0)  # (N, n_heads), suma=1 por cabeza

        # Promedio entre cabezas: (N, n_heads) → (N, 1)
        a = a_heads.mean(dim=1, keepdim=True)  # (N, 1)

        return a


class ABMIL_Base(nn.Module):
    """Clase base con lógica común: proyección, atención y agregación.

    Las subclases definen qué devuelven: logits (ABMIL) o embedding (ABMIL_EMBEDDING).
    """
    def __init__(self, in_dim, num_classes, hidden_dim=256, proj_dim=512, dropout=0.25,
                 multi_head_attention=False, n_attention_heads=4):
        super().__init__()
        # Proyección opcional para estabilizar el entrenamiento
        self.projection = (
            nn.Sequential(nn.Linear(in_dim, proj_dim), nn.LayerNorm(proj_dim), nn.GELU())
            if proj_dim > 0
            else nn.Identity()
        )
        self.ad = proj_dim if proj_dim > 0 else in_dim

        # Elegir arquitectura de atención
        if multi_head_attention:
            self.attention = MultiHeadGatedAttention(self.ad, hidden_dim, n_heads=n_attention_heads)
        else:
            self.attention = GatedAttention(self.ad, hidden_dim)

    def _get_aggregated_embedding(self, h):
        """Proyecta, aplica atención y devuelve embedding ponderado.

        Returns:
            z: embedding ponderado (1, ad)
            a: coeficientes de atención (N, 1)
        """
        h = self.projection(h)
        a = self.attention(h)
        z = (a * h).sum(0, keepdim=True)  # (1, ad)
        return z, a


class ABMIL(ABMIL_Base):
    """ABMIL original: devuelve logits de clasificación."""
    def __init__(self, in_dim, num_classes, hidden_dim=256, proj_dim=512, dropout=0.25,
                 multi_head_attention=False, n_attention_heads=4):
        super().__init__(in_dim, num_classes, hidden_dim, proj_dim, dropout,
                         multi_head_attention, n_attention_heads)
        self.classifier = nn.Sequential(nn.Dropout(dropout), nn.Linear(self.ad, num_classes))

    def forward(self, h):
        z, a = self._get_aggregated_embedding(h)
        return self.classifier(z).squeeze(0), a.squeeze(1)  # logits, atención


class ABMIL_EMBEDDING(ABMIL_Base):
    """ABMIL que devuelve embedding ponderado en lugar de logits.

    Útil para meta-learning: el embedding (dimensión auto=512 ó in_dim) preserva
    más información que la predicción de clase para entrenar el meta-clasificador.
    """
    def __init__(self, in_dim, num_classes, hidden_dim=256, proj_dim=512, dropout=0.25,
                 multi_head_attention=False, n_attention_heads=4):
        # num_classes se acepta pero no se usa (compatibilidad con factory)
        super().__init__(in_dim, num_classes, hidden_dim, proj_dim, dropout,
                         multi_head_attention, n_attention_heads)

    def forward(self, h):
        z, a = self._get_aggregated_embedding(h)
        return z.squeeze(0), a.squeeze(1)  # embedding ponderado (ad), atención

    def get_embedding_dim(self):
        """Devuelve la dimensión del embedding ponderado (útil para metadatos)."""
        return self.ad


# ───────────────────── entrenamiento con early-stopping ─────────────────────
def _grouped_val_split(groups, y, val_frac, seed):
    """Split train/val agrupado por `groups` (case_id) Y ESTRATIFICADO por clase.

    Devuelve (tri, vai) — arrays de índices para train/val.
    Si estratificación falla, fallback sin estratificar (para sklearn viejo o clases únicas).
    """
    groups = np.asarray(groups)
    y = np.asarray(y)
    n_splits = max(2, int(round(1.0 / max(val_frac, 1e-9))))

    try:
        sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        tri, vai = next(sgkf.split(np.zeros(len(y)), y, groups))
        if len(vai) and len(np.unique(y[vai])) > 1 and len(np.unique(y[tri])) > 1:
            return tri, vai
    except Exception:
        pass

    # Fallback: agrupado sin estratificar
    uniq = list(dict.fromkeys(groups.tolist()))
    random.Random(seed).shuffle(uniq)
    sizes = {g: int((groups == g).sum()) for g in uniq}
    target = val_frac * len(groups)
    val, acc = set(), 0
    for g in uniq:
        if acc >= target:
            break
        val.add(g)
        acc += sizes[g]
    vai = np.array([i for i in range(len(groups)) if groups[i] in val])
    tri = np.array([i for i in range(len(groups)) if groups[i] not in val])
    return tri, vai


def _apply(model, feats, graphs, s, device, bag_size=0, rnd=None):
    """Un forward sobre la bolsa `s`, con grafo o sin él.

    `graphs=None` es el camino de siempre y no cambia nada. Cuando se pasa un
    dict de grafos (SpatialABMIL), se le añade el segundo argumento. Es el único
    punto donde el brazo espacial toca el bucle de entrenamiento: así no hay una
    segunda copia del bucle, del early-stopping ni del split interno.

    `bag_size > 0` muestrea ese número de parches sin reemplazo en cada forward,
    lo que convierte cada slide en una bolsa distinta por época (augmentación).
    Solo se activa en el bucle de entrenamiento: en evaluación y predicción se
    usa la bolsa completa, porque muestrear ahí introduciría varianza en la
    métrica. Con `bag_size=0` (defecto) el camino es byte a byte el de siempre.

    No es compatible con `graphs`: el grafo indexa los parches por posición, así
    que un subconjunto invalidaría sus aristas. Se ignora el muestreo en ese caso.
    """
    x = feats[s]
    if bag_size and graphs is None and x.shape[0] > bag_size:
        g = torch.Generator().manual_seed(rnd.randrange(2**31) if rnd else 0)
        idx = torch.randperm(x.shape[0], generator=g)[:bag_size]
        x = x[idx]
    x = x.to(device)
    return model(x) if graphs is None else model(x, graphs[s].to(device))


@torch.no_grad()
def _auc(model, feats, stems, y, device, graphs=None):
    """Evalúa AUC en validación (para early stopping)."""
    model.eval()
    P = []
    for s in stems:
        lo, _ = _apply(model, feats, graphs, s, device)
        P.append(torch.softmax(lo, -1).cpu().numpy())
    P = np.vstack(P)
    y = np.asarray(y)
    nC = P.shape[1]
    try:
        return (
            roc_auc_score(y, P[:, 1])
            if nC == 2
            else roc_auc_score(y, P, multi_class="ovr", average="macro", labels=list(range(nC)))
        )
    except ValueError:
        return 0.5


def train_abmil(
    feats, stems, y, groups, in_dim, n_classes, device, seed,
    max_epochs=40, patience=8, proj_dim=512, dropout=0.25, wd=1e-4, val_frac=0.2,
    graphs=None, model_factory=None, multi_head_attention=False, n_attention_heads=4,
    bag_size=0
):
    """ABMIL con early-stopping sobre validación interna (agrupada y estratificada).

    Args:
        feats: dict {stem: tensor de features}
        stems: list de IDs de bolsas
        y: list de labels
        groups: array de group IDs (case_id) — se mantienen juntos en train/val
        in_dim: dimensión de entrada
        n_classes: número de clases
        device: torch device
        seed: seed para reproducibilidad
        max_epochs: máximo de épocas (honrado — no sobreescrito por la receta)
        patience: paciencia para early stopping (de la receta)
        proj_dim, dropout, wd: parámetros del modelo/entrenamiento
        val_frac: fracción de validación del train
        graphs: dict {stem: SlideGraph} para el brazo espacial. None = ABMIL de
            siempre, sin ningún cambio de comportamiento.
        model_factory: constructor alternativo con la firma
            `factory(in_dim, n_classes, proj_dim=..., dropout=...)`. None = ABMIL.
        bag_size: si > 0, número de parches muestreados sin reemplazo en cada
            forward de entrenamiento (augmentación de bolsa). La evaluación y la
            predicción siguen usando la bolsa completa. 0 = desactivado.

    Returns:
        model: ABMIL entrenado, con atributo ._val_auc (mejor AUC de val)
    """
    stems = list(stems)
    y = list(y)
    groups = np.asarray(groups)

    # Split train/val
    if val_frac:
        tri, vai = _grouped_val_split(groups, y, val_frac, seed)
    else:
        from sklearn.model_selection import GroupKFold
        tri, vai = next(GroupKFold(5).split(stems, y, groups))

    sub = [stems[i] for i in tri]
    suby = [y[i] for i in tri]
    val = [stems[i] for i in vai]
    valy = [y[i] for i in vai]

    # Receta: lr, patience, wd (max_epochs viene del usuario, no de la receta)
    R = _RECIPES[ABMIL_RECIPE]
    patience_use = R["patience"]
    wd_use = R["wd"]
    lr_use = R["lr"]

    torch.manual_seed(seed)
    # Si hay model_factory (ej. spatial ABMIL), úsalo; sino crea ABMIL estándar
    if model_factory:
        model = model_factory(
            in_dim, n_classes, proj_dim=proj_dim, dropout=dropout
        ).to(device)
    else:
        model = ABMIL(
            in_dim, n_classes, proj_dim=proj_dim, dropout=dropout,
            multi_head_attention=multi_head_attention,
            n_attention_heads=n_attention_heads
        ).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr_use, weight_decay=wd_use)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max_epochs)

    # Pesos balanceados para desbalanceo de clases
    cnt = np.bincount(np.asarray(suby), minlength=n_classes).astype(np.float64)
    cw = torch.tensor(
        len(suby) / (n_classes * np.clip(cnt, 1.0, None)),
        dtype=torch.float32, device=device
    )
    crit = nn.CrossEntropyLoss(
        label_smoothing=R["smooth"], weight=cw, reduction="sum"
    )

    order = list(range(len(sub)))
    rnd = random.Random(seed)
    best, best_state, bad = -1.0, None, 0

    for epoch in range(max_epochs):
        model.train()
        rnd.shuffle(order)
        for j in order:
            opt.zero_grad()
            lo, _ = _apply(model, feats, graphs, sub[j], device, bag_size=bag_size, rnd=rnd)
            loss = crit(lo.unsqueeze(0), torch.tensor([suby[j]], device=device))
            loss.backward()
            opt.step()
        sched.step()

        va = _auc(model, feats, val, valy, device, graphs)
        if va > best:
            best = va
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            bad = 0
        else:
            bad += 1
            if bad >= patience_use:
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    model._val_auc = float(best)  # atributo para debug/logging
    return model


# ───────────────────────────── pooling / predicción ─────────────────────────
@torch.no_grad()
def abmil_predict(model, feats, stems, device, graphs=None):
    """Predicción del ABMIL: softmax(logits) sobre cada stem.

    Returns:
        pp: array (N_stems, n_classes) de probabilidades
    """
    model.eval()
    P = []
    for s in stems:
        lo, _ = _apply(model, feats, graphs, s, device)
        P.append(torch.softmax(lo, -1).cpu().numpy())
    return np.vstack(P)


@torch.no_grad()
def abmil_extract_embeddings(model, feats, stems, device, graphs=None):
    """Extrae embeddings ponderados (no predicciones de clase) del ABMIL.

    Típicamente se usa con ABMIL_EMBEDDING para generar meta-features con más
    información que las predicciones de clase. El embedding tiene dimensión
    proj_dim (usualmente 512) en lugar de n_classes (2 o N).

    Returns:
        emb: array (N_stems, embedding_dim) de embeddings ponderados
    """
    model.eval()
    E = []
    for s in stems:
        emb, _ = _apply(model, feats, graphs, s, device)
        E.append(emb.cpu().numpy())
    return np.vstack(E)


# ──────────────────── Idea 3: Atención Stats para Meta-Features ────────────────
def attention_stats_extended(a):
    """Calcula estadísticos normalizados de distribución de atención.

    Args:
        a: tensor torch (N_patches,) con pesos softmax, sum=1

    Returns:
        dict con keys: entropy_norm, ess_norm, top1pct_mass, top5pct_mass, p90, max, std
    """
    a = a.detach().float().flatten().cpu().numpy()
    n = len(a)

    if n <= 0:
        return {
            "entropy_norm": 0.0,
            "ess_norm": 0.0,
            "top1pct_mass": 0.0,
            "top5pct_mass": 0.0,
            "p90": 0.0,
            "max": 0.0,
            "std": 0.0,
        }

    # Entropía Shannon
    p = np.clip(a, 1e-12, 1.0)
    entropy = float(-np.sum(p * np.log(p)))
    entropy_norm = entropy / np.log(n) if n > 1 else 0.0

    # ESS (effective sample size)
    ess = float(1.0 / np.sum(a ** 2))
    ess_norm = ess / n if n > 0 else 0.0

    # Masa acumulada en top-k%
    sorted_a = np.sort(a)[::-1]

    def top_mass(frac):
        k = max(1, int(round(frac * n)))
        return float(np.sum(sorted_a[:k]))

    return {
        "entropy_norm": entropy_norm,
        "ess_norm": ess_norm,
        "top1pct_mass": top_mass(0.01),
        "top5pct_mass": top_mass(0.05),
        "p90": float(np.percentile(a, 90)),
        "max": float(np.max(a)),
        "std": float(np.std(a)),
    }


@torch.no_grad()
def abmil_predict_with_attention(model, feats, stems, device, graphs=None):
    """Predicción del ABMIL + extracción de estadísticos de atención.

    Realiza un único forward pass por stem, extrayendo y almacenando tanto
    predicciones como estadísticos de atención para garantizar alineación
    fila-a-fila.

    Returns:
        preds: array (N_stems, n_classes) de probabilidades
        attn_stats: array (N_stems, n_stats) con columnas [entropy_norm, ess_norm, ...]
        attn_columns: list de nombres de columnas de attn_stats
    """
    model.eval()
    P = []
    A_stats = []

    for s in stems:
        lo, a = _apply(model, feats, graphs, s, device)
        # Predicción: softmax(logits)
        P.append(torch.softmax(lo, -1).cpu().numpy())
        # Atención stats: estadísticos normalizados
        stats = attention_stats_extended(a)
        A_stats.append(stats)

    preds = np.vstack(P)

    # Convertir lista de dicts a array [N_stems, n_stats]
    columns = ["entropy_norm", "ess_norm", "top1pct_mass", "top5pct_mass", "p90", "max", "std"]
    attn_stats = np.array([[A_stats[i][col] for col in columns] for i in range(len(A_stats))])

    return preds, attn_stats, columns
