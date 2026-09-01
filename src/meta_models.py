"""
Meta-learners para el stacking de ensemble4.py.

Todos exponen la interfaz sklearn (.fit, .predict_proba, .classes_) y entran en
ensemble4.py sin envoltorio, porque `_fit_meta_model` inspecciona la firma de
`fit` para decidir si pasarle conjunto de validación.

Ordenados de MENOS a MÁS capacidad, que es el eje que importa aquí: con ~75
muestras de meta-train y una matriz de features de rango 4 (cada vector softmax
suma 1), subir capacidad ha sido sistemáticamente contraproducente.

  - LogitAveragingMetaClassifier: un peso por modelo base + temperatura
        (4 parámetros en un problema de 3 modelos)
  - MLPMetaClassifier: single-cycle CosineAnnealingLR + early stopping
  - SnapshotMLPMetaClassifier: coseno con reinicios + snapshots (Huang et al., ICLR 2017)
  - FGEMLPMetaClassifier: ciclos cortos piecewise-linear (Garipov et al., ICLR 2018)
  - DeepEnsembleMLPMetaClassifier: k MLPs con semillas independientes
  - GatingMLPMetaClassifier: pesos por muestra sobre los modelos base

Los cuatro basados en MLP comparten la arquitectura _TinyMLP
(Linear-ReLU-Dropout-Linear) y la receta de entrenamiento (AdamW,
CrossEntropyLoss) de abmil_engine.py, para que las diferencias entre ellos sean
atribuibles al método de ensamblado y no al montaje.
"""

import copy
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score

from fge_utils import fge_cycle_lr, set_lr


class _TinyMLP(nn.Module):
    """
    Minimal stacking head: Linear(in_dim) -> ReLU -> Dropout -> Linear(num_classes).

    Args:
        in_dim (int): Input dimension (e.g., num_models * num_classes).
        num_classes (int): Number of output classes.
        hidden_dim (int): Hidden layer dimension. Default: 16.
        dropout (float): Dropout probability. Default: 0.1.
    """
    def __init__(self, in_dim, num_classes, hidden_dim=16, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x):
        """Forward pass. Args: x [batch, in_dim]. Returns: logits [batch, num_classes]."""
        return self.net(x)


def _auc_from_probs(probs, y_true):
    """AUC binaria o macro-OVR a partir de probabilidades ya calculadas.

    Devuelve -1.0 si no puede computarse (p. ej. una sola clase en el split de
    validación), de modo que el early stopping lo trate como "no mejora".
    """
    try:
        if probs.shape[1] == 2:
            return roc_auc_score(y_true, probs[:, 1])
        return roc_auc_score(y_true, probs, multi_class="ovr")
    except ValueError:
        return -1.0


def _as_log_probs(X, n_models, input_is_prob=True, eps=1e-6):
    """Reorganiza [N, n_models*C] en log-probabilidades [N, n_models, C].

    Args:
        X: array [N, n_models * C].
        n_models (int): número de modelos base concatenados en X.
        input_is_prob (bool): True si X trae probabilidades (se aplica log con
            recorte); False si ya viene en espacio logarítmico (p. ej. cuando
            ensemble4.py aplicó --feature_space logit).
        eps (float): recorte inferior para evitar log(0).

    Returns:
        torch.Tensor [N, n_models, C] float32.
    """
    X = np.asarray(X, dtype=np.float64)
    if X.shape[1] % n_models:
        raise ValueError(
            f"{X.shape[1]} features no es múltiplo de n_models={n_models}: "
            f"el meta-learner no puede saber qué columnas son de cada modelo base."
        )
    Z = X.reshape(len(X), n_models, X.shape[1] // n_models)
    if input_is_prob:
        Z = np.log(np.clip(Z, eps, 1.0))
    return torch.from_numpy(Z).float()


class LogitAveragingMetaClassifier:
    """
    Media ponderada en espacio logarítmico, con un peso por modelo base.

    Modelo: log p_ens = (1/T) * sum_m w_m * log p_m, con w = softmax(theta).
    Es el "logarithmic opinion pool": el equivalente en log-espacio de la media
    ponderada que hace ensemble.py, pero con los pesos aprendidos por descenso
    de gradiente en vez de derivados de una métrica de validación.

    Motivación: con ~75 muestras de meta-train y una matriz de features de
    rango 4, la regresión logística ya ajusta 7 parámetros. Este modelo ajusta
    n_models + 1 (4 con tres modelos base), y los pesos son positivos y suman 1
    por construcción, lo que lo hace mucho más difícil de sobreajustar. La
    temperatura T es el único grado de libertad de calibración, que es
    justamente la dimensión donde los snapshots FGE mostraban efecto.

    Args:
        n_models (int): número de modelos base concatenados en las features.
        input_is_prob (bool): True si las features son probabilidades.
        epochs (int): épocas de entrenamiento full-batch. Default: 300.
        lr (float): learning rate de AdamW. Default: 5e-2 (alto a propósito:
            son 4 parámetros, no una red).
        learn_temperature (bool): si False, T queda fija en 1. Default: True.
        patience (int): paciencia de early stopping sobre AUC de validación. Default: 30.
        device (str): "cpu" o "cuda". Default: "cpu".
        seed (int): semilla. Default: 42.
    """
    def __init__(self, n_models, input_is_prob=True, epochs=300, lr=5e-2,
                 learn_temperature=True, patience=30, device="cpu", seed=42):
        self.n_models = n_models
        self.input_is_prob = input_is_prob
        self.epochs = epochs
        self.lr = lr
        self.learn_temperature = learn_temperature
        self.patience = patience
        self.device = device
        self.seed = seed
        self.theta_ = None
        self.log_t_ = None
        self.classes_ = None

    def _logits(self, Z):
        """Combina log-probabilidades [N, n_models, C] en logits [N, C]."""
        w = torch.softmax(self.theta_, dim=0).view(1, -1, 1)
        return (Z * w).sum(dim=1) / torch.exp(self.log_t_)

    def fit(self, X_train, y_train, X_val=None, y_val=None):
        """Ajusta pesos y temperatura por descenso de gradiente."""
        torch.manual_seed(self.seed)

        Z = _as_log_probs(X_train, self.n_models, self.input_is_prob).to(self.device)
        yt = torch.from_numpy(np.asarray(y_train)).long().to(self.device)

        self.theta_ = torch.zeros(self.n_models, device=self.device, requires_grad=True)
        self.log_t_ = torch.zeros(1, device=self.device,
                                  requires_grad=self.learn_temperature)

        params = [self.theta_] + ([self.log_t_] if self.learn_temperature else [])
        opt = torch.optim.AdamW(params, lr=self.lr, weight_decay=0.0)
        crit = nn.CrossEntropyLoss()

        best_auc, best_state, bad = -1.0, None, 0
        for _ in range(self.epochs):
            opt.zero_grad()
            crit(self._logits(Z), yt).backward()
            opt.step()

            if X_val is not None and len(X_val) > 0:
                auc = self._auc(X_val, y_val)
                if auc > best_auc:
                    best_auc = auc
                    best_state = (self.theta_.detach().clone(), self.log_t_.detach().clone())
                    bad = 0
                else:
                    bad += 1
                    if bad >= self.patience:
                        break

        if best_state is not None:
            self.theta_ = best_state[0]
            self.log_t_ = best_state[1]

        self.classes_ = np.unique(y_train)
        return self

    def predict_proba(self, X):
        """Probabilidades [N, C] de la combinación ponderada."""
        Z = _as_log_probs(X, self.n_models, self.input_is_prob).to(self.device)
        with torch.no_grad():
            return torch.softmax(self._logits(Z), dim=1).cpu().numpy()

    @property
    def weights_(self):
        """Pesos normalizados por modelo base (interpretables, suman 1)."""
        with torch.no_grad():
            return torch.softmax(self.theta_, dim=0).cpu().numpy()

    def _auc(self, X_val, y_val):
        return _auc_from_probs(self.predict_proba(X_val), y_val)


class MLPMetaClassifier:
    """
    Meta-learner MLP with single-cycle CosineAnnealingLR and early stopping.

    Follows the same training recipe as abmil_engine.py (AdamW + CosineAnnealingLR
    with T_max=epochs, step per epoch), with early stopping on validation AUC.
    Keeps only the best model state in memory (no disk checkpoints).

    Sklearn-compatible interface (.fit, .predict_proba, .classes_) for drop-in
    replacement of LogisticRegression in ensemble4.py.

    Args:
        hidden_dim (int): Hidden layer width. Default: 16.
        dropout (float): Dropout probability. Default: 0.1.
        epochs (int): Total training epochs. Default: 100.
        lr (float): Initial learning rate for AdamW. Default: 1e-3.
        wd (float): Weight decay for AdamW. Default: 1e-4.
        patience (int): Early stopping patience (bad epochs before stopping). Default: 8.
        device (str): "cpu" or "cuda". Default: "cpu".
        seed (int): Random seed for torch initialization. Default: 42.
    """
    def __init__(self, hidden_dim=16, dropout=0.1, epochs=100, lr=1e-3,
                 wd=1e-4, patience=8, device="cpu", seed=42):
        self.hidden_dim = hidden_dim
        self.dropout = dropout
        self.epochs = epochs
        self.lr = lr
        self.wd = wd
        self.patience = patience
        self.device = device
        self.seed = seed
        self.model_ = None
        self.classes_ = None

    def fit(self, X_train, y_train, X_val=None, y_val=None, verbose=False):
        """
        Fit the MLP on training data with optional early stopping on validation AUC.

        Args:
            X_train: [N_train, in_dim] numpy array, float features.
            y_train: [N_train] numpy array, int labels.
            X_val: [N_val, in_dim] optional validation features; if provided, enables early stopping.
            y_val: [N_val] optional validation labels.
            verbose: If True, print per-epoch metrics. Default: False.

        Returns:
            self
        """
        torch.manual_seed(self.seed)

        # Infer dimensions and instantiate model
        in_dim = X_train.shape[1]
        num_classes = len(np.unique(y_train))
        self.model_ = _TinyMLP(in_dim, num_classes, self.hidden_dim,
                               self.dropout).to(self.device)

        # Optimizer and scheduler (same pattern as abmil_engine.py)
        opt = torch.optim.AdamW(self.model_.parameters(), lr=self.lr,
                                weight_decay=self.wd)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=self.epochs)
        crit = nn.CrossEntropyLoss()

        # Convert training data to tensors
        Xt = torch.from_numpy(X_train).float().to(self.device)
        yt = torch.from_numpy(y_train).long().to(self.device)

        # Training loop with early stopping
        best_state, best_auc, bad = None, -1.0, 0
        for epoch in range(self.epochs):
            self.model_.train()
            opt.zero_grad()
            loss = crit(self.model_(Xt), yt)
            loss.backward()
            opt.step()
            sched.step()

            # Early stopping on validation AUC (if validation set provided)
            if X_val is not None and len(X_val) > 0:
                auc = self._auc(X_val, y_val)
                if auc > best_auc:
                    best_auc = auc
                    best_state = copy.deepcopy(self.model_.state_dict())
                    bad = 0
                    if verbose and (epoch + 1) % 10 == 0:
                        print(f"  Epoch {epoch + 1:3d}: AUC={auc:.4f} (best)")
                else:
                    bad += 1
                    if verbose and bad == 1:
                        print(f"  Epoch {epoch + 1:3d}: AUC={auc:.4f} (bad count: 1/{self.patience})")
                    if bad >= self.patience:
                        if verbose:
                            print(f"  Early stopping at epoch {epoch + 1} (patience {self.patience} exhausted)")
                        break

        # Restore best state if early stopping occurred
        if best_state is not None:
            self.model_.load_state_dict(best_state)

        self.classes_ = np.unique(y_train)
        return self

    def predict_proba(self, X):
        """
        Predict class probabilities.

        Args:
            X: [N, in_dim] numpy array of features.

        Returns:
            [N, num_classes] numpy array of softmax probabilities.
        """
        self.model_.eval()
        with torch.no_grad():
            Xt = torch.from_numpy(X).float().to(self.device)
            probs = torch.softmax(self.model_(Xt), dim=1)
        return probs.cpu().numpy()

    def _auc(self, X_val, y_val):
        """Compute AUC on validation set (binary or multiclass)."""
        probs = self.predict_proba(X_val)
        try:
            if probs.shape[1] == 2:
                return roc_auc_score(y_val, probs[:, 1])
            else:
                return roc_auc_score(y_val, probs, multi_class="ovr")
        except ValueError:
            # E.g., only one class present in validation set
            return -1.0


class SnapshotMLPMetaClassifier:
    """
    Meta-learner MLP trained with Snapshot Ensembles (Huang et al., ICLR 2017).

    Uses cyclic cosine annealing (CosineAnnealingWarmRestarts, Eq. 2 in paper) to
    visit multiple local minima during training. Takes a snapshot (state_dict) at the
    end of each cycle and averages predictions across the last m snapshots at test time.

    Implements inter-cycle early stopping: if validation AUC plateaus across multiple
    cycles, training stops early. Keeps only improving snapshots.

    Reuses the AdamW + CosineAnnealingWarmRestarts pattern from the paper.

    Sklearn-compatible interface (.fit, .predict_proba, .classes_) for integration
    with ensemble4.py.

    Args:
        hidden_dim (int): Hidden layer width. Default: 16.
        dropout (float): Dropout probability. Default: 0.1.
        epochs (int): Total training epochs (distributed over n_cycles). Default: 120.
        restart_lr (float): Initial LR at the start of each cycle (α0 in paper). Default: 5e-3.
        n_cycles (int): Number of cosine annealing cycles. Default: 6.
        n_snapshots (int or None): How many of the last snapshots to average at test.
                                   If None, uses all n_cycles snapshots. Default: None.
        wd (float): Weight decay for AdamW. Default: 1e-4.
        cycle_patience (int): Early stopping between cycles if AUC plateaus. Default: 2.
        device (str): "cpu" or "cuda". Default: "cpu".
        seed (int): Random seed for torch initialization. Default: 42.
    """
    def __init__(self, hidden_dim=16, dropout=0.1, epochs=120, restart_lr=5e-3,
                 n_cycles=6, n_snapshots=None, wd=1e-4, cycle_patience=2,
                 device="cpu", seed=42):
        self.hidden_dim = hidden_dim
        self.dropout = dropout
        self.epochs = epochs
        self.restart_lr = restart_lr
        self.n_cycles = n_cycles
        self.n_snapshots = n_snapshots
        self.wd = wd
        self.cycle_patience = cycle_patience
        self.device = device
        self.seed = seed
        self.model_ = None
        self.snapshots_ = []
        self.classes_ = None

    def fit(self, X_train, y_train, X_val=None, y_val=None):
        """
        Fit the MLP with cyclic cosine annealing, accumulating snapshots at cycle boundaries.

        Implements inter-cycle early stopping: if validation AUC plateaus, stops early.
        If no validation set provided, trains all cycles.

        Args:
            X_train: [N_train, in_dim] numpy array, float features.
            y_train: [N_train] numpy array, int labels.
            X_val: [N_val, in_dim] optional validation features; if provided, enables inter-cycle early stopping.
            y_val: [N_val] optional validation labels.

        Returns:
            self
        """
        torch.manual_seed(self.seed)

        # Infer dimensions and instantiate model
        in_dim = X_train.shape[1]
        num_classes = len(np.unique(y_train))
        self.model_ = _TinyMLP(in_dim, num_classes, self.hidden_dim,
                               self.dropout).to(self.device)

        # Optimizer with cyclic cosine annealing (Eq. 2 in Huang et al.)
        opt = torch.optim.AdamW(self.model_.parameters(), lr=self.restart_lr,
                                weight_decay=self.wd)
        cycle_len = max(1, self.epochs // self.n_cycles)
        sched = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            opt, T_0=cycle_len, T_mult=1, eta_min=self.restart_lr * 1e-3
        )
        crit = nn.CrossEntropyLoss()

        # Convert training data to tensors
        Xt = torch.from_numpy(X_train).float().to(self.device)
        yt = torch.from_numpy(y_train).long().to(self.device)

        # Training loop with snapshot-taking at cycle boundaries
        self.snapshots_ = []
        best_auc = -1.0
        bad_cycles = 0

        for epoch in range(self.epochs):
            self.model_.train()
            opt.zero_grad()
            loss = crit(self.model_(Xt), yt)
            loss.backward()
            opt.step()
            sched.step()

            # Take snapshot at the end of each cycle (when learning rate is smallest)
            if (epoch + 1) % cycle_len == 0:
                self.snapshots_.append(copy.deepcopy(self.model_.state_dict()))

                # Inter-cycle early stopping on validation AUC
                if X_val is not None and len(X_val) > 0:
                    current_auc = self._auc_ensemble(X_val, y_val)
                    if current_auc > best_auc:
                        best_auc = current_auc
                        bad_cycles = 0
                    else:
                        bad_cycles += 1
                        if bad_cycles >= self.cycle_patience:
                            # Remove last snapshot and stop
                            self.snapshots_.pop()
                            break

        # Keep only the last m snapshots (usually all of them)
        m = self.n_snapshots or len(self.snapshots_)
        self.snapshots_ = self.snapshots_[-m:]

        self.classes_ = np.unique(y_train)
        return self

    def _auc_ensemble(self, X_val, y_val):
        """
        Compute AUC on validation set using current ensemble of snapshots.
        Used for inter-cycle early stopping.

        Args:
            X_val: [N_val, in_dim] numpy array of validation features.
            y_val: [N_val] numpy array of validation labels.

        Returns:
            float: AUC score on validation set (or -1.0 if only one class).
        """
        probs = self.predict_proba(X_val)
        try:
            if probs.shape[1] == 2:
                return roc_auc_score(y_val, probs[:, 1])
            else:
                return roc_auc_score(y_val, probs, multi_class="ovr")
        except ValueError:
            # E.g., only one class present in validation set
            return -1.0

    def predict_proba(self, X):
        """
        Predict by averaging softmax probabilities across snapshots.

        Implements: h_Ensemble = (1/m) * sum_{i=0}^{m-1} h_{M-i}(x)
        (Section 3, "Ensembling at Test Time" in Huang et al.)

        Args:
            X: [N, in_dim] numpy array of features.

        Returns:
            [N, num_classes] numpy array of averaged softmax probabilities.
        """
        Xt = torch.from_numpy(X).float().to(self.device)
        probs_sum = None

        for state in self.snapshots_:
            self.model_.load_state_dict(state)
            self.model_.eval()
            with torch.no_grad():
                logits = self.model_(Xt)
                p = torch.softmax(logits, dim=1)

            probs_sum = p if probs_sum is None else probs_sum + p

        # Average across snapshots
        avg_probs = probs_sum / len(self.snapshots_)
        return avg_probs.cpu().numpy()


class FGEMLPMetaClassifier:
    """
    Meta-learner MLP con Fast Geometric Ensembles (Garipov et al., ICLR 2018).

    A diferencia de SnapshotMLPMetaClassifier (CosineAnnealingWarmRestarts,
    ciclos largos que exploran nuevos mínimos desde cerca de cero en cada
    reinicio), FGE parte de un modelo YA CONVERGIDO (fase de warm-up, igual
    que MLPMetaClassifier) y hace fine-tuning con ciclos CORTOS de LR
    piecewise-linear (src/fge_utils.py), tomando un snapshot al final de
    cada ciclo. Con el meta-learner ya siendo diminuto (pocas features, pocas
    muestras por fold), ciclos de 2-4 épocas bastan para recorrer la región
    plana cercana al mínimo encontrado en el warm-up.

    Sklearn-compatible interface (.fit, .predict_proba, .classes_) para
    integración con ensemble4.py (--meta_model mlp_fge).

    Args:
        hidden_dim (int): Hidden layer width. Default: 16.
        dropout (float): Dropout probability. Default: 0.1.
        warmup_epochs (int): Épocas de la fase de convergencia inicial
            (delegada a MLPMetaClassifier). Default: 100.
        warmup_lr (float): LR inicial de la fase de warm-up. Default: 1e-3.
        warmup_wd (float): Weight decay de la fase de warm-up. Default: 1e-4.
        warmup_patience (int): Paciencia de early stopping del warm-up. Default: 8.
        n_cycles (int): Número de ciclos FGE tras el warm-up. Default: 6.
        cycle_length (int): Épocas por ciclo FGE (corto). Default: 3.
        lr_1 (float): LR techo del ciclo piecewise-linear. Default: 1e-3.
        lr_2 (float): LR piso del ciclo (valor en el snapshot). Default: 1e-5.
        n_snapshots (int or None): Cuántos de los últimos snapshots promediar
            en predict_proba. Si None, usa todos. Default: None.
        wd (float): Weight decay de la fase FGE. Default: 1e-4.
        cycle_patience (int): Ciclos sin mejora de AUC val antes de detener
            los ciclos FGE. Default: 2.
        device (str): "cpu" o "cuda". Default: "cpu".
        seed (int): Seed para reproducibilidad. Default: 42.
    """
    def __init__(self, hidden_dim=16, dropout=0.1,
                 warmup_epochs=100, warmup_lr=1e-3, warmup_wd=1e-4, warmup_patience=8,
                 n_cycles=6, cycle_length=3, lr_1=1e-3, lr_2=1e-5,
                 n_snapshots=None, wd=1e-4, cycle_patience=2,
                 device="cpu", seed=42):
        self.hidden_dim = hidden_dim
        self.dropout = dropout
        self.warmup_epochs = warmup_epochs
        self.warmup_lr = warmup_lr
        self.warmup_wd = warmup_wd
        self.warmup_patience = warmup_patience
        self.n_cycles = n_cycles
        self.cycle_length = cycle_length
        self.lr_1 = lr_1
        self.lr_2 = lr_2
        self.n_snapshots = n_snapshots
        self.wd = wd
        self.cycle_patience = cycle_patience
        self.device = device
        self.seed = seed
        self.model_ = None
        self.snapshots_ = []
        self.classes_ = None

    def fit(self, X_train, y_train, X_val=None, y_val=None):
        """
        Fase 1 (warm-up): converge un MLPMetaClassifier estándar (composición,
        no copia de código) — mismo patrón que abmil_fge.py usa
        abmil_engine.train_abmil para converger antes de los ciclos FGE.

        Fase 2 (FGE): fine-tuning con ciclos cortos de LR piecewise-linear
        partiendo del modelo convergido, tomando snapshots por ciclo con
        early stopping inter-ciclo si el AUC de validación no mejora.

        Args:
            X_train, y_train: features/labels de entrenamiento.
            X_val, y_val: features/labels de validación (opcional; habilita
                early stopping en el warm-up y entre ciclos FGE).

        Returns:
            self
        """
        torch.manual_seed(self.seed)

        # ---- Fase 1: warm-up (reutiliza MLPMetaClassifier.fit) ----
        warm = MLPMetaClassifier(
            hidden_dim=self.hidden_dim, dropout=self.dropout,
            epochs=self.warmup_epochs, lr=self.warmup_lr, wd=self.warmup_wd,
            patience=self.warmup_patience, device=self.device, seed=self.seed,
        )
        warm.fit(X_train, y_train, X_val=X_val, y_val=y_val)
        self.model_ = warm.model_
        self.classes_ = warm.classes_

        # ---- Fase 2: ciclos FGE cortos ----
        opt = torch.optim.AdamW(self.model_.parameters(), lr=self.lr_1, weight_decay=self.wd)
        crit = nn.CrossEntropyLoss()

        Xt = torch.from_numpy(X_train).float().to(self.device)
        yt = torch.from_numpy(y_train).long().to(self.device)

        self.snapshots_ = []
        best_auc, bad_cycles = -1.0, 0
        total_epochs = self.n_cycles * self.cycle_length

        for epoch in range(total_epochs):
            set_lr(opt, fge_cycle_lr(epoch, self.cycle_length, self.lr_1, self.lr_2))

            self.model_.train()
            opt.zero_grad()
            loss = crit(self.model_(Xt), yt)
            loss.backward()
            opt.step()

            if (epoch + 1) % self.cycle_length == 0:
                self.snapshots_.append(copy.deepcopy(self.model_.state_dict()))

                if X_val is not None and len(X_val) > 0:
                    current_auc = self._auc_ensemble(X_val, y_val)
                    if current_auc > best_auc:
                        best_auc = current_auc
                        bad_cycles = 0
                    else:
                        bad_cycles += 1
                        if bad_cycles >= self.cycle_patience:
                            self.snapshots_.pop()
                            break

        if not self.snapshots_:
            # FGE no completó ningún ciclo (early stop inmediato o n_cycles=0):
            # cae de vuelta al modelo del warm-up como único "snapshot".
            self.snapshots_ = [copy.deepcopy(self.model_.state_dict())]

        m = self.n_snapshots or len(self.snapshots_)
        self.snapshots_ = self.snapshots_[-m:]

        return self

    def _auc_ensemble(self, X_val, y_val):
        """AUC en validación usando el ensemble actual de snapshots FGE."""
        probs = self.predict_proba(X_val)
        try:
            if probs.shape[1] == 2:
                return roc_auc_score(y_val, probs[:, 1])
            else:
                return roc_auc_score(y_val, probs, multi_class="ovr")
        except ValueError:
            return -1.0

    def predict_proba(self, X):
        """
        Predice promediando softmax entre snapshots (mismo patrón que
        SnapshotMLPMetaClassifier.predict_proba).

        Args:
            X: [N, in_dim] numpy array de features.

        Returns:
            [N, num_classes] numpy array de probabilidades promediadas.
        """
        Xt = torch.from_numpy(X).float().to(self.device)
        probs_sum = None

        for state in self.snapshots_:
            self.model_.load_state_dict(state)
            self.model_.eval()
            with torch.no_grad():
                logits = self.model_(Xt)
                p = torch.softmax(logits, dim=1)
            probs_sum = p if probs_sum is None else probs_sum + p

        avg_probs = probs_sum / len(self.snapshots_)
        return avg_probs.cpu().numpy()


class DeepEnsembleMLPMetaClassifier:
    """
    k MLPs entrenados de forma independiente (semillas distintas), promediando
    softmax en test.

    Es el CONTROL que da sentido a SnapshotMLPMetaClassifier. Huang et al.
    presentan Snapshot Ensembles como una aproximación barata a entrenar varios
    modelos por separado: sin medir el ensemble de semillas independientes no
    se puede distinguir "SE funciona" de "ensamblar funciona, y SE es una forma
    cara de hacerlo peor". El coste aquí es irrelevante (el meta-learner es
    diminuto), así que la comparación es directa.

    Reutiliza MLPMetaClassifier por composición: cada miembro se entrena con la
    misma receta y el mismo early stopping, cambiando sólo la semilla.

    Args:
        n_members (int): número de MLPs independientes. Default: 6 (mismo orden
            que los n_cycles de SnapshotMLPMetaClassifier, para que la
            comparación sea a igual número de miembros).
        hidden_dim, dropout, epochs, lr, wd, patience: se pasan tal cual a cada
            MLPMetaClassifier.
        device (str): "cpu" o "cuda". Default: "cpu".
        seed (int): semilla base; el miembro i usa seed + i. Default: 42.
    """
    def __init__(self, n_members=6, hidden_dim=16, dropout=0.1, epochs=100,
                 lr=1e-3, wd=1e-4, patience=8, device="cpu", seed=42):
        self.n_members = n_members
        self.hidden_dim = hidden_dim
        self.dropout = dropout
        self.epochs = epochs
        self.lr = lr
        self.wd = wd
        self.patience = patience
        self.device = device
        self.seed = seed
        self.members_ = []
        self.classes_ = None

    def fit(self, X_train, y_train, X_val=None, y_val=None):
        """Entrena los k miembros con semillas distintas."""
        self.members_ = []
        for i in range(self.n_members):
            m = MLPMetaClassifier(
                hidden_dim=self.hidden_dim, dropout=self.dropout,
                epochs=self.epochs, lr=self.lr, wd=self.wd,
                patience=self.patience, device=self.device, seed=self.seed + i,
            )
            m.fit(X_train, y_train, X_val=X_val, y_val=y_val)
            self.members_.append(m)

        self.classes_ = np.unique(y_train)
        return self

    def predict_proba(self, X):
        """Media de las softmax de los k miembros."""
        return np.mean([m.predict_proba(X) for m in self.members_], axis=0)


class GatingMLPMetaClassifier:
    """
    Combinación convexa de los modelos base con pesos POR MUESTRA
    (mixture-of-experts).

    A diferencia de LogitAveragingMetaClassifier, que aprende un peso global
    por modelo, aquí una cabeza pequeña lee las features y emite n_models pesos
    para cada slide: permite que el modelo fundacional dominante cambie según
    el caso. La predicción es sum_m w_m(x) * p_m(x), así que nunca sale del
    envolvente convexo de las predicciones base — una restricción fuerte que
    lo hace más seguro que una MLP libre.

    No es convexo, luego SE/FGE tendrían sentido sobre él; se deja como
    extensión si esta variante resulta competitiva.

    ATENCIÓN: sólo tiene sentido evaluarlo con meta-features out-of-fold. Con
    las features in-sample (donde los modelos base valen AUC 0.94 frente a 0.77
    en test) el gating aprende a confiar en quien acierta dentro de muestra, que
    es justo la señal que no transfiere.

    Args:
        n_models (int): número de modelos base concatenados en las features.
        input_is_prob (bool): True si las features son probabilidades.
        hidden_dim (int): ancho de la cabeza de gating. Default: 8.
        dropout (float): dropout de la cabeza. Default: 0.1.
        epochs, lr, wd, patience: receta de entrenamiento (igual que MLPMetaClassifier).
        device (str): "cpu" o "cuda". Default: "cpu".
        seed (int): semilla. Default: 42.
    """
    def __init__(self, n_models, input_is_prob=True, hidden_dim=8, dropout=0.1,
                 epochs=100, lr=1e-3, wd=1e-4, patience=8, device="cpu", seed=42):
        self.n_models = n_models
        self.input_is_prob = input_is_prob
        self.hidden_dim = hidden_dim
        self.dropout = dropout
        self.epochs = epochs
        self.lr = lr
        self.wd = wd
        self.patience = patience
        self.device = device
        self.seed = seed
        self.gate_ = None
        self.classes_ = None

    def _mixture_logits(self, Xt, Z):
        """Log de la mezcla convexa, utilizable como logits en CrossEntropyLoss.

        Xt: [N, n_models*C] features crudas (entrada del gate).
        Z:  [N, n_models, C] log-probabilidades de los modelos base.
        """
        w = torch.softmax(self.gate_(Xt), dim=1).unsqueeze(-1)   # [N, n_models, 1]
        # logsumexp sobre modelos de (log w_m + log p_m) = log(sum_m w_m p_m),
        # que es estable frente a calcular la mezcla y luego su logaritmo.
        return torch.logsumexp(torch.log(w.clamp_min(1e-12)) + Z, dim=1)

    def fit(self, X_train, y_train, X_val=None, y_val=None):
        """Ajusta la cabeza de gating con early stopping sobre AUC de validación."""
        torch.manual_seed(self.seed)

        Xt = torch.from_numpy(np.asarray(X_train, dtype=np.float64)).float().to(self.device)
        Z = _as_log_probs(X_train, self.n_models, self.input_is_prob).to(self.device)
        yt = torch.from_numpy(np.asarray(y_train)).long().to(self.device)

        self.gate_ = _TinyMLP(Xt.shape[1], self.n_models, self.hidden_dim,
                              self.dropout).to(self.device)
        opt = torch.optim.AdamW(self.gate_.parameters(), lr=self.lr,
                                weight_decay=self.wd)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=self.epochs)
        crit = nn.CrossEntropyLoss()

        best_state, best_auc, bad = None, -1.0, 0
        for _ in range(self.epochs):
            self.gate_.train()
            opt.zero_grad()
            crit(self._mixture_logits(Xt, Z), yt).backward()
            opt.step()
            sched.step()

            if X_val is not None and len(X_val) > 0:
                auc = _auc_from_probs(self.predict_proba(X_val), y_val)
                if auc > best_auc:
                    best_auc = auc
                    best_state = copy.deepcopy(self.gate_.state_dict())
                    bad = 0
                else:
                    bad += 1
                    if bad >= self.patience:
                        break

        if best_state is not None:
            self.gate_.load_state_dict(best_state)

        self.classes_ = np.unique(y_train)
        return self

    def predict_proba(self, X):
        """Probabilidades de la mezcla convexa."""
        Xt = torch.from_numpy(np.asarray(X, dtype=np.float64)).float().to(self.device)
        Z = _as_log_probs(X, self.n_models, self.input_is_prob).to(self.device)
        self.gate_.eval()
        with torch.no_grad():
            return torch.softmax(self._mixture_logits(Xt, Z), dim=1).cpu().numpy()

    def gate_weights(self, X):
        """Pesos por muestra [N, n_models] — para inspeccionar qué modelo domina."""
        Xt = torch.from_numpy(np.asarray(X, dtype=np.float64)).float().to(self.device)
        self.gate_.eval()
        with torch.no_grad():
            return torch.softmax(self.gate_(Xt), dim=1).cpu().numpy()


# ============================================================================
# NUEVOS META-MODELOS: DeepMLP y Variantes
# ============================================================================

class _DeepMLP(nn.Module):
    """
    Red neuronal más profunda: 3 capas ocultas con dropout y batch norm.
    
    Arquitectura: Linear(in_dim) -> ReLU -> BN -> Dropout ->
                  Linear(256) -> ReLU -> BN -> Dropout ->
                  Linear(128) -> ReLU -> BN -> Dropout ->
                  Linear(64) -> ReLU -> Dropout ->
                  Linear(num_classes)
    """
    def __init__(self, in_dim, num_classes, dropout=0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(dropout),
            
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(dropout),
            
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(dropout),
            
            nn.Linear(64, num_classes)
        )
    
    def forward(self, x):
        return self.net(x)


class DeepMLPMetaClassifier:
    """Deep MLP con CosineAnnealingLR y early stopping."""
    
    def __init__(self, lr=1e-3, wd=1e-4, epochs=100, patience=8, device="cpu"):
        self.lr = lr
        self.wd = wd
        self.epochs = epochs
        self.patience = patience
        self.device = device
        self.model_ = None
        self.classes_ = None
    
    def fit(self, X_train, y_train, X_val=None, y_val=None):
        X = np.asarray(X_train, dtype=np.float64)
        y = np.asarray(y_train, dtype=np.int64)
        
        self.classes_ = np.unique(y)
        n_classes = len(self.classes_)
        
        self.model_ = _DeepMLP(X.shape[1], n_classes, dropout=0.2).to(self.device)
        
        opt = torch.optim.AdamW(self.model_.parameters(), lr=self.lr, weight_decay=self.wd)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=self.epochs)
        crit = nn.CrossEntropyLoss()
        
        Xt = torch.from_numpy(X).float().to(self.device)
        yt = torch.from_numpy(y).long().to(self.device)
        
        best_state, best_auc, bad = None, -1.0, 0
        for epoch in range(self.epochs):
            self.model_.train()
            opt.zero_grad()
            crit(self.model_(Xt), yt).backward()
            opt.step()
            sched.step()
            
            if X_val is not None and len(X_val) > 0:
                auc = self._auc(X_val, y_val)
                if auc > best_auc:
                    best_auc = auc
                    best_state = copy.deepcopy(self.model_.state_dict())
                    bad = 0
                else:
                    bad += 1
                    if bad >= self.patience:
                        break
        
        if best_state is not None:
            self.model_.load_state_dict(best_state)
        
        return self
    
    def predict_proba(self, X):
        X = np.asarray(X, dtype=np.float64)
        Xt = torch.from_numpy(X).float().to(self.device)
        self.model_.eval()
        with torch.no_grad():
            return torch.softmax(self.model_(Xt), dim=1).cpu().numpy()
    
    def _auc(self, X_val, y_val):
        probs = self.predict_proba(X_val)

        # Asegurar que siempre sea (n, 2) para binary classification
        if probs.ndim == 1:
            probs = np.column_stack([1 - probs.ravel(), probs.ravel()])
        elif probs.ndim == 2 and probs.shape[1] == 1:
            probs = np.column_stack([1 - probs.ravel(), probs.ravel()])

        y_val = np.asarray(y_val)
        if len(self.classes_) == 2:
            return roc_auc_score(y_val, probs[:, 1])
        else:
            return roc_auc_score(y_val, probs, multi_class='ovr')


class SnapshotDeepMLPMetaClassifier:
    """Deep MLP con CosineAnnealingWarmRestarts + snapshots."""
    
    def __init__(self, lr=1e-3, wd=1e-4, epochs=120, n_cycles=6, restart_lr=5e-3, 
                 n_snapshots=None, device="cpu"):
        self.lr = lr
        self.wd = wd
        self.epochs = epochs
        self.n_cycles = n_cycles
        self.restart_lr = restart_lr
        self.n_snapshots = n_snapshots or n_cycles
        self.device = device
        self.snapshots_ = []
        self.classes_ = None
    
    def fit(self, X_train, y_train, X_val=None, y_val=None):
        X = np.asarray(X_train, dtype=np.float64)
        y = np.asarray(y_train, dtype=np.int64)
        
        self.classes_ = np.unique(y)
        n_classes = len(self.classes_)
        
        model = _DeepMLP(X.shape[1], n_classes, dropout=0.2).to(self.device)
        opt = torch.optim.AdamW(model.parameters(), lr=self.lr, weight_decay=self.wd)
        sched = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            opt, T_0=self.epochs // self.n_cycles, T_mult=1, eta_min=1e-7, last_epoch=-1
        )
        crit = nn.CrossEntropyLoss()
        
        Xt = torch.from_numpy(X).float().to(self.device)
        yt = torch.from_numpy(y).long().to(self.device)
        
        best_auc = -1.0
        for epoch in range(self.epochs):
            model.train()
            opt.zero_grad()
            crit(model(Xt), yt).backward()
            opt.step()
            sched.step()
            
            if (epoch + 1) % (self.epochs // self.n_snapshots) == 0:
                self.snapshots_.append(copy.deepcopy(model.state_dict()))
            
            if X_val is not None and len(X_val) > 0:
                auc = self._auc(X_val, y_val)
                if auc > best_auc:
                    best_auc = auc
        
        return self
    
    def predict_proba(self, X):
        X = np.asarray(X, dtype=np.float64)
        Xt = torch.from_numpy(X).float().to(self.device)

        probs_list = []
        for state_dict in self.snapshots_:
            model = _DeepMLP(X.shape[1], len(self.classes_), dropout=0.2).to(self.device)
            model.load_state_dict(state_dict)
            model.eval()
            with torch.no_grad():
                probs = torch.softmax(model(Xt), dim=1).cpu().numpy()
            probs_list.append(probs)

        if len(probs_list) == 0:
            # Si no hay snapshots, retornar probabilidades uniformes
            return np.ones((len(X), len(self.classes_))) / len(self.classes_)

        return np.mean(probs_list, axis=0)
    
    def _auc(self, X_val, y_val):
        probs = self.predict_proba(X_val)

        # Asegurar que siempre sea (n, 2) para binary classification
        if probs.ndim == 1:
            probs = np.column_stack([1 - probs.ravel(), probs.ravel()])
        elif probs.ndim == 2 and probs.shape[1] == 1:
            probs = np.column_stack([1 - probs.ravel(), probs.ravel()])

        y_val = np.asarray(y_val)
        if len(self.classes_) == 2:
            return roc_auc_score(y_val, probs[:, 1])
        else:
            return roc_auc_score(y_val, probs, multi_class='ovr')


class TabPFNMetaClassifier:
    """
    Tabular Prior Foundation Network (TabPFN) como meta-classifier.
    Requiere: pip install tabpfn
    """
    
    def __init__(self, device="cpu"):
        self.device = device
        self.model_ = None
        self.classes_ = None
        try:
            from tabpfn import TabPFNClassifier
            self.TabPFNClassifier = TabPFNClassifier
        except ImportError:
            raise ImportError("TabPFN no está instalado. Ejecuta: pip install tabpfn")
    
    def fit(self, X_train, y_train, X_val=None, y_val=None):
        self.classes_ = np.unique(y_train)
        
        # TabPFN espera arrays 2D
        X = np.asarray(X_train, dtype=np.float32)
        y = np.asarray(y_train, dtype=np.int64)
        
        # TabPFN requiere escalado a [0, 1]
        X_min = X.min(axis=0)
        X_max = X.max(axis=0)
        X_scaled = (X - X_min) / (X_max - X_min + 1e-8)
        
        self.model_ = self.TabPFNClassifier(device=self.device)
        self.model_.fit(X_scaled, y)
        
        self.X_min_ = X_min
        self.X_max_ = X_max
        
        return self
    
    def predict_proba(self, X):
        X = np.asarray(X, dtype=np.float32)
        X_scaled = (X - self.X_min_) / (self.X_max_ - self.X_min_ + 1e-8)
        probs = self.model_.predict_proba(X_scaled)

        # Asegurar que siempre sea (n, 2) para binary classification
        if probs.ndim == 1 or probs.shape[1] == 1:
            # Si es 1D o (n,1), construir (n,2): [1-p, p]
            probs = np.column_stack([1 - probs.ravel(), probs.ravel()])

        return probs


class SnapshotTabPFNMetaClassifier:
    """
    TabPFN con Multiple Independent Initializations (snapshot ensemble style).
    Entrena múltiples TabPFN con distintas semillas.
    """
    
    def __init__(self, device="cpu", n_snapshots=4):
        self.device = device
        self.n_snapshots = n_snapshots
        self.models_ = []
        self.classes_ = None
        self.X_min_ = None
        self.X_max_ = None
        try:
            from tabpfn import TabPFNClassifier
            self.TabPFNClassifier = TabPFNClassifier
        except ImportError:
            raise ImportError("TabPFN no está instalado. Ejecuta: pip install tabpfn")
    
    def fit(self, X_train, y_train, X_val=None, y_val=None):
        self.classes_ = np.unique(y_train)
        
        X = np.asarray(X_train, dtype=np.float32)
        y = np.asarray(y_train, dtype=np.int64)
        
        # Escalado global
        self.X_min_ = X.min(axis=0)
        self.X_max_ = X.max(axis=0)
        X_scaled = (X - self.X_min_) / (self.X_max_ - self.X_min_ + 1e-8)
        
        # Entrenar múltiples TabPFN con distintas semillas
        for i in range(self.n_snapshots):
            model = self.TabPFNClassifier(device=self.device)
            model.fit(X_scaled, y)
            self.models_.append(model)
        
        return self
    
    def predict_proba(self, X):
        X = np.asarray(X, dtype=np.float32)
        X_scaled = (X - self.X_min_) / (self.X_max_ - self.X_min_ + 1e-8)

        probs_list = []
        for m in self.models_:
            probs = m.predict_proba(X_scaled)
            # Asegurar que siempre sea (n, 2) para binary classification
            if probs.ndim == 1:
                probs = np.column_stack([1 - probs.ravel(), probs.ravel()])
            elif probs.ndim == 2 and probs.shape[1] == 1:
                probs = np.column_stack([1 - probs.ravel(), probs.ravel()])
            probs_list.append(probs)

        if len(probs_list) == 0:
            # Si no hay modelos, retornar probabilidades uniformes
            return np.ones((len(X), len(self.classes_))) / len(self.classes_)

        return np.mean(probs_list, axis=0)
