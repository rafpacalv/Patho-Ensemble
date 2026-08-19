import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
import os
import argparse
import inspect
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from pathlib import Path

from utils import Metrics, MetricDistance
from meta_models import (MLPMetaClassifier, SnapshotMLPMetaClassifier, FGEMLPMetaClassifier,
                         LogitAveragingMetaClassifier, DeepEnsembleMLPMetaClassifier,
                         GatingMLPMetaClassifier, DeepMLPMetaClassifier,
                         SnapshotDeepMLPMetaClassifier, TabPFNMetaClassifier,
                         SnapshotTabPFNMetaClassifier)


def _fit_meta_model(meta_model, X_train, y_train, X_val=None, y_val=None):
    """
    Helper to call fit() on a meta-model, automatically checking whether it
    accepts X_val/y_val parameters (for neural network models) or not
    (for sklearn models like LogisticRegression).

    Args:
        meta_model: The meta-learner instance (LogisticRegression, MLPMetaClassifier, etc.)
        X_train, y_train: Training features and labels.
        X_val, y_val: Optional validation features and labels.
    """
    params = inspect.signature(meta_model.fit).parameters
    if "X_val" in params and "y_val" in params:
        # Neural network models accept validation set for early stopping
        meta_model.fit(X_train, y_train, X_val=X_val, y_val=y_val)
    else:
        # sklearn models (LogisticRegression, etc.) do not
        meta_model.fit(X_train, y_train)


class Ensemble(nn.Module):
    """
    Meta-learner ensemble usando stacking.

    Entrena una Logistic Regression sobre predicciones concatenadas de múltiples modelos base (ABMIL).
    Combina predicciones de diferentes modelos fundacionales para mejorar generalización.

    Args:
        xdirs (list): Rutas a directorios con predicciones de training set para cada modelo.
                     Cada directorio debe contener:
                     - labels.npy: array [N_train] con etiquetas
                     - preds.npy: array [N_train, num_classes] con predicciones
        vdirs (list): Rutas a directorios con predicciones de validation set.
        tdirs (list): Rutas a directorios con predicciones de test set.
        meta_model: Modelo sklearn para entrenar (ej: LogisticRegression(max_iter=1000))
        device (str): "cuda" o "cpu"

    Returns (en forward):
        tuple: (final_probs, tlabels)
            - final_probs: Tensor [N_test, num_classes] con probabilidades normalizadas
            - tlabels: Tensor [N_test] con true labels del test set

    Example:
        >>> ensemble = Ensemble(
        ...     xdirs=['model0/train', 'model1/train'],
        ...     vdirs=['model0/val', 'model1/val'],
        ...     tdirs=['model0/test', 'model1/test'],
        ...     meta_model=LogisticRegression(max_iter=1000),
        ...     device="cuda"
        ... )
        >>> probs, labels = ensemble()
        >>> print(probs.shape)  # [N_test, 2]
    """

    def __init__(self, xdirs, vdirs, tdirs, meta_model, device="cpu", results_subdir="ensemble4",
                 feature_space="prob", drop_redundant_class=False, model_names=None,
                 reweight_by_model_importance=False, reweight_temperature=1.0):
        super().__init__()
        self.xdirs = xdirs
        self.vdirs = vdirs
        self.tdirs = tdirs
        self.meta_model = meta_model
        self.device = device
        self.results_subdir = results_subdir
        self.feature_space = feature_space
        self.drop_redundant_class = drop_redundant_class
        self.model_names = model_names
        self.reweight_by_model_importance = reweight_by_model_importance
        self.reweight_temperature = reweight_temperature
        self.reweight_weights = None  # Dict {model: weight}, se calcula en forward()
        self.reweight_scales = None   # Array [n_features], se calcula en forward()

        # Validar inputs
        self._validate_directories()

    def _transform(self, X, num_models):
        """Aplica las transformaciones de features pedidas, idénticas en train/val/test.

        Args:
            X: [N, num_models * num_classes] de probabilidades (o embeddings sin transformación).
            num_models (int): modelos base concatenados en X.

        Returns:
            [N, D] transformado.
        """
        # Si num_models * D no divide limpiamente X.shape[1], probablemente son embeddings
        # de dimensiones variables. En ese caso, devolver sin transformar.
        if X.shape[1] % num_models != 0:
            # Dimensiones variables (probablemente embeddings) - no aplicar transformaciones
            return X

        num_classes = X.shape[1] // num_models

        if self.feature_space == "logit":
            # Espacio natural para combinar clasificadores: los log-odds. El
            # recorte evita infinitos cuando un ABMIL satura la softmax.
            p = np.clip(X, 1e-6, 1 - 1e-6)
            X = np.log(p / (1 - p))
        elif self.feature_space != "prob":
            raise ValueError(f"feature_space desconocido: {self.feature_space!r}")

        if self.drop_redundant_class and num_classes > 1:
            # Cada vector softmax suma 1, así que la última columna de cada
            # modelo es combinación lineal de las demás: en cptac_brca son 6
            # columnas con rango 4. Quitar una clase por modelo elimina esa
            # colinealidad sin perder información.
            X = X.reshape(len(X), num_models, num_classes)[:, :, :-1]
            X = X.reshape(len(X), -1)

        return X

    def _validate_directories(self):
        """Valida que todos los directorios y archivos necesarios existan."""
        all_dirs = self.xdirs + self.vdirs + self.tdirs

        for d in all_dirs:
            if not os.path.exists(d):
                raise FileNotFoundError(f"Directory not found: {d}")

            labels_path = os.path.join(d, 'labels.npy')
            preds_path = os.path.join(d, 'preds.npy')

            if not os.path.exists(labels_path):
                raise FileNotFoundError(f"Missing labels.npy in {d}")
            if not os.path.exists(preds_path):
                raise FileNotFoundError(f"Missing preds.npy in {d}")

    def _compute_reweight_scales(self, X_train, y_train, num_models):
        """
        Entrena un LogisticRegression auxiliar para determinar la importancia de
        cada modelo base, y calcula un vector de escalas para reponderar features.

        Args:
            X_train: features de entrenamiento
            y_train: labels de entrenamiento
            num_models: número de modelos base

        Returns:
            scales: array [n_features] con pesos por feature
            weights: dict {model_name/idx: weight} con importancia por modelo
        """
        from utils import model_importance_from_coefs

        # Entrenar LR auxiliar
        lr_aux = LogisticRegression(max_iter=1000)
        lr_aux.fit(X_train, y_train)

        # Calcular pesos por bloque
        num_classes = X_train.shape[1] // num_models
        weights = model_importance_from_coefs(
            lr_aux.coef_[0], num_models,
            model_names=self.model_names,
            temperature=self.reweight_temperature
        )

        # Construir vector de escalas [n_features]
        scales = np.ones(X_train.shape[1])
        for i in range(num_models):
            model_name = self.model_names[i] if self.model_names else i
            weight = weights[model_name]
            start = i * num_classes
            end = (i + 1) * num_classes
            scales[start:end] = weight

        return scales, weights

    def forward(self):
        """
        Ejecuta el ensamble combinando predicciones.

        Flujo:
        1. Carga predicciones y labels de training/val/test
        2. Valida consistencia de labels entre modelos
        3. Reshape: [num_models, N, num_classes] → [N, num_models*num_classes]
        4. Entrena Logistic Regression en training set
        5. Guarda coeficientes y odds ratios
        6. Predice en test set

        Returns:
            tuple: (final_probs, tlabels) con predicciones normalizadas
        """

        # ========== TRAINING SET ==========
        xlabels_ref, xpreds = None, []

        for idx, d in enumerate(self.xdirs):
            labels_i = torch.from_numpy(np.load(os.path.join(d, 'labels.npy'))).to(self.device)
            preds_i = torch.from_numpy(np.load(os.path.join(d, 'preds.npy'))).to(self.device)

            # Validar que todos los modelos tienen los mismos labels
            if idx == 0:
                xlabels_ref = labels_i
            else:
                if not torch.equal(xlabels_ref, labels_i):
                    raise AssertionError(f"Labels mismatch between model 0 and {idx}")

            xpreds.append(preds_i)

        # ========== VALIDATION SET ==========
        vlabels_ref, vpreds = None, []

        for idx, d in enumerate(self.vdirs):
            labels_i = torch.from_numpy(np.load(os.path.join(d, 'labels.npy'))).to(self.device)
            preds_i = torch.from_numpy(np.load(os.path.join(d, 'preds.npy'))).to(self.device)

            if idx == 0:
                vlabels_ref = labels_i
            else:
                if not torch.equal(vlabels_ref, labels_i):
                    raise AssertionError(f"Validation labels mismatch between model 0 and {idx}")

            vpreds.append(preds_i)

        # ========== RESHAPE PARA META-LEARNER ==========
        # Detectar si todos los modelos tienen la misma dimensión de features
        dims = [p.shape[1] for p in xpreds]
        all_same_dim = len(set(dims)) == 1

        if all_same_dim:
            # Caso típico: todas las predicciones/embeddings tienen la misma dimensión
            # Stack: [num_models, N_train, dim] → [N_train, num_models, dim]
            # Reshape: [N_train, num_models, dim] → [N_train, num_models*dim]
            xstacked = torch.stack(xpreds, dim=0).to(self.device)
            vstacked = torch.stack(vpreds, dim=0).to(self.device)

            num_models = xstacked.shape[0]
            X_train = xstacked.permute(1, 0, 2).reshape(xstacked.shape[1], -1).cpu().numpy()
            X_val = vstacked.permute(1, 0, 2).reshape(vstacked.shape[1], -1).cpu().numpy()
        else:
            # Caso flexible: dimensiones variables (ej. embeddings de tamaño diferente por modelo)
            # Concatenar directamente en dimensión 1: [N, dim1] + [N, dim2] + ... = [N, dim1+dim2+...]
            num_models = len(xpreds)
            X_train = torch.cat(xpreds, dim=1).cpu().numpy()  # Concatenar en dim 1 (features)
            X_val = torch.cat(vpreds, dim=1).cpu().numpy()

            print(f"⚠️  Embeddings con dimensiones variables: {dims} → concatenados a {X_train.shape[1]}")

        y_train = xlabels_ref.cpu().numpy()
        y_val = vlabels_ref.cpu().numpy()

        X_train = self._transform(X_train, num_models)
        X_val = self._transform(X_val, num_models)

        # ========== REESCALADO OPCIONAL POR IMPORTANCIA DE MODELO (PRE-ENTRENAMIENTO) ==========
        if self.reweight_by_model_importance:
            self.reweight_scales, self.reweight_weights = self._compute_reweight_scales(
                X_train, y_train, num_models
            )
            X_train = X_train * self.reweight_scales
            X_val = X_val * self.reweight_scales
            print(f"✓ Features reescaladas por importancia de modelo: {self.reweight_weights}")
        else:
            self.reweight_scales = None
            self.reweight_weights = None

        # ========== ENTRENAR META-LEARNER ==========
        _fit_meta_model(self.meta_model, X_train, y_train, X_val, y_val)

        # ========== GUARDAR ARTEFACTOS ==========
        base_path = Path(self.xdirs[0]).parent.parent.parent / self.results_subdir
        base_path.mkdir(parents=True, exist_ok=True)

        # Guardar coefs/odds ratios solo para modelos lineales (LogisticRegression)
        if hasattr(self.meta_model, "coef_"):
            coefs_path = base_path / "coefs.npy"
            odds_path = base_path / "odds_ratios.npy"

            # Coeficientes del meta-learner
            coefs = np.array(self.meta_model.coef_[0])
            odds = np.exp(coefs)

            # Append o create
            if coefs_path.exists():
                old = np.load(coefs_path)
                new = np.vstack([old, coefs])
            else:
                new = np.array([coefs])
            np.save(coefs_path, new)

            if odds_path.exists():
                old = np.load(odds_path)
                new = np.vstack([old, odds])
            else:
                new = np.array([odds])
            np.save(odds_path, new)

        # Guardar pesos de reweighting si se aplicaron
        if self.reweight_weights is not None:
            import json
            # Guardar como JSON para legibilidad
            weights_path = base_path / "model_importance_weights.json"
            with open(weights_path, "w") as f:
                json.dump(self.reweight_weights, f, indent=2)
            print(f"✓ Model importance weights guardados en {weights_path}")

        # ========== TEST SET ==========
        tlabels_ref, tpreds = None, []

        for idx, d in enumerate(self.tdirs):
            labels_i = torch.from_numpy(np.load(os.path.join(d, 'labels.npy'))).to(self.device)
            preds_i = torch.from_numpy(np.load(os.path.join(d, 'preds.npy'))).to(self.device)

            if idx == 0:
                tlabels_ref = labels_i

            tpreds.append(preds_i)

        # Aplicar la misma lógica de reshape (flexible)
        if all_same_dim:
            tstacked = torch.stack(tpreds, dim=0).to(self.device)
            N_test = tstacked.shape[1]
            X_test = tstacked.permute(1, 0, 2).reshape(N_test, num_models * dims[0]).cpu().numpy()
        else:
            N_test = tpreds[0].shape[0]
            X_test = torch.cat(tpreds, dim=1).cpu().numpy()

        X_test = self._transform(X_test, num_models)

        # ========== REESCALADO OPCIONAL POR IMPORTANCIA DE MODELO (POST-ENTRENAMIENTO) ==========
        # Se aplica después de haber entrenado el meta-modelo, para rescalar X_test
        # con los mismos pesos que se usaron en train/val
        if self.reweight_by_model_importance and self.reweight_scales is not None:
            X_test = X_test * self.reweight_scales

        # ========== PREDICCIÓN ==========
        probs = self.meta_model.predict_proba(X_test)
        final_probs = torch.from_numpy(probs).to(self.device).float()

        return final_probs, tlabels_ref


def main(foundational_models, work_dir, train_source, tissue_patching, task_name,
         meta_model="logreg", meta_hidden_dim=16, meta_dropout=0.1, meta_epochs=None,
         meta_lr=1e-3, meta_wd=1e-4, meta_patience=8, n_cycles=6, restart_lr=5e-3,
         n_snapshots_ensemble=None, results_subdir_suffix="", base_source="standard",
         fge_n_cycles=6, fge_cycle_length=3, fge_lr_1=1e-3, fge_lr_2=1e-5,
         fge_cycle_patience=2, meta_features="insample", feature_space="prob",
         drop_redundant_class=False, n_members=6, meta_svm_c=1.0, meta_svm_kernel="linear",
         meta_knn_k=5, reweight_by_model_importance=False, reweight_temperature=1.0):
    """
    Ejecuta ensamble meta-learner para múltiples folds.

    Args:
        foundational_models (list): Lista de modelos base (ej: ['ctranspath', 'uni_v2'])
        work_dir (str): Directorio raíz del proyecto
        train_source (str): Dataset (ej: 'cptac_brca')
        tissue_patching (str): Estrategia de patching (ej: '20x_224px_0px_overlap')
        task_name (str): Tarea de clasificación (ej: 'TP53_mutation')
        meta_model (str): Tipo de meta-learner ('logreg', 'mlp', 'mlp_snapshot', 'mlp_fge',
            'tabpfn'). Default: 'logreg'
        meta_hidden_dim (int): Ancho capa oculta (MLP). Default: 16
        meta_dropout (float): Dropout (MLP). Default: 0.1
        meta_epochs (int): Épocas (None=auto: 100 mlp, 120 mlp_snapshot). Default: None
        meta_lr (float): LR inicial (solo mlp). Default: 1e-3
        meta_wd (float): Weight decay. Default: 1e-4
        meta_patience (int): Early stopping patience (solo mlp). Default: 8
        n_cycles (int): Número ciclos (mlp_snapshot). Default: 6
        restart_lr (float): LR reinicio (mlp_snapshot). Default: 5e-3
        n_snapshots_ensemble (int): Snapshots a promediar (None=todos). Default: None
        base_source (str): 'standard' (ABMIL base sin ciclos, checkpoint único) o
            'fge' (ABMIL base diversificado con Fast Geometric Ensembles, ver
            train_abmil_fge.py/test_abmil_fge.py). Default: 'standard'
        fge_n_cycles, fge_cycle_length, fge_lr_1, fge_lr_2, fge_cycle_patience:
            hiperparámetros de FGEMLPMetaClassifier (solo meta_model='mlp_fge').
    """
    # Auto-set epochs if not provided
    if meta_epochs is None:
        meta_epochs = 120 if meta_model == "mlp_snapshot" else 100

    # Construir rutas de directorios base
    dirs = [
        os.path.join(work_dir, train_source, task_name, 'abmil', f'{f}_{tissue_patching}')
        for f in foundational_models
    ]

    # base_source es libre: "standard" usa los árboles sin sufijo; cualquier otro
    # valor (fge, fge_wbase, fge_lowlr...) es la etiqueta de una variante FGE y
    # debe coincidir con el --fge_tag usado en train/test_abmil_fge.py.
    assert base_source, "base_source no puede estar vacío"
    # logit_avg y gating tratan cada bloque de columnas como una distribución
    # completa sobre clases (la reconstruyen con _as_log_probs). Quitar una
    # clase rompe esa semántica en silencio en vez de fallar.
    if drop_redundant_class and meta_model in ("logit_avg", "gating"):
        raise ValueError(
            f"--drop_redundant_class es incompatible con --meta_model {meta_model}: "
            f"necesita la distribución completa sobre clases de cada modelo base."
        )

    # Sufijo de rutas: bases FGE se guardan en árboles paralelos
    # (val_outputs_fge/test_outputs_fge/_train_eval_fge) que no pisan las
    # predicciones del ABMIL sin ciclos — ver train_abmil_fge.py/test_abmil_fge.py.
    suffix = "" if base_source == "standard" else f"_{base_source}"

    # Sufijo independiente para las features del meta-learner. Las predicciones
    # de test_abmil.py son in-sample (los modelos base valen AUC 0.94 sobre
    # ellas frente a 0.77 en test), lo que penaliza a los meta-modelos
    # flexibles; build_oof_features.py genera la alternativa out-of-fold.
    # Cualquier valor distinto de "insample" es la etiqueta de un árbol
    # out-of-fold generado por build_oof_features.py --oof_tag (oof, oof5,
    # oof10...), lo que permite comparar varios n_inner entre sí.
    if meta_features == "insample":
        xsuffix = suffix
    else:
        if base_source != "standard":
            raise ValueError(
                f"--meta_features {meta_features} sólo está disponible con "
                f"--base_source standard: no se han generado features out-of-fold "
                f"para las variantes de snapshots."
            )
        xsuffix = f"_{meta_features}"

    xdirs = []
    vdirs = []
    tdirs = []
    folds = []

    # Detectar número de folds
    for idx, d in enumerate(dirs):
        val_outputs_path = os.path.join(d, f'val_outputs{suffix}')

        if any(name.startswith("fold_") for name in os.listdir(val_outputs_path)):
            # Multi-fold: busca fold_* subdirectories
            folds.append(len(os.listdir(val_outputs_path)))
        else:
            # Single fold (solo un modelo o un fold)
            folds.append(1)

        # test_abmil_fge.py escribe en {..}_train_eval_fge/val_outputs_fge/, así que
        # el sufijo aplica también al subdirectorio, no sólo al dir padre.
        xdirs.append(os.path.join(f'{d}_train_eval{xsuffix}', f'val_outputs{xsuffix}'))
        # Cuando meta_features != "insample" (ej. embeddings), validación y test también usan el sufijo
        # porque test_abmil.py genera val_outputs_embeddings y test_outputs_embeddings
        vdirs.append(os.path.join(d, f'val_outputs{xsuffix if xsuffix else suffix}'))
        tdirs.append(os.path.join(d, f'test_outputs{xsuffix if xsuffix else suffix}'))

    assert len(set(folds)) == 1, f"Mismatch in number of folds: {set(folds)}"

    n_folds = folds[0]
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Mapeo de meta-model a nombre de subdirectorio de resultados
    results_subdir_map = {
        "logreg": "ensemble4",
        "mlp": "ensemble4_mlp",
        "mlp_snapshot": "ensemble4_mlp_snapshot",
        "mlp_fge": "ensemble4_mlp_fge",
        "deep_mlp": "ensemble4_deep_mlp",
        "deep_mlp_snapshot": "ensemble4_deep_mlp_snapshot",
        "tabpfn": "ensemble4_tabpfn",
        "tabpfn_snapshot": "ensemble4_tabpfn_snapshot",
        "logit_avg": "ensemble4_logit_avg",
        "mlp_deepens": "ensemble4_mlp_deepens",
        "gating": "ensemble4_gating",
        "lightgbm": "ensemble4_lightgbm",
        "svm": "ensemble4_svm",
        "knn": "ensemble4_knn",
        "nb": "ensemble4_nb",
    }
    results_subdir = results_subdir_map[meta_model]

    # Las variantes de features escriben en su propio subdirectorio: comparar
    # regímenes exige que los resultados de cada uno se conserven por separado.
    if meta_features != "insample":
        results_subdir = f"{results_subdir}_{meta_features}"
    if feature_space != "prob":
        results_subdir = f"{results_subdir}_{feature_space}"
    if drop_redundant_class:
        results_subdir = f"{results_subdir}_dropc"

    # Bases FGE -> resultados en subdirectorio propio (no pisa base_source=standard)
    if base_source != "standard":
        results_subdir = f"{results_subdir}_{base_source}"

    # Append suffix if provided (for grid search trials)
    if results_subdir_suffix:
        results_subdir = f"{results_subdir}{results_subdir_suffix}"

    all_preds = []
    all_labels = []

    print(f"\n\nEvaluating ensemble with meta-model: {meta_model}")
    print(f"Results will be saved to: .../{results_subdir}/")

    # coefs.npy y odds_ratios.npy se acumulan fold a fold con np.vstack. Si quedan
    # de una corrida anterior con distinto nº de modelos base, el vstack revienta
    # (p.ej. 5 modelos x 4 clases = 20 columnas contra 3 x 4 = 12). Se borran aquí
    # para que cada corrida empiece limpia en vez de heredar la forma antigua.
    coefs_dir = Path(work_dir) / train_source / task_name / 'abmil' / results_subdir
    for stale in ('coefs.npy', 'odds_ratios.npy'):
        (coefs_dir / stale).unlink(missing_ok=True)
    with tqdm(total=n_folds, unit="fold") as pbar:
        for f in range(n_folds):
            pbar.set_description(f"Evaluating fold {f+1}/{n_folds}")

            # Rutas para este fold
            xdirs_fold = [os.path.join(d, f'fold_{f}') if n_folds > 1 else d for d in xdirs]
            vdirs_fold = [os.path.join(d, f'fold_{f}') if n_folds > 1 else d for d in vdirs]
            tdirs_fold = [os.path.join(d, f'fold_{f}') if n_folds > 1 else d for d in tdirs]

            try:
                # Instantiate meta-learner based on meta_model choice
                if meta_model == "logreg":
                    meta_learner = LogisticRegression(max_iter=1000)
                elif meta_model == "mlp":
                    meta_learner = MLPMetaClassifier(
                        hidden_dim=meta_hidden_dim,
                        dropout=meta_dropout,
                        epochs=meta_epochs,
                        lr=meta_lr,
                        wd=meta_wd,
                        patience=meta_patience,
                        device=device,
                        seed=42 + f
                    )
                elif meta_model == "mlp_snapshot":
                    meta_learner = SnapshotMLPMetaClassifier(
                        hidden_dim=meta_hidden_dim,
                        dropout=meta_dropout,
                        epochs=meta_epochs,
                        restart_lr=restart_lr,
                        n_cycles=n_cycles,
                        n_snapshots=n_snapshots_ensemble,
                        wd=meta_wd,
                        device=device,
                        seed=42 + f
                    )
                elif meta_model == "mlp_fge":
                    meta_learner = FGEMLPMetaClassifier(
                        hidden_dim=meta_hidden_dim,
                        dropout=meta_dropout,
                        warmup_epochs=meta_epochs,
                        warmup_lr=meta_lr,
                        warmup_wd=meta_wd,
                        warmup_patience=meta_patience,
                        n_cycles=fge_n_cycles,
                        cycle_length=fge_cycle_length,
                        lr_1=fge_lr_1,
                        lr_2=fge_lr_2,
                        n_snapshots=n_snapshots_ensemble,
                        wd=meta_wd,
                        cycle_patience=fge_cycle_patience,
                        device=device,
                        seed=42 + f
                    )
                elif meta_model == "logit_avg":
                    meta_learner = LogitAveragingMetaClassifier(
                        n_models=len(foundational_models),
                        input_is_prob=(feature_space == "prob"),
                        device=device,
                        seed=42 + f
                    )
                elif meta_model == "mlp_deepens":
                    meta_learner = DeepEnsembleMLPMetaClassifier(
                        n_members=n_members,
                        hidden_dim=meta_hidden_dim,
                        dropout=meta_dropout,
                        epochs=meta_epochs,
                        lr=meta_lr,
                        wd=meta_wd,
                        patience=meta_patience,
                        device=device,
                        seed=42 + f
                    )
                elif meta_model == "gating":
                    meta_learner = GatingMLPMetaClassifier(
                        n_models=len(foundational_models),
                        input_is_prob=(feature_space == "prob"),
                        dropout=meta_dropout,
                        epochs=meta_epochs,
                        lr=meta_lr,
                        wd=meta_wd,
                        patience=meta_patience,
                        device=device,
                        seed=42 + f
                    )
                elif meta_model == "lightgbm":
                    # Import local, igual que tabpfn: no rompe entornos sin la librería.
                    from lightgbm import LGBMClassifier
                    meta_learner = LGBMClassifier(
                        n_estimators=200, learning_rate=0.05, num_leaves=7,
                        min_child_samples=5, subsample=0.8, colsample_bytree=0.8,
                        random_state=42 + f, verbose=-1
                    )
                elif meta_model == "tabpfn":
                    # Import local: no rompe entornos donde tabpfn no esté instalado
                    # (p.ej. grid_search_mlp.py / ablation_study_mlp.py, que no lo usan).
                    from tabpfn import TabPFNClassifier
                    meta_learner = TabPFNClassifier(device=device, random_state=42 + f)
                elif meta_model == "deep_mlp":
                    meta_learner = DeepMLPMetaClassifier(
                        lr=meta_lr,
                        wd=meta_wd,
                        epochs=meta_epochs,
                        patience=meta_patience,
                        device=device
                    )
                elif meta_model == "deep_mlp_snapshot":
                    meta_learner = SnapshotDeepMLPMetaClassifier(
                        lr=meta_lr,
                        wd=meta_wd,
                        epochs=meta_epochs,
                        n_cycles=n_cycles,
                        restart_lr=restart_lr,
                        n_snapshots=n_snapshots_ensemble,
                        device=device
                    )
                elif meta_model == "tabpfn_snapshot":
                    meta_learner = SnapshotTabPFNMetaClassifier(
                        device=device,
                        n_snapshots=4
                    )
                elif meta_model == "svm":
                    meta_learner = SVC(
                        probability=True,
                        kernel=meta_svm_kernel,
                        C=meta_svm_c,
                        random_state=42 + f
                    )
                elif meta_model == "knn":
                    meta_learner = KNeighborsClassifier(
                        n_neighbors=meta_knn_k,
                        weights="distance"
                    )
                elif meta_model == "nb":
                    meta_learner = GaussianNB()
                else:
                    raise ValueError(f"Unknown meta_model: {meta_model}")

                preds, labels = Ensemble(
                    xdirs_fold,
                    vdirs_fold,
                    tdirs_fold,
                    meta_learner,
                    device=device,
                    results_subdir=results_subdir,
                    feature_space=feature_space,
                    drop_redundant_class=drop_redundant_class,
                    model_names=foundational_models,
                    reweight_by_model_importance=reweight_by_model_importance,
                    reweight_temperature=reweight_temperature
                )()

                all_preds.append(preds.detach().cpu().numpy())
                all_labels.append(labels.detach().cpu().numpy())

            except Exception as e:
                tqdm.write(f"❌ Error en fold {f}: {e}")
                raise

            pbar.update(1)

    tqdm.write("🔎 Computing metrics across all folds...")
    Metrics(
        task_type='classification',
        model_kwargs={'num_classes': len(set(all_labels[0]))},
        num_bootstraps=100,
        results_dir=os.path.join(work_dir, train_source, task_name, 'abmil', results_subdir),
        split='test',
        num_folds=n_folds,
        all_labels_across_folds=all_labels,
        all_preds_across_folds=all_preds,
    ).run()

    tqdm.write("✅ Metrics computation finished.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ensemble meta-learner evaluation")
    parser.add_argument("--foundational_models", nargs='+', type=str, required=True,
                       help="List of base models (e.g., ctranspath uni_v2 virchow)")
    parser.add_argument("--work_dir", type=str,
                       default="/home/JKP6679/Patho-Ensemble/PARADIS/datos/patches",
                       help="Root working directory")
    parser.add_argument("--train_source", type=str, required=True,
                       help="Dataset name (e.g., cptac_brca)")
    parser.add_argument("--tissue_patching", type=str, required=True,
                       help="Patching strategy (e.g., 20x_224px_0px_overlap)")
    parser.add_argument("--task_name", type=str, required=True,
                       help="Task name (e.g., TP53_mutation)")

    # Meta-learner selection and hyperparameters
    parser.add_argument("--meta_model", type=str,
                       choices=["logreg", "mlp", "mlp_snapshot", "mlp_fge", "deep_mlp",
                                "deep_mlp_snapshot", "tabpfn", "tabpfn_snapshot",
                                "logit_avg", "mlp_deepens", "gating", "lightgbm",
                                "svm", "knn", "nb"],
                       default="logreg",
                       help="Meta-learner type. Default: logreg (for backward compatibility)")
    parser.add_argument("--meta_features", type=str, default="insample",
                       help="Origen de las features del meta-learner. 'insample' usa las "
                            "predicciones de test_abmil.py sobre el propio train del fold; "
                            "cualquier otro valor es el --oof_tag de un árbol generado por "
                            "build_oof_features.py (oof, oof5, oof10...). Default: insample.")
    parser.add_argument("--feature_space", type=str,
                       choices=["prob", "logit"], default="prob",
                       help="Espacio en el que se combinan las salidas base: probabilidades "
                            "o log-odds. Default: prob.")
    parser.add_argument("--drop_redundant_class", action="store_true",
                       help="Elimina una columna softmax por modelo base. Cada vector suma 1, "
                            "así que esa columna es redundante (en cptac_brca, 6 columnas de "
                            "rango 4).")
    parser.add_argument("--n_members", type=int, default=6,
                       help="Miembros del ensemble de semillas independientes (mlp_deepens).")
    parser.add_argument("--base_source", type=str,
                       default="standard",
                       help="Origen de las predicciones base ABMIL: 'standard' (checkpoint "
                            "único, como hoy) o 'fge' (bases diversificadas por Fast "
                            "Geometric Ensembles, ver train_abmil_fge.py). Default: standard")
    parser.add_argument("--meta_hidden_dim", type=int, default=16,
                       help="Hidden layer dimension for MLP models. Default: 16")
    parser.add_argument("--meta_dropout", type=float, default=0.1,
                       help="Dropout probability for MLP models. Default: 0.1")
    parser.add_argument("--meta_epochs", type=int, default=None,
                       help="Training epochs (auto: 100 for mlp, 120 for mlp_snapshot). Default: None")
    parser.add_argument("--meta_lr", type=float, default=1e-3,
                       help="Initial learning rate for MLPMetaClassifier. Default: 1e-3")
    parser.add_argument("--meta_wd", type=float, default=1e-4,
                       help="Weight decay for MLP models. Default: 1e-4")
    parser.add_argument("--meta_patience", type=int, default=8,
                       help="Early stopping patience for MLPMetaClassifier. Default: 8")
    parser.add_argument("--n_cycles", type=int, default=6,
                       help="Number of cosine annealing cycles (mlp_snapshot only). Default: 6")
    parser.add_argument("--restart_lr", type=float, default=5e-3,
                       help="LR restart value at cycle start (mlp_snapshot only). Default: 5e-3")
    parser.add_argument("--n_snapshots_ensemble", type=int, default=None,
                       help="Number of snapshots to average (mlp_snapshot/mlp_fge; None=all). Default: None")
    parser.add_argument("--fge_n_cycles", type=int, default=6,
                       help="Number of FGE cycles after warm-up (mlp_fge only). Default: 6")
    parser.add_argument("--fge_cycle_length", type=int, default=3,
                       help="Epochs per FGE cycle, short vs mlp_snapshot (mlp_fge only). Default: 3")
    parser.add_argument("--fge_lr_1", type=float, default=1e-3,
                       help="FGE cycle ceiling LR (mlp_fge only). Default: 1e-3")
    parser.add_argument("--fge_lr_2", type=float, default=1e-5,
                       help="FGE cycle floor LR / snapshot LR (mlp_fge only). Default: 1e-5")
    parser.add_argument("--fge_cycle_patience", type=int, default=2,
                       help="Cycles without val AUC improvement before stopping FGE (mlp_fge only). Default: 2")
    parser.add_argument("--results_subdir_suffix", type=str, default="",
                       help="Suffix for results directory (e.g., '_trial_1' → ensemble4_mlp_trial_1). Default: ''")
    parser.add_argument("--meta_svm_c", type=float, default=1.0,
                       help="Regularization parameter C for SVM (svm only). Default: 1.0")
    parser.add_argument("--meta_svm_kernel", type=str, choices=["linear", "rbf"], default="linear",
                       help="Kernel type for SVM (svm only). Default: linear")
    parser.add_argument("--meta_knn_k", type=int, default=5,
                       help="Number of neighbors for KNN (knn only). Default: 5")
    parser.add_argument("--reweight_by_model_importance", action="store_true",
                       help="Reweight features by model importance (via auxiliary LogisticRegression). Default: False")
    parser.add_argument("--reweight_temperature", type=float, default=1.0,
                       help="Softening exponent for reweighting (0=no-op, 1=linear, >1=accentuate). Default: 1.0")

    args = parser.parse_args()
    main(**vars(args))
