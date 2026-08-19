import os
import glob
from tqdm import tqdm
import json
import pandas as pd
import h5py
from contextlib import ExitStack
import numpy as np
from sklearn.metrics import f1_score, roc_auc_score, log_loss
from optuna import create_study
import optuna
import pygad
import matplotlib.pyplot as plt
from optuna.samplers import TPESampler
from optuna.trial import TrialState
from optuna.distributions import FloatDistribution

from patho_bench.experiments.utils.ClassificationMixin import ClassificationMixin
from patho_bench.experiments.BaseExperiment import BaseExperiment


def get_features_dir(work_dir, train_source, foundational_model, features_root=None):
    """
    Resuelve el directorio real de features HDF5 para un foundational_model,
    sin asumir ni crear symlinks.

    En PARADIS los embeddings y los splits NO cuelgan del mismo raíz: los
    features viven en `datos/features/{dataset}` mientras que los splits y los
    resultados viven en `datos/patches/{dataset}/{task}`. Por eso se prueban dos
    bases: `{work_dir}/features` (convención antigua, work_dir=.../datos) y
    `{work_dir}/../features` (work_dir=.../datos/patches). Dentro de cada base se
    prueba primero la convención con sufijo `_monai` y luego la que no lo lleva
    (p.ej. cptac_brca tiene `features_uni_v2` sin sufijo).

    Args:
        work_dir (str): Directorio raíz de trabajo (ej. PARADIS/datos/patches).
        train_source (str): Dataset (ej. cptac_brca).
        foundational_model (str): Nombre del modelo (ej. uni_v2, ctranspath).
        features_root (str, optional): Raíz explícita de features. Si se indica,
            se usa sólo esa y se ignora la deducción a partir de work_dir.

    Returns:
        str: Ruta absoluta a la carpeta de features existente.

    Raises:
        FileNotFoundError: si ninguna combinación existe en disco.
    """
    if features_root:
        bases = [f"{features_root}/{train_source}"]
    else:
        parent = os.path.dirname(os.path.normpath(work_dir))
        bases = [
            f"{work_dir}/features/{train_source}",   # work_dir=.../datos
            f"{parent}/features/{train_source}",     # work_dir=.../datos/patches
        ]

    tried = []
    for base in bases:
        for suffix in ("_monai", ""):
            path = f"{base}/features_{foundational_model}{suffix}"
            tried.append(path)
            if os.path.isdir(path):
                return path

    raise FileNotFoundError(
        "No se encontró carpeta de features para "
        f"foundational_model='{foundational_model}' en train_source='{train_source}'. "
        "Rutas probadas:\n  - " + "\n  - ".join(tried)
    )


def detect_latent_dim(features_dir):
    """
    Lee la dimensión real de embedding del primer .h5 del directorio.

    Evita el error recurrente de pasar un --latent_dim que no coincide con los
    embeddings (768 vs 1536 vs 2560), que se manifiesta como un size mismatch al
    construir o cargar el modelo.

    Args:
        features_dir (str): Carpeta devuelta por get_features_dir().

    Returns:
        int: Dimensión del embedding (features.shape[1]).

    Raises:
        FileNotFoundError: si la carpeta no contiene ningún .h5.
    """
    h5_files = sorted(glob.glob(f"{features_dir}/*.h5"))
    if not h5_files:
        raise FileNotFoundError(f"Sin ficheros .h5 en {features_dir}")
    with h5py.File(h5_files[0], "r") as f:
        return int(f["features"].shape[1])


def resolve_latent_dim(work_dir, train_source, foundational_model,
                       latent_dim=None, features_root=None):
    """
    Devuelve la dimensión de embedding a usar: la explícita si se pasó, o la
    auto-detectada del .h5 en caso contrario.

    Pensado para llamarse una vez al principio de main(), de modo que
    --latent_dim sea opcional en la CLI sin cambiar la firma de
    load_features_cached().
    """
    if latent_dim is not None:
        return latent_dim
    feats_dir = get_features_dir(work_dir, train_source, foundational_model, features_root)
    dim = detect_latent_dim(feats_dir)
    print(f"  latent_dim auto-detectado: {dim}  (de {os.path.basename(feats_dir)})")
    return dim


class Metrics(ClassificationMixin, BaseExperiment):
    def __init__(self,
                 task_type: str,
                 model_kwargs: dict,
                 num_bootstraps: int,
                 results_dir: str,
                 split: str,
                 num_folds: int,
                 all_labels_across_folds: list,
                 all_preds_across_folds: list):
        """
        Base class for all experiments.

        Args:
            task_type (str): Type of task. Can be 'classification' or 'survival'.
            dataset (BaseDataset): Dataset object
            batch_size (int): Batch size.
            model_constructor (callable): Model class which can be called to create model instance.
            model_kwargs: Arguments passed to model_constructor.
            num_epochs (int): Number of epochs.
            accumulation_steps (int): Number of batches to accumulate gradients over before stepping optimizer.
            optimizer_config: Optimizer config.
            scheduler_config: LR scheduler config.
            save_which_checkpoints (str): Mode of saving checkpoints.
            num_bootstraps (int): Number of bootstraps to use for computing 95% CI.
            precision (torch.dtype): Precision to use for training.
            device (str): Device to use for training.
            results_dir (str): Where to save results.
            view_progress (str, optional): How to log progress. Can be 'bar' or 'verbose'. Defaults to 'bar'.
            lr_logging_interval (int, optional): Interval at which to log learning rate to dashboard (in number of accumulation steps). Defaults to None (do not log).
            seed (int): Seed for reproducibility.
            **kwargs: Additional arguments to save in config.json
        """
        self.task_type = task_type
        self.model_kwargs = model_kwargs
        self.num_bootstraps = num_bootstraps
        self.results_dir = results_dir
        self.split = split
        self.num_folds = num_folds
        self.all_labels_across_folds = all_labels_across_folds
        self.all_preds_across_folds = all_preds_across_folds

    def run(self):
        labels = []
        preds = []
        scores = []

        if self.num_folds == 1:
            # If only one fold or one sample per fold, will save results at end across all folds
            labels = self.all_labels_across_folds
            preds = self.all_preds_across_folds
        else:
            # If multiple folds and multiple samples per fold, save per-fold results
            for f in range(self.num_folds):
                fold_labels = self.all_labels_across_folds[f]
                fold_preds = self.all_preds_across_folds[f]

                # Skip empty folds (e.g., folds with no test data)
                if len(fold_labels) == 0 or len(fold_preds) == 0:
                    print(f"  Skipping fold {f} (empty: {len(fold_labels)} labels, {len(fold_preds)} preds)")
                    continue

                per_fold_save_dir = os.path.join(self.results_dir, f'{self.split}_metrics', f'fold_{f}')
                scores.append(self._compute_metrics(fold_labels, fold_preds, per_fold_save_dir))

        # After collecting all folds, either do bootstrapping or an average across folds
        summary = self._finalize_metrics(self.split, labels, preds, scores)

        with open(os.path.join(self.results_dir, f'{self.split}_metrics_summary.json'), 'w') as f:
            json.dump(summary, f, indent=4)

    def _compute_metrics(self, labels, preds, save_dir):
        """
        Save metrics to file and return a dictionary of metrics.

        Args:
            labels (np.array or dict): Ground truth labels
            preds (np.array): Predictions
            save_dir (str): Directory to save metrics to
        """
        if self.task_type == 'classification':
            self.auc_roc(labels, preds, self.model_kwargs['num_classes'], saveto = os.path.join(save_dir, "roc_curves.png"))
            self.confusion_matrix(labels, preds, self.model_kwargs['num_classes'], saveto = os.path.join(save_dir, "confusion_matrices.png"))
            self.precision_recall(labels, preds, self.model_kwargs['num_classes'], saveto = os.path.join(save_dir, "pr_curves.png"))
            scores = self.classification_metrics(labels, preds, self.model_kwargs['num_classes'], saveto = os.path.join(save_dir, "metrics.json"))
            return scores['overall']

    def _finalize_metrics(self, split, labels_across_folds, preds_across_folds, scores_across_folds):
        """
        Combine per-fold results or do bootstrapping if single fold

        Arguments:
            split (str): Split name ('val' or 'test')
            labels_across_folds (list): List of labels across folds
            preds_across_folds (list): List of predictions across folds
            scores_across_folds (list): List of scores across folds

        Returns:
            summary (dict): Dictionary of summary metrics
        """
        if len(labels_across_folds) > 0:
            # Perform bootstrapping and calculate 95% CI
            bootstraps = self.bootstrap(labels_across_folds, preds_across_folds, self.num_bootstraps)
            if self.task_type == 'classification':
                scores_across_folds = [self.classification_metrics(labels, preds, self.model_kwargs['num_classes'])['overall'] for labels, preds in tqdm(bootstraps, desc=f'Computing {self.num_bootstraps} bootstraps')]

            # Save bootstraps
            folder_path = os.path.join(self.results_dir, f"{split}_metrics")
            os.makedirs(folder_path, exist_ok=True)
            for idx, metrics_dict in enumerate(scores_across_folds):
                folder_path_curr = os.path.join(folder_path, f"bootstrap_{idx}")
                os.makedirs(folder_path_curr, exist_ok=True)

                file_path = os.path.join(folder_path_curr, "metrics.json")
                with open(file_path, "w") as f:
                    json.dump(metrics_dict, f, indent=4)

            return self.get_95_ci(scores_across_folds)
        elif len(scores_across_folds) > 0:
            # Report mean ± SE across folds (if we have per-fold scores)
            return self.get_mean_se(scores_across_folds)
        else:
            # No data available — return empty summary
            print(f"⚠️  No {split} data available to compute metrics")
            return {}


class MetricDistance:
    def __init__(self,
                 foundational_models: list,
                 work_dir: str,
                 train_source: str,
                 tissue_patching: str,
                 task_name: str,
                 metric: str,
                 fold: int):
        self.metric = metric
        self.fold = fold
        self.work_dir = work_dir
        self.train_source = train_source
        self.tissue_patching = tissue_patching
        self.task_name = task_name
        self.foundational_models = foundational_models

        # Try val_metrics first, fall back to test_metrics if not found
        self.metric_type = 'val'
        self.dirs = [os.path.join(work_dir, train_source, task_name, "abmil", f'{f}_{tissue_patching}', 'val_metrics') for f in foundational_models]

        # Check if any val_metrics directories exist
        if not any(os.path.exists(d) for d in self.dirs):
            # Fall back to test_metrics
            print(f"  ℹ️  No validation metrics found, using test metrics for weighting")
            self.metric_type = 'test'
            self.dirs = [os.path.join(work_dir, train_source, task_name, "abmil", f'{f}_{tissue_patching}', 'test_metrics') for f in foundational_models]

        for d in self.dirs:
            if not os.path.exists(d):
                raise FileNotFoundError(f"Missing path: {d}")

    def _map_metric_name(self, metric_key):
        """Map metric name variations to actual JSON keys."""
        # Map user-friendly names to actual metric keys
        metric_mapping = {
            'auc_roc': ['macro-ovr-auc', 'macro-ovo-auc', 'roc_auc', 'auc-roc'],
            'auc-roc': ['macro-ovr-auc', 'macro-ovo-auc', 'roc_auc', 'auc_roc'],
            'f1': ['macro-f1', 'weighted-f1', 'f1-score'],
            'accuracy': ['acc', 'accuracy'],
        }

        # If exact match exists, use it
        if metric_key in metric_mapping:
            return metric_mapping[metric_key]
        # Otherwise return the key as-is (might be exact match)
        return [metric_key]

    def _get_metric_value(self, metrics_dict, metric_key):
        """Extract metric value from metrics dict, trying multiple possible keys."""
        overall = metrics_dict.get("overall", {})

        # Try direct key first
        if metric_key in overall:
            return overall[metric_key]

        # Try mapped variations
        possible_keys = self._map_metric_name(metric_key)
        for key in possible_keys:
            if key in overall:
                return overall[key]

        # If nothing found, raise informative error
        available_keys = list(overall.keys())
        raise KeyError(f"Metric '{metric_key}' not found. Available metrics: {available_keys}")

    def run(self):
        fold_values = []
        # Show which metric will be used (for debugging - print only once)
        possible_keys = self._map_metric_name(self.metric)

        for model_idx, dir in enumerate(self.dirs):
            if model_idx == 0:  # Print only for first model to avoid spam
                print(f"  Using metric: {self.metric} (will try: {possible_keys})")
            metrics_path = os.path.join(dir, f"fold_{self.fold}/metrics.json")

            # Try fold-specific metrics first
            if os.path.exists(metrics_path):
                try:
                    with open(metrics_path, "r") as f:
                        metrics = json.load(f)
                    value = self._get_metric_value(metrics, self.metric)
                    fold_values.append(value)
                except KeyError as e:
                    print(f"  ⚠️  Error in {dir}: {e}")
                    print(f"       Using equal weight for this fold")
                    fold_values.append(1.0)
            else:
                # Try bootstrap metrics
                metrics_path = os.path.join(dir, f"bootstrap_{self.fold}/metrics.json")
                if os.path.exists(metrics_path):
                    try:
                        with open(metrics_path, "r") as f:
                            metrics = json.load(f)
                        value = self._get_metric_value(metrics, self.metric)
                        fold_values.append(value)
                    except KeyError as e:
                        print(f"  ⚠️  Error in {dir}: {e}")
                        print(f"       Using equal weight for this fold")
                        fold_values.append(1.0)
                else:
                    # Try summary file as fallback
                    summary_path = os.path.join(dir, f"{self.metric_type}_metrics_summary.json")
                    if os.path.exists(summary_path):
                        try:
                            with open(summary_path, "r") as f:
                                metrics = json.load(f)
                            # Try to extract mean value from summary
                            possible_keys = self._map_metric_name(self.metric)
                            found = False
                            for key in possible_keys:
                                if key in metrics and "mean" in metrics[key]:
                                    fold_values.append(metrics[key]["mean"])
                                    found = True
                                    break
                            if not found:
                                print(f"  ⚠️  Could not extract metric from {summary_path}, using equal weight")
                                fold_values.append(1.0)
                        except Exception as e:
                            print(f"  ⚠️  Error reading {summary_path}: {e}")
                            fold_values.append(1.0)
                    else:
                        print(f"  ⚠️  No metrics found in {dir} for fold {self.fold}, using equal weight")
                        fold_values.append(1.0)

        fold_values = np.array(fold_values, dtype=float)

        # Compute weights using softmax normalization
        if fold_values.sum() == 0 or np.all(fold_values == 1.0):
            # All values are 0, NaN, or all fallback values — use equal weights
            weights = np.ones_like(fold_values) / len(fold_values)
        else:
            # Normalize using min-max to [0, 1] then softmax
            min_val = fold_values.min()
            max_val = fold_values.max()
            if max_val > min_val:
                normalized = (fold_values - min_val) / (max_val - min_val)
            else:
                normalized = fold_values
            weights = normalized / normalized.sum()

        return np.array(weights)


def model_importance_from_coefs(coefs, num_models, model_names=None, temperature=1.0):
    """
    Agrega coeficientes de un modelo lineal (coef_) por bloque de modelo base
    y devuelve pesos normalizados por modelo (media 1.0 = neutro).

    Calcula la norma L2 de los coeficientes de cada modelo base y normaliza
    para que el promedio sea 1.0, permitiendo interpretar qué modelo pesa más
    en las predicciones del meta-clasificador lineal.

    Args:
        coefs: array de coeficientes
            - Si 1D: [n_features] — coeficientes de un fold individual
            - Si 2D: [n_folds, n_features] — acumulado de múltiples folds
        num_models (int): número de modelos base concatenados en las features.
            block_size = n_features // num_models
        model_names (list, optional): lista de nombres de modelos en orden.
            Si se proporciona, devuelve dict {name: weight}; si no, dict {idx: weight}.
        temperature (float): exponente de suavizado
            - 0.0: todos los pesos = 1.0 (no-op)
            - 1.0 (default): lineal, pesos proporcionales a norms
            - >1.0: acentúa diferencias (winner-take-more)

    Returns:
        dict: {model_name/index: weight}
              - Promedio de pesos = 1.0
              - Orden coincide con el de foundational_models en el CLI
    """
    coefs = np.asarray(coefs)

    if coefs.ndim == 1:
        coefs = coefs[np.newaxis, :]  # reshape a [1, n_features]

    n_features = coefs.shape[1]
    block_size = n_features // num_models

    if n_features % num_models != 0:
        raise ValueError(
            f"n_features ({n_features}) no es divisible entre num_models ({num_models}). "
            f"Comprueba drop_redundant_class: ¿reduce columnas uniformemente?"
        )

    # Calcular norma L2 por modelo base y fold
    norms_per_fold = []
    for fold_coefs in coefs:
        fold_norms = []
        for i in range(num_models):
            start = i * block_size
            end = (i + 1) * block_size
            norm_i = np.linalg.norm(fold_coefs[start:end], ord=2)
            fold_norms.append(norm_i)
        norms_per_fold.append(fold_norms)

    # Promediar entre folds
    norms_per_fold = np.array(norms_per_fold)  # [n_folds, num_models]
    mean_norms = norms_per_fold.mean(axis=0)  # [num_models]

    # Normalizar para que promedio sea 1.0
    if mean_norms.sum() > 0:
        if temperature == 0.0:
            weights = np.ones(num_models)
        else:
            # weight_i = (norm_i / mean(norm)) ** temperature
            norm_ratio = mean_norms / mean_norms.mean()
            weights = np.power(norm_ratio, temperature)
            weights /= weights.mean()  # renormalizar a media 1.0
    else:
        weights = np.ones(num_models)

    # Mapear a nombres o índices
    if model_names is not None:
        if len(model_names) != num_models:
            raise ValueError(f"len(model_names)={len(model_names)} != num_models={num_models}")
        return {name: w for name, w in zip(model_names, weights)}
    else:
        return {i: w for i, w in enumerate(weights)}
