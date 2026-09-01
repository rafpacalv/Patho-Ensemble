import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
import os
import argparse
from sklearn.linear_model import LogisticRegression
from pathlib import Path

from utils import Metrics, MetricDistance


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

    def __init__(self, xdirs, vdirs, tdirs, meta_model, device="cpu"):
        super().__init__()
        self.xdirs = xdirs
        self.vdirs = vdirs
        self.tdirs = tdirs
        self.meta_model = meta_model
        self.device = device

        # Validar inputs
        self._validate_directories()

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
        # Stack: [num_models, N_train, num_classes] → [N_train, num_models, num_classes]
        # Reshape: [N_train, num_models, num_classes] → [N_train, num_models*num_classes]

        xstacked = torch.stack(xpreds, dim=0).to(self.device)
        vstacked = torch.stack(vpreds, dim=0).to(self.device)

        X_train = xstacked.permute(1, 0, 2).reshape(xstacked.shape[1], -1).cpu().numpy()
        y_train = xlabels_ref.cpu().numpy()

        X_val = vstacked.permute(1, 0, 2).reshape(vstacked.shape[1], -1).cpu().numpy()
        y_val = vlabels_ref.cpu().numpy()

        # ========== ENTRENAR META-LEARNER ==========
        self.meta_model.fit(X_train, y_train)

        # ========== GUARDAR ARTEFACTOS ==========
        base_path = Path(self.xdirs[0]).parent.parent.parent / "ensemble4"
        base_path.mkdir(parents=True, exist_ok=True)

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

        # ========== TEST SET ==========
        tlabels_ref, tpreds = None, []

        for idx, d in enumerate(self.tdirs):
            labels_i = torch.from_numpy(np.load(os.path.join(d, 'labels.npy'))).to(self.device)
            preds_i = torch.from_numpy(np.load(os.path.join(d, 'preds.npy'))).to(self.device)

            if idx == 0:
                tlabels_ref = labels_i

            tpreds.append(preds_i)

        tstacked = torch.stack(tpreds, dim=0).to(self.device)
        num_models, N_test, num_classes = tstacked.shape
        X_test = tstacked.permute(1, 0, 2).reshape(N_test, num_models * num_classes).cpu().numpy()

        # ========== PREDICCIÓN ==========
        probs = self.meta_model.predict_proba(X_test)
        final_probs = torch.from_numpy(probs).to(self.device).float()

        return final_probs, tlabels_ref


def main(foundational_models, work_dir, train_source, tissue_patching, task_name):
    """
    Ejecuta ensamble meta-learner para múltiples folds.

    Args:
        foundational_models (list): Lista de modelos base (ej: ['ctranspath', 'uni_v2'])
        work_dir (str): Directorio raíz del proyecto
        train_source (str): Dataset (ej: 'cptac_brca')
        tissue_patching (str): Estrategia de patching (ej: '20x_224px_0px_overlap')
        task_name (str): Tarea de clasificación (ej: 'TP53_mutation')
    """

    # Construir rutas de directorios base
    dirs = [
        os.path.join(work_dir, train_source, task_name, 'abmil', f'{f}_{tissue_patching}')
        for f in foundational_models
    ]

    xdirs = []
    vdirs = []
    tdirs = []
    folds = []

    # Detectar número de folds
    for idx, d in enumerate(dirs):
        val_outputs_path = os.path.join(d, 'val_outputs')

        if any(name.startswith("fold_") for name in os.listdir(val_outputs_path)):
            # Multi-fold: busca fold_* subdirectories
            folds.append(len(os.listdir(val_outputs_path)))
        else:
            # Single fold (solo un modelo o un fold)
            folds.append(1)

        xdirs.append(os.path.join(f'{d}_train_eval', 'val_outputs'))
        vdirs.append(os.path.join(d, 'val_outputs'))
        tdirs.append(os.path.join(d, 'test_outputs'))

    assert len(set(folds)) == 1, f"Mismatch in number of folds: {set(folds)}"

    n_folds = folds[0]
    device = "cuda" if torch.cuda.is_available() else "cpu"

    all_preds = []
    all_labels = []

    print("\n\nEvaluating ensemble...")
    with tqdm(total=n_folds, unit="fold") as pbar:
        for f in range(n_folds):
            pbar.set_description(f"Evaluating fold {f+1}/{n_folds}")

            # Rutas para este fold
            xdirs_fold = [os.path.join(d, f'fold_{f}') if n_folds > 1 else d for d in xdirs]
            vdirs_fold = [os.path.join(d, f'fold_{f}') if n_folds > 1 else d for d in vdirs]
            tdirs_fold = [os.path.join(d, f'fold_{f}') if n_folds > 1 else d for d in tdirs]

            try:
                preds, labels = Ensemble(
                    xdirs_fold,
                    vdirs_fold,
                    tdirs_fold,
                    LogisticRegression(max_iter=1000),
                    device=device,
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
        results_dir=os.path.join(work_dir, train_source, task_name, 'abmil', 'ensemble4'),
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
                       default="/shared/home/JKP6679/Patho-Ensemble/PARADIS/datos",
                       help="Root working directory")
    parser.add_argument("--train_source", type=str, required=True,
                       help="Dataset name (e.g., cptac_brca)")
    parser.add_argument("--tissue_patching", type=str, required=True,
                       help="Patching strategy (e.g., 20x_224px_0px_overlap)")
    parser.add_argument("--task_name", type=str, required=True,
                       help="Task name (e.g., TP53_mutation)")

    args = parser.parse_args()
    main(**vars(args))
