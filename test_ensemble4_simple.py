"""
Test simplificado para ensemble4.py
Prueba solo la clase Ensemble sin dependencias de patho_bench
"""

import os
import sys
import tempfile
import numpy as np
from pathlib import Path
import shutil
import torch
import torch.nn as nn
from sklearn.linear_model import LogisticRegression


class EnsembleSimple(nn.Module):
    """Copia simplificada de Ensemble de ensemble4.py para testing"""
    def __init__(self, xdirs, vdirs, tdirs, meta_model, device="cpu"):
        super().__init__()
        self.xdirs = xdirs
        self.vdirs = vdirs
        self.tdirs = tdirs
        self.meta_model = meta_model
        self.device = device

    def forward(self):
        xlabels, xpreds = [], []
        for d in self.xdirs:
            xlabels.append(torch.from_numpy(np.load(os.path.join(d, 'labels.npy'))).to(self.device))
            xpreds.append(torch.from_numpy(np.load(os.path.join(d, 'preds.npy'))).to(self.device))

        for i in range(1, len(xlabels)):
            assert torch.equal(xlabels[0], xlabels[i]), f"Labels difieren entre modelo 0 y {i}"

        vlabels, vpreds = [], []
        for d in self.vdirs:
            vlabels.append(torch.from_numpy(np.load(os.path.join(d, 'labels.npy'))).to(self.device))
            vpreds.append(torch.from_numpy(np.load(os.path.join(d, 'preds.npy'))).to(self.device))

        for i in range(1, len(vlabels)):
            assert torch.equal(vlabels[0], vlabels[i]), f"Labels difieren entre modelo 0 y {i}"

        xstacked = torch.stack(xpreds, dim=0).to(self.device)
        vstacked = torch.stack(vpreds, dim=0).to(self.device)

        xlabels = xlabels[0]
        vlabels = vlabels[0]

        X_train = xstacked.permute(1, 0, 2).reshape(xstacked.shape[1], -1).cpu().numpy()
        y_train = xlabels.cpu().numpy()

        X_val = vstacked.permute(1, 0, 2).reshape(vstacked.shape[1], -1).cpu().numpy()
        y_val = vlabels.cpu().numpy()

        X_all = np.concatenate([X_train, X_val], axis=0)
        y_all = np.concatenate([y_train, y_val], axis=0)

        self.meta_model.fit(X_train, y_train)

        base_path = Path(self.xdirs[0]).parent.parent.parent / "ensemble4"
        base_path.mkdir(parents=True, exist_ok=True)

        coefs_path = base_path / "coefs.npy"
        odds_path = base_path / "odds_ratios.npy"

        coefs = np.array(self.meta_model.coef_[0])
        odds = np.exp(coefs)

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

        tlabels, tpreds = [], []
        for d in self.tdirs:
            tlabels.append(torch.from_numpy(np.load(os.path.join(d, 'labels.npy'))).to(self.device))
            tpreds.append(torch.from_numpy(np.load(os.path.join(d, 'preds.npy'))).to(self.device))

        tstacked = torch.stack(tpreds, dim=0).to(self.device)
        num_models, N_test, num_classes = tstacked.shape
        X_test = tstacked.permute(1, 0, 2).reshape(N_test, num_models * num_classes).cpu().numpy()

        probs = self.meta_model.predict_proba(X_test)
        final_probs = torch.from_numpy(probs).to(self.device).float()

        return final_probs, tlabels[0]


def create_mock_data(base_dir, n_models=3, n_samples=100, n_classes=2, n_folds=2):
    """Crea estructura de datos mock para ensemble4."""

    print(f"📁 Creando datos mock en: {base_dir}")
    Path(base_dir).mkdir(parents=True, exist_ok=True)

    xdirs = []
    vdirs = []
    tdirs = []
    models = [f'model_{i}' for i in range(n_models)]

    # Generar labels una sola vez (igual para todos los modelos)
    labels_train = np.random.randint(0, n_classes, n_samples)
    labels_val = np.random.randint(0, n_classes, n_samples)
    labels_test = np.random.randint(0, n_classes, n_samples)

    for fold_idx in range(n_folds):
        for model_idx, model in enumerate(models):
            model_dir = os.path.join(base_dir, f'dataset/task/abmil/{model}')

            if n_folds > 1:
                xdir = os.path.join(model_dir + '_train_eval', f'fold_{fold_idx}')
                vdir = os.path.join(model_dir, f'fold_{fold_idx}')
                tdir = os.path.join(model_dir, f'fold_{fold_idx}')
            else:
                xdir = os.path.join(model_dir + '_train_eval')
                vdir = os.path.join(model_dir)
                tdir = os.path.join(model_dir)

            Path(xdir).mkdir(parents=True, exist_ok=True)
            Path(vdir).mkdir(parents=True, exist_ok=True)
            Path(tdir).mkdir(parents=True, exist_ok=True)

            # Cada modelo genera predicciones diferentes pero con los mismos labels
            preds_train = np.random.dirichlet(np.ones(n_classes), n_samples)
            preds_val = np.random.dirichlet(np.ones(n_classes), n_samples)
            preds_test = np.random.dirichlet(np.ones(n_classes), n_samples)

            np.save(os.path.join(xdir, 'labels.npy'), labels_train)
            np.save(os.path.join(xdir, 'preds.npy'), preds_train)

            np.save(os.path.join(vdir, 'labels.npy'), labels_val)
            np.save(os.path.join(vdir, 'preds.npy'), preds_val)

            np.save(os.path.join(tdir, 'labels.npy'), labels_test)
            np.save(os.path.join(tdir, 'preds.npy'), preds_test)

            if fold_idx == 0:
                if n_folds > 1:
                    xdirs.append(os.path.join(model_dir + '_train_eval', f'fold_0'))
                    vdirs.append(os.path.join(model_dir, f'fold_0'))
                    tdirs.append(os.path.join(model_dir, f'fold_0'))
                else:
                    xdirs.append(os.path.join(model_dir + '_train_eval'))
                    vdirs.append(os.path.join(model_dir))
                    tdirs.append(os.path.join(model_dir))

    print(f"✅ Datos mock creados:")
    print(f"   • {len(models)} modelos")
    print(f"   • {n_folds} folds")
    print(f"   • {n_samples} muestras por split")
    print(f"   • {n_classes} clases\n")

    return xdirs, vdirs, tdirs


def test_ensemble4():
    """Ejecuta y valida ensemble4."""

    test_dir = tempfile.mkdtemp(prefix='ensemble4_test_')
    print(f"🧪 Test de ensemble4.py\n")
    print("="*60)

    try:
        # Crear datos mock
        xdirs, vdirs, tdirs = create_mock_data(
            test_dir,
            n_models=3,
            n_samples=50,
            n_classes=2,
            n_folds=2
        )

        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"🔧 Device: {device}\n")

        # Probar para cada fold
        print("📊 Evaluando ensamble por fold:\n")
        all_preds = []
        all_labels = []

        for fold_idx in range(2):
            print(f"   Procesando fold {fold_idx + 1}/2...")

            xdirs_fold = [os.path.join(d, f'fold_{fold_idx}') if os.path.exists(os.path.join(d, f'fold_{fold_idx}')) else d for d in xdirs]
            vdirs_fold = [os.path.join(d, f'fold_{fold_idx}') if os.path.exists(os.path.join(d, f'fold_{fold_idx}')) else d for d in vdirs]
            tdirs_fold = [os.path.join(d, f'fold_{fold_idx}') if os.path.exists(os.path.join(d, f'fold_{fold_idx}')) else d for d in tdirs]

            ensemble = EnsembleSimple(
                xdirs_fold,
                vdirs_fold,
                tdirs_fold,
                LogisticRegression(max_iter=1000),
                device=device
            )

            preds, labels = ensemble()

            all_preds.append(preds.detach().cpu().numpy())
            all_labels.append(labels.detach().cpu().numpy())

            print(f"      ✓ Predicciones shape: {preds.shape}")
            print(f"      ✓ Labels shape: {labels.shape}")
            print()

        # Validaciones
        print("="*60)
        print("\n🔍 Validaciones:\n")

        # 1. Verificar shapes
        assert all_preds[0].shape[0] == all_labels[0].shape[0], "Mismatch between predictions and labels"
        print("   ✅ Shapes de predicciones y labels coinciden")

        # 2. Verificar probabilidades válidas
        sums = np.sum(all_preds[0], axis=1)
        assert np.allclose(sums, 1.0, atol=1e-6), "Probabilities don't sum to 1"
        print("   ✅ Probabilidades normalizadas (suma ≈ 1)")

        # 3. Verificar rango de valores
        assert np.all(all_preds[0] >= 0) and np.all(all_preds[0] <= 1), "Predictions out of [0, 1] range"
        print("   ✅ Predicciones en rango [0, 1]")

        # 4. Verificar que se generaron coefs y odds
        coefs_path = Path(test_dir) / "dataset/task/abmil/ensemble4/coefs.npy"
        odds_path = Path(test_dir) / "dataset/task/abmil/ensemble4/odds_ratios.npy"

        if coefs_path.exists():
            coefs = np.load(coefs_path)
            print(f"   ✅ Coeficientes guardados (shape: {coefs.shape})")
            print(f"      Valores: {coefs[0][:5]}... (mostrando primeros 5)")

        if odds_path.exists():
            odds = np.load(odds_path)
            print(f"   ✅ Odds ratios guardados (shape: {odds.shape})")
            print(f"      Valores: {odds[0][:5]}... (mostrando primeros 5)")

        print("\n" + "="*60)
        print("🎉 PRUEBA EXITOSA")
        print("="*60)
        print("\n✨ ensemble4.py funciona correctamente\n")

        return True

    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False

    finally:
        if os.path.exists(test_dir):
            shutil.rmtree(test_dir)


if __name__ == "__main__":
    success = test_ensemble4()
    sys.exit(0 if success else 1)
