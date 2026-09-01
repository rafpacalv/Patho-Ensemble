"""
Genera predicciones sobre el split de ENTRENAMIENTO (no val/test) para cada
fold, usando los snapshots FGE ya entrenados por train_abmil_fge.py.

Análogo a test_abmil.py, pero carga los N snapshots de cada fold (en vez de
un único checkpoint) y guarda tanto el promedio como las predicciones por
snapshot, para que ensemble4.py (--base_source fge) pueda alimentar al
meta-learner con las bases ABMIL diversificadas por FGE.

Salida:
    {work_dir}/{train_source}/{task_name}/abmil/
        {foundational_model}_{tissue_patching}_train_eval_fge/val_outputs/fold_k/
            labels.npy, preds.npy (promedio de snapshots), preds_per_snapshot.npy

El sufijo "_train_eval_fge/val_outputs" replica la convención ya usada por
test_abmil.py ("_train_eval/val_outputs"): el contenido real son las
predicciones FGE de ese fold sobre su propio split de TRAIN, no sobre val.
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch

import abmil_engine
from abmil_engine import ABMIL
from train_abmil import load_config_yaml, load_features_cached, create_fold_split
from train_abmil_fge import save_fold_outputs_fge
from utils import resolve_latent_dim


def abmil_fge_predict_train_fold(
    work_dir, train_source, foundational_model, tissue_patching, task_name,
    fold_k, n_folds, task_col, num_classes, latent_dim, df, feats, fge_tag="fge"
):
    """Carga los snapshots FGE del fold_k y predice sobre su propio split de train.

    Devuelve (labels, preds_avg) o (None, None) si faltan snapshots o datos.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"

    output_dir = f"{work_dir}/{train_source}/{task_name}/abmil/{foundational_model}_{tissue_patching}"
    ckpt_dir = Path(f"{output_dir}/checkpoints_{fge_tag}/fold_{fold_k}")

    snapshot_paths = sorted(ckpt_dir.glob("snapshot_*.pt"),
                             key=lambda p: int(p.stem.split("_")[1]))
    if not snapshot_paths:
        print(f"  ⚠️  Fold {fold_k}: no hay snapshots FGE en {ckpt_dir}, se omite.")
        return None, None

    tr_df, _, _ = create_fold_split(df, fold_k, task_col)
    if tr_df is None or tr_df.empty:
        print(f"  ⚠️  Fold {fold_k}: sin datos de train, se omite.")
        return None, None

    tr_stems = tr_df['slide_id'].tolist()
    tr_y = tr_df[task_col].values

    preds_per_snapshot = []
    for snap_path in snapshot_paths:
        model = ABMIL(in_dim=latent_dim, num_classes=num_classes,
                       hidden_dim=256, proj_dim=512, dropout=0.25).to(device)
        model.load_state_dict(torch.load(snap_path, map_location=device))
        model.eval()
        preds_per_snapshot.append(abmil_engine.abmil_predict(model, feats, tr_stems, device))

    preds_per_snapshot = np.stack(preds_per_snapshot, axis=0)

    eval_dir = f"{output_dir}_train_eval_{fge_tag}"
    save_fold_outputs_fge(eval_dir, fold_k, n_folds, tr_y, preds_per_snapshot,
                          split="val", tag=fge_tag)

    return tr_y, preds_per_snapshot.mean(axis=0)


def abmil_fge_test(work_dir, train_source, foundational_model, tissue_patching, task_name,
                    latent_dim, epochs=None, fge_tag="fge"):
    """Recorre todos los folds de k=all.tsv y genera predicciones FGE de train para cada uno."""
    config_path = f"{work_dir}/{train_source}/{task_name}/config.yaml"
    task_col, num_classes = load_config_yaml(config_path)

    split_file = f"{work_dir}/{train_source}/{task_name}/k=all.tsv"
    df = pd.read_csv(split_file, sep="\t")
    df = df.dropna(subset=[task_col]).reset_index(drop=True)

    fold_cols = [c for c in df.columns if c.startswith("fold_")]
    n_folds = len(fold_cols)

    print(f"\n{'='*60}")
    print(f"ABMIL-FGE Test (predicciones sobre train): {foundational_model}")
    print(f"  Dataset: {train_source}/{task_name}")
    print(f"  Latent dim: {latent_dim}, Num classes: {num_classes}, Folds: {n_folds}")
    print(f"  Variante FGE: {fge_tag}")
    print(f"{'='*60}\n")

    all_slide_ids = set(df['slide_id'].unique())
    feats = load_features_cached(work_dir, train_source, foundational_model, all_slide_ids, latent_dim)

    n_ok, n_skipped = 0, 0
    for fold_k in range(n_folds):
        labels, preds = abmil_fge_predict_train_fold(
            work_dir, train_source, foundational_model, tissue_patching, task_name,
            fold_k, n_folds, task_col, num_classes, latent_dim, df, feats, fge_tag
        )
        if labels is None:
            n_skipped += 1
        else:
            n_ok += 1

    print(f"\n✅ Predicciones FGE de train generadas para {n_ok}/{n_folds} folds "
          f"({n_skipped} omitidos por falta de snapshots/datos).")
    if n_skipped:
        print("   ⚠️  Revisa que train_abmil_fge.py haya completado esos folds "
              "antes de correr ensemble4.py --base_source fge.")


def main():
    parser = argparse.ArgumentParser(
        description="Genera predicciones sobre el split de train (features del "
                     "meta-learner de ensemble4.py --base_source fge) usando "
                     "snapshots FGE ya entrenados por train_abmil_fge.py"
    )
    parser.add_argument("--foundational_model", type=str, required=True)
    parser.add_argument("--latent_dim", type=int, default=None,
                        help="Dimensión del embedding. Si se omite, se auto-detecta del .h5.")
    parser.add_argument("--work_dir", type=str,
                        default="/home/JKP6679/Patho-Ensemble/PARADIS/datos/patches")
    parser.add_argument("--train_source", type=str, required=True)
    parser.add_argument("--tissue_patching", type=str, required=True)
    parser.add_argument("--task_name", type=str, required=True)
    parser.add_argument("--epochs", type=int, default=None,
                         help="No usado (solo inferencia); se acepta por "
                              "compatibilidad con train_base_models_fge.sbatch")
    parser.add_argument("--fge_tag", type=str, default="fge",
                        help="Etiqueta de la variante FGE cuyos snapshots se leen "
                             "(checkpoints_{tag}/). Debe coincidir con la usada en "
                             "train_abmil_fge.py.")
    args = parser.parse_args()

    latent_dim = resolve_latent_dim(
        args.work_dir, args.train_source, args.foundational_model, args.latent_dim
    )

    abmil_fge_test(
        work_dir=args.work_dir,
        train_source=args.train_source,
        foundational_model=args.foundational_model,
        tissue_patching=args.tissue_patching,
        task_name=args.task_name,
        latent_dim=latent_dim,
        epochs=args.epochs,
        fge_tag=args.fge_tag,
    )


if __name__ == "__main__":
    main()
