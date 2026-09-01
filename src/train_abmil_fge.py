"""
Entrenamiento de ABMIL con Fast Geometric Ensembles (FGE, Garipov et al. ICLR 2018)
sobre features HDF5 de PARADIS.

Análogo a train_abmil.py: reutiliza load_config_yaml / load_features_cached /
create_fold_split de ese módulo para no duplicar lógica de splits ni carga de
features. Por cada fold: (1) converge un ABMIL exactamente como train_abmil.py
(mismo seed, misma receta), (2) fine-tunea ese modelo convergido con ciclos
FGE cortos (abmil_fge.train_abmil_fge), tomando varios snapshots.

Salida (paralela a la de train_abmil.py, no la pisa):
    {output_dir}/checkpoints_fge/fold_k/
        base_model.pt          # modelo convergido (punto de partida FGE)
        snapshot_0.pt ... snapshot_{n-1}.pt
    {output_dir}/val_outputs_fge/fold_k/
        labels.npy, preds.npy (promedio de snapshots), preds_per_snapshot.npy
    {output_dir}/test_outputs_fge/fold_k/
        labels.npy, preds.npy (promedio de snapshots), preds_per_snapshot.npy
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

import abmil_engine
import abmil_fge
from train_abmil import load_config_yaml, load_features_cached, create_fold_split
from utils import Metrics, resolve_latent_dim


def save_fold_outputs_fge(output_dir, fold_k, n_folds, labels, preds_per_snapshot,
                          split="val", tag="fge"):
    """Guarda labels.npy, preds.npy (promedio) y preds_per_snapshot.npy en
    {split}_outputs_{tag}/[fold_k/] (según n_folds).

    `tag` separa los árboles de cada variante FGE (fge, fge_wbase, fge_lowlr...)
    para que varias configuraciones puedan convivir sin pisarse.

    preds_per_snapshot: array (n_snapshots, N, num_classes).
    """
    if n_folds > 1:
        output_path = f"{output_dir}/{split}_outputs_{tag}/fold_{fold_k}"
    else:
        output_path = f"{output_dir}/{split}_outputs_{tag}"

    Path(output_path).mkdir(parents=True, exist_ok=True)
    preds_avg = preds_per_snapshot.mean(axis=0)
    np.save(f"{output_path}/labels.npy", labels)
    np.save(f"{output_path}/preds.npy", preds_avg)
    np.save(f"{output_path}/preds_per_snapshot.npy", preds_per_snapshot)


def _predict_snapshots(snapshots, base_kwargs, feats, stems, device):
    """Corre abmil_predict con cada snapshot sobre `stems`.

    Returns:
        array (n_snapshots, N, num_classes)
    """
    from abmil_engine import ABMIL
    preds = []
    for state in snapshots:
        model = ABMIL(**base_kwargs).to(device)
        model.load_state_dict(state)
        preds.append(abmil_engine.abmil_predict(model, feats, stems, device))
    return np.stack(preds, axis=0)


def abmil_fge_train_test(
    work_dir, train_source, foundational_model, tissue_patching, task_name,
    latent_dim, epochs, fold_k, n_folds, task_col, num_classes,
    n_cycles, cycle_length, lr_1, lr_2, cycle_patience,
    fge_tag="fge", include_base=False, max_folds=None, schedule="fge",
):
    """Entrena (convergencia + FGE) y evalúa ABMIL-FGE para un fold específico.

    Devuelve (val_labels, val_preds_avg, test_labels, test_preds_avg) o
    arrays vacíos si el fold no tiene datos.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"

    output_dir = f"{work_dir}/{train_source}/{task_name}/abmil/{foundational_model}_{tissue_patching}"
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    split_file = f"{work_dir}/{train_source}/{task_name}/k=all.tsv"
    if fold_k != 'all' and Path(f"{work_dir}/{train_source}/{task_name}/k={fold_k}.tsv").exists():
        split_file = f"{work_dir}/{train_source}/{task_name}/k={fold_k}.tsv"

    df = pd.read_csv(split_file, sep="\t")
    label_col = task_col
    if label_col not in df.columns:
        raise ValueError(f"Column {label_col} not found in {split_file}")
    df = df.dropna(subset=[label_col]).reset_index(drop=True)

    if fold_k == 'all':
        all_val_labels, all_val_preds = [], []
        all_test_labels, all_test_preds = [], []
        fold_cols = [c for c in df.columns if c.startswith("fold_")]
        n_run = min(max_folds, len(fold_cols)) if max_folds else len(fold_cols)
        if n_run < len(fold_cols):
            print(f"  (usando {n_run} de {len(fold_cols)} folds disponibles)")

        for fk in range(n_run):
            vl, vp, tl, tp = abmil_fge_train_test(
                work_dir, train_source, foundational_model, tissue_patching, task_name,
                latent_dim, epochs, str(fk), len(fold_cols), task_col, num_classes,
                n_cycles, cycle_length, lr_1, lr_2, cycle_patience,
                fge_tag, include_base, max_folds, schedule,
            )
            if len(tl) > 0 and len(tp) > 0:
                all_val_labels.append(vl)
                all_val_preds.append(vp)
                all_test_labels.append(tl)
                all_test_preds.append(tp)
            else:
                print(f"  ⚠️  Fold {fk}: skipped (no test data)")

        if all_val_labels:
            print("\nCalculando métricas de validación (FGE, promedio de snapshots)...")
            Metrics(
                task_type='classification',
                model_kwargs={'num_classes': num_classes},
                num_bootstraps=100,
                results_dir=output_dir,
                split=f'val_{fge_tag}',
                num_folds=len(all_val_labels),
                all_labels_across_folds=all_val_labels,
                all_preds_across_folds=all_val_preds
            ).run()

        if all_test_labels:
            print("\nCalculando métricas de test (FGE, promedio de snapshots)...")
            Metrics(
                task_type='classification',
                model_kwargs={'num_classes': num_classes},
                num_bootstraps=100,
                results_dir=output_dir,
                split=f'test_{fge_tag}',
                num_folds=len(all_test_labels),
                all_labels_across_folds=all_test_labels,
                all_preds_across_folds=all_test_preds
            ).run()
            print(f"✅ Métricas FGE de test completadas ({len(all_test_labels)} folds con datos)")
        else:
            print("\n❌ ERROR: No test data found for any FGE fold")

        return None, None, None, None

    fold_k_int = int(fold_k)
    tr_df, va_df, te_df = create_fold_split(df, fold_k_int, label_col)

    if tr_df is None or te_df is None or te_df.empty:
        print(f"Fold {fold_k}: sin datos")
        return np.array([]), np.array([]), np.array([]), np.array([])

    all_slide_ids = set(df['slide_id'].unique())
    feats = load_features_cached(work_dir, train_source, foundational_model, all_slide_ids, latent_dim)

    tr_stems = tr_df['slide_id'].tolist()
    tr_y = tr_df[label_col].values
    tr_groups = tr_df['case_id'].values

    te_stems = te_df['slide_id'].tolist()
    te_y = te_df[label_col].values

    if va_df is None or va_df.empty:
        va_stems, va_y = [], np.array([])
    else:
        va_stems = va_df['slide_id'].tolist()
        va_y = va_df[label_col].values

    print(f"\nFold {fold_k} (FGE): train={len(tr_stems)}, val={len(va_stems)}, test={len(te_stems)}")

    # Paso 1: convergencia estándar (idéntica a train_abmil.py, mismo seed)
    seed = 42 + fold_k_int
    model = abmil_engine.train_abmil(
        feats=feats, stems=tr_stems, y=tr_y, groups=tr_groups,
        in_dim=latent_dim, n_classes=num_classes, device=device, seed=seed,
        max_epochs=epochs, patience=8, proj_dim=512, dropout=0.25, wd=1e-4,
        val_frac=0.2 if va_stems == [] else None,
    )
    print(f"Fold {fold_k}: base convergido (val_auc={model._val_auc:.4f}), iniciando ciclos FGE...")

    # Paso 2: ciclos FGE cortos partiendo del modelo convergido
    snapshots, val_aucs = abmil_fge.train_abmil_fge(
        feats=feats, stems=tr_stems, y=tr_y, groups=tr_groups,
        in_dim=latent_dim, n_classes=num_classes, device=device, seed=seed,
        warm_start_state=model.state_dict(),
        n_cycles=n_cycles, cycle_length=cycle_length, lr_1=lr_1, lr_2=lr_2,
        proj_dim=512, dropout=0.25, wd=1e-4,
        val_frac=0.2 if va_stems == [] else None, cycle_patience=cycle_patience,
        include_base=include_base, schedule=schedule,
    )
    print(f"Fold {fold_k}: {len(snapshots)} snapshots FGE (val_auc por snapshot: "
          f"{', '.join(f'{a:.4f}' for a in val_aucs)})")

    if not snapshots:
        print(f"  ⚠️  Fold {fold_k}: FGE no produjo snapshots (early stop inmediato), "
              f"usando solo el modelo convergido como único snapshot.")
        snapshots = [model.state_dict()]

    # Guardar checkpoints
    ckpt_dir = f"{output_dir}/checkpoints_{fge_tag}/fold_{fold_k}"
    Path(ckpt_dir).mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), f"{ckpt_dir}/base_model.pt")
    for i, state in enumerate(snapshots):
        torch.save(state, f"{ckpt_dir}/snapshot_{i}.pt")

    base_kwargs = dict(in_dim=latent_dim, num_classes=num_classes,
                        hidden_dim=256, proj_dim=512, dropout=0.25)

    if va_stems:
        va_preds_per_snap = _predict_snapshots(snapshots, base_kwargs, feats, va_stems, device)
        save_fold_outputs_fge(output_dir, fold_k_int, n_folds if fold_k != 'all' else 1,
                               va_y, va_preds_per_snap, split="val", tag=fge_tag)
        va_preds_avg = va_preds_per_snap.mean(axis=0)
    else:
        va_preds_avg, va_y = np.array([]), np.array([])

    te_preds_per_snap = _predict_snapshots(snapshots, base_kwargs, feats, te_stems, device)
    save_fold_outputs_fge(output_dir, fold_k_int, n_folds if fold_k != 'all' else 1,
                           te_y, te_preds_per_snap, split="test", tag=fge_tag)
    te_preds_avg = te_preds_per_snap.mean(axis=0)

    return va_y, va_preds_avg, te_y, te_preds_avg


def main():
    parser = argparse.ArgumentParser(description="Train ABMIL with FGE (Fast Geometric Ensembles)")
    parser.add_argument("--foundational_model", type=str, required=True)
    parser.add_argument("--latent_dim", type=int, default=None,
                        help="Dimensión del embedding. Si se omite, se auto-detecta del .h5.")
    parser.add_argument("--work_dir", type=str,
                        default="/home/JKP6679/Patho-Ensemble/PARADIS/datos/patches")
    parser.add_argument("--train_source", type=str, required=True)
    parser.add_argument("--tissue_patching", type=str, required=True)
    parser.add_argument("--task_name", type=str, required=True)
    parser.add_argument("--epochs", type=int, default=100,
                         help="Épocas para la fase de convergencia inicial (igual que train_abmil.py)")
    parser.add_argument("--fold", type=str, default="all")
    parser.add_argument("--n_cycles", type=int, default=6, help="Número de ciclos FGE")
    parser.add_argument("--cycle_length", type=int, default=4, help="Épocas por ciclo FGE")
    parser.add_argument("--lr_1", type=float, default=2e-4, help="LR techo del ciclo")
    parser.add_argument("--lr_2", type=float, default=2e-5, help="LR piso del ciclo (valor en snapshot)")
    parser.add_argument("--cycle_patience", type=int, default=2,
                         help="Ciclos sin mejora de AUC val antes de detener FGE")
    parser.add_argument("--max_folds", type=int, default=None,
                        help="Limitar el número de folds a recorrer con --fold all.")
    parser.add_argument("--fge_tag", type=str, default="fge",
                        help="Etiqueta de la variante FGE. Determina los directorios de "
                             "salida (checkpoints_{tag}/, val_outputs_{tag}/...), de modo "
                             "que varias configuraciones no se pisen entre sí.")
    parser.add_argument("--include_base", action="store_true",
                        help="Incluir el modelo convergido como miembro del ensemble, "
                             "además de los snapshots de los ciclos.")
    parser.add_argument("--schedule", type=str, default="fge", choices=["fge", "se"],
                        help="Forma de la LR dentro del ciclo: 'fge' (piecewise-linear, "
                             "Garipov et al.) o 'se' (coseno con reinicios, Huang et al.). "
                             "Con 'se', --lr_1 es la LR del reinicio y --lr_2 la mínima.")

    args = parser.parse_args()

    config_path = f"{args.work_dir}/{args.train_source}/{args.task_name}/config.yaml"
    task_col, num_classes = load_config_yaml(config_path)

    args.latent_dim = resolve_latent_dim(
        args.work_dir, args.train_source, args.foundational_model, args.latent_dim
    )

    print(f"\n{'='*60}")
    print(f"ABMIL-FGE Training: {args.foundational_model}")
    print(f"  Dataset: {args.train_source}/{args.task_name}")
    print(f"  Latent dim: {args.latent_dim}, Num classes: {num_classes}")
    print(f"  Epochs (convergencia): {args.epochs}, Fold: {args.fold}")
    print(f"  Schedule: {args.schedule}  "
          f"({'piecewise-linear, Garipov et al.' if args.schedule == 'fge' else 'coseno con reinicios, Huang et al.'})")
    print(f"  Ciclos: n_cycles={args.n_cycles}, cycle_length={args.cycle_length}, "
          f"lr_1={args.lr_1}, lr_2={args.lr_2}, cycle_patience={args.cycle_patience}")
    print(f"  Variante: {args.fge_tag}  (include_base={args.include_base})")
    print(f"{'='*60}\n")

    split_file = f"{args.work_dir}/{args.train_source}/{args.task_name}/k=all.tsv"
    df_check = pd.read_csv(split_file, sep="\t")
    fold_cols = [c for c in df_check.columns if c.startswith("fold_")]
    n_folds = len(fold_cols)

    abmil_fge_train_test(
        work_dir=args.work_dir, train_source=args.train_source,
        foundational_model=args.foundational_model, tissue_patching=args.tissue_patching,
        task_name=args.task_name, latent_dim=args.latent_dim, epochs=args.epochs,
        fold_k=args.fold, n_folds=n_folds, task_col=task_col, num_classes=num_classes,
        n_cycles=args.n_cycles, cycle_length=args.cycle_length,
        lr_1=args.lr_1, lr_2=args.lr_2, cycle_patience=args.cycle_patience,
        fge_tag=args.fge_tag, include_base=args.include_base,
        max_folds=args.max_folds, schedule=args.schedule,
    )

    print(f"\n✅ FGE training completed")


if __name__ == "__main__":
    main()
