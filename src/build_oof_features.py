"""
Genera meta-features OUT-OF-FOLD para el meta-learner de ensemble4.py.

Motivación
----------
`test_abmil.py` produce las features del meta-learner prediciendo con el modelo
del fold sobre su PROPIO split de train. Esas predicciones son in-sample: medido
en cptac_brca/TP53_mutation, los modelos base alcanzan AUC 0.93-0.94 sobre ellas
frente a 0.76-0.79 en test. El meta-learner aprende entonces a combinar señales
mucho más separables de las que verá al predecir, que es el fallo clásico del
stacking (Wolpert 1992, Breiman 1996) y penaliza sobre todo a los meta-modelos
flexibles: cuanto mejor ajustan una distribución mal especificada, peor
transfieren.

La corrección estándar es usar predicciones out-of-fold: dentro del split de
train de cada fold externo se hace una CV interna, y cada muestra recibe la
predicción de un modelo que NO la vio entrenar.

Salida (árbol paralelo, no pisa el de test_abmil.py):
    {work_dir}/{train_source}/{task_name}/abmil/
        {foundational_model}_{tissue_patching}_train_eval_oof/val_outputs_oof/fold_k/
            labels.npy, preds.npy

El orden de las filas es el de `tr_df`, idéntico entre modelos base, porque
ensemble4.py exige que las etiquetas coincidan exactamente entre modelos.
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import StratifiedGroupKFold, GroupKFold

import abmil_engine
from train_abmil import load_config_yaml, load_features_cached, create_fold_split
from train_abmil_fge import save_fold_outputs_fge
from utils import resolve_latent_dim


def _inner_folds(groups, y, n_inner, seed):
    """Particiona en `n_inner` folds agrupados por case_id y estratificados.

    Mismo criterio que `abmil_engine._grouped_val_split`, pero devolviendo TODAS
    las particiones en vez de sólo la primera: aquí cada muestra tiene que caer
    exactamente una vez en el conjunto retenido.

    Devuelve lista de (train_idx, held_out_idx).
    """
    groups = np.asarray(groups)
    y = np.asarray(y)

    try:
        sgkf = StratifiedGroupKFold(n_splits=n_inner, shuffle=True, random_state=seed)
        splits = list(sgkf.split(np.zeros(len(y)), y, groups))
        # Un fold interno sin ambas clases en train haría inútil al modelo.
        if all(len(np.unique(y[tr])) > 1 for tr, _ in splits):
            return splits
    except ValueError:
        pass

    # Fallback: agrupado sin estratificar (clases muy escasas o sklearn viejo).
    return list(GroupKFold(n_splits=n_inner).split(np.zeros(len(y)), y, groups))


def build_oof_for_fold(feats, tr_df, task_col, latent_dim, num_classes,
                       device, seed, epochs, n_inner, use_embeddings=False):
    """Predicciones (o embeddings) out-of-fold para el split de train de un fold externo.

    Args:
        use_embeddings: si True, extrae embeddings ponderados; si False, predicciones.

    Returns:
        (labels, preds) alineados con el orden de filas de `tr_df`, o
        (None, None) si el fold no admite CV interna.
    """
    stems = tr_df['slide_id'].tolist()
    y = tr_df[task_col].values
    groups = tr_df['case_id'].values

    if len(np.unique(y)) < 2:
        print("    ⚠️  una sola clase en train, se omite")
        return None, None
    if len(np.unique(groups)) < n_inner:
        print(f"    ⚠️  menos pacientes ({len(np.unique(groups))}) que folds internos "
              f"({n_inner}), se omite")
        return None, None

    # Primera predicción (o embedding) para determinar dimensión
    oof = None

    for i, (tr_idx, ho_idx) in enumerate(_inner_folds(groups, y, n_inner, seed)):
        inner_stems = [stems[j] for j in tr_idx]
        inner_y = y[tr_idx]
        inner_groups = groups[tr_idx]

        # val_frac=0.2 hace que train_abmil reserve su validación de early
        # stopping DENTRO del train interno; el fold retenido nunca se toca.
        model = abmil_engine.train_abmil(
            feats=feats, stems=inner_stems, y=inner_y, groups=inner_groups,
            in_dim=latent_dim, n_classes=num_classes, device=device,
            seed=seed + 1000 * i, max_epochs=epochs, patience=8,
            proj_dim=512, dropout=0.25, wd=1e-4, val_frac=0.2,
        )

        ho_stems = [stems[j] for j in ho_idx]
        if use_embeddings:
            preds = abmil_engine.abmil_extract_embeddings(model, feats, ho_stems, device)
            embedding_dim = model.get_embedding_dim() if hasattr(model, 'get_embedding_dim') else None
            attn_stats = None
            attn_columns = None
        else:
            preds, attn_stats, attn_columns = abmil_engine.abmil_predict_with_attention(
                model, feats, ho_stems, device
            )
            embedding_dim = None

        # Inicializar oof con la dimensión correcta en la primera iteración
        if oof is None:
            out_dim = preds.shape[1]
            oof = np.full((len(stems), out_dim), np.nan, dtype=np.float64)

        # Inicializar attn_oof también (solo si no embeddings)
        if not use_embeddings and attn_stats is not None:
            if 'attn_oof' not in locals():
                attn_oof = np.full((len(stems), attn_stats.shape[1]), np.nan, dtype=np.float64)
            attn_oof[ho_idx] = attn_stats

        oof[ho_idx] = preds
        print(f"    interno {i + 1}/{n_inner}: train={len(tr_idx)} retenido={len(ho_idx)} "
              f"(val_auc={model._val_auc:.4f})")

    if np.isnan(oof).any():
        n_missing = int(np.isnan(oof).any(axis=1).sum())
        raise RuntimeError(
            f"{n_missing} muestras sin predicción out-of-fold: la partición interna "
            f"no cubre todo el train. Revisa _inner_folds."
        )

    # Guardar la dimensión del embedding si se usó
    embedding_dim = oof.shape[1] if use_embeddings else None

    # Devolver también attn_oof y attn_columns si no se usaron embeddings
    attn_oof_return = locals().get('attn_oof', None)
    attn_columns_return = locals().get('attn_columns', None)
    return y, oof, embedding_dim, attn_oof_return, attn_columns_return


def build_oof(work_dir, train_source, foundational_model, tissue_patching,
              task_name, latent_dim, epochs, n_inner, max_folds=None, oof_tag="oof",
              use_embeddings=False):
    """Recorre todos los folds externos y genera sus meta-features out-of-fold.

    Args:
        use_embeddings: si True, extrae embeddings ponderados; si False, predicciones.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"

    config_path = f"{work_dir}/{train_source}/{task_name}/config.yaml"
    task_col, num_classes = load_config_yaml(config_path)

    split_file = f"{work_dir}/{train_source}/{task_name}/k=all.tsv"
    df = pd.read_csv(split_file, sep="\t")
    df = df.dropna(subset=[task_col]).reset_index(drop=True)

    fold_cols = [c for c in df.columns if c.startswith("fold_")]
    n_folds = len(fold_cols)
    n_run = min(max_folds, n_folds) if max_folds else n_folds

    output_dir = (f"{work_dir}/{train_source}/{task_name}/abmil/"
                  f"{foundational_model}_{tissue_patching}")
    eval_dir = f"{output_dir}_train_eval_{oof_tag}"
    if use_embeddings:
        eval_dir += "_embeddings"

    output_type = "embeddings" if use_embeddings else "predicciones"
    print(f"\n{'=' * 60}")
    print(f"Meta-features out-of-fold ({output_type}): {foundational_model}")
    print(f"  Dataset: {train_source}/{task_name}")
    print(f"  Latent dim: {latent_dim}, clases: {num_classes}")
    print(f"  Folds externos: {n_run}/{n_folds}, folds internos: {n_inner}")
    print(f"  Salida: {eval_dir}/val_outputs_{oof_tag}/")
    print(f"{'=' * 60}\n")

    all_slide_ids = set(df['slide_id'].unique())
    feats = load_features_cached(work_dir, train_source, foundational_model,
                                 all_slide_ids, latent_dim)

    n_ok, n_skipped = 0, 0
    for fold_k in range(n_run):
        tr_df, _, _ = create_fold_split(df, fold_k, task_col)
        if tr_df is None or tr_df.empty:
            print(f"  Fold {fold_k}: sin datos de train, se omite")
            n_skipped += 1
            continue

        print(f"  Fold {fold_k}: train={len(tr_df)}")
        labels, preds, embedding_dim, attn_oof, attn_columns = build_oof_for_fold(
            feats, tr_df, task_col, latent_dim, num_classes, device,
            seed=42 + fold_k, epochs=epochs, n_inner=n_inner, use_embeddings=use_embeddings,
        )
        if labels is None:
            n_skipped += 1
            continue

        # save_fold_outputs_fge espera (n_snapshots, N, C); con un único "snapshot"
        # escribe preds.npy = esa misma matriz y el árbol val_outputs_oof/ que
        # ensemble4.py consume con --meta_features oof.
        save_fold_outputs_fge(eval_dir, fold_k, n_folds, labels,
                              preds[None, ...], split="val", tag=oof_tag)

        # Guardar también atención stats si no se usan embeddings
        if attn_oof is not None and attn_columns is not None:
            from train_abmil import save_fold_outputs_attn
            save_fold_outputs_attn(eval_dir, fold_k, n_folds, attn_oof, attn_columns, split="val")

        # Guardar metadatos de dimensión si es embedding
        if use_embeddings and embedding_dim is not None:
            import json
            metadata_path = f"{eval_dir}/fold_{fold_k}/embedding_dim.json"
            Path(metadata_path).parent.mkdir(parents=True, exist_ok=True)
            with open(metadata_path, 'w') as f:
                json.dump({"embedding_dim": embedding_dim}, f)

        n_ok += 1

    print(f"\n✅ Features out-of-fold ({output_type}) generadas para {n_ok}/{n_run} folds "
          f"({n_skipped} omitidos).")
    return n_ok


def main():
    parser = argparse.ArgumentParser(
        description="Genera meta-features out-of-fold mediante CV anidada agrupada "
                    "por paciente (corrige el sesgo in-sample de test_abmil.py)"
    )
    parser.add_argument("--foundational_model", type=str, required=True)
    parser.add_argument("--latent_dim", type=int, default=None,
                        help="Dimensión del embedding. Si se omite, se auto-detecta del .h5.")
    parser.add_argument("--work_dir", type=str,
                        default="/home/JKP6679/Patho-Ensemble/PARADIS/datos/patches")
    parser.add_argument("--train_source", type=str, required=True)
    parser.add_argument("--tissue_patching", type=str, required=True)
    parser.add_argument("--task_name", type=str, required=True)
    parser.add_argument("--epochs", type=int, default=100,
                        help="Épocas de cada entrenamiento interno (igual que train_abmil.py)")
    parser.add_argument("--n_inner", type=int, default=3,
                        help="Folds de la CV interna. El coste total es n_inner veces "
                             "el del entrenamiento base.")
    parser.add_argument("--max_folds", type=int, default=None,
                        help="Limitar el número de folds externos a recorrer.")
    parser.add_argument("--oof_tag", type=str, default="oof",
                        help="Etiqueta del árbol de salida "
                             "(_train_eval_{tag}/val_outputs_{tag}/). Permite que "
                             "varios valores de --n_inner convivan sin pisarse.")
    parser.add_argument("--use_embeddings", action="store_true", default=False,
                        help="Extrae embeddings ponderados (512-dim) en lugar de predicciones de clase.")

    args = parser.parse_args()

    latent_dim = resolve_latent_dim(
        args.work_dir, args.train_source, args.foundational_model, args.latent_dim
    )

    build_oof(
        work_dir=args.work_dir, train_source=args.train_source,
        foundational_model=args.foundational_model,
        tissue_patching=args.tissue_patching, task_name=args.task_name,
        latent_dim=latent_dim, epochs=args.epochs, n_inner=args.n_inner,
        max_folds=args.max_folds, oof_tag=args.oof_tag, use_embeddings=args.use_embeddings,
    )


if __name__ == "__main__":
    main()
