"""
Genera predicciones sobre el split de ENTRENAMIENTO (no val/test) para cada
fold, usando los modelos ABMIL ya entrenados por train_abmil.py.

Motor: abmil_engine.py (el mismo que usa train_abmil.py; sin dependencia de
patho_bench). Reutiliza load_config_yaml / load_features_cached /
create_fold_split / save_fold_outputs de train_abmil.py para no duplicar
lógica de splits ni de carga de features.

Salida:
    {work_dir}/{train_source}/{task_name}/abmil/
        {foundational_model}_{tissue_patching}_train_eval/val_outputs/fold_k/
            labels.npy, preds.npy

El sufijo "_train_eval/val_outputs" se conserva únicamente porque
ensemble4.py ya lee esa ruta exacta como `xdirs` (features de entrenamiento
del meta-learner); el contenido real son las predicciones del ABMIL de ese
fold sobre su propio split de TRAIN, no sobre val.

--epochs se acepta solo por compatibilidad de CLI con train_base_models.sbatch
(no se usa: aquí no se entrena nada, solo se corre inferencia con checkpoints
ya guardados por train_abmil.py).

--use_embeddings y --topk son dos modos de embedding alternativos (mutuamente
excluyentes), cada uno con su propio árbol de salida (val_outputs_embeddings,
val_outputs_topk{K}) para no pisarse ni con las predicciones de clase ni entre
sí: --use_embeddings agrega con la atención blanda del modelo sobre toda la
bolsa; --topk K agrega sólo los K parches de mayor atención, renormalizada
(ver abmil_engine.abmil_extract_topk_embeddings).
"""
import argparse
from pathlib import Path

import pandas as pd
import torch

import abmil_engine
from train_abmil import (
    load_config_yaml,
    load_features_cached,
    load_graphs_cached,
    create_fold_split,
    save_fold_outputs,
    save_fold_outputs_attn,
)
from train_abmil_fge import save_fold_outputs_fge
from utils import resolve_latent_dim


def abmil_predict_train_fold(
    work_dir, train_source, foundational_model, tissue_patching, task_name,
    fold_k, n_folds, task_col, num_classes, latent_dim, df, feats,
    arch_tag="", graphs=None, use_embeddings=False, topk=0
):
    """Carga el checkpoint del fold_k y predice sobre su propio split de train.

    Args:
        use_embeddings: si True, extrae embeddings ponderados (atención blanda
            sobre toda la bolsa); si False, predicciones de clase.
        topk: si >0, extrae embeddings usando sólo los `topk` parches de mayor
            atención (pooling duro, ver abmil_engine.abmil_extract_topk_embeddings).
            Implica embeddings (ignora `use_embeddings`) y escribe en un árbol
            propio (tag "topk{topk}") para no pisar ni el árbol de
            predicciones de clase ni el de embeddings de atención blanda.

    Devuelve (labels, preds) o (None, None) si falta el checkpoint o no hay
    datos de train para ese fold.
    """
    if topk and use_embeddings:
        raise ValueError("--topk y --use_embeddings son modos de embedding "
                          "alternativos; usa sólo uno de los dos.")
    extract_topk = topk > 0
    embed = use_embeddings or extract_topk
    device = "cuda" if torch.cuda.is_available() else "cpu"

    tag = f"_{arch_tag}" if arch_tag else ""
    output_dir = f"{work_dir}/{train_source}/{task_name}/abmil/{foundational_model}{tag}_{tissue_patching}"
    ckpt_path = f"{output_dir}/checkpoints/fold_{fold_k}/model.pt"

    if not Path(ckpt_path).exists():
        print(f"  ⚠️  Fold {fold_k}: checkpoint no encontrado ({ckpt_path}), se omite.")
        return None, None

    tr_df, _, _ = create_fold_split(df, fold_k, task_col)
    if tr_df is None or tr_df.empty:
        print(f"  ⚠️  Fold {fold_k}: sin datos de train, se omite.")
        return None, None

    tr_stems = tr_df['slide_id'].tolist()
    tr_y = tr_df[task_col].values

    # Obtener los splits de validación y test para generar predicciones de ambos cuando se usan embeddings
    _, va_df, te_df = create_fold_split(df, fold_k, task_col)
    va_stems = va_df['slide_id'].tolist() if va_df is not None and not va_df.empty else []
    va_y = va_df[task_col].values if va_df is not None and not va_df.empty else None
    te_stems = te_df['slide_id'].tolist() if te_df is not None and not te_df.empty else []
    te_y = te_df[task_col].values if te_df is not None and not te_df.empty else None

    # Los brazos con grafo guardan {state_dict, arch, n_layers, ...}, así que la
    # arquitectura se LEE del checkpoint en vez de reconstruirse a mano. Los
    # checkpoints de siempre son un state_dict pelado y siguen cargando igual.
    ckpt = torch.load(ckpt_path, map_location=device)
    if isinstance(ckpt, dict) and "state_dict" in ckpt:
        import spatial_abmil
        model_class = spatial_abmil.make_factory(
            ckpt["arch"], n_layers=ckpt.get("n_layers", 2),
            alpha_init=ckpt.get("alpha_init", 0.0)
        )
        model = model_class(latent_dim, num_classes, proj_dim=512, dropout=0.25).to(device)
        state = ckpt["state_dict"]
    else:
        # Arquitectura idéntica a la usada por abmil_engine.train_abmil()
        # (ver src/abmil_engine.py::ABMIL — hidden_dim=256 por defecto, no se
        # pasa explícito en train_abmil.py, proj_dim=512 y dropout=0.25 sí).
        ModelClass = abmil_engine.ABMIL_EMBEDDING if embed else abmil_engine.ABMIL
        model = ModelClass(
            in_dim=latent_dim, num_classes=num_classes,
            hidden_dim=256, proj_dim=512, dropout=0.25,
        ).to(device)
        state = ckpt
    # Cuando embed=True (use_embeddings o topk), el checkpoint tiene el
    # classifier que no existe en ABMIL_EMBEDDING; strict=False lo ignora.
    strict = not embed
    model.load_state_dict(state, strict=strict)
    model.eval()

    # embed_tag namespaces el árbol de salida (val_outputs_{embed_tag}/...),
    # igual que --fge_tag en train_abmil_fge.py: "embeddings" (atención
    # blanda de toda la bolsa) y "topk{k}" (pooling duro sobre los k parches
    # de más peso) conviven sin pisarse. No confundir con `tag` de arriba
    # (línea 69), que namespaces el checkpoint por --arch_tag.
    if extract_topk:
        extract_fn = lambda stems: abmil_engine.abmil_extract_topk_embeddings(
            model, feats, stems, device, graphs, k=topk)
        embed_tag = f"topk{topk}"
    elif use_embeddings:
        extract_fn = lambda stems: abmil_engine.abmil_extract_embeddings(
            model, feats, stems, device, graphs)
        embed_tag = "embeddings"

    if embed:
        tr_preds = extract_fn(tr_stems)
        embedding_dim = model.get_embedding_dim()
        attn_columns = None
    else:
        tr_preds, tr_attn_stats, attn_columns = abmil_engine.abmil_predict_with_attention(
            model, feats, tr_stems, device, graphs
        )
        embedding_dim = None

    eval_dir = f"{output_dir}_train_eval"
    if embed:
        eval_dir += f"_{embed_tag}"
        # save_fold_outputs_fge espera [n_snapshots, N, C]; con embeddings es [N, D]
        # así que le pasamos [1, N, D] y guarda con ese tag.
        # Esto genera: {eval_dir}/val_outputs_{embed_tag}/fold_k/
        save_fold_outputs_fge(eval_dir, fold_k, n_folds, tr_y,
                             tr_preds[None, ...], split="val", tag=embed_tag)
    else:
        save_fold_outputs(eval_dir, fold_k, n_folds, tr_y, tr_preds, split="val")
        save_fold_outputs_attn(eval_dir, fold_k, n_folds, tr_attn_stats, attn_columns, split="val")

    # Guardar metadatos de dimensión si es embedding
    if embed and embedding_dim is not None:
        import json
        metadata_path = f"{eval_dir}/fold_{fold_k}/embedding_dim.json"
        Path(metadata_path).parent.mkdir(parents=True, exist_ok=True)
        with open(metadata_path, 'w') as f:
            json.dump({"embedding_dim": embedding_dim}, f)

    # Generar también predicciones de VALIDACIÓN y TEST cuando se usan embeddings
    # (ensemble4.py necesita val_outputs_{embed_tag} y test_outputs_{embed_tag} para vdirs y tdirs)
    if embed:
        # Validación
        if len(va_stems) > 0 and va_y is not None:
            va_preds = extract_fn(va_stems)
            # Guardar embeddings de validación en el directorio base con split="val"
            save_fold_outputs_fge(output_dir, fold_k, n_folds, va_y,
                                 va_preds[None, ...], split="val", tag=embed_tag)

        # Test
        if len(te_stems) > 0 and te_y is not None:
            te_preds = extract_fn(te_stems)
            # Guardar embeddings de test en el directorio base con split="test"
            save_fold_outputs_fge(output_dir, fold_k, n_folds, te_y,
                                 te_preds[None, ...], split="test", tag=embed_tag)

    return tr_y, tr_preds


def abmil_test(work_dir, train_source, foundational_model, tissue_patching, task_name,
                latent_dim, epochs=None, arch="abmil", arch_tag="",
                graph_mode="lattice", shuffle_coords=False, use_embeddings=False, topk=0):
    """Recorre todos los folds de k=all.tsv y genera predicciones (o embeddings) de train para cada uno.

    Args:
        use_embeddings: si True, extrae embeddings ponderados; si False, predicciones de clase.
        topk: si >0, embeddings top-k (ver abmil_predict_train_fold). Incompatible con use_embeddings.
    """
    config_path = f"{work_dir}/{train_source}/{task_name}/config.yaml"
    task_col, num_classes = load_config_yaml(config_path)

    split_file = f"{work_dir}/{train_source}/{task_name}/k=all.tsv"
    df = pd.read_csv(split_file, sep="\t")
    df = df.dropna(subset=[task_col]).reset_index(drop=True)

    fold_cols = [c for c in df.columns if c.startswith("fold_")]
    n_folds = len(fold_cols)

    output_type = f"embeddings top{topk}" if topk else ("embeddings" if use_embeddings else "predicciones")
    print(f"\n{'='*60}")
    print(f"ABMIL Test ({output_type} sobre train): {foundational_model}")
    print(f"  Dataset: {train_source}/{task_name}")
    print(f"  Latent dim: {latent_dim}, Num classes: {num_classes}, Folds: {n_folds}")
    print(f"{'='*60}\n")

    # Features de todas las slides, una sola vez (evita recargar H5 por fold)
    all_slide_ids = set(df['slide_id'].unique())
    feats = load_features_cached(work_dir, train_source, foundational_model, all_slide_ids, latent_dim)

    # Mismos grafos que en entrenamiento: la semilla del control sale del
    # slide_id por sha256, así que la permutación coincide con la de train.
    graphs = None
    if arch != "abmil":
        graphs = load_graphs_cached(
            work_dir, train_source, foundational_model, all_slide_ids,
            mode=graph_mode, shuffle=shuffle_coords,
        )

    n_ok, n_skipped = 0, 0
    for fold_k in range(n_folds):
        labels, preds = abmil_predict_train_fold(
            work_dir, train_source, foundational_model, tissue_patching, task_name,
            fold_k, n_folds, task_col, num_classes, latent_dim, df, feats,
            arch_tag=arch_tag, graphs=graphs, use_embeddings=use_embeddings, topk=topk
        )
        if labels is None:
            n_skipped += 1
        else:
            n_ok += 1

    print(f"\n✅ {output_type.capitalize()} de train generadas para {n_ok}/{n_folds} folds "
          f"({n_skipped} omitidos por falta de checkpoint/datos).")
    if n_skipped:
        print("   ⚠️  Revisa que train_abmil.py haya completado esos folds "
              "antes de correr ensemble4.py.")


def main():
    parser = argparse.ArgumentParser(
        description="Genera predicciones sobre el split de train (features del "
                     "meta-learner de ensemble4.py) usando checkpoints ya "
                     "entrenados por train_abmil.py"
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
                              "compatibilidad con train_base_models.sbatch")
    # Deben coincidir con los de train_abmil.py: el tag localiza el checkpoint
    # y los grafos tienen que ser los mismos que se usaron al entrenar.
    parser.add_argument("--arch", type=str, default="abmil",
                        choices=["abmil", "smooth", "sage_bn", "sage"])
    parser.add_argument("--arch_tag", type=str, default="")
    parser.add_argument("--graph_mode", type=str, default="lattice",
                        choices=["lattice", "knn", "self"])
    parser.add_argument("--shuffle_coords", action="store_true", default=False)
    parser.add_argument("--use_embeddings", action="store_true", default=False,
                        help="Extrae embeddings ponderados (512-dim) en lugar de predicciones de clase.")
    parser.add_argument("--topk", type=int, default=0,
                        help="Si >0, extrae embeddings usando sólo los K parches de mayor "
                             "atención en vez de la bolsa completa (pooling duro). Escribe en "
                             "un árbol propio (val_outputs_topk{K}/...) y es incompatible con "
                             "--use_embeddings.")
    args = parser.parse_args()

    if args.arch != "abmil" and not args.arch_tag:
        parser.error("--arch != abmil requiere --arch_tag (el mismo del entrenamiento)")
    if args.topk and args.use_embeddings:
        parser.error("--topk y --use_embeddings son modos de embedding alternativos; usa sólo uno.")
    if args.topk and args.arch != "abmil":
        parser.error("--topk no soporta --arch != abmil (SpatialABMIL): ver "
                      "abmil_extract_topk_embeddings en abmil_engine.py.")

    latent_dim = resolve_latent_dim(
        args.work_dir, args.train_source, args.foundational_model, args.latent_dim
    )

    abmil_test(
        work_dir=args.work_dir,
        train_source=args.train_source,
        foundational_model=args.foundational_model,
        tissue_patching=args.tissue_patching,
        task_name=args.task_name,
        latent_dim=latent_dim,
        epochs=args.epochs,
        arch=args.arch,
        arch_tag=args.arch_tag,
        graph_mode=args.graph_mode,
        shuffle_coords=args.shuffle_coords,
        use_embeddings=args.use_embeddings,
        topk=args.topk,
    )


if __name__ == "__main__":
    main()
