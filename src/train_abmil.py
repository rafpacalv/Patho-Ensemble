"""
Entrenamiento de ABMIL individual sobre features HDF5 de PARADIS.

Motor: abmil_engine.py (autocontenido, sin dependencia de patho_bench).
Salida: checkpoints, val_outputs/{labels,preds}.npy, test_outputs/{labels,preds}.npy,
        {val,test}_metrics/fold_k/metrics.json + {val,test}_metrics_summary.json.

Split: Lee k=all.tsv o k={fold}.tsv.
  Si la columna fold_k trae 'val', lo usa; si no, genera split train/val interno.
"""
import argparse
import os
import yaml
import json
import pandas as pd
import numpy as np
import torch
from pathlib import Path
import h5py
from tqdm import tqdm

import abmil_engine
from utils import Metrics, get_features_dir, resolve_latent_dim


def load_config_yaml(path):
    """Carga config.yaml, devuelve (task_col, num_classes)."""
    with open(path, 'r') as f:
        cfg = yaml.safe_load(f)
    task_col = cfg.get("task_col", "task")
    label_dict = cfg.get("label_dict", {})
    num_classes = len(label_dict) if label_dict else 2  # default: binary
    return task_col, num_classes


def load_features_cached(work_dir, train_source, foundational_model, slide_ids, latent_dim):
    """Carga features HDF5 una sola vez, cachea en memoria.

    Devuelve dict {slide_id: torch.Tensor (n_patches, latent_dim)}
    """
    feats_dir = get_features_dir(work_dir, train_source, foundational_model)
    feats = {}

    print(f"Cargando features de {feats_dir}...")
    for slide_id in tqdm(slide_ids, desc="Cargando features"):
        h5_path = f"{feats_dir}/{slide_id}.h5"
        if not Path(h5_path).exists():
            raise FileNotFoundError(f"Feature file not found: {h5_path}")

        with h5py.File(h5_path, 'r') as h:
            x = h["features"][:]  # (n_patches, dim)
            if x.shape[1] != latent_dim:
                print(f"  WARNING: {slide_id} has dim {x.shape[1]}, expected {latent_dim}")
            feats[slide_id] = torch.tensor(x, dtype=torch.float32)

    return feats


def load_graphs_cached(work_dir, train_source, foundational_model, slide_ids,
                       mode="lattice", shuffle=False):
    """Construye un grafo espacial por slide desde los `coords` del MISMO .h5.

    Se deja aparte de load_features_cached en vez de engordar su valor de
    retorno: esa función la importan test_abmil, train_abmil_fge, test_abmil_fge
    y build_oof_features, y los cuatro hacen `.to(device)` sobre los valores del
    dict. Cambiar el tipo de retorno obligaría a tocar los cuatro y el bucle del
    motor, sobre el camino de código que produjo todos los números del informe.

    Los coords vienen alineados fila a fila con las features en el propio .h5, y
    nada del pipeline reordena ni submuestrea parches, así que el grafo
    construido aquí sigue siendo válido en train, val, test y OOF.
    """
    import graph_utils

    feats_dir = get_features_dir(work_dir, train_source, foundational_model)
    graphs, infos = {}, []

    print(f"Construyendo grafos espaciales (mode={mode}, shuffle={shuffle})...")
    for slide_id in tqdm(slide_ids, desc="Grafos"):
        with h5py.File(f"{feats_dir}/{slide_id}.h5", 'r') as h:
            coords = h["coords"][:]
            # ctranspath_monai no trae attrs en coords; el default de 448 es el
            # que corresponde a 20x_224px sobre slides de 40x, y se verifica
            # abajo mirando cuántas slides caen en retícula.
            stride = int(h["coords"].attrs.get("patch_size_level0", 448))

        # Semilla derivada del slide_id por sha256: fija por slide e idéntica
        # entre folds y entre splits. Con hash() de Python cambiaría por
        # proceso y el control dejaría de ser un control.
        seed = graph_utils.seed_from_slide_id(slide_id) if shuffle else None
        g, info = graph_utils.slide_graph(coords, stride, mode=mode, shuffle_seed=seed)
        graphs[slide_id] = g
        infos.append(info)

    n_tiles = sum(i["n"] for i in infos)
    n_iso = sum(i["n_isolated"] for i in infos)
    n_fb = sum(i["method"] == "knn_fallback" for i in infos)
    mean_deg = float(np.average([i["mean_deg"] for i in infos],
                                weights=[i["n"] for i in infos]))
    print(f"  {len(graphs)} grafos, grado medio {mean_deg:.2f}, "
          f"aislados {n_iso}/{n_tiles} ({100 * n_iso / n_tiles:.2f} %), "
          f"fallback k-NN en {n_fb} slides")

    # La guarda va sobre lo que de verdad indica que el grafo está ROTO, no
    # sobre una banda de grado calibrada con un solo modelo. El grado medio
    # legítimo depende de lo agresiva que fuese la extracción al filtrar
    # parches: uni_v2 da 7.75 y ctranspath_monai 5.34 sobre la MISMA retícula
    # de 448, porque conserva 1.7x menos parches. Rechazar 5.34 fue un error de
    # calibración, no una detección.
    if mode == "lattice":
        if n_fb > 0.5 * len(infos):
            raise ValueError(
                f"{n_fb}/{len(infos)} slides fuera de retícula: el stride "
                "asumido no es el de esta extracción"
            )
        if n_iso > 0.25 * n_tiles:
            raise ValueError(
                f"{100 * n_iso / n_tiles:.1f} % de nodos sin vecinos: el grafo "
                "no aporta contexto espacial"
            )
        if mean_deg < 1.0:
            raise ValueError(f"grado medio {mean_deg:.2f}: grafo degenerado")
    return graphs


def create_fold_split(df, fold_k, task_col):
    """Extrae train/test/val del fold_k-ésimo.

    Devuelve (tr_df, va_df, te_df) — DataFrames filtrados.
    """
    fold_col = f"fold_{fold_k}"
    if fold_col not in df.columns:
        return None, None, None

    tr_df = df[df[fold_col] == "train"].reset_index(drop=True)
    te_df = df[df[fold_col] == "test"].reset_index(drop=True)
    va_df = df[df[fold_col] == "val"].reset_index(drop=True) if "val" in df[fold_col].values else None

    return tr_df, va_df, te_df


def save_fold_outputs(output_dir, fold_k, n_folds, labels, preds, split="val"):
    """Guarda labels.npy y preds.npy en {split}_outputs/[fold_k/] (según n_folds)."""
    if n_folds > 1:
        output_path = f"{output_dir}/{split}_outputs/fold_{fold_k}"
    else:
        output_path = f"{output_dir}/{split}_outputs"

    Path(output_path).mkdir(parents=True, exist_ok=True)
    np.save(f"{output_path}/labels.npy", labels)
    np.save(f"{output_path}/preds.npy", preds)


def save_fold_outputs_attn(output_dir, fold_k, n_folds, attn_stats, columns, split="val"):
    """Guarda atención stats: attn_stats.npy + attn_stats_columns.json."""
    if n_folds > 1:
        output_path = f"{output_dir}/{split}_outputs/fold_{fold_k}"
    else:
        output_path = f"{output_dir}/{split}_outputs"

    Path(output_path).mkdir(parents=True, exist_ok=True)
    np.save(f"{output_path}/attn_stats.npy", attn_stats)

    import json
    with open(f"{output_path}/attn_stats_columns.json", "w") as f:
        json.dump({"columns": columns, "n_stats": len(columns)}, f)


def abmil_train_test(
    work_dir, train_source, foundational_model, tissue_patching, task_name,
    latent_dim, epochs, fold_k, n_folds, task_col, num_classes, max_folds=None,
    arch="abmil", n_layers=2, arch_tag="", graph_mode="lattice", shuffle_coords=False,
    alpha_init=0.0, multi_head_attention=False, n_attention_heads=4, bag_size=0
):
    """Entrena y evalúa ABMIL para un fold específico.

    Con arch='abmil' (por defecto) el comportamiento es idéntico al de siempre:
    ni se leen coords, ni se construyen grafos, ni cambia el directorio.

    Devuelve (val_labels, val_preds, test_labels, test_preds)
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # El tag va en el nombre del MODELO, no en un --base_source, para que las
    # variantes coexistan y se puedan MEZCLAR en un mismo stack: ensemble4.py
    # compone las rutas por concatenación (f'{f}_{tissue_patching}') y nunca
    # llama a get_features_dir, así que un "modelo" ctranspath_gnn resuelve solo.
    # Es el mismo precedente que --fge_tag.
    tag = f"_{arch_tag}" if arch_tag else ""
    output_dir = f"{work_dir}/{train_source}/{task_name}/abmil/{foundational_model}{tag}_{tissue_patching}"
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Leer splits
    split_file = f"{work_dir}/{train_source}/{task_name}/k=all.tsv"
    if fold_k != 'all' and Path(f"{work_dir}/{train_source}/{task_name}/k={fold_k}.tsv").exists():
        split_file = f"{work_dir}/{train_source}/{task_name}/k={fold_k}.tsv"

    df = pd.read_csv(split_file, sep="\t")

    # Mapear labels
    label_col = task_col
    if label_col not in df.columns:
        raise ValueError(f"Column {label_col} not found in {split_file}")

    df = df.dropna(subset=[label_col]).reset_index(drop=True)

    if fold_k == 'all':
        # Recorre todos los folds
        all_val_labels, all_val_preds = [], []
        all_test_labels, all_test_preds = [], []
        fold_cols = [c for c in df.columns if c.startswith("fold_")]
        # max_folds acota cuántos folds se recorren. Con datasets caros (muchos
        # parches por slide) 50 folds pueden ser días de GPU; 10 folds con un
        # test grande dan intervalos igual de estrechos por bootstrap.
        n_run = min(max_folds, len(fold_cols)) if max_folds else len(fold_cols)
        if n_run < len(fold_cols):
            print(f"  (usando {n_run} de {len(fold_cols)} folds disponibles)")

        for fk in range(n_run):
            # OJO: esta llamada es recursiva y hay que reenviar TODOS los
            # parámetros de arquitectura. Si no se hace, cada fold se entrena
            # con los defaults (arch='abmil', arch_tag='') y escribe en el
            # directorio del BASELINE — entrena el modelo equivocado y además
            # pisa la referencia contra la que se compara.
            vl, vp, tl, tp = abmil_train_test(
                work_dir, train_source, foundational_model, tissue_patching, task_name,
                latent_dim, epochs, str(fk), len(fold_cols), task_col, num_classes,
                max_folds=None, arch=arch, n_layers=n_layers, arch_tag=arch_tag,
                graph_mode=graph_mode, shuffle_coords=shuffle_coords,
                alpha_init=alpha_init, multi_head_attention=multi_head_attention,
                n_attention_heads=n_attention_heads, bag_size=bag_size
            )
            # Only add non-empty folds to aggregation
            if len(tl) > 0 and len(tp) > 0:
                all_val_labels.append(vl)
                all_val_preds.append(vp)
                all_test_labels.append(tl)
                all_test_preds.append(tp)
            else:
                print(f"  ⚠️  Fold {fk}: skipped (no test data)")

        # Agregación de métricas (solo si hay validación)
        if all_val_labels:
            print("\nCalculando métricas de validación...")
            Metrics(
                task_type='classification',
                model_kwargs={'num_classes': num_classes},
                num_bootstraps=100,
                results_dir=output_dir,
                split='val',
                num_folds=len(all_val_labels),  # Usar número real de folds con datos
                all_labels_across_folds=all_val_labels,
                all_preds_across_folds=all_val_preds
            ).run()
            print("✅ Métricas de validación completadas")
        else:
            print("\n⚠️  No validation data found for any fold (this is OK if split file has no explicit val splits)")

        print("\nCalculando métricas de test...")
        if all_test_labels:
            Metrics(
                task_type='classification',
                model_kwargs={'num_classes': num_classes},
                num_bootstraps=100,
                results_dir=output_dir,
                split='test',
                num_folds=len(all_test_labels),  # Usar número real de folds con datos
                all_labels_across_folds=all_test_labels,
                all_preds_across_folds=all_test_preds
            ).run()
            print(f"✅ Métricas de test completadas ({len(all_test_labels)} folds con datos)")
        else:
            print("\n❌ ERROR: No test data found for any fold")

        return None, None, None, None

    # Fold específico
    fold_k_int = int(fold_k)
    tr_df, va_df, te_df = create_fold_split(df, fold_k_int, label_col)

    if tr_df is None or te_df is None or te_df.empty:
        print(f"Fold {fold_k}: sin datos")
        return np.array([]), np.array([]), np.array([]), np.array([])

    # Cargar features
    all_slide_ids = set(df['slide_id'].unique())
    feats = load_features_cached(work_dir, train_source, foundational_model, all_slide_ids, latent_dim)

    # Grafos espaciales: sólo para los brazos con grafo. Con arch='abmil' esto
    # es None y el motor toma exactamente el camino de siempre.
    graphs, model_factory = None, None
    if arch != "abmil":
        import spatial_abmil
        graphs = load_graphs_cached(
            work_dir, train_source, foundational_model, all_slide_ids,
            mode=graph_mode, shuffle=shuffle_coords,
        )
        model_factory = spatial_abmil.make_factory(arch, n_layers=n_layers,
                                                   alpha_init=alpha_init)

    # Extraer datos
    tr_stems = tr_df['slide_id'].tolist()
    tr_y = tr_df[label_col].values
    tr_groups = tr_df['case_id'].values

    te_stems = te_df['slide_id'].tolist()
    te_y = te_df[label_col].values

    if va_df is None or va_df.empty:
        va_stems = []
        va_y = np.array([])
    else:
        va_stems = va_df['slide_id'].tolist()
        va_y = va_df[label_col].values

    print(f"\nFold {fold_k}: train={len(tr_stems)}, val={len(va_stems)}, test={len(te_stems)}")

    # Entrenar
    model = abmil_engine.train_abmil(
        feats=feats,
        stems=tr_stems,
        y=tr_y,
        groups=tr_groups,
        in_dim=latent_dim,
        n_classes=num_classes,
        device=device,
        seed=42 + fold_k_int,
        max_epochs=epochs,
        patience=8,
        proj_dim=512,
        dropout=0.25,
        wd=1e-4,
        val_frac=0.2 if va_stems == [] else None,  # genera split interno si no hay val explícita
        graphs=graphs,
        model_factory=model_factory,
        multi_head_attention=multi_head_attention,
        n_attention_heads=n_attention_heads,
        bag_size=bag_size
    )

    extra = ""
    if arch == "smooth":
        # El alpha aprendido es el diagnóstico principal: alpha ~ 0 significa
        # que el modelo ha descartado el promediado espacial, y eso se lee
        # directamente, sin pasar por un delta de AUC dentro del ruido.
        extra = f", alphas={[round(a, 4) for a in model.alphas()]}"
    print(f"Fold {fold_k}: ABMIL entrenado (val_auc={model._val_auc:.4f}{extra})")

    # Salvar checkpoint
    ckpt_dir = f"{output_dir}/checkpoints/fold_{fold_k}"
    Path(ckpt_dir).mkdir(parents=True, exist_ok=True)
    if arch == "abmil":
        # Formato de siempre: los ~150 checkpoints ya guardados se leen con
        # torch.load + load_state_dict directo y deben seguir haciéndolo.
        torch.save(model.state_dict(), f"{ckpt_dir}/model.pt")
    else:
        # Los brazos con grafo guardan la arquitectura junto a los pesos, para
        # que test_abmil.py no tenga que reconstruirla a mano. Hay ya tres
        # sitios que la reconstruyen a mano y divergen; no se añade un cuarto.
        torch.save({
            "state_dict": model.state_dict(),
            "arch": arch,
            "n_layers": n_layers,
            "graph_mode": graph_mode,
            "shuffle_coords": shuffle_coords,
            "alpha_init": alpha_init,
        }, f"{ckpt_dir}/model.pt")

    # Predicción en validación (si no existe split explícita, se usó la del train — pero igual predicimos)
    if va_stems:
        va_preds, va_attn_stats, attn_columns = abmil_engine.abmil_predict_with_attention(
            model, feats, va_stems, device, graphs
        )
        save_fold_outputs(output_dir, fold_k_int, n_folds if fold_k != 'all' else 1, va_y, va_preds, split="val")
        save_fold_outputs_attn(output_dir, fold_k_int, n_folds if fold_k != 'all' else 1, va_attn_stats, attn_columns, split="val")
    else:
        # No hay validación explícita; devolver vacío
        va_preds = np.array([])
        va_y = np.array([])

    # Predicción en test
    te_preds, te_attn_stats, attn_columns = abmil_engine.abmil_predict_with_attention(
        model, feats, te_stems, device, graphs
    )
    save_fold_outputs(output_dir, fold_k_int, n_folds if fold_k != 'all' else 1, te_y, te_preds, split="test")
    save_fold_outputs_attn(output_dir, fold_k_int, n_folds if fold_k != 'all' else 1, te_attn_stats, attn_columns, split="test")

    return va_y, va_preds, te_y, te_preds


def main():
    parser = argparse.ArgumentParser(description="Train ABMIL model from HDF5 features")
    parser.add_argument("--foundational_model", type=str, required=True)
    parser.add_argument("--latent_dim", type=int, default=None,
                        help="Dimensión del embedding. Si se omite, se auto-detecta del .h5.")
    parser.add_argument("--work_dir", type=str,
                        default="/home/JKP6679/Patho-Ensemble/PARADIS/datos/patches")
    parser.add_argument("--train_source", type=str, required=True)
    parser.add_argument("--tissue_patching", type=str, required=True)
    parser.add_argument("--task_name", type=str, required=True)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--mtr", type=str, default="")
    parser.add_argument("--create_val", action='store_true', default=False)
    parser.add_argument("--n_folds", type=int, default=50)
    parser.add_argument("--max_folds", type=int, default=None,
                        help="Limitar el número de folds a recorrer con --fold all "
                             "(por defecto: todos los del k=all.tsv).")
    parser.add_argument("--fold", type=str, default="all")

    # ── brazo espacial ──────────────────────────────────────────────────────
    parser.add_argument("--arch", type=str, default="abmil",
                        choices=["abmil", "smooth", "sage_bn", "sage"],
                        help="Arquitectura. 'abmil' (por defecto) es el camino "
                             "de siempre y no lee coords. El resto añaden "
                             "message passing sobre el grafo de tiles ANTES de "
                             "la atención, en orden creciente de capacidad.")
    parser.add_argument("--n_layers", type=int, default=2,
                        help="Rondas de message passing (K). Con paso de 448 px "
                             "a 40x, K=2 da un contexto de ~560 um.")
    parser.add_argument("--arch_tag", type=str, default="",
                        help="Sufijo del directorio de salida, p.ej. 'gnn' -> "
                             "ctranspath_gnn_20x_224px_0px_overlap/. Las features "
                             "se siguen leyendo de --foundational_model.")
    parser.add_argument("--graph_mode", type=str, default="lattice",
                        choices=["lattice", "knn", "self"],
                        help="'self' es el control negativo que elimina el "
                             "vecindario manteniendo los parámetros.")
    parser.add_argument("--alpha_init", type=float, default=0.0,
                        help="Valor inicial de alpha en --arch smooth. 0.0 hace que el\nmodelo arranque siendo ABMIL exacto. Un valor sustantivo (0.3) sirve para\ncomprobar si el entrenamiento EMPUJA alpha de vuelta a 0: eso distingue\n'el espacio no aporta' de 'el gradiente no llego a mover alpha'.")
    parser.add_argument("--shuffle_coords", action="store_true", default=False,
                        help="Control negativo: relabela los nodos con una "
                             "permutación fija por slide. El grafo queda "
                             "isomorfo (mismo grado, mismo E, mismos FLOPs) y "
                             "sólo se destruye el vínculo feature<->posición.")
    parser.add_argument("--bag_size", type=int, default=0,
                       help="Muestrea N parches sin reemplazo en cada forward de "
                            "entrenamiento (augmentación de bolsa). Evaluación y "
                            "predicción usan siempre la bolsa completa. 0 = desactivado "
                            "(comportamiento de siempre). Incompatible con --arch != abmil.")
    parser.add_argument("--multi_head_attention", action="store_true", default=False,
                        help="Usa MultiHeadGatedAttention en lugar de single-head. "
                             "Cada cabeza es independiente; combinación por mean-pool.")
    parser.add_argument("--n_attention_heads", type=int, default=4,
                        help="Número de cabezas en multi-head attention (ignorado si "
                             "--multi_head_attention es False).")

    args = parser.parse_args()

    if args.arch != "abmil" and not args.arch_tag:
        # Sin tag, el brazo espacial escribiría encima de los outputs del
        # baseline. Se exige explícito en vez de inventar uno.
        parser.error("--arch != abmil requiere --arch_tag (p.ej. --arch_tag gnn)")

    # Leer config
    config_path = f"{args.work_dir}/{args.train_source}/{args.task_name}/config.yaml"
    task_col, num_classes = load_config_yaml(config_path)

    args.latent_dim = resolve_latent_dim(
        args.work_dir, args.train_source, args.foundational_model, args.latent_dim
    )

    print(f"\n{'='*60}")
    print(f"ABMIL Training: {args.foundational_model}")
    print(f"  Dataset: {args.train_source}/{args.task_name}")
    print(f"  Latent dim: {args.latent_dim}, Num classes: {num_classes}")
    print(f"  Epochs: {args.epochs}, Fold: {args.fold}")
    if args.arch != "abmil":
        print(f"  Arch: {args.arch} (K={args.n_layers}), tag: {args.arch_tag}, "
              f"grafo: {args.graph_mode}, shuffle: {args.shuffle_coords}, "
              f"alpha_init: {args.alpha_init}")
    print(f"{'='*60}\n")

    # Determinar número de folds
    split_file = f"{args.work_dir}/{args.train_source}/{args.task_name}/k=all.tsv"
    df_check = pd.read_csv(split_file, sep="\t")
    fold_cols = [c for c in df_check.columns if c.startswith("fold_")]
    n_folds = len(fold_cols)

    # Entrenar
    abmil_train_test(
        work_dir=args.work_dir,
        train_source=args.train_source,
        foundational_model=args.foundational_model,
        tissue_patching=args.tissue_patching,
        task_name=args.task_name,
        latent_dim=args.latent_dim,
        epochs=args.epochs,
        fold_k=args.fold,
        n_folds=n_folds,
        task_col=task_col,
        num_classes=num_classes,
        arch=args.arch,
        n_layers=args.n_layers,
        arch_tag=args.arch_tag,
        graph_mode=args.graph_mode,
        shuffle_coords=args.shuffle_coords,
        alpha_init=args.alpha_init,
        multi_head_attention=args.multi_head_attention,
        n_attention_heads=args.n_attention_heads,
        bag_size=args.bag_size
    )

    print(f"\n✅ Training completed")


if __name__ == "__main__":
    main()
