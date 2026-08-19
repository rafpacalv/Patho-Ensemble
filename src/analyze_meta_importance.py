#!/usr/bin/env python3
"""
Analiza la importancia de cada modelo base en un meta-clasificador lineal.

Carga los coeficientes guardados por ensemble4.py y calcula qué modelo pesa
más en las predicciones, útil para entender qué modelos base son redundantes
o cuál es el dominante.

Uso:
    python src/analyze_meta_importance.py \\
        --results_dir .../abmil/ensemble4 \\
        --foundational_models ctranspath uni_v2 virchow_v1 \\
        [--temperature 1.0] [--output_json model_importance.json]

Requiere:
    - coefs.npy existente en --results_dir (generado por ensemble4.py)
    - --foundational_models en EL MISMO ORDEN usado al correr ensemble4.py
      (es crucial, porque ese orden no se persiste en el artefacto coefs.npy)
"""

import argparse
import json
import os
from pathlib import Path

import numpy as np

from utils import model_importance_from_coefs


def analyze_importance(results_dir, foundational_models, temperature=1.0, output_json=None):
    """
    Carga coefs.npy y calcula la importancia por modelo base.

    Args:
        results_dir (str): directorio donde está coefs.npy
        foundational_models (list): lista de nombres de modelos en orden
        temperature (float): exponente de suavizado
        output_json (str): ruta para guardar model_importance.json
                          (default: model_importance.json en results_dir)

    Returns:
        dict: {model_name: weight}
    """
    results_dir = Path(results_dir)
    coefs_path = results_dir / "coefs.npy"

    if not coefs_path.exists():
        raise FileNotFoundError(
            f"coefs.npy no encontrado en {results_dir}.\n"
            f"¿Se ejecutó ensemble4.py con --meta_model logreg u otro modelo lineal?"
        )

    coefs = np.load(coefs_path)
    print(f"📊 Loaded coefs.npy shape: {coefs.shape}")
    print(f"   Models: {foundational_models} ({len(foundational_models)} total)")

    num_models = len(foundational_models)
    weights = model_importance_from_coefs(
        coefs, num_models, model_names=foundational_models, temperature=temperature
    )

    # Computar rango entre folds si es 2D
    if coefs.ndim == 2:
        norms_per_fold = []
        block_size = coefs.shape[1] // num_models
        for fold_coefs in coefs:
            fold_norms = []
            for i in range(num_models):
                start = i * block_size
                end = (i + 1) * block_size
                norm_i = np.linalg.norm(fold_coefs[start:end], ord=2)
                fold_norms.append(norm_i)
            norms_per_fold.append(fold_norms)

        norms_per_fold = np.array(norms_per_fold)  # [n_folds, num_models]
        norm_mean = norms_per_fold.mean(axis=0)
        norm_std = norms_per_fold.std(axis=0)
        norm_min = norms_per_fold.min(axis=0)
        norm_max = norms_per_fold.max(axis=0)
    else:
        norm_mean = None
        norm_std = None
        norm_min = None
        norm_max = None

    # Imprimir tabla
    print(f"\n{'Model':<20} {'Weight':>10} {'L2 norm':>12} {'Range':>20} {'Rank':>6}")
    print("-" * 68)

    sorted_models = sorted(foundational_models, key=lambda m: weights[m], reverse=True)
    for rank, model in enumerate(sorted_models, 1):
        w = weights[model]
        if norm_mean is not None:
            idx = foundational_models.index(model)
            mean_norm = norm_mean[idx]
            std_norm = norm_std[idx]
            min_norm = norm_min[idx]
            max_norm = norm_max[idx]
            range_str = f"{min_norm:.3f}–{max_norm:.3f} (σ={std_norm:.3f})"
            print(f"{model:<20} {w:>10.3f} {mean_norm:>12.3f} {range_str:>20} {rank:>6}")
        else:
            print(f"{model:<20} {w:>10.3f} {'—':>12} {'—':>20} {rank:>6}")

    # Guardar JSON
    if output_json is None:
        output_json = results_dir / "model_importance.json"
    else:
        output_json = Path(output_json)

    with open(output_json, "w") as f:
        json.dump(weights, f, indent=2)

    print(f"\n✅ Importancias guardadas en {output_json}")
    return weights


def main():
    parser = argparse.ArgumentParser(
        description="Analiza la importancia de cada modelo base en un meta-clasificador."
    )
    parser.add_argument(
        "--results_dir",
        type=str,
        required=True,
        help="Directorio con coefs.npy (p.ej. .../abmil/ensemble4)",
    )
    parser.add_argument(
        "--foundational_models",
        nargs="+",
        type=str,
        required=True,
        help="Lista de nombres de modelos base EN EL MISMO ORDEN que se usó en ensemble4.py",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help="Exponente de suavizado (0=no-op, 1=lineal, >1=acentúa). Default: 1.0",
    )
    parser.add_argument(
        "--output_json",
        type=str,
        default=None,
        help="Ruta para guardar model_importance.json (default: {results_dir}/model_importance.json)",
    )

    args = parser.parse_args()
    analyze_importance(
        args.results_dir,
        args.foundational_models,
        temperature=args.temperature,
        output_json=args.output_json,
    )


if __name__ == "__main__":
    main()
