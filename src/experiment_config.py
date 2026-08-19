#!/usr/bin/env python3
"""
Lector de la matriz de experimentos (experiments.yaml).

Fusiona `defaults` con cada entrada de `experiments`, filtra por `enabled` y
emite el resultado en un formato que bash puede consumir con `eval`, de modo que
los .sbatch no necesiten parsear YAML.

Uso desde bash:

    # Listar experimentos activos (una línea "dataset<TAB>task" por experimento)
    python src/experiment_config.py --list-active

    # Volcar la config de un experimento como variables de shell
    eval "$(python src/experiment_config.py --emit-shell --dataset cptac_brca --task TP53_mutation)"
    echo "$WORK_DIR $MODELS $TISSUE_PATCHING"

Uso desde Python:

    from experiment_config import load_experiments
    for exp in load_experiments():
        print(exp["dataset"], exp["task"], exp["models"])
"""
import argparse
import os
import shlex
import sys
from pathlib import Path

import yaml

DEFAULT_CONFIG = Path(__file__).resolve().parent.parent / "experiments.yaml"


def load_config(config_path=None):
    """Carga y valida el YAML de experimentos."""
    path = Path(config_path or os.environ.get("EXPERIMENTS_CONFIG") or DEFAULT_CONFIG)
    if not path.exists():
        raise FileNotFoundError(f"No se encontró el fichero de experimentos: {path}")
    with open(path) as f:
        cfg = yaml.safe_load(f)

    for key in ("work_dir", "defaults", "experiments"):
        if key not in cfg:
            raise ValueError(f"'{key}' ausente en {path}")
    return cfg


def load_experiments(config_path=None, only_enabled=True):
    """
    Devuelve la lista de experimentos con los `defaults` ya fusionados.

    Cada entrada de `experiments` puede sobrescribir cualquier clave de
    `defaults` (p. ej. `models` para un dataset con pocos embeddings).
    """
    cfg = load_config(config_path)
    defaults = cfg["defaults"]
    work_dir = cfg["work_dir"]

    resolved = []
    for entry in cfg["experiments"]:
        if only_enabled and not entry.get("enabled", False):
            continue
        exp = dict(defaults)      # copia de los valores por defecto
        exp.update(entry)         # la fila concreta tiene prioridad
        exp["work_dir"] = work_dir
        if "dataset" not in exp or "task" not in exp:
            raise ValueError(f"Experimento sin 'dataset'/'task': {entry}")
        resolved.append(exp)
    return resolved


def find_experiment(dataset, task, config_path=None):
    """
    Busca un experimento concreto, ignorando el flag `enabled`.

    Permite lanzar una combinación puntual con DATASET=... TASK=... aunque en el
    YAML esté desactivada. Si no figura en el fichero, se construye a partir de
    `defaults` para no obligar a editar el YAML por una prueba suelta.
    """
    for exp in load_experiments(config_path, only_enabled=False):
        if exp["dataset"] == dataset and exp["task"] == task:
            return exp

    cfg = load_config(config_path)
    exp = dict(cfg["defaults"])
    exp.update({"dataset": dataset, "task": task, "work_dir": cfg["work_dir"]})
    print(f"# aviso: {dataset}/{task} no figura en experiments.yaml; se usan defaults",
          file=sys.stderr)
    return exp


FGE_DEFAULTS = {"lr_1": 2e-4, "lr_2": 2e-5, "cycle_length": 4,
                "n_cycles": 6, "cycle_patience": 2, "include_base": False,
                "schedule": "fge"}

META_DEFAULTS = {"base_source": "standard", "meta_features": "insample",
                 "feature_space": "prob", "drop_redundant_class": False, "reweight": False}


def emit_shell(exp):
    """Imprime la config como asignaciones de shell listas para `eval`."""
    metas = exp.get("meta_configs", [])
    variants = exp.get("fge_variants", [])

    lines = [
        f"WORK_DIR={shlex.quote(str(exp['work_dir']))}",
        f"DATASET={shlex.quote(str(exp['dataset']))}",
        f"TASK={shlex.quote(str(exp['task']))}",
        f"TISSUE_PATCHING={shlex.quote(str(exp['tissue_patching']))}",
        f"EPOCHS={shlex.quote(str(exp['epochs']))}",
        f"MODELS={shlex.quote(' '.join(exp['models']))}",
        f"MAX_FOLDS={shlex.quote(str(exp.get('max_folds', '') or ''))}",
        # Nombres de las variantes FGE, en orden
        f"FGE_VARIANTS={shlex.quote(' '.join(v['name'] for v in variants))}",
        # Configs de meta-learner, un campo por posición:
        #   name:meta_model:base_source:meta_features:feature_space:drop_redundant:reweight
        # Los campos 3-7 se añadieron después; los valores por defecto
        # (META_DEFAULTS) reproducen el comportamiento anterior, así que las
        # filas del YAML que no los declaran siguen significando lo mismo.
        "META_CONFIGS={}".format(shlex.quote(" ".join(
            ":".join([
                m["name"],
                m["meta_model"],
                c["base_source"],
                c["meta_features"],
                c["feature_space"],
                "1" if c["drop_redundant_class"] else "0",
                "1" if c["reweight"] else "0",
            ])
            for m in metas
            for c in [{**META_DEFAULTS, **m}]
        ))),
    ]

    # Una línea por variante: FGE_CFG_<NOMBRE>="lr1 lr2 clen ncyc pat incbase schedule"
    # El orden de los campos lo consume `read -r` en train_base_models_fge_v2.sbatch;
    # los campos nuevos van al final para no romper esa lectura posicional.
    for v in variants:
        c = {**FGE_DEFAULTS, **v}
        key = "FGE_CFG_" + str(v["name"]).upper().replace("-", "_")
        val = " ".join(str(c[k]) for k in
                       ("lr_1", "lr_2", "cycle_length", "n_cycles", "cycle_patience"))
        val += " 1" if c["include_base"] else " 0"
        val += f" {c['schedule']}"
        lines.append(f"{key}={shlex.quote(val)}")

    print("\n".join(lines))


def main():
    p = argparse.ArgumentParser(description="Lector de experiments.yaml")
    p.add_argument("--config", default=None, help="Ruta al YAML (por defecto experiments.yaml)")
    p.add_argument("--list-active", action="store_true",
                   help="Lista 'dataset<TAB>task' de los experimentos con enabled: true")
    p.add_argument("--list", action="store_true", help="Lista legible de experimentos activos")
    p.add_argument("--emit-shell", action="store_true", help="Emite variables de shell")
    p.add_argument("--dataset", default=None)
    p.add_argument("--task", default=None)
    args = p.parse_args()

    if args.list_active:
        for exp in load_experiments(args.config):
            print(f"{exp['dataset']}\t{exp['task']}")
        return

    if args.list:
        exps = load_experiments(args.config)
        if not exps:
            print("No hay experimentos activos (enabled: true) en experiments.yaml")
            return
        print(f"Experimentos activos ({len(exps)}):\n")
        for exp in exps:
            variants = exp.get("fge_variants", [])
            metas = exp.get("meta_configs", [])
            print(f"  {exp['dataset']}/{exp['task']}")
            print(f"    modelos:  {' '.join(exp['models'])}")
            mf = exp.get('max_folds')
            print(f"    patching: {exp['tissue_patching']}   epochs: {exp['epochs']}"
                  f"   folds: {mf if mf else 'todos'}")
            print(f"    variantes FGE ({len(variants)}):")
            for v in variants:
                c = {**FGE_DEFAULTS, **v}
                print(f"      - {c['name']:<12} [{c['schedule']}] lr_1={c['lr_1']:<7} "
                      f"clen={c['cycle_length']:<3} pat={c['cycle_patience']}  "
                      f"include_base={c['include_base']}")
            print(f"    meta-learners ({len(metas)}):")
            for m in metas:
                c = {**META_DEFAULTS, **m}
                extra = []
                if c["feature_space"] != "prob":
                    extra.append(c["feature_space"])
                if c["drop_redundant_class"]:
                    extra.append("drop-clase")
                print(f"      - {m['name']:<24} {m['meta_model']:<12} "
                      f"bases={c['base_source']:<10} features={c['meta_features']:<9}"
                      f"{' ' + ','.join(extra) if extra else ''}")
            print()
        return

    if args.emit_shell:
        if not args.dataset or not args.task:
            p.error("--emit-shell requiere --dataset y --task")
        emit_shell(find_experiment(args.dataset, args.task, args.config))
        return

    p.print_help()


if __name__ == "__main__":
    main()
