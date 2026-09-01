#!/usr/bin/env python3
"""
Ablation Study: Desacoplar arquitectura del modelo vs técnica de entrenamiento (snapshot)

Objetivo: Determinar si la mejora en snapshot ensemble viene de:
  A) La técnica de snapshot ensemble en sí
  B) La arquitectura más grande (hidden_dim)
  C) Una combinación de ambos

Experiments:
  Phase 1: Single-Cycle MLP CON RED GRANDE (hidden_dim=[128,256,512])
           → Comparar vs grid search original (hidden_dim=[32,64,128])
           → Si aún es malo → snapshot es la clave
           → Si mejora mucho → arquitectura es importante

  Phase 2: Snapshot MLP CON RED PEQUEÑA (hidden_dim=[32,64,128])
           → Comparar vs grid search original (hidden_dim=[64,128,256])
           → Si sigue siendo bueno → snapshot funciona incluso pequeño
           → Si empeora → necesitas red grande para snapshot
"""

import json
import subprocess
import sys
from pathlib import Path
from datetime import datetime
from itertools import product
import re

# Configuration
PROJECT_DIR = Path("/shared/home/JKP6679/Patho-Ensemble")
WORK_DIR = Path("/home/JKP6679/Patho-Ensemble/PARADIS/datos/patches")
TRAIN_SOURCE = "cptac_brca"
TISSUE_PATCHING = "20x_224px_0px_overlap"
TASK_NAME = "TP53_mutation"
RESULTS_DIR = WORK_DIR / TRAIN_SOURCE / TASK_NAME / "abmil"

# Baseline (LogReg) results for comparison
BASELINE_AUC = 0.744
BASELINE_F1 = 0.672
BASELINE_ACC = 0.701

# Grid search parameters for ABLATION STUDY
# Phase 1: Single-Cycle with LARGE architecture (same as snapshot in original grid)
SINGLE_CYCLE_LARGE_GRID = {
    "hidden_dim": [128, 256, 512],      # LARGE (same as snapshot)
    "dropout": [0.3, 0.4],              # Same as snapshot
    "meta_lr": [5e-4, 1e-3],            # Same as snapshot
    "meta_wd": [1e-5, 1e-4],            # Same as snapshot
}
# 3 × 2 × 2 × 2 = 24 trials

# Phase 2: Snapshot with SMALL architecture (same as single-cycle in original grid)
SNAPSHOT_SMALL_GRID = {
    "hidden_dim": [32, 64, 128],        # SMALL (same as single-cycle)
    "dropout": [0.3, 0.4, 0.5],         # Same as single-cycle
    "meta_lr": [5e-4, 1e-3, 5e-3],      # Same as single-cycle
    "meta_wd": [1e-5, 1e-4, 1e-3],      # Same as single-cycle
    "n_cycles": [4, 6],                 # Reduced cycles
    "restart_lr": [5e-3, 1e-2],
    "n_snapshots": [4, 6],
}
# 3 × 3 × 3 × 3 × 2 × 2 × 2 = 216 trials


def extract_metrics_from_json(json_path):
    """Extract AUC, F1, Accuracy from test_metrics_summary.json"""
    try:
        with open(json_path) as f:
            data = json.load(f)

        auc_data = data.get("macro-ovr-auc", {})
        f1_data = data.get("macro-f1", {})
        acc_data = data.get("acc", {})

        return {
            "auc": auc_data.get("mean", None),
            "auc_se": auc_data.get("se", None),
            "f1": f1_data.get("mean", None),
            "f1_se": f1_data.get("se", None),
            "acc": acc_data.get("mean", None),
            "acc_se": acc_data.get("se", None),
        }
    except Exception as e:
        print(f"  ⚠️  Error reading {json_path}: {e}", file=sys.stderr)
        return None


def run_single_cycle_large_trial(trial_id, hidden_dim, dropout, meta_lr, meta_wd):
    """Run a single-cycle MLP trial with LARGE architecture"""
    output_suffix = f"_ablation_sc_large_{trial_id}"

    # Build command
    cmd = [
        "python", "src/ensemble4.py",
        "--foundational_models", "ctranspath", "uni_v2", "virchow_v1",
        "--work_dir", str(WORK_DIR),
        "--train_source", TRAIN_SOURCE,
        "--tissue_patching", TISSUE_PATCHING,
        "--task_name", TASK_NAME,
        "--meta_model", "mlp",
        "--meta_hidden_dim", str(hidden_dim),
        "--meta_dropout", str(dropout),
        "--meta_epochs", "100",
        "--meta_lr", str(meta_lr),
        "--meta_wd", str(meta_wd),
        "--meta_patience", "10",
        "--results_subdir_suffix", output_suffix,
    ]

    print(f"\n[Ablation Trial {trial_id}] Single-Cycle MLP (LARGE architecture)")
    print(f"  Params: hidden_dim={hidden_dim}, dropout={dropout:.1f}, "
          f"lr={meta_lr:.1e}, wd={meta_wd:.1e}")
    print(f"  Running: {' '.join(cmd[1:4])}...")

    try:
        result = subprocess.run(
            cmd,
            cwd=PROJECT_DIR,
            capture_output=True,
            text=True,
            timeout=1800,
        )

        if result.returncode != 0:
            print(f"  ❌ Failed (exit code {result.returncode})")
            if "Error" in result.stderr:
                err_lines = result.stderr.strip().split("\n")[-3:]
                for line in err_lines:
                    print(f"     {line}")
            return None

        # Extract metrics
        output_dir = f"ensemble4_mlp{output_suffix}"
        metrics_file = RESULTS_DIR / output_dir / "test_metrics_summary.json"
        metrics = extract_metrics_from_json(metrics_file)

        if metrics is None:
            return None

        print(f"  ✅ AUC={metrics['auc']:.4f}±{metrics['auc_se']:.4f}, "
              f"F1={metrics['f1']:.4f}±{metrics['f1_se']:.4f}, "
              f"Acc={metrics['acc']:.4f}±{metrics['acc_se']:.4f}")

        return {
            "trial_id": trial_id,
            "phase": "ablation_single_cycle_large",
            "type": "single_cycle",
            "params": {
                "hidden_dim": hidden_dim,
                "dropout": dropout,
                "meta_lr": meta_lr,
                "meta_wd": meta_wd,
            },
            **metrics,
        }

    except subprocess.TimeoutExpired:
        print(f"  ⏱️  Timeout (30 min exceeded)")
        return None
    except Exception as e:
        print(f"  ❌ Exception: {e}")
        return None


def run_snapshot_small_trial(trial_id, hidden_dim, dropout, meta_lr, meta_wd,
                             n_cycles, restart_lr, n_snapshots):
    """Run a snapshot ensemble MLP trial with SMALL architecture"""
    output_suffix = f"_ablation_snap_small_{trial_id}"

    cmd = [
        "python", "src/ensemble4.py",
        "--foundational_models", "ctranspath", "uni_v2", "virchow_v1",
        "--work_dir", str(WORK_DIR),
        "--train_source", TRAIN_SOURCE,
        "--tissue_patching", TISSUE_PATCHING,
        "--task_name", TASK_NAME,
        "--meta_model", "mlp_snapshot",
        "--meta_hidden_dim", str(hidden_dim),
        "--meta_dropout", str(dropout),
        "--meta_epochs", "120",
        "--meta_lr", str(meta_lr),
        "--meta_wd", str(meta_wd),
        "--n_cycles", str(n_cycles),
        "--restart_lr", str(restart_lr),
        "--n_snapshots_ensemble", str(n_snapshots),
        "--results_subdir_suffix", output_suffix,
    ]

    print(f"\n[Ablation Trial {trial_id}] Snapshot Ensemble MLP (SMALL architecture)")
    print(f"  Params: hidden_dim={hidden_dim}, dropout={dropout:.1f}, "
          f"lr={meta_lr:.1e}, wd={meta_wd:.1e}, cycles={n_cycles}, "
          f"restart_lr={restart_lr:.1e}, snapshots={n_snapshots}")
    print(f"  Running: {' '.join(cmd[1:4])}...")

    try:
        result = subprocess.run(
            cmd,
            cwd=PROJECT_DIR,
            capture_output=True,
            text=True,
            timeout=2400,  # 40 min
        )

        if result.returncode != 0:
            print(f"  ❌ Failed (exit code {result.returncode})")
            if "Error" in result.stderr:
                err_lines = result.stderr.strip().split("\n")[-3:]
                for line in err_lines:
                    print(f"     {line}")
            return None

        output_dir = f"ensemble4_mlp_snapshot{output_suffix}"
        metrics_file = RESULTS_DIR / output_dir / "test_metrics_summary.json"
        metrics = extract_metrics_from_json(metrics_file)

        if metrics is None:
            return None

        print(f"  ✅ AUC={metrics['auc']:.4f}±{metrics['auc_se']:.4f}, "
              f"F1={metrics['f1']:.4f}±{metrics['f1_se']:.4f}, "
              f"Acc={metrics['acc']:.4f}±{metrics['acc_se']:.4f}")

        return {
            "trial_id": trial_id,
            "phase": "ablation_snapshot_small",
            "type": "snapshot",
            "params": {
                "hidden_dim": hidden_dim,
                "dropout": dropout,
                "meta_lr": meta_lr,
                "meta_wd": meta_wd,
                "n_cycles": n_cycles,
                "restart_lr": restart_lr,
                "n_snapshots": n_snapshots,
            },
            **metrics,
        }

    except subprocess.TimeoutExpired:
        print(f"  ⏱️  Timeout (40 min exceeded)")
        return None
    except Exception as e:
        print(f"  ❌ Exception: {e}")
        return None


def main():
    print("=" * 90)
    print("ABLATION STUDY: Arquitectura vs Snapshot Ensemble")
    print("=" * 90)
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Work dir: {WORK_DIR}")
    print(f"Baseline LogReg: AUC={BASELINE_AUC:.4f}, F1={BASELINE_F1:.4f}, "
          f"Acc={BASELINE_ACC:.4f}")
    print()

    print("Objetivo:")
    print("  Phase 1: ¿Single-Cycle mejora si usamos red GRANDE (como snapshot)?")
    print("  Phase 2: ¿Snapshot sigue siendo bueno con red PEQUEÑA (como single-cycle)?")
    print()

    # Generate all trial combinations
    sc_large_params = list(product(*SINGLE_CYCLE_LARGE_GRID.values()))
    snap_small_params = list(product(*SNAPSHOT_SMALL_GRID.values()))

    print(f"Single-Cycle (LARGE) Grid: {len(sc_large_params)} combinations")
    print(f"Snapshot (SMALL) Grid: {len(snap_small_params)} combinations")
    print(f"Total trials: {len(sc_large_params) + len(snap_small_params)}")
    print()

    all_results = []
    trial_counter = 1

    # Phase 1: Single-cycle with LARGE architecture
    print("=" * 90)
    print("PHASE 1: Single-Cycle MLP with LARGE Architecture")
    print("=" * 90)
    print("(Comparar vs original single-cycle small: ¿mejora la arquitectura?)")
    print()
    for hidden_dim, dropout, meta_lr, meta_wd in sc_large_params:
        result = run_single_cycle_large_trial(trial_counter, hidden_dim, dropout,
                                              meta_lr, meta_wd)
        if result:
            all_results.append(result)
        trial_counter += 1

    # Phase 2: Snapshot with SMALL architecture
    print("\n" + "=" * 90)
    print("PHASE 2: Snapshot Ensemble MLP with SMALL Architecture")
    print("=" * 90)
    print("(Comparar vs original snapshot large: ¿funciona snapshot con red pequeña?)")
    print()
    for hidden_dim, dropout, meta_lr, meta_wd, n_cycles, restart_lr, n_snapshots in snap_small_params:
        result = run_snapshot_small_trial(trial_counter, hidden_dim, dropout, meta_lr,
                                          meta_wd, n_cycles, restart_lr, n_snapshots)
        if result:
            all_results.append(result)
        trial_counter += 1

    # Analyze and report results
    print("\n" + "=" * 90)
    print("ABLATION STUDY RESULTS")
    print("=" * 90)

    if not all_results:
        print("❌ No successful trials!")
        return

    # Separate by phase
    sc_large_results = [r for r in all_results if r["phase"] == "ablation_single_cycle_large"]
    snap_small_results = [r for r in all_results if r["phase"] == "ablation_snapshot_small"]

    # Report single-cycle large
    if sc_large_results:
        print("\n" + "=" * 90)
        print("PHASE 1: Single-Cycle MLP (LARGE Architecture)")
        print("=" * 90)
        sc_large_sorted = sorted(sc_large_results, key=lambda x: x["auc"], reverse=True)

        print("\nTop 5 Trials (LARGE):")
        print("-" * 90)
        print(f"{'Trial':<8} {'AUC':<12} {'F1':<12} {'Acc':<12} {'Hidden Dim':<12} {'Dropout':<10}")
        print("-" * 90)
        for i, r in enumerate(sc_large_sorted[:5], 1):
            p = r["params"]
            print(f"{r['trial_id']:<8} "
                  f"{r['auc']:.4f}±{r['auc_se']:.3f}  "
                  f"{r['f1']:.4f}±{r['f1_se']:.3f}  "
                  f"{r['acc']:.4f}±{r['acc_se']:.3f}  "
                  f"{p['hidden_dim']:<12} "
                  f"{p['dropout']:<10.1f}")

        best_large = sc_large_sorted[0]
        print(f"\nBest Single-Cycle LARGE: AUC={best_large['auc']:.4f}")
        print(f"  (Original small single-cycle best was ≈ 0.71)")
        if best_large['auc'] > 0.73:
            print("  → CONCLUSIÓN: La arquitectura grande ayuda MUCHO")
        else:
            print("  → CONCLUSIÓN: La arquitectura grande NO ayuda lo suficiente")

    # Report snapshot small
    if snap_small_results:
        print("\n" + "=" * 90)
        print("PHASE 2: Snapshot Ensemble MLP (SMALL Architecture)")
        print("=" * 90)
        snap_small_sorted = sorted(snap_small_results, key=lambda x: x["auc"], reverse=True)

        print("\nTop 5 Trials (SMALL):")
        print("-" * 90)
        print(f"{'Trial':<8} {'AUC':<12} {'F1':<12} {'Acc':<12} {'Hidden Dim':<12} {'Cycles':<8}")
        print("-" * 90)
        for i, r in enumerate(snap_small_sorted[:5], 1):
            p = r["params"]
            print(f"{r['trial_id']:<8} "
                  f"{r['auc']:.4f}±{r['auc_se']:.3f}  "
                  f"{r['f1']:.4f}±{r['f1_se']:.3f}  "
                  f"{r['acc']:.4f}±{r['acc_se']:.3f}  "
                  f"{p['hidden_dim']:<12} "
                  f"{p['n_cycles']:<8}")

        best_small = snap_small_sorted[0]
        print(f"\nBest Snapshot SMALL: AUC={best_small['auc']:.4f}")
        print(f"  (Original large snapshot best was ≈ 0.745)")
        if best_small['auc'] > 0.74:
            print("  → CONCLUSIÓN: Snapshot funciona BIEN incluso con red pequeña")
        else:
            print("  → CONCLUSIÓN: Snapshot necesita red grande para ser efectivo")

    # Overall comparison
    print("\n" + "=" * 90)
    print("CONCLUSIONES DEL ABLATION STUDY")
    print("=" * 90)

    if sc_large_results and snap_small_results:
        best_sc_large = max(sc_large_results, key=lambda x: x["auc"])
        best_snap_small = max(snap_small_results, key=lambda x: x["auc"])

        print(f"\nBest Single-Cycle (LARGE):  AUC = {best_sc_large['auc']:.4f} ± {best_sc_large['auc_se']:.4f}")
        print(f"Best Snapshot (SMALL):      AUC = {best_snap_small['auc']:.4f} ± {best_snap_small['auc_se']:.4f}")
        print(f"Diferencia:                 {abs(best_snap_small['auc'] - best_sc_large['auc']):.4f}")

        print("\nInterpretación:")
        if best_snap_small['auc'] > best_sc_large['auc'] + 0.01:
            print("  ✅ Snapshot ensemble ES mejor, incluso con red pequeña")
            print("     → La mejora NO viene solo de la arquitectura grande")
        elif best_snap_small['auc'] > best_sc_large['auc'] - 0.01:
            print("  🟡 Snapshot y Single-Cycle (LARGE) son similares")
            print("     → La mejora en grid original probablemente viene de arquitectura + snapshot")
        else:
            print("  ❌ Single-Cycle (LARGE) es mejor que Snapshot (SMALL)")
            print("     → La arquitectura grande es crítica para obtener AUC > 0.74")

    # Save all results to JSON
    output_file = PROJECT_DIR / "ablation_study_results.json"
    with open(output_file, "w") as f:
        json.dump({
            "timestamp": datetime.now().isoformat(),
            "baseline": {
                "auc": BASELINE_AUC,
                "f1": BASELINE_F1,
                "acc": BASELINE_ACC,
            },
            "all_trials": all_results,
        }, f, indent=2)

    print(f"\n✅ All results saved to: {output_file}")
    print("=" * 90)


if __name__ == "__main__":
    main()
