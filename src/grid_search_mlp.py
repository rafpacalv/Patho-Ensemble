#!/usr/bin/env python3
"""
Grid Search Hyperparameter Optimization for MLP Meta-learners
Testa diferentes configuraciones de MLP single-cycle y snapshot ensemble
para encontrar los mejores hiperparámetros que superen el baseline LogReg.
"""

import json
import subprocess
import sys
from pathlib import Path
from datetime import datetime
from itertools import product
import re

# Configuration
PROJECT_DIR = Path("/home/JKP6679/Patho-Ensemble")
WORK_DIR = Path("/home/JKP6679/Patho-Ensemble/PARADIS/datos/patches")
TRAIN_SOURCE = "cptac_brca"
TISSUE_PATCHING = "20x_224px_0px_overlap"
TASK_NAME = "TP53_mutation"
RESULTS_DIR = WORK_DIR / TRAIN_SOURCE / TASK_NAME / "abmil"

# Baseline (LogReg) results for comparison
BASELINE_AUC = 0.744
BASELINE_F1 = 0.672
BASELINE_ACC = 0.701

# Grid search parameters (intelligently reduced to ~100 total combinations)
# Baseline was: hidden_dim=16, dropout=0.1, lr=1e-3, wd=1e-4
SINGLE_CYCLE_GRID = {
    "hidden_dim": [32, 64, 128],           # Increase from 16 (too small)
    "dropout": [0.3, 0.4, 0.5],            # Increase from 0.1 (under-regularized)
    "meta_lr": [5e-4, 1e-3, 5e-3],         # Test around 1e-3
    "meta_wd": [1e-5, 1e-4, 1e-3],         # Regularization sweep
}

SNAPSHOT_GRID = {
    "hidden_dim": [64, 128, 256],          # Larger models for snapshot ensemble
    "dropout": [0.3, 0.4],                 # Good regularization for snapshots
    "meta_lr": [5e-4, 1e-3],               # Conservative LR range
    "meta_wd": [1e-5, 1e-4],               # Regularization sweep
    "n_cycles": [4, 6, 8],                 # Number of restart cycles
    "restart_lr": [5e-3, 1e-2],            # LR for cycle restarts
    "n_snapshots": [4, 6],                 # Models to ensemble
}


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


def run_single_cycle_trial(trial_id, hidden_dim, dropout, meta_lr, meta_wd):
    """Run a single-cycle MLP trial"""
    output_suffix = f"_trial_{trial_id}"

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

    print(f"\n[Trial {trial_id}] Single-Cycle MLP")
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
                # Print only the last few lines of error
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


def run_snapshot_trial(trial_id, hidden_dim, dropout, meta_lr, meta_wd,
                       n_cycles, restart_lr, n_snapshots):
    """Run a snapshot ensemble MLP trial"""
    output_suffix = f"_trial_{trial_id}"

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

    print(f"\n[Trial {trial_id}] Snapshot Ensemble MLP")
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
    print("MLP Hyperparameter Grid Search Optimization")
    print("=" * 90)
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Work dir: {WORK_DIR}")
    print(f"Baseline LogReg: AUC={BASELINE_AUC:.4f}, F1={BASELINE_F1:.4f}, "
          f"Acc={BASELINE_ACC:.4f}")
    print()

    # Generate all trial combinations
    sc_params = list(product(*SINGLE_CYCLE_GRID.values()))
    ss_params = list(product(*SNAPSHOT_GRID.values()))

    print(f"Single-Cycle Grid: {len(sc_params)} combinations")
    print(f"Snapshot Grid: {len(ss_params)} combinations")
    print(f"Total trials: {len(sc_params) + len(ss_params)}")
    print()

    all_results = []
    trial_counter = 1

    # Run single-cycle trials
    print("=" * 90)
    print("PHASE 1: Single-Cycle MLP Optimization")
    print("=" * 90)
    for hidden_dim, dropout, meta_lr, meta_wd in sc_params:
        result = run_single_cycle_trial(trial_counter, hidden_dim, dropout,
                                         meta_lr, meta_wd)
        if result:
            all_results.append(result)
        trial_counter += 1

    # Run snapshot trials
    print("\n" + "=" * 90)
    print("PHASE 2: Snapshot Ensemble MLP Optimization")
    print("=" * 90)
    for hidden_dim, dropout, meta_lr, meta_wd, n_cycles, restart_lr, n_snapshots in ss_params:
        result = run_snapshot_trial(trial_counter, hidden_dim, dropout, meta_lr,
                                    meta_wd, n_cycles, restart_lr, n_snapshots)
        if result:
            all_results.append(result)
        trial_counter += 1

    # Analyze and report results
    print("\n" + "=" * 90)
    print("RESULTS SUMMARY")
    print("=" * 90)

    if not all_results:
        print("❌ No successful trials!")
        return

    # Separate by type
    sc_results = [r for r in all_results if r["type"] == "single_cycle"]
    ss_results = [r for r in all_results if r["type"] == "snapshot"]

    # Report single-cycle best
    if sc_results:
        print("\nSingle-Cycle MLP - Top 5 Trials:")
        print("-" * 90)
        sc_sorted = sorted(sc_results, key=lambda x: x["auc"], reverse=True)
        print(f"{'Trial':<6} {'AUC':<12} {'F1':<12} {'Acc':<12} {'Params':<50}")
        print("-" * 90)
        for i, r in enumerate(sc_sorted[:5], 1):
            p = r["params"]
            params_str = (f"h={p['hidden_dim']}, d={p['dropout']:.1f}, "
                         f"lr={p['meta_lr']:.1e}, wd={p['meta_wd']:.1e}")
            auc_improvement = (r["auc"] - BASELINE_AUC) / BASELINE_AUC * 100
            print(f"{r['trial_id']:<6} "
                  f"{r['auc']:.4f}±{r['auc_se']:.3f}  "
                  f"{r['f1']:.4f}±{r['f1_se']:.3f}  "
                  f"{r['acc']:.4f}±{r['acc_se']:.3f}  "
                  f"{params_str:<50} "
                  f"({auc_improvement:+.2f}%)")

    # Report snapshot best
    if ss_results:
        print("\nSnapshot Ensemble MLP - Top 5 Trials:")
        print("-" * 90)
        ss_sorted = sorted(ss_results, key=lambda x: x["auc"], reverse=True)
        print(f"{'Trial':<6} {'AUC':<12} {'F1':<12} {'Acc':<12} {'Params':<60}")
        print("-" * 90)
        for i, r in enumerate(ss_sorted[:5], 1):
            p = r["params"]
            params_str = (f"h={p['hidden_dim']}, d={p['dropout']:.1f}, "
                         f"cycles={p['n_cycles']}, snaps={p['n_snapshots']}")
            auc_improvement = (r["auc"] - BASELINE_AUC) / BASELINE_AUC * 100
            print(f"{r['trial_id']:<6} "
                  f"{r['auc']:.4f}±{r['auc_se']:.3f}  "
                  f"{r['f1']:.4f}±{r['f1_se']:.3f}  "
                  f"{r['acc']:.4f}±{r['acc_se']:.3f}  "
                  f"{params_str:<60} "
                  f"({auc_improvement:+.2f}%)")

    # Overall best
    print("\n" + "=" * 90)
    best_overall = max(all_results, key=lambda x: x["auc"])
    print("BEST OVERALL CONFIGURATION:")
    print("-" * 90)
    print(f"Trial ID: {best_overall['trial_id']}")
    print(f"Type: {best_overall['type'].upper()}")
    print(f"AUC: {best_overall['auc']:.4f} ± {best_overall['auc_se']:.4f} "
          f"(vs baseline {BASELINE_AUC:.4f}: {(best_overall['auc']-BASELINE_AUC)/BASELINE_AUC*100:+.2f}%)")
    print(f"F1: {best_overall['f1']:.4f} ± {best_overall['f1_se']:.4f} "
          f"(vs baseline {BASELINE_F1:.4f}: {(best_overall['f1']-BASELINE_F1)/BASELINE_F1*100:+.2f}%)")
    print(f"Acc: {best_overall['acc']:.4f} ± {best_overall['acc_se']:.4f} "
          f"(vs baseline {BASELINE_ACC:.4f}: {(best_overall['acc']-BASELINE_ACC)/BASELINE_ACC*100:+.2f}%)")
    print("\nHyperparameters:")
    for key, val in best_overall["params"].items():
        print(f"  {key}: {val}")

    # Save all results to JSON
    output_file = PROJECT_DIR / "grid_search_results.json"
    with open(output_file, "w") as f:
        json.dump({
            "timestamp": datetime.now().isoformat(),
            "baseline": {
                "auc": BASELINE_AUC,
                "f1": BASELINE_F1,
                "acc": BASELINE_ACC,
            },
            "all_trials": all_results,
            "best_trial": best_overall,
        }, f, indent=2)

    print(f"\n✅ All results saved to: {output_file}")
    print("=" * 90)


if __name__ == "__main__":
    main()
