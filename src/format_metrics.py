"""
Utility para formatear y enriquecer métricas de clasificación.

Asegura que todas las métricas importantes (Accuracy, Precision, Recall, F1,
versiones macro y weighted) estén presentes en el resumen final.
"""
import json
import numpy as np
from pathlib import Path
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    balanced_accuracy_score, roc_auc_score, matthews_corrcoef
)


def compute_all_metrics(y_true, y_pred_proba, num_classes):
    """Calcula TODAS las métricas de clasificación (macro, weighted, binarias).

    Args:
        y_true: array de labels verdaderos
        y_pred_proba: array de probabilidades (n_samples, num_classes)
        num_classes: número de clases

    Returns:
        dict con todas las métricas
    """
    y_pred = np.argmax(y_pred_proba, axis=1)
    metrics = {}

    # AUROC
    try:
        if num_classes == 2:
            metrics["auc_roc"] = float(roc_auc_score(y_true, y_pred_proba[:, 1]))
        else:
            metrics["auc_roc"] = float(
                roc_auc_score(y_true, y_pred_proba, multi_class="ovr", average="macro", labels=list(range(num_classes)))
            )
    except (ValueError, IndexError):
        metrics["auc_roc"] = 0.5

    # Accuracy
    metrics["accuracy"] = float(accuracy_score(y_true, y_pred))

    # Balanced Accuracy (media de recall por clase)
    metrics["balanced_accuracy"] = float(balanced_accuracy_score(y_true, y_pred))

    # Precision (macro y weighted)
    metrics["precision_macro"] = float(precision_score(y_true, y_pred, average="macro", zero_division=0))
    metrics["precision_weighted"] = float(precision_score(y_true, y_pred, average="weighted", zero_division=0))

    # Recall (macro y weighted)
    metrics["recall_macro"] = float(recall_score(y_true, y_pred, average="macro", zero_division=0))
    metrics["recall_weighted"] = float(recall_score(y_true, y_pred, average="weighted", zero_division=0))

    # F1 (macro y weighted)
    metrics["f1_macro"] = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    metrics["f1_weighted"] = float(f1_score(y_true, y_pred, average="weighted", zero_division=0))

    # MCC (solo binario)
    if num_classes == 2:
        metrics["mcc"] = float(matthews_corrcoef(y_true, y_pred))

    return metrics


def enrich_metrics_summary(metrics_summary_path, y_true_list, y_pred_list, num_classes):
    """Enriquece un metrics_summary.json con cálculos adicionales explícitos.

    Lee el JSON existente y añade un bloque 'all_metrics' con todas las métricas.

    Args:
        metrics_summary_path: ruta al test_metrics_summary.json
        y_true_list: lista de arrays de labels por fold
        y_pred_list: lista de arrays de predicciones (probabilidades) por fold
        num_classes: número de clases
    """
    # Cargar JSON existente
    with open(metrics_summary_path, 'r') as f:
        summary = json.load(f)

    # Agregar métricas agregadas
    all_y = np.concatenate(y_true_list)
    all_p = np.concatenate(y_pred_list)

    all_metrics = compute_all_metrics(all_y, all_p, num_classes)
    summary["all_metrics"] = all_metrics

    # Guardar versión enriquecida
    with open(metrics_summary_path, 'w') as f:
        json.dump(summary, f, indent=2)


def print_metrics_table(metrics_dict):
    """Imprime métricas de forma legible (tabla ASCII).

    Args:
        metrics_dict: dict con métricas (p.ej. de 'all_metrics')
    """
    print("\n" + "="*60)
    print("MÉTRICAS DETALLADAS")
    print("="*60)

    # Orden: primero AUROC, luego accuracy, luego precision/recall/f1
    key_order = [
        "auc_roc",
        "accuracy",
        "balanced_accuracy",
        "precision_macro",
        "precision_weighted",
        "recall_macro",
        "recall_weighted",
        "f1_macro",
        "f1_weighted",
        "mcc"
    ]

    for key in key_order:
        if key in metrics_dict:
            val = metrics_dict[key]
            print(f"  {key:25s}: {val:.4f}")

    print("="*60 + "\n")


def generate_comparison_report(ensemble_metrics_path, ensemble_nm_metrics_path, output_path=None):
    """Genera un reporte comparativo entre dos métodos de ensemble.

    Args:
        ensemble_metrics_path: ruta a abmil/ensemble/test_metrics_summary.json
        ensemble_nm_metrics_path: ruta a abmil/ensemble_nm/test_metrics_summary.json
        output_path: ruta de salida (opcional; si None, imprime a stdout)
    """
    with open(ensemble_metrics_path, 'r') as f:
        ensemble = json.load(f)
    with open(ensemble_nm_metrics_path, 'r') as f:
        ensemble_nm = json.load(f)

    # Extraer "all_metrics" o lo disponible
    ens_metrics = ensemble.get("all_metrics", ensemble.get("overall", {}))
    ens_nm_metrics = ensemble_nm.get("all_metrics", ensemble_nm.get("overall", {}))

    report = []
    report.append("="*80)
    report.append("COMPARATIVA: ENSEMBLE PROMEDIADO vs NELDER-MEAD")
    report.append("="*80)
    report.append("")

    key_order = [
        "auc_roc", "accuracy", "balanced_accuracy",
        "precision_macro", "precision_weighted",
        "recall_macro", "recall_weighted",
        "f1_macro", "f1_weighted", "mcc"
    ]

    report.append(f"{'Métrica':<30s} {'Promediado':>15s} {'Nelder-Mead':>15s} {'Δ':>10s}")
    report.append("-"*80)

    for key in key_order:
        ens_val = ens_metrics.get(key, np.nan)
        ens_nm_val = ens_nm_metrics.get(key, np.nan)

        if np.isnan(ens_val) or np.isnan(ens_nm_val):
            continue

        delta = ens_nm_val - ens_val
        delta_str = f"{delta:+.4f}" if delta != 0 else "—"

        report.append(f"{key:<30s} {ens_val:>15.4f} {ens_nm_val:>15.4f} {delta_str:>10s}")

    report.append("="*80)
    report.append("")

    report_text = "\n".join(report)

    if output_path:
        with open(output_path, 'w') as f:
            f.write(report_text)
        print(f"✅ Reporte guardado en {output_path}")
    else:
        print(report_text)

    return report_text
