from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

try:
    import matplotlib.pyplot as plt
except Exception:  # pragma: no cover
    plt = None


def write_rows(path: str | Path, rows: list[dict]) -> None:
    if not rows:
        return
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def curve_point_rows(config_name: str, split_name: str, evaluation: dict, kind: str) -> list[dict]:
    if kind == "pr":
        return [
            {"config": config_name, "split": split_name, "kind": kind, "point_index": index, "recall": recall, "precision": precision}
            for index, (recall, precision) in enumerate(zip(evaluation["recall"], evaluation["precision"]))
        ]
    if kind == "roc":
        return [
            {"config": config_name, "split": split_name, "kind": kind, "point_index": index, "fpr": fpr, "tpr": tpr}
            for index, (fpr, tpr) in enumerate(zip(evaluation["fpr"], evaluation["tpr"]))
        ]
    return []


def save_pr_plot(reference_eval, compare_eval, out_path: str | Path, title: str, *, reference_label: str, compare_label: str) -> bool:
    if plt is None or len(reference_eval["recall"]) == 0 or len(compare_eval["recall"]) == 0:
        return False
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    ax.plot(reference_eval["recall"], reference_eval["precision"], label=f"{reference_label} AP={reference_eval['average_precision']:.4f}", linewidth=2)
    ax.plot(compare_eval["recall"], compare_eval["precision"], label=f"{compare_label} AP={compare_eval['average_precision']:.4f}", linewidth=2)
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title(title)
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)
    return True


def save_roc_plot(reference_eval, compare_eval, out_path: str | Path, title: str, *, reference_label: str, compare_label: str) -> bool:
    if plt is None or len(reference_eval["fpr"]) == 0 or len(compare_eval["fpr"]) == 0:
        return False
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    ax.plot(reference_eval["fpr"], reference_eval["tpr"], label=f"{reference_label} AUC={reference_eval['roc_auc']:.4f}", linewidth=2)
    ax.plot(compare_eval["fpr"], compare_eval["tpr"], label=f"{compare_label} AUC={compare_eval['roc_auc']:.4f}", linewidth=2)
    ax.plot([0, 1], [0, 1], linestyle="--", linewidth=1)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(title)
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)
    return True


def save_convergence_plot(rows: list[dict], out_path: str | Path, title: str) -> bool:
    if plt is None or not rows:
        return False
    loglik_rows = [row for row in rows if np.isfinite(row.get("loglik", np.nan))]
    subgrad_rows = [row for row in rows if np.isfinite(row.get("subgradient_l2", np.nan))]
    if not loglik_rows and not subgrad_rows:
        return False
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 1, figsize=(8, 8), sharex=True)
    if loglik_rows:
        axes[0].plot([row["iteration"] for row in loglik_rows], [row["loglik"] for row in loglik_rows], linewidth=2)
    if subgrad_rows:
        axes[1].plot([row["iteration"] for row in subgrad_rows], [row["subgradient_l2"] for row in subgrad_rows], linewidth=2)
    axes[0].set_title(title)
    axes[0].set_ylabel("Log-likelihood")
    axes[1].set_ylabel("Subgradient L2 norm")
    axes[1].set_xlabel("Iteration")
    axes[0].grid(True, alpha=0.3)
    axes[1].grid(True, alpha=0.3)
    if subgrad_rows and all(row["subgradient_l2"] > 0 for row in subgrad_rows):
        axes[1].set_yscale("log")
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)
    return True

