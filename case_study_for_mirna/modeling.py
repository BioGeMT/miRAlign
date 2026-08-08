from __future__ import annotations

import pickle
import time
from pathlib import Path

import numpy as np

from src.miralign import miRAlign
from src.optimization_functions import baseline_parameters, create_subgradient_step
from src.positional_alignment import pos_aware_align_glocal, pos_aware_align_local

ALIGNERS = {
    "local": pos_aware_align_local,
    "glocal": pos_aware_align_glocal,
}


def label_prior_from_name(name: str):
    if name == "none":
        return None
    if name == "symmetric_95_5":
        return np.array([[950, 50], [50, 950]], dtype=float)
    if name == "symmetric_90_10":
        return np.array([[900, 100], [100, 900]], dtype=float)
    if name == "symmetric_80_20":
        return np.array([[800, 200], [200, 800]], dtype=float)
    raise ValueError(f"Unknown label prior: {name}")


def sample_weight_from_name(name: str, frame) -> np.ndarray | None:
    if name == "none":
        return None
    weights = np.ones(len(frame), dtype=float)
    if len(frame) == 0:
        return weights
    if name == "mirna_balanced":
        counts = frame["noncodingRNA"].map(frame["noncodingRNA"].value_counts()).to_numpy(dtype=float)
        weights = len(frame) / (frame["noncodingRNA"].nunique() * counts)
    elif name == "gene_balanced":
        counts = frame["gene"].map(frame["gene"].value_counts()).to_numpy(dtype=float)
        weights = len(frame) / (frame["gene"].nunique() * counts)
    elif name == "pair_balanced":
        pair_counts = frame.groupby(["noncodingRNA", "gene"]).size()
        counts = np.array(
            [pair_counts.loc[(mirna, gene)] for mirna, gene in zip(frame["noncodingRNA"], frame["gene"])],
            dtype=float,
        )
        weights = len(frame) / (len(pair_counts) * counts)
    elif name == "mirna_gene_sqrt":
        mirna_counts = frame["noncodingRNA"].map(frame["noncodingRNA"].value_counts()).to_numpy(dtype=float)
        gene_counts = frame["gene"].map(frame["gene"].value_counts()).to_numpy(dtype=float)
        weights = 1 / np.sqrt(mirna_counts * gene_counts)
        weights = weights / np.mean(weights)
    else:
        raise ValueError(f"Unknown sample weight: {name}")
    return weights


def priors_from_precision(prior_precision: float, mirna_length: int):
    if float(prior_precision) <= 0:
        return None, None, None
    params = baseline_parameters(mirna_length)
    return params["M"], params["G_miR"], params["G_gene"]


def fit_configuration(fit_inputs, config: dict):
    start_time = time.perf_counter()
    model_length = int(config["model_length"])
    M_prior, G_miR_prior, G_gene_prior = priors_from_precision(float(config["prior_precision"]), model_length)
    result = miRAlign(
        mirna_list=fit_inputs[0],
        gene_list=fit_inputs[1],
        label_list=fit_inputs[2],
        aligner=ALIGNERS[config["aligner"]],
        step_function=create_subgradient_step(
            float(config["step_scale"]),
            float(config["step_power"]),
            int(config["step_decay_burnin"]),
        ),
        M_prior=M_prior,
        G_miR_prior=G_miR_prior,
        G_gene_prior=G_gene_prior,
        prior_precision=float(config["prior_precision"]),
        label_prior=label_prior_from_name(config["label_prior"]),
        model_length=model_length,
        sample_weight=sample_weight_from_name(config["sample_weight"], fit_inputs[3]),
        MAX_ITER=int(config["max_iter"]),
        num_threads=int(config["num_threads"]),
        verbose=False,
    )
    return result, time.perf_counter() - start_time


def model_from_result(result: dict, config: dict):
    return {
        "aligner": ALIGNERS[config["aligner"]],
        "aligner_name": config["aligner"],
        "M": result["M"],
        "G_miR": result["G_miR"],
        "G_gene": result["G_gene"],
        "alpha": result["alpha"],
        "label_observation_probs": result.get("label_observation_probs"),
    }


def summarize_result(config: dict, result: dict, runtime_seconds: float) -> dict:
    optimizer_warnings = result.get("optimizer_warnings", [])
    return {
        **config,
        "status": "ok",
        "error": "",
        "optimizer_warning_count": len(optimizer_warnings),
        "optimizer_warnings": "; ".join(optimizer_warnings),
        "final_loglik": result.get("final_loglik", np.nan),
        "final_train_auprc": result.get("auprc_trajectory", [np.nan])[-1],
        "alpha": result.get("alpha", np.nan),
        "runtime_seconds": round(runtime_seconds, 3),
        "iterations_completed": len(result.get("subgradient_norm_trajectory", [])),
    }


def build_trajectory_rows(config: dict, result: dict) -> list[dict]:
    loglik = list(result.get("loglik_trajectory", []))
    auprc = list(result.get("auprc_trajectory", []))
    subgrad = list(result.get("subgradient_norm_trajectory", []))
    rows = []
    for iteration in range(max(len(loglik), len(auprc), len(subgrad))):
        rows.append(
            {
                **config,
                "iteration": iteration,
                "loglik": loglik[iteration] if iteration < len(loglik) else np.nan,
                "train_auprc": auprc[iteration] if iteration < len(auprc) else np.nan,
                "subgradient_l2": subgrad[iteration] if iteration < len(subgrad) else np.nan,
            }
        )
    return rows


def rank_successful(summary_rows: list[dict]) -> list[dict]:
    successful = [row for row in summary_rows if row.get("status") == "ok"]
    return sorted(successful, key=lambda row: (row.get("ap_validation", -np.inf), row.get("roc_auc_validation", -np.inf)), reverse=True)


def save_model(path: str | Path, model: dict, config: dict, summary: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as handle:
        pickle.dump({"model": model, "config": config, "summary": summary}, handle)
