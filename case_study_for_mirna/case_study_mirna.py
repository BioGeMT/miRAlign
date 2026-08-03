from __future__ import annotations

import argparse
import json
import pickle
import shutil
import sys
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd
from Bio.Seq import Seq
from joblib import Parallel, delayed
from sklearn.model_selection import GroupShuffleSplit
from sklearn.model_selection import train_test_split

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from case_study_for_mirna.import_mirbench_datasets import get_dataset_dataframe
from case_study_for_mirna.modeling import (
    build_trajectory_rows,
    fit_configuration,
    model_from_result,
    rank_successful,
    save_model,
    summarize_result,
)
from case_study_for_mirna.outputs import curve_point_rows, save_convergence_plot, save_pr_plot, save_roc_plot, write_rows
from case_study_for_mirna.scoring import evaluate_scores, score_pairs_with_model
from src.optimization_functions import baseline_parameters

PAIRED_DATASET_SPLITS = {"hejret": ("hejret_train", "hejret_test"), "manakov": ("manakov_train", "manakov_test")}
SUPPORTED_DATASET_SPLITS = ["hejret_train", "hejret_test", "manakov_train", "manakov_test", "manakov_leftout", "klimentova_test"]
REQUIRED_EVALUATION_COLUMNS = ["noncodingRNA", "gene", "label"]
CONFIG_KEYS = [
    "dataset",
    "aligner",
    "step_scale",
    "step_power",
    "step_decay_burnin",
    "prior_precision",
    "label_prior",
    "model_length",
    "split_strategy",
    "num_threads",
]


def csv_values(raw: str) -> list[str]:
    return [value.strip() for value in raw.split(",") if value.strip()]


def json_safe(value):
    if isinstance(value, dict):
        return {key: json_safe(val) for key, val in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(val) for val in value]
    if isinstance(value, np.ndarray):
        return json_safe(value.tolist())
    if isinstance(value, np.generic):
        return json_safe(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    if isinstance(value, Path):
        return str(value)
    return value


def write_json(path: str | Path, payload: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(json_safe(payload), handle, indent=2, sort_keys=True)
        handle.write("\n")


def parse_named_path(raw_value: str) -> tuple[str, Path]:
    if "=" in raw_value:
        name, path = raw_value.split("=", 1)
        return name.strip(), Path(path.strip())
    path = Path(raw_value.strip())
    return path.stem, path


def load_evaluation_file(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Evaluation file does not exist: {path}")
    frame = pd.read_csv(path, sep="\t" if path.suffix.lower() in {".tsv", ".tab"} else ",")
    missing = [column for column in REQUIRED_EVALUATION_COLUMNS if column not in frame.columns]
    if missing:
        raise ValueError(f"Evaluation file {path} is missing required columns {missing}.")
    return frame[REQUIRED_EVALUATION_COLUMNS].copy().reset_index(drop=True)


def load_custom_evaluation_frames(raw_eval_files: str | None) -> dict[str, pd.DataFrame]:
    if not raw_eval_files:
        return {}
    frames = {}
    for raw_value in csv_values(raw_eval_files):
        name, path = parse_named_path(raw_value)
        if name in frames:
            raise ValueError(f"Duplicate evaluation split name {name!r}.")
        frames[name] = load_evaluation_file(path)
    return frames


def parse_args():
    parser = argparse.ArgumentParser(description="Run the miRAlign miRNA case-study workflow.")
    parser.add_argument("--dataset", default="hejret", choices=sorted(PAIRED_DATASET_SPLITS))
    parser.add_argument("--eval-splits", default=None, help="Comma-separated miRBench evaluation split aliases.")
    parser.add_argument("--eval-files", default=None, help="Comma-separated user evaluation files, name=path.tsv or path.tsv.")
    parser.add_argument("--results-dir", default="results/case_study_for_mirna")
    parser.add_argument("--run-tag", default="")
    parser.add_argument("--validation-fraction", type=float, default=0.2)
    parser.add_argument("--split-seed", type=int, default=42)
    parser.add_argument("--split-strategy", default="random", choices=["random", "group_mirna", "group_gene"])
    parser.add_argument("--aligners", default="local,glocal")
    parser.add_argument("--step-scales", default="0.00001,0.000012,0.00005,0.0001,0.0005")
    parser.add_argument("--step-power", type=float, default=0.5)
    parser.add_argument(
        "--step-decay-burnins",
        default="300",
        help="Comma-separated iteration counts before learning-rate decay starts.",
    )
    parser.add_argument("--prior-precisions", default="0,1")
    parser.add_argument("--label-priors", default="none,symmetric_95_5,symmetric_90_10,symmetric_80_20")
    parser.add_argument("--max-iters", default="100")
    parser.add_argument("--final-max-iter", type=int, default=500)
    parser.add_argument("--num-threads", type=int, default=1)
    parser.add_argument("--config-workers", type=int, default=1, help="Number of grid configurations to fit concurrently.")
    parser.add_argument("--limit-configs", type=int, default=0)
    parser.add_argument("--top-entities", type=int, default=20, help="Rows per split/entity to write to top_entities.csv.")
    parser.add_argument("--trained-model", default="", help="Path to a saved model.pkl artifact for evaluation-only use.")
    parser.add_argument("--evaluate-only", action="store_true")
    parser.add_argument("--skip-evaluation", action="store_true")
    args = parser.parse_args()
    if args.evaluate_only and not args.trained_model:
        raise ValueError("--evaluate-only requires --trained-model.")
    return args


def dataset_summary_rows(split_name: str, frame: pd.DataFrame) -> list[dict]:
    if frame.empty:
        mean_mirnas_per_gene = np.nan
    else:
        mean_mirnas_per_gene = (
            frame.groupby("gene")["noncodingRNA"].nunique().mean()
        )
    rows = [{
        "split": split_name,
        "row_type": "all_lengths",
        "length": "all",
        "pairs": len(frame),
        "unique_mirnas": frame["noncodingRNA"].nunique(),
        "unique_genes": frame["gene"].nunique(),
        "unique_pairs": frame[["noncodingRNA", "gene"]].drop_duplicates().shape[0],
        "mean_mirnas_per_gene": float(mean_mirnas_per_gene) if np.isfinite(mean_mirnas_per_gene) else np.nan,
        "positive_pairs": int(frame["label"].astype(int).sum()) if not frame.empty else 0,
        "negative_pairs": int((frame["label"].astype(int) == 0).sum()) if not frame.empty else 0,
    }]
    if not frame.empty:
        frame = frame.copy()
        frame["length"] = frame["noncodingRNA"].astype(str).str.len()
        for length, length_frame in frame.groupby("length", sort=True):
            rows.append({
                "split": split_name,
                "row_type": "length",
                "length": int(length),
                "pairs": len(length_frame),
                "unique_mirnas": length_frame["noncodingRNA"].nunique(),
                "unique_genes": length_frame["gene"].nunique(),
                "unique_pairs": length_frame[["noncodingRNA", "gene"]].drop_duplicates().shape[0],
                "mean_mirnas_per_gene": float(length_frame.groupby("gene")["noncodingRNA"].nunique().mean()),
                "positive_pairs": int(length_frame["label"].astype(int).sum()),
                "negative_pairs": int((length_frame["label"].astype(int) == 0).sum()),
            })
    return rows


def write_dataset_summary(path: str | Path, frames_by_split: dict[str, pd.DataFrame]) -> None:
    rows = []
    for split_name, frame in frames_by_split.items():
        rows.extend(dataset_summary_rows(split_name, frame))
    write_rows(path, rows)


def _distribution_stats(values) -> dict:
    values = pd.Series(list(values), dtype=float)
    if values.empty:
        return {"mean_pairs": np.nan, "median_pairs": np.nan, "p95_pairs": np.nan, "max_pairs": 0}
    return {
        "mean_pairs": float(values.mean()),
        "median_pairs": float(values.median()),
        "p95_pairs": float(values.quantile(0.95)),
        "max_pairs": int(values.max()),
    }


def entity_frequency_summary_rows(frames_by_split: dict[str, pd.DataFrame]) -> list[dict]:
    rows = []
    for split_name, frame in frames_by_split.items():
        for entity_type, column in [("mirna", "noncodingRNA"), ("gene", "gene")]:
            label_groups = [("all", frame)]
            label_groups.extend((str(label), group) for label, group in frame.groupby("label", sort=True))
            for label_scope, group in label_groups:
                counts = group.groupby(column).size() if not group.empty else pd.Series(dtype=int)
                rows.append(
                    {
                        "split": split_name,
                        "entity_type": entity_type,
                        "label": label_scope,
                        "entities": int(counts.shape[0]),
                        **_distribution_stats(counts.tolist()),
                    }
                )
    return rows


def top_entity_rows(frames_by_split: dict[str, pd.DataFrame], top_n: int) -> list[dict]:
    rows = []
    if top_n <= 0:
        return rows
    for split_name, frame in frames_by_split.items():
        for entity_type, column in [("mirna", "noncodingRNA"), ("gene", "gene")]:
            label_groups = [("all", frame)]
            label_groups.extend((str(label), group) for label, group in frame.groupby("label", sort=True))
            for label_scope, group in label_groups:
                counts = group.groupby(column).size().sort_values(ascending=False)
                for rank, (entity, count) in enumerate(counts.head(top_n).items(), start=1):
                    rows.append(
                        {
                            "split": split_name,
                            "entity_type": entity_type,
                            "label": label_scope,
                            "rank": rank,
                            "entity": entity,
                            "pairs": int(count),
                        }
                    )
    return rows


def _entity_set(frame: pd.DataFrame, entity_type: str) -> set:
    if entity_type == "pair":
        return set(map(tuple, frame[["noncodingRNA", "gene"]].itertuples(index=False, name=None)))
    return set(frame[entity_type])


def split_overlap_rows(frames_by_split: dict[str, pd.DataFrame]) -> list[dict]:
    rows = []
    items = list(frames_by_split.items())
    for idx, (left_name, left_frame) in enumerate(items):
        for right_name, right_frame in items[idx + 1:]:
            for entity_type, column in [("noncodingRNA", "noncodingRNA"), ("gene", "gene"), ("pair", None)]:
                left = _entity_set(left_frame, entity_type if entity_type == "pair" else column)
                right = _entity_set(right_frame, entity_type if entity_type == "pair" else column)
                overlap = left.intersection(right)
                rows.append(
                    {
                        "left_split": left_name,
                        "right_split": right_name,
                        "entity_type": "mirna" if entity_type == "noncodingRNA" else entity_type,
                        "left_entities": len(left),
                        "right_entities": len(right),
                        "overlap_entities": len(overlap),
                        "left_overlap_fraction": len(overlap) / len(left) if left else np.nan,
                        "right_overlap_fraction": len(overlap) / len(right) if right else np.nan,
                    }
                )
    return rows


def write_dataset_diagnostics(run_dir: str | Path, frames_by_split: dict[str, pd.DataFrame], top_n: int) -> None:
    run_dir = Path(run_dir)
    write_dataset_summary(run_dir / "dataset_summary.csv", frames_by_split)
    write_rows(run_dir / "entity_frequency_summary.csv", entity_frequency_summary_rows(frames_by_split))
    write_rows(run_dir / "top_entities.csv", top_entity_rows(frames_by_split, top_n))
    write_rows(run_dir / "split_overlap.csv", split_overlap_rows(frames_by_split))


def require_training_frame(frame: pd.DataFrame, split_name: str) -> None:
    if frame.empty:
        raise ValueError(f"No rows are available in {split_name}.")
    if frame["label"].astype(int).nunique() < 2:
        raise ValueError(f"{split_name} must contain both positive and negative labels.")


def split_train_validation(train_frame: pd.DataFrame, args) -> tuple[pd.DataFrame, pd.DataFrame]:
    if args.split_strategy == "random":
        fit_frame, validation_frame = train_test_split(
            train_frame,
            test_size=args.validation_fraction,
            random_state=args.split_seed,
            stratify=train_frame["label"].astype(int),
        )
        return fit_frame.reset_index(drop=True), validation_frame.reset_index(drop=True)

    group_column = "noncodingRNA" if args.split_strategy == "group_mirna" else "gene"
    splitter = GroupShuffleSplit(
        n_splits=50,
        test_size=args.validation_fraction,
        random_state=args.split_seed,
    )
    labels = train_frame["label"].astype(int)
    groups = train_frame[group_column].astype(str)
    for fit_idx, validation_idx in splitter.split(train_frame, labels, groups):
        fit_frame = train_frame.iloc[fit_idx].reset_index(drop=True)
        validation_frame = train_frame.iloc[validation_idx].reset_index(drop=True)
        if fit_frame["label"].astype(int).nunique() == 2 and validation_frame["label"].astype(int).nunique() == 2:
            return fit_frame, validation_frame
    raise ValueError(f"Could not create a two-class validation split with strategy {args.split_strategy!r}.")


def prepare_inputs(frame: pd.DataFrame):
    seq_a = frame["noncodingRNA"].astype(str).tolist()
    seq_b = [str(Seq(seq).reverse_complement()) for seq in frame["gene"].astype(str)]
    labels = frame["label"].astype(int).tolist()
    return seq_a, seq_b, labels


def infer_model_length(frames: list[pd.DataFrame]) -> int:
    lengths = []
    for frame in frames:
        if not frame.empty:
            lengths.extend(frame["noncodingRNA"].astype(str).str.len().tolist())
    if not lengths:
        raise ValueError("Cannot infer model length from empty datasets.")
    return int(max(lengths))


def load_frames(args):
    summary_frames = {}
    custom_evaluation_frames = {} if args.skip_evaluation else load_custom_evaluation_frames(args.eval_files)
    train_alias, default_test_alias = PAIRED_DATASET_SPLITS[args.dataset]
    if args.skip_evaluation:
        evaluation_frames = {}
    else:
        split_names = csv_values(args.eval_splits) if args.eval_splits else [default_test_alias]
        invalid = [name for name in split_names if name not in SUPPORTED_DATASET_SPLITS]
        if invalid:
            raise ValueError(f"Invalid evaluation split aliases: {invalid}")
        raw_evaluation_frames = {name: get_dataset_dataframe(name).reset_index(drop=True) for name in split_names}
        overlap = sorted(set(raw_evaluation_frames).intersection(custom_evaluation_frames))
        if overlap:
            raise ValueError(f"Custom evaluation names duplicate miRBench split names: {overlap}")
        raw_evaluation_frames.update(custom_evaluation_frames)
        evaluation_frames = raw_evaluation_frames
        summary_frames.update(raw_evaluation_frames)
    if args.evaluate_only:
        empty_frame = pd.DataFrame(columns=REQUIRED_EVALUATION_COLUMNS)
        return empty_frame, empty_frame, empty_frame, evaluation_frames, summary_frames

    raw_train_frame = get_dataset_dataframe(train_alias).reset_index(drop=True)
    train_frame = raw_train_frame
    summary_frames[train_alias] = train_frame
    require_training_frame(train_frame, train_alias)
    fit_frame, validation_frame = split_train_validation(train_frame, args)
    summary_frames["fit"] = fit_frame
    summary_frames["validation"] = validation_frame
    return train_frame, fit_frame, validation_frame, evaluation_frames, summary_frames


def build_config(dataset_label, index, values, args):
    aligner, step_scale, step_decay_burnin, prior_precision, label_prior, max_iter = values
    config_name = (
        f"cfg_{index:04d}_{aligner}_s{step_scale}_d{step_decay_burnin}"
        f"_p{prior_precision}_{label_prior}_i{max_iter}"
    )
    return {
        "dataset": dataset_label,
        "config_index": index,
        "config": config_name,
        "aligner": aligner,
        "step_scale": float(step_scale),
        "step_power": float(args.step_power),
        "step_decay_burnin": int(step_decay_burnin),
        "prior_precision": float(prior_precision),
        "label_prior": label_prior,
        "model_length": int(args.model_length),
        "split_strategy": args.split_strategy,
        "max_iter": int(max_iter),
        "num_threads": int(args.num_threads),
    }


def model_parameters_payload(model: dict, config: dict, summary: dict) -> dict:
    baseline = baseline_parameters(model["M"].shape[-1])
    return {
        "config": config,
        "summary": summary,
        "baseline": {
            "M_shape": list(baseline["M"].shape),
            "G_miR_shape": list(baseline["G_miR"].shape),
            "G_gene_shape": list(baseline["G_gene"].shape),
            "alpha": baseline["alpha"],
        },
        "model": {
            "aligner": model["aligner_name"],
            "alpha": model["alpha"],
            "M_shape": list(model["M"].shape),
            "G_miR": model["G_miR"],
            "G_gene": model["G_gene"],
            "label_observation_probs": model.get("label_observation_probs"),
        },
    }


def export_model_artifacts(model: dict, config: dict, row: dict, trajectories: list[dict], artifact_dir: str | Path) -> dict:
    artifact_dir = Path(artifact_dir)
    artifact_dir.mkdir(parents=True, exist_ok=True)
    save_model(artifact_dir / "model.pkl", model, config, row)
    write_json(artifact_dir / "model_parameters.json", model_parameters_payload(model, config, row))
    write_rows(artifact_dir / "trajectory.csv", trajectories)
    save_convergence_plot(trajectories, artifact_dir / "convergence.png", config["config"])
    return {
        "model_artifact_dir": str(artifact_dir),
        "model_pickle_path": str(artifact_dir / "model.pkl"),
        "model_parameters_path": str(artifact_dir / "model_parameters.json"),
    }


def evaluate_model(config, model, inputs_by_split, run_dir):
    updates, metrics, pr_points, roc_points, split_stats = {}, [], [], [], {}
    for split_name, inputs in inputs_by_split.items():
        print(f"  Scoring split {split_name} ({len(inputs[2])} pairs)", flush=True)
        scores = score_pairs_with_model(inputs[0], inputs[1], model, num_threads=int(config["num_threads"]))
        stats = evaluate_scores(inputs[2], scores)
        split_stats[split_name] = stats
        updates[f"ap_{split_name}"] = stats["average_precision"]
        updates[f"roc_auc_{split_name}"] = stats["roc_auc"]
        metrics.append({**config, "split": split_name, "average_precision": stats["average_precision"], "roc_auc": stats["roc_auc"], "status": "ok"})
        pr_points.extend(curve_point_rows(config["config"], split_name, stats, "pr"))
        roc_points.extend(curve_point_rows(config["config"], split_name, stats, "roc"))
    reference_split = "fit" if "fit" in split_stats else "train"
    if reference_split in split_stats:
        for split_name, stats in split_stats.items():
            if split_name == reference_split:
                continue
            save_pr_plot(split_stats[reference_split], stats, run_dir / "pr_curves" / f"{config['config']}_{reference_split}_vs_{split_name}.png", f"{config['config']}: {reference_split} vs {split_name}", reference_label=reference_split, compare_label=split_name)
            save_roc_plot(split_stats[reference_split], stats, run_dir / "roc_curves" / f"{config['config']}_{reference_split}_vs_{split_name}.png", f"{config['config']}: {reference_split} vs {split_name}", reference_label=reference_split, compare_label=split_name)
    return updates, metrics, pr_points, roc_points


def run_configuration(index, total_configs, config, fit_inputs, inputs_by_split, run_dir, skip_evaluation=False):
    print(f"Starting {index}/{total_configs}: {config['config']}", flush=True)
    try:
        result, runtime = fit_configuration(fit_inputs, config)
        row = summarize_result(config, result, runtime)
        model = model_from_result(result, config)
        updates, metrics, pr_points, roc_points = ({}, [], [], []) if skip_evaluation else evaluate_model(config, model, inputs_by_split, run_dir)
        row.update(updates)
        trajectories = build_trajectory_rows(config, result)
        row.update(export_model_artifacts(model, config, row, trajectories, run_dir / "model_artifacts" / config["config"]))
        print(f"Completed {index}/{total_configs}: ok", flush=True)
        return {"summary": row, "errors": [], "metrics": metrics, "pr_points": pr_points, "roc_points": roc_points, "trajectories": trajectories}
    except Exception as exc:
        row = {**config, "status": "error", "error": repr(exc), "final_loglik": np.nan}
        print(f"Completed {index}/{total_configs}: error {row['error']}", flush=True)
        return {"summary": row, "errors": [row], "metrics": [], "pr_points": [], "roc_points": [], "trajectories": []}


def evaluate_existing_model(config: dict, model: dict, inputs_by_split: dict, out_dir: str | Path, summary: dict | None = None) -> dict:
    out_dir = Path(out_dir)
    row = {**config, "status": "ok", "error": ""}
    if summary:
        row.update({key: value for key, value in summary.items() if key not in row})
    updates, metric_rows, pr_rows, roc_rows = evaluate_model(config, model, inputs_by_split, out_dir)
    row.update(updates)
    write_rows(out_dir / "summary.csv", [row])
    write_rows(out_dir / "metrics.csv", metric_rows)
    write_rows(out_dir / "pr_points.csv", pr_rows)
    write_rows(out_dir / "roc_points.csv", roc_rows)
    return {
        "summary": row,
        "metrics": metric_rows,
        "pr_points": pr_rows,
        "roc_points": roc_rows,
    }


def load_trained_model(path: str | Path) -> dict:
    with Path(path).open("rb") as handle:
        payload = pickle.load(handle)
    if "model" not in payload:
        raise ValueError(f"Trained model {path} does not contain a model payload.")
    return payload


def copy_artifact_dir(source_dir: str | Path, destination_dir: str | Path) -> None:
    source_dir = Path(source_dir)
    destination_dir = Path(destination_dir)
    if not source_dir.exists():
        return
    if destination_dir.exists():
        shutil.rmtree(destination_dir)
    shutil.copytree(source_dir, destination_dir)


def main():
    args = parse_args()
    train_frame, fit_frame, validation_frame, evaluation_frames, summary_frames = load_frames(args)
    run_name = args.run_tag if args.run_tag else args.dataset
    run_dir = Path(args.results_dir) / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    args.model_length = infer_model_length([train_frame, *evaluation_frames.values()])
    write_dataset_diagnostics(run_dir, summary_frames, args.top_entities)
    evaluation_inputs = {name: prepare_inputs(frame) for name, frame in evaluation_frames.items()}

    if args.evaluate_only:
        payload = load_trained_model(args.trained_model)
        config = {**payload.get("config", {}), "config": f"evaluate_{Path(args.trained_model).stem}", "max_iter": 0, "num_threads": args.num_threads, "evaluation_only": True}
        evaluation = evaluate_existing_model(config, payload["model"], evaluation_inputs, run_dir, payload.get("summary", {}))
        write_json(run_dir / "best_grid_model" / "selected_summary.json", {"selected_from": "trained_model", "summary": evaluation["summary"]})
        print(f"Wrote {run_dir}")
        return

    fit_inputs = prepare_inputs(fit_frame)
    validation_inputs = prepare_inputs(validation_frame)
    train_inputs = prepare_inputs(train_frame)
    grid_inputs_by_split = {} if args.skip_evaluation else {"fit": fit_inputs, "validation": validation_inputs}
    grid = list(
        product(
            csv_values(args.aligners),
            [float(value) for value in csv_values(args.step_scales)],
            [int(value) for value in csv_values(args.step_decay_burnins)],
            [float(value) for value in csv_values(args.prior_precisions)],
            csv_values(args.label_priors),
            [int(value) for value in csv_values(args.max_iters)],
        )
    )
    if args.limit_configs:
        grid = grid[: args.limit_configs]
    configs = [build_config(args.dataset, index, values, args) for index, values in enumerate(grid, start=1)]
    print(f"Running {len(configs)} configurations...", flush=True)

    summary_rows, error_rows, metric_rows, pr_rows, roc_rows, trajectory_rows = [], [], [], [], [], []
    if args.config_workers == 1:
        results = [
            run_configuration(index, len(configs), config, fit_inputs, grid_inputs_by_split, run_dir, args.skip_evaluation)
            for index, config in enumerate(configs, start=1)
        ]
    else:
        results = Parallel(n_jobs=args.config_workers, return_as="list")(
            delayed(run_configuration)(index, len(configs), config, fit_inputs, grid_inputs_by_split, run_dir, args.skip_evaluation)
            for index, config in enumerate(configs, start=1)
        )
    for result in results:
        summary_rows.append(result["summary"])
        error_rows.extend(result["errors"])
        metric_rows.extend(result["metrics"])
        pr_rows.extend(result["pr_points"])
        roc_rows.extend(result["roc_points"])
        trajectory_rows.extend(result["trajectories"])

    ranked = rank_successful(summary_rows)
    if ranked:
        best = ranked[0]
        copy_artifact_dir(best.get("model_artifact_dir", ""), run_dir / "best_grid_model")
        write_json(run_dir / "best_grid_model" / "selected_summary.json", {"selected_from": "grid", "summary": best})
        if not args.skip_evaluation and evaluation_inputs:
            payload = load_trained_model(run_dir / "best_grid_model" / "model.pkl")
            best_eval_inputs = {"fit": fit_inputs, "validation": validation_inputs, **evaluation_inputs}
            best_evaluation = evaluate_existing_model(
                best,
                payload["model"],
                best_eval_inputs,
                run_dir / "best_grid_model" / "evaluation",
                best,
            )
            write_json(run_dir / "best_grid_model" / "selected_summary.json", {"selected_from": "grid", "summary": best_evaluation["summary"]})
    if ranked and args.final_max_iter > 0 and not args.skip_evaluation:
        final_config = {key: ranked[0][key] for key in CONFIG_KEYS}
        final_config.update({"config_index": 0, "config": f"final_refit_{ranked[0]['config']}", "max_iter": args.final_max_iter})
        result, runtime = fit_configuration(train_inputs, final_config)
        final_row = summarize_result(final_config, result, runtime)
        final_model = model_from_result(result, final_config)
        final_updates, final_metrics, final_pr, final_roc = evaluate_model(final_config, final_model, {"train": train_inputs, **evaluation_inputs}, run_dir / "final_refit")
        final_row.update(final_updates)
        final_trajectories = build_trajectory_rows(final_config, result)
        final_row.update(export_model_artifacts(final_model, final_config, final_row, final_trajectories, run_dir / "final_refit" / "model"))
        write_rows(run_dir / "final_refit" / "summary.csv", [final_row])
        write_rows(run_dir / "final_refit" / "metrics.csv", final_metrics)
        write_rows(run_dir / "final_refit" / "pr_points.csv", final_pr)
        write_rows(run_dir / "final_refit" / "roc_points.csv", final_roc)
        write_rows(run_dir / "final_refit" / "trajectory.csv", final_trajectories)

    write_rows(run_dir / "summary.csv", summary_rows)
    write_rows(run_dir / "errors.csv", error_rows)
    write_rows(run_dir / "metrics.csv", metric_rows)
    write_rows(run_dir / "pr_points.csv", pr_rows)
    write_rows(run_dir / "roc_points.csv", roc_rows)
    write_rows(run_dir / "trajectories.csv", trajectory_rows)
    print(f"Wrote {run_dir}")


if __name__ == "__main__":
    main()
