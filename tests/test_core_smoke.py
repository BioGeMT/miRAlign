import numpy as np
import pandas as pd

from case_study_for_mirna.modeling import label_prior_from_name, sample_weight_from_name
from case_study_for_mirna.case_study_mirna import (
    evaluate_existing_model,
    prepare_inputs,
    split_train_validation,
    write_dataset_diagnostics,
    write_dataset_summary,
)
from src.miralign import miRAlign
from src.optimization_functions import (
    BASELINE_ALPHA,
    BASELINE_GAP_SCORE,
    BASELINE_MATCH_SCORE,
    BASELINE_MISMATCH_SCORE,
    baseline_parameters,
    create_subgradient_step,
    logreg_starting_point,
)
from src.positional_alignment import pos_aware_align_glocal, pos_aware_align_local


def test_baseline_parameters_match_hejret_discrimalign_constants():
    params = baseline_parameters()

    assert params["alpha"] == BASELINE_ALPHA
    assert np.all(params["G_miR"] == BASELINE_GAP_SCORE)
    assert np.all(params["G_gene"] == BASELINE_GAP_SCORE)
    assert np.isclose(params["M"][0, 0, 0], BASELINE_MATCH_SCORE)
    assert np.isclose(params["M"][0, 1, 0], BASELINE_MISMATCH_SCORE)


def test_logreg_starting_point_uses_fixed_baseline():
    params = logreg_starting_point(["A" * 22], ["T" * 30], [1])

    assert params["M"].shape == (4, 4, 22)
    assert params["G_miR"].shape == (21,)
    assert params["G_gene"].shape == (22,)
    assert params["alpha"] == BASELINE_ALPHA


def test_glocal_alignment_wrapper_handles_gaps():
    params = baseline_parameters()
    score, aligned_mir, aligned_gene, mir_coords, gene_coords = pos_aware_align_glocal(
        "A" * 22,
        "A" * 21,
        params["M"],
        params["G_miR"],
        params["G_gene"],
        backtrack=True,
    )

    assert isinstance(score, float)
    assert "-" in aligned_gene
    assert len(aligned_mir) == len(aligned_gene) == len(mir_coords) == len(gene_coords)


def test_miralign_one_iteration_smoke():
    mirnas = ["A" * 22, "C" * 22]
    genes = ["A" * 30, "T" * 30]
    labels = [1, 0]
    result = miRAlign(
        mirnas,
        genes,
        labels,
        pos_aware_align_local,
        create_subgradient_step(0.00001, 0.5, 300),
        MAX_ITER=1,
        num_threads=1,
    )

    assert result["M"].shape == (4, 4, 22)
    assert np.isfinite(result["alpha"])
    assert len(result["loglik_trajectory"]) == 1


def test_miralign_one_iteration_with_mixed_mirna_lengths():
    result = miRAlign(
        ["A" * 21, "C" * 22],
        ["A" * 30, "T" * 30],
        [1, 0],
        pos_aware_align_local,
        create_subgradient_step(0.00001, 0.5, 300),
        model_length=22,
        MAX_ITER=1,
        num_threads=1,
    )

    assert result["M"].shape == (4, 4, 22)
    assert result["G_miR"].shape == (21,)
    assert result["G_gene"].shape == (22,)


def test_miralign_one_iteration_with_sample_weights():
    result = miRAlign(
        ["A" * 21, "C" * 22],
        ["A" * 30, "T" * 30],
        [1, 0],
        pos_aware_align_local,
        create_subgradient_step(0.00001, 0.5, 300),
        model_length=22,
        sample_weight=np.array([2.0, 1.0]),
        MAX_ITER=1,
        num_threads=1,
    )

    assert np.isfinite(result["final_loglik"])


def test_miralign_one_iteration_with_label_prior():
    result = miRAlign(
        ["A" * 22, "C" * 22],
        ["A" * 30, "T" * 30],
        [1, 0],
        pos_aware_align_local,
        create_subgradient_step(0.00001, 0.5, 300),
        label_prior=np.array([[900, 100], [100, 900]], dtype=float),
        MAX_ITER=1,
        num_threads=1,
    )

    assert result["label_observation_probs"].shape == (2, 2)
    assert np.isfinite(result["final_loglik"])


def test_case_study_label_prior_names():
    assert label_prior_from_name("none") is None
    assert label_prior_from_name("symmetric_95_5")[0, 0] == 950
    assert label_prior_from_name("symmetric_90_10")[0, 1] == 100
    assert label_prior_from_name("symmetric_80_20")[1, 0] == 200


def test_sample_weight_names_balance_entities():
    frame = pd.DataFrame(
        {
            "noncodingRNA": ["mir1", "mir1", "mir2", "mir3"],
            "gene": ["gene1", "gene2", "gene2", "gene2"],
            "label": [1, 0, 1, 0],
        }
    )

    mirna_weights = sample_weight_from_name("mirna_balanced", frame)
    gene_weights = sample_weight_from_name("gene_balanced", frame)
    pair_weights = sample_weight_from_name("pair_balanced", frame)
    combined_weights = sample_weight_from_name("mirna_gene_sqrt", frame)

    assert mirna_weights[0] == mirna_weights[1]
    assert mirna_weights[2] > mirna_weights[0]
    assert gene_weights[0] > gene_weights[1]
    assert np.allclose(pair_weights, 1.0)
    assert np.isclose(combined_weights.mean(), 1.0)


def test_grouped_validation_split_holds_out_mirnas():
    frame = pd.DataFrame(
        {
            "noncodingRNA": [f"mir{i}" for i in range(8) for _ in range(2)],
            "gene": [f"gene{i}_{j}" for i in range(8) for j in range(2)],
            "label": [0, 1] * 8,
        }
    )
    args = type("Args", (), {"split_strategy": "group_mirna", "validation_fraction": 0.25, "split_seed": 1})

    fit, validation = split_train_validation(frame, args)

    assert set(fit["noncodingRNA"]).isdisjoint(set(validation["noncodingRNA"]))
    assert set(fit["label"]) == {0, 1}
    assert set(validation["label"]) == {0, 1}


def test_grouped_validation_split_holds_out_genes():
    frame = pd.DataFrame(
        {
            "noncodingRNA": [f"mir{i}_{j}" for i in range(8) for j in range(2)],
            "gene": [f"gene{i}" for i in range(8) for _ in range(2)],
            "label": [0, 1] * 8,
        }
    )
    args = type("Args", (), {"split_strategy": "group_gene", "validation_fraction": 0.25, "split_seed": 1})

    fit, validation = split_train_validation(frame, args)

    assert set(fit["gene"]).isdisjoint(set(validation["gene"]))
    assert set(fit["label"]) == {0, 1}
    assert set(validation["label"]) == {0, 1}


def test_evaluate_existing_model_writes_requested_splits(tmp_path):
    params = baseline_parameters()
    model = {
        "aligner": pos_aware_align_local,
        "aligner_name": "local",
        "M": params["M"],
        "G_miR": params["G_miR"],
        "G_gene": params["G_gene"],
        "alpha": params["alpha"],
    }
    config = {"config": "baseline", "num_threads": 1}
    fit_frame = pd.DataFrame(
        {
            "noncodingRNA": ["A" * 22, "C" * 21],
            "gene": ["A" * 30, "T" * 30],
            "label": [1, 0],
        }
    )
    heldout_frame = pd.DataFrame(
        {
            "noncodingRNA": ["A" * 22, "G" * 21],
            "gene": ["A" * 30, "G" * 30],
            "label": [1, 0],
        }
    )
    inputs = {"fit": prepare_inputs(fit_frame), "heldout": prepare_inputs(heldout_frame)}

    evaluate_existing_model(config, model, inputs, tmp_path)

    metrics = pd.read_csv(tmp_path / "metrics.csv")
    assert set(metrics["split"]) == {"fit", "heldout"}
    slices = pd.read_csv(tmp_path / "metric_slices.csv")
    assert {"mirna_length", "mirna_seen", "gene_seen", "mirna_frequency_bin", "gene_frequency_bin"}.issubset(set(slices["slice_type"]))


def test_write_dataset_summary_reports_length_counts(tmp_path):
    raw = pd.DataFrame(
        {
            "noncodingRNA": ["A" * 22, "C" * 22, "G" * 21],
            "gene": ["gene1", "gene1", "gene2"],
            "label": [1, 0, 1],
        }
    )
    write_dataset_summary(tmp_path / "dataset_summary.csv", {"train": raw})

    summary = pd.read_csv(tmp_path / "dataset_summary.csv")
    row = summary[summary["row_type"] == "all_lengths"].iloc[0]
    assert row["split"] == "train"
    assert row["pairs"] == 3
    assert row["unique_mirnas"] == 3
    assert row["unique_genes"] == 2
    assert row["unique_pairs"] == 3
    assert row["mean_mirnas_per_gene"] == 1.5
    assert set(summary["length"].astype(str)) == {"all", "21", "22"}


def test_write_dataset_diagnostics_reports_frequency_and_overlap(tmp_path):
    train = pd.DataFrame(
        {
            "noncodingRNA": ["mir1", "mir1", "mir2"],
            "gene": ["gene1", "gene2", "gene2"],
            "label": [1, 0, 1],
        }
    )
    heldout = pd.DataFrame(
        {
            "noncodingRNA": ["mir1", "mir3"],
            "gene": ["gene9", "gene2"],
            "label": [0, 1],
        }
    )

    write_dataset_diagnostics(tmp_path, {"train": train, "heldout": heldout}, top_n=1)

    frequency = pd.read_csv(tmp_path / "entity_frequency_summary.csv")
    assert {"mirna", "gene"} == set(frequency["entity_type"])
    assert frequency[(frequency["split"] == "train") & (frequency["entity_type"] == "mirna") & (frequency["label"].astype(str) == "all")]["max_pairs"].iloc[0] == 2

    top = pd.read_csv(tmp_path / "top_entities.csv")
    assert set(top["rank"]) == {1}

    overlap = pd.read_csv(tmp_path / "split_overlap.csv")
    mirna_overlap = overlap[overlap["entity_type"] == "mirna"].iloc[0]
    assert mirna_overlap["overlap_entities"] == 1
