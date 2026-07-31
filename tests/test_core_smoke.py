import numpy as np
import pandas as pd

from case_study_for_mirna.modeling import label_prior_from_name
from case_study_for_mirna.case_study_mirna import evaluate_existing_model
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


def test_miralign_one_iteration_with_sample_weights():
    result = miRAlign(
        ["A" * 22, "C" * 22],
        ["A" * 30, "T" * 30],
        [1, 0],
        pos_aware_align_local,
        create_subgradient_step(0.00001, 0.5, 300),
        sample_weight=np.array([2.0, 1.0]),
        MAX_ITER=1,
        num_threads=1,
    )

    assert np.isfinite(result["final_loglik"])
    assert len(result["subgradient_norm_trajectory"]) == 1


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
    inputs = {
        "fit": (["A" * 22, "C" * 22], ["A" * 30, "T" * 30], [1, 0]),
        "heldout": (["A" * 22, "C" * 22], ["A" * 30, "T" * 30], [1, 0]),
    }

    evaluate_existing_model(config, model, inputs, tmp_path)

    metrics = pd.read_csv(tmp_path / "metrics.csv")
    assert set(metrics["split"]) == {"fit", "heldout"}
