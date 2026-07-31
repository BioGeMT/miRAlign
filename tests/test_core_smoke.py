import numpy as np

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
    assert params["M"][0, 0, 0] == BASELINE_MATCH_SCORE
    assert params["M"][0, 1, 0] == BASELINE_MISMATCH_SCORE


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
        "A" * 11,
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

