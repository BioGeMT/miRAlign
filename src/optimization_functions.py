"""Optimization helpers for miRAlign."""

from __future__ import annotations

import numpy as np

from .likelihood_and_subgradients import logit_subderivative_theta

BASELINE_MATCH_SCORE = 0.724709
BASELINE_MISMATCH_SCORE = -0.647892
BASELINE_GAP_SCORE = -0.901264
BASELINE_ALPHA = -5.226262


def baseline_parameters(mirna_length=22):
    """Return the Hejret/DiscrimAlign-derived baseline positional parameters."""
    M = np.zeros((4, 4, mirna_length))
    for position in range(mirna_length):
        M[..., position] = np.eye(4) * (BASELINE_MATCH_SCORE - BASELINE_MISMATCH_SCORE)
        M[..., position] += BASELINE_MISMATCH_SCORE
    return {
        "M": M,
        "G_miR": np.zeros(mirna_length - 1) + BASELINE_GAP_SCORE,
        "G_gene": np.zeros(mirna_length) + BASELINE_GAP_SCORE,
        "alpha": BASELINE_ALPHA,
    }


def logreg_starting_point(
    mirna_list,
    gene_list,
    label_list,
    model_length=None,
    match_weight=5,
    mismatch_weight=-4,
    gap_weight=-6,
    num_threads=1,
    max_sample_size=10000,
):
    """Return fixed baseline parameters from the Hejret simple alignment model.

    The signature preserves the cleanup-era call site in ``src.miralign``. The
    historical miRAlign notebooks initialized from these Hejret/DiscrimAlign
    constants rather than fitting a new starting point for each run.
    """
    del gene_list, label_list, match_weight, mismatch_weight, gap_weight, num_threads, max_sample_size
    if model_length is None:
        model_length = max(len(str(mirna)) for mirna in mirna_list)
    return baseline_parameters(int(model_length))


def _subgradient_descent_step(
    step_factor,
    step_power,
    power_offset,
    iter_nb,
    posloc_alignments,
    labels,
    alpha,
    current_M=None,
    current_G_miR=None,
    current_G_gene=None,
    M_prior=None,
    G_miR_prior=None,
    G_gene_prior=None,
    lbd=0,
    label_observation_probs=None,
    verbose=False,
):
    del verbose
    mirna_length = current_M.shape[-1] if current_M is not None else 22
    M_subd, G_miR_subd, G_gene_subd = logit_subderivative_theta(
        posloc_alignments,
        labels,
        alpha,
        label_observation_probs,
        mirna_length=mirna_length,
    )

    if current_M is not None and M_prior is not None:
        M_subd -= 2 * lbd * (current_M - M_prior)
    if current_G_miR is not None and G_miR_prior is not None:
        G_miR_subd -= 2 * lbd * (current_G_miR - G_miR_prior)
    if current_G_gene is not None and G_gene_prior is not None:
        G_gene_subd -= 2 * lbd * (current_G_gene - G_gene_prior)

    if iter_nb > power_offset:
        stepsize = step_factor / (iter_nb - power_offset) ** step_power
    else:
        stepsize = step_factor

    return {
        "G_miR_step": G_miR_subd * stepsize,
        "G_gene_step": G_gene_subd * stepsize,
        "M_step": M_subd * stepsize,
        "G_miR_subgradient": G_miR_subd,
        "G_gene_subgradient": G_gene_subd,
        "M_subgradient": M_subd,
    }


def create_subgradient_step(step_factor, step_power, power_offset):
    def stepfunction(
        iter_nb,
        posloc_alignments,
        labels,
        alpha,
        current_M=None,
        current_G_miR=None,
        current_G_gene=None,
        M_prior=None,
        G_miR_prior=None,
        G_gene_prior=None,
        lbd=0,
        label_observation_probs=None,
        verbose=False,
    ):
        return _subgradient_descent_step(
            step_factor=step_factor,
            step_power=step_power,
            power_offset=power_offset,
            iter_nb=iter_nb,
            posloc_alignments=posloc_alignments,
            labels=labels,
            alpha=alpha,
            current_M=current_M,
            current_G_miR=current_G_miR,
            current_G_gene=current_G_gene,
            M_prior=M_prior,
            G_miR_prior=G_miR_prior,
            G_gene_prior=G_gene_prior,
            lbd=lbd,
            label_observation_probs=label_observation_probs,
            verbose=verbose,
        )

    return stepfunction
