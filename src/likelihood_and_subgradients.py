"""Likelihood and subgradient utilities for miRAlign."""

from __future__ import annotations

import numpy as np

from .shared_global_vars import NUCL_DICT


def logit_logl(
    scores_pos,
    scores_neg,
    alpha,
    label_observation_parameters=None,
    label_observation_probs=None,
    M=None,
    G_miR=None,
    G_gene=None,
    M_prior=None,
    G_miR_prior=None,
    G_gene_prior=None,
    lbd=0,
    weights_pos=None,
    weights_neg=None,
):
    """Return the penalized logistic log-likelihood up to a constant."""
    scores_pos = np.asarray(scores_pos, dtype=float)
    scores_neg = np.asarray(scores_neg, dtype=float)
    weights_pos = np.ones_like(scores_pos) if weights_pos is None else np.asarray(weights_pos, dtype=float)
    weights_neg = np.ones_like(scores_neg) if weights_neg is None else np.asarray(weights_neg, dtype=float)
    value = 0.0

    if M_prior is not None and M is not None:
        value -= lbd * np.sum((M - M_prior) ** 2)
    if G_miR_prior is not None and G_miR is not None:
        value -= lbd * np.sum((G_miR - G_miR_prior) ** 2)
    if G_gene_prior is not None and G_gene is not None:
        value -= lbd * np.sum((G_gene - G_gene_prior) ** 2)

    if label_observation_parameters is not None:
        value += np.sum(label_observation_parameters * np.log(label_observation_probs))
        pos_part = label_observation_probs[1, 1] / (1 + np.exp(-alpha - scores_pos))
        pos_part += label_observation_probs[0, 1] / (1 + np.exp(alpha + scores_pos))
        neg_part = label_observation_probs[1, 0] / (1 + np.exp(-alpha - scores_neg))
        neg_part += label_observation_probs[0, 0] / (1 + np.exp(alpha + scores_neg))
        value += np.sum(weights_pos * np.log(pos_part)) + np.sum(weights_neg * np.log(neg_part))
    else:
        value -= np.sum(weights_pos * np.logaddexp(0, -alpha - scores_pos))
        value -= np.sum(weights_neg * np.logaddexp(0, alpha + scores_neg))

    return value


def logit_lhood_vect(scores, alpha):
    return 1 / (1 + np.exp(-alpha - np.asarray(scores, dtype=float)))


def logit_derivative_alpha(
    scores_pos,
    scores_neg,
    alpha,
    label_observation_probs=None,
    weights_pos=None,
    weights_neg=None,
):
    scores_pos = np.asarray(scores_pos, dtype=float)
    scores_neg = np.asarray(scores_neg, dtype=float)
    weights_pos = np.ones_like(scores_pos) if weights_pos is None else np.asarray(weights_pos, dtype=float)
    weights_neg = np.ones_like(scores_neg) if weights_neg is None else np.asarray(weights_neg, dtype=float)
    if label_observation_probs is None:
        pos_part = 1 / (1 + np.exp(alpha + scores_pos))
        neg_part = 1 / (1 + np.exp(-alpha - scores_neg))
        return np.sum(weights_pos * pos_part) - np.sum(weights_neg * neg_part)

    pos_factor = label_observation_probs[1, 1] - label_observation_probs[0, 1]
    neg_factor = label_observation_probs[1, 0] - label_observation_probs[0, 0]
    pos_denominator = label_observation_probs[0, 1] * (1 + np.exp(-alpha - scores_pos))
    pos_denominator += label_observation_probs[1, 1] * (1 + np.exp(alpha + scores_pos))
    neg_denominator = label_observation_probs[0, 0] * (1 + np.exp(-alpha - scores_neg))
    neg_denominator += label_observation_probs[1, 0] * (1 + np.exp(alpha + scores_neg))
    return pos_factor * np.sum(weights_pos / pos_denominator) + neg_factor * np.sum(weights_neg / neg_denominator)


def logit_derivative_label_probs(
    scores_pos,
    scores_neg,
    alpha,
    label_observation_parameters,
    label_observation_probs,
    weights_pos=None,
    weights_neg=None,
):
    """Derivative with respect to probabilities of correct labels eta_00 and eta_11."""
    scores_pos = np.asarray(scores_pos, dtype=float)
    scores_neg = np.asarray(scores_neg, dtype=float)
    weights_pos = np.ones_like(scores_pos) if weights_pos is None else np.asarray(weights_pos, dtype=float)
    weights_neg = np.ones_like(scores_neg) if weights_neg is None else np.asarray(weights_neg, dtype=float)

    d_00_prior = label_observation_parameters[0, 0] / label_observation_probs[0, 0]
    d_00_prior -= label_observation_parameters[0, 1] / label_observation_probs[0, 1]
    d_00_neg = 1 / (1 + np.exp(alpha + scores_neg))
    d_00_neg /= (
        label_observation_probs[0, 0] / (1 + np.exp(-alpha - scores_neg))
        + label_observation_probs[1, 0] / (1 + np.exp(alpha + scores_neg))
    )
    d_00_pos = -1 / (1 + np.exp(alpha + scores_pos))
    d_00_pos /= (
        label_observation_probs[0, 1] * (1 + np.exp(-alpha - scores_pos))
        + label_observation_probs[1, 1] * (1 + np.exp(alpha + scores_pos))
    )

    d_11_prior = label_observation_parameters[1, 1] / label_observation_probs[1, 1]
    d_11_prior -= label_observation_parameters[1, 0] / label_observation_probs[1, 0]
    d_11_neg = -1 / (1 + np.exp(-alpha - scores_neg))
    d_11_neg /= (
        label_observation_probs[1, 0] / (1 + np.exp(-alpha - scores_neg))
        + label_observation_probs[0, 0] / (1 + np.exp(alpha + scores_neg))
    )
    d_11_pos = 1 / (1 + np.exp(-alpha - scores_pos))
    d_11_pos /= (
        label_observation_probs[1, 1] / (1 + np.exp(-alpha - scores_pos))
        + label_observation_probs[0, 1] / (1 + np.exp(alpha + scores_pos))
    )

    return np.array(
        [
            d_00_prior + np.sum(weights_neg * d_00_neg) + np.sum(weights_pos * d_00_pos),
            d_11_prior + np.sum(weights_neg * d_11_neg) + np.sum(weights_pos * d_11_pos),
        ]
    )


def logit_subderivative_theta(alignments, labels, alpha, label_observation_probs=None, mirna_length=22, sample_weight=None):
    """Return subderivatives with respect to M, G_miR, and G_gene."""
    M_subd = np.zeros((4, 4, mirna_length))
    G_miR_subd = np.zeros(max(mirna_length - 1, 0))
    G_gene_subd = np.zeros(mirna_length)
    if sample_weight is None:
        sample_weight = np.ones(len(labels))
    for aln, lab, weight in zip(alignments, labels, sample_weight):
        score, aligned_miR, aligned_mR, crd_miR, _ = aln
        if lab == 1:
            if label_observation_probs is None:
                scaling_factor = 1 / (1 + np.exp(alpha + score))
            else:
                denominator = label_observation_probs[0, 1] * (1 + np.exp(-alpha - score))
                denominator += label_observation_probs[1, 1] * (1 + np.exp(alpha + score))
                scaling_factor = (label_observation_probs[1, 1] - label_observation_probs[0, 1]) / denominator
        elif lab == 0:
            if label_observation_probs is None:
                scaling_factor = -1 / (1 + np.exp(-alpha - score))
            else:
                denominator = label_observation_probs[0, 0] * (1 + np.exp(-alpha - score))
                denominator += label_observation_probs[1, 0] * (1 + np.exp(alpha + score))
                scaling_factor = (label_observation_probs[1, 0] - label_observation_probs[0, 0]) / denominator
        else:
            raise ValueError("Labels must be 0 or 1.")

        last_mir_crd = -1
        mir_gap_count = 0
        for aln_crd, mir_crd in enumerate(crd_miR):
            if mir_crd != -1:
                if last_mir_crd > -1:
                    G_miR_subd[last_mir_crd] += mir_gap_count * scaling_factor * weight
                last_mir_crd = mir_crd
                mir_gap_count = 0
                nt1 = aligned_miR[aln_crd]
                nt2 = aligned_mR[aln_crd]
                if nt2 == "-":
                    G_gene_subd[mir_crd] += scaling_factor * weight
                else:
                    M_subd[NUCL_DICT[nt1], NUCL_DICT[nt2], mir_crd] += scaling_factor * weight
            else:
                mir_gap_count += 1
    return M_subd, G_miR_subd, G_gene_subd
