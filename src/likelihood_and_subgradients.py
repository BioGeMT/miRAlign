"""Likelihood and subgradient utilities for miRAlign."""

from __future__ import annotations

import numpy as np

from .shared_global_vars import NUCL_DICT


TINY = np.finfo(float).tiny


def _sigmoid(values):
    values = np.asarray(values, dtype=float)
    result = np.empty_like(values, dtype=float)
    positive = values >= 0
    result[positive] = 1 / (1 + np.exp(-values[positive]))
    exp_values = np.exp(values[~positive])
    result[~positive] = exp_values / (1 + exp_values)
    return result


def _safe_label_probs(label_observation_probs):
    return np.clip(np.asarray(label_observation_probs, dtype=float), TINY, 1.0)


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
        label_observation_probs = _safe_label_probs(label_observation_probs)
        value += np.sum(label_observation_parameters * np.log(label_observation_probs))
        pos_prob = _sigmoid(alpha + scores_pos)
        neg_prob = _sigmoid(alpha + scores_neg)
        pos_part = label_observation_probs[1, 1] * pos_prob
        pos_part += label_observation_probs[0, 1] * (1 - pos_prob)
        neg_part = label_observation_probs[1, 0] * neg_prob
        neg_part += label_observation_probs[0, 0] * (1 - neg_prob)
        value += np.sum(weights_pos * np.log(np.maximum(pos_part, TINY)))
        value += np.sum(weights_neg * np.log(np.maximum(neg_part, TINY)))
    else:
        value -= np.sum(weights_pos * np.logaddexp(0, -alpha - scores_pos))
        value -= np.sum(weights_neg * np.logaddexp(0, alpha + scores_neg))

    return value


def logit_lhood_vect(scores, alpha):
    return _sigmoid(alpha + np.asarray(scores, dtype=float))


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
    pos_prob = _sigmoid(alpha + scores_pos)
    neg_prob = _sigmoid(alpha + scores_neg)
    if label_observation_probs is None:
        pos_part = 1 - pos_prob
        neg_part = neg_prob
        return np.sum(weights_pos * pos_part) - np.sum(weights_neg * neg_part)

    label_observation_probs = _safe_label_probs(label_observation_probs)
    pos_factor = label_observation_probs[1, 1] - label_observation_probs[0, 1]
    neg_factor = label_observation_probs[1, 0] - label_observation_probs[0, 0]
    pos_mix = label_observation_probs[1, 1] * pos_prob + label_observation_probs[0, 1] * (1 - pos_prob)
    neg_mix = label_observation_probs[1, 0] * neg_prob + label_observation_probs[0, 0] * (1 - neg_prob)
    pos_slope = pos_prob * (1 - pos_prob)
    neg_slope = neg_prob * (1 - neg_prob)
    pos_part = pos_factor * pos_slope / np.maximum(pos_mix, TINY)
    neg_part = neg_factor * neg_slope / np.maximum(neg_mix, TINY)
    return np.sum(weights_pos * pos_part) + np.sum(weights_neg * neg_part)


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
    label_observation_probs = _safe_label_probs(label_observation_probs)
    pos_prob = _sigmoid(alpha + scores_pos)
    neg_prob = _sigmoid(alpha + scores_neg)
    pos_mix = label_observation_probs[1, 1] * pos_prob + label_observation_probs[0, 1] * (1 - pos_prob)
    neg_mix = label_observation_probs[1, 0] * neg_prob + label_observation_probs[0, 0] * (1 - neg_prob)
    pos_mix = np.maximum(pos_mix, TINY)
    neg_mix = np.maximum(neg_mix, TINY)

    d_00_prior = label_observation_parameters[0, 0] / label_observation_probs[0, 0]
    d_00_prior -= label_observation_parameters[0, 1] / label_observation_probs[0, 1]
    d_00_neg = (1 - neg_prob) / neg_mix
    d_00_pos = -(1 - pos_prob) / pos_mix

    d_11_prior = label_observation_parameters[1, 1] / label_observation_probs[1, 1]
    d_11_prior -= label_observation_parameters[1, 0] / label_observation_probs[1, 0]
    d_11_neg = -neg_prob / neg_mix
    d_11_pos = pos_prob / pos_mix

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
        score_prob = _sigmoid(alpha + score)
        if lab == 1:
            if label_observation_probs is None:
                scaling_factor = 1 - score_prob
            else:
                label_observation_probs = _safe_label_probs(label_observation_probs)
                mix = label_observation_probs[1, 1] * score_prob
                mix += label_observation_probs[0, 1] * (1 - score_prob)
                slope = score_prob * (1 - score_prob)
                scaling_factor = (label_observation_probs[1, 1] - label_observation_probs[0, 1]) * slope / max(mix, TINY)
        elif lab == 0:
            if label_observation_probs is None:
                scaling_factor = -score_prob
            else:
                label_observation_probs = _safe_label_probs(label_observation_probs)
                mix = label_observation_probs[1, 0] * score_prob
                mix += label_observation_probs[0, 0] * (1 - score_prob)
                slope = score_prob * (1 - score_prob)
                scaling_factor = (label_observation_probs[1, 0] - label_observation_probs[0, 0]) * slope / max(mix, TINY)
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
