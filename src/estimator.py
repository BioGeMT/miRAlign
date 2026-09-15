"""Scikit-learn compatible estimator for miRAlign."""

from __future__ import annotations

from math import ceil

import numpy as np
from scipy.special import expit
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.multiclass import check_classification_targets, unique_labels
from sklearn.utils.validation import check_is_fitted

from .miralign import miRAlign as _fit_miralign
from .optimization_functions import create_subgradient_step
from .positional_alignment import pos_aware_align_glocal, pos_aware_align_local


_ALIGNERS = {
    "local": pos_aware_align_local,
    "glocal": pos_aware_align_glocal,
}
_ALLOWED_NUCLEOTIDES = frozenset("ATCG")


def _split_sequence_pairs(X):
    """Validate paired sequence input and return normalized sequence lists."""
    pairs = np.asarray(X, dtype=object)
    if pairs.ndim != 2 or pairs.shape[1] != 2:
        raise ValueError("X must be array-like with shape (n_samples, 2).")
    if pairs.shape[0] == 0:
        raise ValueError("X must contain at least one sequence pair.")

    mirnas = []
    genes = []
    for index, (mirna, gene) in enumerate(pairs):
        if not isinstance(mirna, str) or not isinstance(gene, str):
            raise TypeError(f"X row {index} must contain two sequence strings.")
        # miRNA/target inputs may be provided in RNA notation. The core
        # positional aligners use the A/C/G/T alphabet, so U is equivalent to T.
        mirna = mirna.upper().replace("U", "T")
        gene = gene.upper().replace("U", "T")
        if not mirna or not gene:
            raise ValueError(f"X row {index} contains an empty sequence.")
        invalid = (set(mirna) | set(gene)) - _ALLOWED_NUCLEOTIDES
        if invalid:
            raise ValueError(
                f"X row {index} contains unsupported nucleotide(s): {sorted(invalid)}."
            )
        mirnas.append(mirna)
        genes.append(gene)
    return mirnas, genes


def _score_pair_chunk(pair_chunk, aligner, M, G_miR, G_gene):
    return [
        aligner(mirna, gene, M, G_miR, G_gene, backtrack=False)[0]
        for mirna, gene in pair_chunk
    ]


def _pair_chunks(mirnas, genes, chunk_size):
    pair_count = len(mirnas)
    for start in range(0, pair_count, chunk_size):
        stop = min(start + chunk_size, pair_count)
        yield list(zip(mirnas[start:stop], genes[start:stop]))


class MiRAlignClassifier(ClassifierMixin, BaseEstimator):
    """Binary classifier using learned position-aware miRNA-target alignments.

    Parameters are learned by the existing :func:`src.miralign.miRAlign`
    implementation. The estimator adds the standard scikit-learn fit and
    prediction protocol without duplicating the alignment model.

    ``X`` must have shape ``(n_samples, 2)`` with miRNA sequences in column 0
    and target sequences in column 1. Input is case-insensitive and RNA ``U`` is
    normalized to ``T`` for the underlying A/C/G/T alignment alphabet. Mixed
    miRNA lengths are supported. If
    ``model_length`` is omitted, the longest miRNA seen during ``fit`` defines
    the learned parameter length, exactly as in the core implementation.
    """

    def __init__(
        self,
        aligner="local",
        step_scale=1e-4,
        step_power=0.5,
        step_decay_burnin=300,
        M_prior=None,
        G_miR_prior=None,
        G_gene_prior=None,
        prior_precision=0.0,
        label_prior=None,
        model_length=None,
        max_iter=100,
        tol=1e-3,
        num_threads=1,
        verbose=False,
    ):
        self.aligner = aligner
        self.step_scale = step_scale
        self.step_power = step_power
        self.step_decay_burnin = step_decay_burnin
        self.M_prior = M_prior
        self.G_miR_prior = G_miR_prior
        self.G_gene_prior = G_gene_prior
        self.prior_precision = prior_precision
        self.label_prior = label_prior
        self.model_length = model_length
        self.max_iter = max_iter
        self.tol = tol
        self.num_threads = num_threads
        self.verbose = verbose

    def _resolve_aligner(self):
        try:
            return _ALIGNERS[self.aligner]
        except KeyError as exc:
            raise ValueError(
                f"aligner must be one of {sorted(_ALIGNERS)}; got {self.aligner!r}."
            ) from exc

    def fit(self, X, y, sample_weight=None):
        """Fit the alignment model and return ``self``."""
        mirnas, genes = _split_sequence_pairs(X)
        y = np.asarray(y)
        if y.ndim != 1:
            raise ValueError("y must be a one-dimensional binary target.")
        if len(y) != len(mirnas):
            raise ValueError("X and y have inconsistent lengths.")
        check_classification_targets(y)
        classes = unique_labels(y)
        if len(classes) != 2:
            raise ValueError("MiRAlignClassifier requires exactly two classes.")
        if int(self.max_iter) < 1:
            raise ValueError("max_iter must be at least 1.")
        if int(self.num_threads) < 1:
            raise ValueError("num_threads must be at least 1.")
        if self.tol is not None:
            tol = float(self.tol)
            if not np.isfinite(tol) or tol < 0:
                raise ValueError("tol must be a finite non-negative number or None.")

        binary_y = (y == classes[1]).astype(int)
        aligner_function = self._resolve_aligner()
        step_function = create_subgradient_step(
            float(self.step_scale),
            float(self.step_power),
            int(self.step_decay_burnin),
        )

        # Keep fitted state atomic: nothing ending in '_' is assigned until the
        # complete core fit succeeds. A failed refit therefore leaves the last
        # successful fitted model internally consistent.
        result = _fit_miralign(
            mirna_list=mirnas,
            gene_list=genes,
            label_list=binary_y,
            aligner=aligner_function,
            step_function=step_function,
            M_prior=self.M_prior,
            G_miR_prior=self.G_miR_prior,
            G_gene_prior=self.G_gene_prior,
            prior_precision=float(self.prior_precision),
            label_prior=self.label_prior,
            model_length=self.model_length,
            sample_weight=sample_weight,
            MAX_ITER=int(self.max_iter),
            tol=self.tol,
            num_threads=int(self.num_threads),
            verbose=bool(self.verbose),
        )

        self.classes_ = classes
        self.M_ = np.array(result["M"], copy=True)
        self.G_miR_ = np.array(result["G_miR"], copy=True)
        self.G_gene_ = np.array(result["G_gene"], copy=True)
        self.alpha_ = float(np.asarray(result["alpha"]).ravel()[0])
        self.model_length_ = int(self.M_.shape[-1])
        self.aligner_function_ = aligner_function
        self.aligner_name_ = self.aligner
        self.result_ = result
        self.auprc_trajectory_ = list(result.get("auprc_trajectory", []))
        self.loglik_trajectory_ = list(result.get("loglik_trajectory", []))
        self.subgradient_norm_trajectory_ = list(
            result.get("subgradient_norm_trajectory", [])
        )
        self.final_loglik_ = float(result["final_loglik"])
        self.label_observation_probs_ = result.get("label_observation_probs")
        self.optimizer_warnings_ = list(result.get("optimizer_warnings", []))
        self.n_iter_ = int(result.get("n_iter", len(self.subgradient_norm_trajectory_)))
        self.converged_ = bool(result.get("converged", False))
        self.n_features_in_ = 2
        return self

    def decision_function(self, X):
        """Return raw logits (alignment score plus learned intercept)."""
        check_is_fitted(
            self,
            ["classes_", "M_", "G_miR_", "G_gene_", "alpha_", "model_length_"],
        )
        mirnas, genes = _split_sequence_pairs(X)
        too_long = sorted({len(mirna) for mirna in mirnas if len(mirna) > self.model_length_})
        if too_long:
            raise ValueError(
                "miRNA sequences cannot be longer than the fitted model length; "
                f"model_length_={self.model_length_}, observed lengths {too_long}."
            )

        pair_count = len(mirnas)
        if int(self.num_threads) == 1 or pair_count == 0:
            scores = _score_pair_chunk(
                list(zip(mirnas, genes)),
                self.aligner_function_,
                self.M_,
                self.G_miR_,
                self.G_gene_,
            )
        else:
            from joblib import Parallel, delayed

            n_jobs = min(int(self.num_threads), pair_count)
            chunk_size = max(1, ceil(pair_count / (n_jobs * 4)))
            chunked_scores = Parallel(
                n_jobs=n_jobs,
                prefer="processes",
                return_as="list",
            )(
                delayed(_score_pair_chunk)(
                    pair_chunk,
                    self.aligner_function_,
                    self.M_,
                    self.G_miR_,
                    self.G_gene_,
                )
                for pair_chunk in _pair_chunks(mirnas, genes, chunk_size)
            )
            scores = [score for chunk in chunked_scores for score in chunk]

        return np.asarray(scores, dtype=float) + self.alpha_

    def predict_proba(self, X):
        """Return class probabilities ordered as ``classes_``."""
        positive_proba = expit(self.decision_function(X))
        return np.column_stack([1.0 - positive_proba, positive_proba])

    def predict(self, X):
        """Predict class labels using the standard zero-logit threshold."""
        positive = self.decision_function(X) > 0.0
        return self.classes_[positive.astype(int)]


# Canonical short name requested for the reusable model API.
MiRAlign = MiRAlignClassifier

__all__ = ["MiRAlign", "MiRAlignClassifier"]
