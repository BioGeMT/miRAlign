import pickle

import numpy as np
import pytest
from sklearn.base import clone
from sklearn.model_selection import GridSearchCV
from sklearn.utils.validation import check_is_fitted

from src import MiRAlign, MiRAlignClassifier, miRAlign
from src.optimization_functions import create_subgradient_step
from src.positional_alignment import pos_aware_align_local


@pytest.fixture
def sequence_pairs():
    X = np.array(
        [
            ["AAAA", "AAAAAAAA"],
            ["AAAT", "AAAATAAA"],
            ["CCCC", "CCCCCCCC"],
            ["GGGG", "GGGGGGGG"],
            ["AAAA", "TTTTTTTT"],
            ["AAAT", "CCCCCCCC"],
            ["CCCC", "AAAAAAAA"],
            ["GGGG", "TTTTTTTT"],
        ],
        dtype=object,
    )
    y = np.array([1, 1, 1, 1, 0, 0, 0, 0])
    return X, y


def test_estimator_alias_and_original_function_remain_available():
    assert MiRAlign is MiRAlignClassifier
    assert callable(miRAlign)


def test_fit_sets_atomic_sklearn_state_and_learned_parameters(sequence_pairs):
    X, y = sequence_pairs
    estimator = MiRAlign(max_iter=1, tol=None)

    fitted = estimator.fit(X, y)

    assert fitted is estimator
    check_is_fitted(estimator, ["M_", "G_miR_", "G_gene_", "alpha_"])
    assert estimator.classes_.tolist() == [0, 1]
    assert estimator.n_features_in_ == 2
    assert estimator.model_length_ == 4
    assert estimator.M_.shape == (4, 4, 4)
    assert estimator.G_miR_.shape == (3,)
    assert estimator.G_gene_.shape == (4,)
    assert estimator.n_iter_ == 1


def test_estimator_matches_core_fit_and_scoring(sequence_pairs):
    X, y = sequence_pairs
    estimator = MiRAlign(max_iter=1, tol=None, step_scale=1e-4).fit(X, y)

    core = miRAlign(
        mirna_list=X[:, 0].tolist(),
        gene_list=X[:, 1].tolist(),
        label_list=y,
        aligner=pos_aware_align_local,
        step_function=create_subgradient_step(1e-4, 0.5, 300),
        MAX_ITER=1,
        tol=None,
        num_threads=1,
    )

    np.testing.assert_allclose(estimator.M_, core["M"])
    np.testing.assert_allclose(estimator.G_miR_, core["G_miR"])
    np.testing.assert_allclose(estimator.G_gene_, core["G_gene"])
    assert estimator.alpha_ == pytest.approx(core["alpha"])

    manual = np.array(
        [
            pos_aware_align_local(
                mirna,
                gene,
                estimator.M_,
                estimator.G_miR_,
                estimator.G_gene_,
                backtrack=False,
            )[0]
            for mirna, gene in X
        ],
        dtype=float,
    ) + estimator.alpha_
    np.testing.assert_allclose(estimator.decision_function(X), manual)


def test_mixed_lengths_are_preserved(sequence_pairs):
    X, y = sequence_pairs
    mixed = X.copy()
    mixed[0, 0] = "AAA"
    mixed[1, 0] = "AAT"

    estimator = MiRAlign(max_iter=1, tol=None).fit(mixed, y)

    assert estimator.model_length_ == 4
    assert estimator.decision_function(mixed).shape == (len(mixed),)


def test_predict_proba_and_predict_follow_binary_class_order(sequence_pairs):
    X, y = sequence_pairs
    labels = np.where(y == 1, "target", "non_target")
    estimator = MiRAlign(max_iter=1, tol=None).fit(X, labels)

    probabilities = estimator.predict_proba(X)
    predictions = estimator.predict(X)

    assert probabilities.shape == (len(X), 2)
    np.testing.assert_allclose(probabilities.sum(axis=1), 1.0)
    assert set(predictions) <= set(estimator.classes_)
    expected = estimator.classes_[(estimator.decision_function(X) > 0.0).astype(int)]
    np.testing.assert_array_equal(predictions, expected)


def test_fit_accepts_sample_weight(sequence_pairs):
    X, y = sequence_pairs
    weights = np.linspace(0.5, 1.5, len(y))

    estimator = MiRAlign(max_iter=1, tol=None).fit(X, y, sample_weight=weights)

    assert np.isfinite(estimator.final_loglik_)


def test_failed_initial_fit_does_not_create_fitted_state(sequence_pairs):
    X, y = sequence_pairs
    estimator = MiRAlign(model_length=3, max_iter=1, tol=None)

    with pytest.raises(ValueError, match="longer than model_length"):
        estimator.fit(X, y)

    for attribute in ("classes_", "M_", "G_miR_", "G_gene_", "alpha_"):
        assert not hasattr(estimator, attribute)


def test_failed_refit_preserves_previous_consistent_model(sequence_pairs):
    X, y = sequence_pairs
    estimator = MiRAlign(model_length=4, max_iter=1, tol=None).fit(X, y)
    old_scores = estimator.decision_function(X).copy()
    old_classes = estimator.classes_.copy()

    estimator.set_params(model_length=3)
    with pytest.raises(ValueError, match="longer than model_length"):
        estimator.fit(X, y)

    np.testing.assert_array_equal(estimator.classes_, old_classes)
    np.testing.assert_allclose(estimator.decision_function(X), old_scores)


def test_clone_and_grid_search_use_standard_parameter_protocol(sequence_pairs):
    X, y = sequence_pairs
    estimator = MiRAlign(max_iter=1, tol=None)

    cloned = clone(estimator)
    assert cloned.get_params()["step_scale"] == estimator.step_scale

    search = GridSearchCV(
        MiRAlign(max_iter=1, tol=None),
        {"step_scale": [1e-4, 5e-5]},
        cv=2,
    )
    search.fit(X, y)

    assert isinstance(search.best_estimator_, MiRAlignClassifier)
    check_is_fitted(search.best_estimator_, ["M_", "alpha_"])


def test_pickle_round_trip_preserves_predictions(sequence_pairs):
    X, y = sequence_pairs
    estimator = MiRAlign(max_iter=1, tol=None).fit(X, y)

    restored = pickle.loads(pickle.dumps(estimator))

    np.testing.assert_allclose(
        restored.decision_function(X),
        estimator.decision_function(X),
    )
    np.testing.assert_array_equal(restored.predict(X), estimator.predict(X))


def test_core_tol_can_stop_outer_optimization(sequence_pairs):
    X, y = sequence_pairs
    result = miRAlign(
        mirna_list=X[:, 0].tolist(),
        gene_list=X[:, 1].tolist(),
        label_list=y,
        aligner=pos_aware_align_local,
        step_function=create_subgradient_step(1e-4, 0.5, 300),
        MAX_ITER=5,
        tol=1e12,
        num_threads=1,
    )

    assert result["converged"] is True
    assert result["n_iter"] == 1
    assert len(result["subgradient_norm_trajectory"]) == 1


def test_estimator_rejects_bad_input_and_unknown_aligner(sequence_pairs):
    X, y = sequence_pairs

    with pytest.raises(ValueError, match="shape"):
        MiRAlign(max_iter=1).fit(["AAAA", "CCCC"], y[:2])

    with pytest.raises(ValueError, match="aligner must be"):
        MiRAlign(aligner="unknown", max_iter=1).fit(X, y)

    bad = X.copy()
    bad[0, 0] = "AANA"
    with pytest.raises(ValueError, match="unsupported nucleotide"):
        MiRAlign(max_iter=1).fit(bad, y)
