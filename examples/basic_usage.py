"""Minimal end-to-end use of the public miRAlign estimator API.

Run from the repository root with:

    uv run python examples/basic_usage.py
"""

from __future__ import annotations

import pickle
import sys
from pathlib import Path

import numpy as np

# This repository is currently used directly from a checkout rather than as an
# installed Python package. Add the repository root so the public ``src`` API is
# importable when this file is executed as ``python examples/basic_usage.py``.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src import MiRAlign


# Tiny synthetic dataset for API demonstration only.
# X has two columns: [miRNA sequence, target sequence].
# y=1 means interacting; y=0 means non-interacting.
# RNA U is accepted and normalized internally to T.
X_train = np.array(
    [
        ["UGAGGUAGUAGGUUGUAUAGUU", "TCTACAACTACCTACTACCTCAGTAC"],
        ["UAGCAGCACGUAAAUAUUGGCG", "CCAATATTTACGTGCTGCTATGGAAC"],
        ["UGAGGUAGUAGGUUGUAUAGUU", "ACTACCTACTACCTCATGGTAACTGA"],
        ["UAGCAGCACGUAAAUAUUGGCG", "ATATTTACGTGCTGCTATCGGACCAA"],
        ["UGAGGUAGUAGGUUGUAUAGUU", "GCGCGCGCGCGCGCGCGCGCGCGCGC"],
        ["UAGCAGCACGUAAAUAUUGGCG", "AAAAAAAAAAAAAAAAAAAAAAAAAA"],
        ["UGAGGUAGUAGGUUGUAUAGUU", "CCCCCCCCCCCCCCCCCCCCCCCCCC"],
        ["UAGCAGCACGUAAAUAUUGGCG", "GGGGGGGGGGGGGGGGGGGGGGGGGG"],
    ],
    dtype=object,
)
y_train = np.array([1, 1, 1, 1, 0, 0, 0, 0])

# This fits a NEW model to the supplied labels. It does not load pretrained weights.
# A small max_iter keeps the example fast; use the defaults or tune hyperparameters
# for real experiments.
model = MiRAlign(max_iter=5, tol=None)
model.fit(X_train, y_train)

print("Model length:", model.model_length_)
print("Iterations:", model.n_iter_, "(fixed-iteration demo)")
print("Decision scores:", np.round(model.decision_function(X_train), 3))
print("Interaction probabilities:", np.round(model.predict_proba(X_train)[:, 1], 3))
print("Predictions:", model.predict(X_train))

# Learned position-aware parameters are available directly.
print("M_ shape:", model.M_.shape)
print("G_miR_ shape:", model.G_miR_.shape)
print("G_gene_ shape:", model.G_gene_.shape)
print("alpha_:", round(model.alpha_, 3))

# Fitted estimators can be serialized. This in-memory round trip avoids leaving
# a demo artifact on disk.
restored = pickle.loads(pickle.dumps(model))
np.testing.assert_allclose(
    restored.predict_proba(X_train),
    model.predict_proba(X_train),
)
print("Pickle round trip: OK")
