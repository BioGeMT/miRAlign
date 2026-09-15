# miRAlign
Elucidating miRNA-target interactions through a position-aware alignment model with learnable weights.

miRAlign is a **trainable binary classifier**, not a pretrained interaction predictor.
Given labeled miRNA-target sequence pairs, it learns position-specific substitution
scores, gap penalties, and a logistic intercept. The recommended public interface is
the scikit-learn-compatible `MiRAlign` estimator.

## Quick start

The repository currently uses Python 3.10-3.12 and `uv` for environment management.
Clone the repository, create the locked environment, and run the included example:

```bash
git clone https://github.com/BioGeMT/miRAlign.git
cd miRAlign
uv sync
uv run python examples/basic_usage.py
```

The project is currently intended to be used from a repository checkout, so the
public estimator is imported as:

```python
from src import MiRAlign
```

Packaging it as an installable `miralign` Python package is a separate follow-up.

## Recommended estimator API

`MiRAlign` follows the standard scikit-learn estimator pattern. `X` has shape
`(n_samples, 2)`, with the miRNA in column 0 and the target sequence in column 1.
For interaction prediction, use `y=1` for interacting pairs and `y=0` for
non-interacting pairs. With these labels, `predict_proba(X)[:, 1]` is the
estimated interaction probability.

The snippets below use a tiny synthetic dataset only to demonstrate the API;
they are not benchmark data and do not represent a pretrained predictor.

```python
import numpy as np
from src import MiRAlign

X_train = np.array([
    ["UGAGGUAGUAGGUUGUAUAGUU", "TCTACAACTACCTACTACCTCAGTAC"],
    ["UAGCAGCACGUAAAUAUUGGCG",  "CCAATATTTACGTGCTGCTATGGAAC"],
    ["UGAGGUAGUAGGUUGUAUAGUU", "GCGCGCGCGCGCGCGCGCGCGCGCGC"],
    ["UAGCAGCACGUAAAUAUUGGCG",  "AAAAAAAAAAAAAAAAAAAAAAAAAA"],
], dtype=object)
y_train = np.array([1, 1, 0, 0])

model = MiRAlign()
model.fit(X_train, y_train)

scores = model.decision_function(X_train)
probabilities = model.predict_proba(X_train)
predictions = model.predict(X_train)
```

Lowercase input is accepted. RNA `U` is normalized internally to `T`, because the
underlying alignment alphabet is `A/C/G/T`. Ambiguous nucleotides such as `N` are
not currently supported.

For a complete runnable example, including model inspection and serialization, run:

```bash
uv run python examples/basic_usage.py
```

### Learned model state

After `fit`, the main learned attributes are:

- `M_`: position-specific nucleotide substitution scores, shape `(4, 4, L)`;
- `G_miR_`: gap penalties between miRNA positions, shape `(L - 1,)`;
- `G_gene_`: penalties for a gap opposite each miRNA position, shape `(L,)`;
- `alpha_`: logistic intercept;
- `model_length_`: the learned parameter length `L`;
- `n_iter_` and `converged_`: optimization diagnostics.

Mixed miRNA lengths are supported. If `model_length` is omitted, `L` is the
longest miRNA seen during fitting. Shorter miRNAs remain shorter sequences: absent
tail positions are not padded and do not contribute alignment scores or updates.
Prediction rejects a miRNA longer than the fitted `model_length_`.

### Saving and loading a fitted estimator

A fitted estimator can be serialized with `pickle`:

```python
import pickle

with open("miralign_model.pkl", "wb") as handle:
    pickle.dump(model, handle)

with open("miralign_model.pkl", "rb") as handle:
    restored = pickle.load(handle)

restored.predict_proba(X_train)
```

## What model does `MiRAlign` fit?

For each miRNA-target pair, miRAlign computes a learned position-aware alignment
score `s` and combines it with a learned intercept `alpha`:

```text
logit = s(miRNA, target) + alpha
P(interaction) = sigmoid(logit)
```

The default estimator uses `aligner="local"`, a position-aware local alignment.
`aligner="glocal"` is also available when the full miRNA should be consumed while
the matching target region remains local.

Training starts from the fixed Hejret/DiscrimAlign baseline parameters:

```text
match      =  0.724709
mismatch   = -0.647892
gap        = -0.901264
alpha      = -5.226262
```

These values are **initialization**, not a pretrained final predictor. Calling
`fit()` learns a new set of position-specific parameters from the supplied labels.
No pretrained miRAlign weights are loaded by the estimator. Likewise, `MiRAlign()`
does not automatically load the best configuration from the miRBench case study;
it constructs a new estimator with the documented defaults.

The estimator uses these main defaults:

```python
MiRAlign(
    aligner="local",
    step_scale=1e-4,
    step_power=0.5,
    step_decay_burnin=300,
    prior_precision=0.0,
    label_prior=None,
    model_length=None,
    max_iter=100,
    tol=1e-3,
    num_threads=1,
)
```

`tol` controls stopping by subgradient norm. Set `tol=None` to run exactly
`max_iter` outer iterations. The low-level `miRAlign(...)` function keeps
`tol=None` as its default so existing fixed-iteration workflows remain unchanged.

Because `MiRAlign` follows the scikit-learn parameter protocol, it can also be
cloned and used in model-selection utilities such as `GridSearchCV`. `fit` accepts
optional `sample_weight`.

### Low-level API

For normal use, prefer:

```python
from src import MiRAlign
```

The historical lower-level fitting function remains available as
`src.miralign.miRAlign` (and as `miRAlign` from `src`) for research workflows that
need direct access to the optimizer inputs and result dictionary. The names differ
only by capitalization, so new code should use `MiRAlign` unless the low-level API
is specifically required.

## Repository structure

```text
miRAlign/
├── src/                       core miRAlign model and estimator API
├── examples/                  small runnable usage examples
├── case_study_for_mirna/      miRBench case-study workflow and documentation
├── tests/                     core and estimator tests
├── results/                   generated experiment outputs
├── *.ipynb                    development notebooks
├── pyproject.toml             Python project metadata and dependencies
└── uv.lock                    locked Python dependency versions
```

The estimator API is the shortest path for fitting miRAlign to your own labeled
sequence pairs. The case-study workflow is the end-to-end research pipeline used
for miRBench model selection and held-out evaluation.

## miRNA case study

The miRNA case study is the repository's end-to-end demonstration of miRAlign
for microRNA-target interaction prediction. It treats miRAlign as an
interpretable, position-aware alignment model: learn position-specific
substitution weights and gap penalties from labeled miRNA-target sequence pairs,
then evaluate the learned model on held-out benchmark splits.

The current case study uses an all-length miRAlign strategy rather than a
fixed-length workflow. Instead of filtering to one
miRNA length, it loads all selected miRNA lengths, allocates parameters up to
the longest loaded miRNA, and records performance slices by miRNA length,
seen/unseen entities, and frequency bins.

As a final product, the case study provides:

- a command-line workflow for training and evaluating miRAlign on miRBench;
- mixed-length modeling with a shared position-specific parameter set;
- validation strategies for random rows, unseen miRNAs, or unseen genes;
- grid search over aligner, optimization, prior, label-noise, and sample-weight
  settings;
- validation-AUPRC model selection followed by full-train refitting;
- dataset diagnostics, held-out AUPRC/ROC-AUC summaries, metric slices, PR/ROC
  curves, fitted models, and learned parameters under
  `results/case_study_for_mirna/`.

Performance should be interpreted with the held-out splits and diagnostic
slices, not only aggregate AUPRC. The strongest generalization claims should use
the `manakov_leftout` split, `group_mirna` validation, and frequency-sliced
metrics because miRBench was designed to reduce miRNA frequency-class artifacts.

The miRNA case study uses datasets from miRBench:

- Sammut et al., "miRBench: novel benchmark datasets for microRNA binding site
  prediction that mitigate against prevalent microRNA frequency class bias",
  *Bioinformatics* 41(Supplement_1), i542-i551 (2025),
  https://academic.oup.com/bioinformatics/article/41/Supplement_1/i542/8199406

See the case-study README for the full workflow details:

```text
case_study_for_mirna/README.md
```

The workflow lives under `case_study_for_mirna/`.

The workflow lets the user choose the training dataset, validation split design,
evaluation splits, and optional user-provided evaluation files. It initializes a
model length from the longest loaded miRNA, selects configurations by validation
AUPRC, refits the selected configuration, and evaluates on held-out miRBench or
custom splits. Prefer `--split-strategy group_mirna` when tuning for claims
about unseen-miRNA generalization.

Example:

```bash
uv run python case_study_for_mirna/case_study_mirna.py \
  --dataset hejret \
  --eval-splits hejret_test,manakov_test,manakov_leftout \
  --split-strategy group_mirna \
  --sample-weights none,mirna_gene_sqrt
```

DiscrimAlign references used by the case study:

- Ciach et al., "Discriminative learning of substitution matrices and gap
  penalties for pairwise alignment of biological sequences",
  https://doi.org/10.64898/2026.05.14.725168
- DiscrimAlign GitHub repository: https://github.com/BioGeMT/DiscrimAlign/
