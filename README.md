# miRAlign
Elucidating miRNA-target interactions through a position-aware local alignment with learnable weights

## Repository structure

```text
miRAlign/
├── src/                       core miRAlign package and model code
├── case_study_for_mirna/      miRBench case-study workflow and documentation
├── tests/                     smoke tests for core modeling and outputs
├── results/                   output directory for generated experiment results
├── *.ipynb                    notebooks used during development
├── pyproject.toml             Python project metadata and dependencies
└── uv.lock                    locked Python dependency versions
```

The main implementation lives in `src/`. The miRNA case-study scripts and their
detailed instructions live in `case_study_for_mirna/`; generated outputs are
written under `results/case_study_for_mirna/` by default.

## miRNA case study

The miRNA case study is the repository's end-to-end demonstration of miRAlign
for microRNA-target interaction prediction. It treats miRAlign as an
interpretable, position-aware alignment model: learn position-specific
substitution weights and gap penalties from labeled miRNA-target sequence pairs,
then evaluate the learned model on held-out benchmark splits.

This branch should be interpreted as an alternative all-length miRAlign
strategy, not as a fixed-length case-study workflow. Instead of filtering to one
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
