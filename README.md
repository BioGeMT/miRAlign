# miRAlign
Elucidating miRNA-target interactions through a position-aware local alignment with learnable weights

## Repository structure

```text
miRAlign/
├── src/miralign/              core miRAlign package and model code
├── case_study_for_mirna/      miRBench case-study workflow and documentation
├── results/                   output directory for generated experiment results
├── *.ipynb                    exploratory notebooks used during development
├── pyproject.toml             Python project metadata and dependencies
└── uv.lock                    locked Python dependency versions
```

The main implementation lives in `src/miralign/`. The miRNA case-study scripts
and their detailed instructions live in `case_study_for_mirna/`; generated
outputs are written under `results/case_study_for_mirna/`.

## miRNA case study

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

The workflow lets the user choose the training dataset and miRNA length. It
filters the selected miRBench training split to that length, initializes
length-specific miRAlign parameters, selects configurations by validation AUPRC,
refits the selected configuration, and evaluates on held-out miRBench splits.

Example:

```bash
uv run python case_study_for_mirna/case_study_mirna.py \
  --dataset hejret \
  --mirna-length 22 \
  --eval-splits hejret_test,manakov_test,manakov_leftout
```

DiscrimAlign references used by the case study:

- Ciach et al., "Discriminative learning of substitution matrices and gap
  penalties for pairwise alignment of biological sequences",
  https://doi.org/10.64898/2026.05.14.725168
- DiscrimAlign GitHub repository: https://github.com/BioGeMT/DiscrimAlign/
