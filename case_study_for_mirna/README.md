# miRNA case study

This directory contains the miRBench-based miRNA case-study workflow for miRAlign.
The workflow follows the same structure used for the DiscrimAlign miRNA case
study: train on a named miRBench training split, select a configuration using a
validation split, refit the selected configuration, and report AUPRC/ROC-AUC on
held-out miRBench splits.

## Methodology

The case study uses the miRBench dataset interface and the following aliases:

```text
hejret_train
hejret_test
manakov_train
manakov_test
manakov_leftout
klimentova_test
```

Training data are split into fit and validation partitions with stratified
sampling. Configurations are ranked by validation AUPRC. The selected
configuration is then refit on the full training split and evaluated on the
requested held-out splits.

miRAlign initializes its position-specific substitution matrix and gap vectors
from the Hejret/DiscrimAlign simple-alignment baseline:

```text
match = 0.724709
mismatch = -0.647892
gap = -0.901264
alpha = -5.226262
```

When prior regularization is enabled, the same baseline should be used as the
default prior for `M`, `G_miR`, and `G_gene`.

`prior_precision` controls how strongly the fitted position-specific parameters
are pulled back toward that baseline. `0` means no prior regularization. Larger
values penalize deviations from the baseline more strongly, which can reduce
overfitting but may limit how much miRAlign adapts to the training data.

`step_decay_burnin` controls the learning-rate schedule. miRAlign keeps the
initial `step_scale` constant through this many iterations, then decays it as
`step_scale / (iteration - step_decay_burnin) ** step_power`. With the default
grid-search setting of 100 iterations and `step_decay_burnin=300`, grid runs use
a constant step size; the 500-iteration final refit starts decaying after
iteration 300.

`label_prior` controls whether miRAlign models noisy observed labels. `none`
uses the observed labels directly. The symmetric options start from 2x2 priors
where true negatives and true positives have the same assumed correctness rate:

```text
symmetric_95_5:
[[950,  50],
 [ 50, 950]]

symmetric_90_10:
[[900, 100],
 [100, 900]]

symmetric_80_20:
[[800, 200],
 [200, 800]]
```

This is different from `class_weight`: `label_prior` models possible label
noise, while `class_weight` changes how much positive and negative examples
contribute to the optimization.

## Grid

The default grid is intentionally bounded because miRAlign learns
position-specific parameters. The step scales include the DiscrimAlign table
runner values `0.00001,0.00005,0.0001,0.0005` plus `0.000012`, the value used
in the miRAlign notebook recommendation.

```text
aligner: local, glocal
step_scale: 0.00001, 0.000012, 0.00005, 0.0001, 0.0005
step_decay_burnin: 300
prior_precision: 0, 1
label_prior: none, symmetric_95_5, symmetric_90_10, symmetric_80_20
class_weight: none, balanced, pos2
max_iter: 100
```

The workflow writes:

```text
results/case_study_for_mirna/<dataset>_<run-tag>/
  summary.csv
  metrics.csv
  pr_points.csv
  roc_points.csv
  trajectories.csv
  errors.csv
  best_grid_model/
    model.pkl
    model_parameters.json
    selected_summary.json
```

## Run the workflow

To reproduce both training-family rows for the AUPRC table:

```bash
uv run python case_study_for_mirna/run_mirna_auprc_table.py
```

The runner executes the following full runs.

### Hejret-trained run

```bash
uv run python case_study_for_mirna/case_study_mirna.py \
  --dataset hejret \
  --eval-splits hejret_test,manakov_test,manakov_leftout \
  --aligners local,glocal \
  --step-scales 0.00001,0.000012,0.00005,0.0001,0.0005 \
  --step-decay-burnins 300 \
  --prior-precisions 0,1 \
  --label-priors none,symmetric_95_5,symmetric_90_10,symmetric_80_20 \
  --class-weights none,balanced,pos2 \
  --max-iters 100 \
  --final-max-iter 500 \
  --num-threads 65 \
  --run-tag train_hejret_eval_all
```

### Manakov-trained run

```bash
uv run python case_study_for_mirna/case_study_mirna.py \
  --dataset manakov \
  --eval-splits hejret_test,manakov_test,manakov_leftout \
  --aligners local,glocal \
  --step-scales 0.00001,0.000012,0.00005,0.0001,0.0005 \
  --step-decay-burnins 300 \
  --prior-precisions 0,1 \
  --label-priors none,symmetric_95_5,symmetric_90_10,symmetric_80_20 \
  --class-weights none,balanced,pos2 \
  --max-iters 100 \
  --final-max-iter 500 \
  --num-threads 65 \
  --run-tag train_manakov_eval_all
```

This follows the DiscrimAlign case-study approach: run the grid at
`--max-iters 100`, select by validation AUPRC, then refit the selected
configuration at `--final-max-iter 500` on the full training split.

For a quick smoke run, limit the grid and skip the final refit:

```bash
uv run python case_study_for_mirna/case_study_mirna.py \
  --dataset hejret \
  --eval-splits hejret_test \
  --aligners local \
  --step-scales 0.00001 \
  --step-decay-burnins 300 \
  --prior-precisions 0 \
  --label-priors none \
  --class-weights none \
  --max-iters 1 \
  --final-max-iter 0 \
  --limit-configs 1 \
  --run-tag smoke
```

Saved models can be evaluated without refitting:

```bash
uv run python case_study_for_mirna/case_study_mirna.py \
  --evaluate-only \
  --trained-model results/case_study_for_mirna/hejret_smoke/best_grid_model/model.pkl \
  --dataset hejret \
  --eval-splits hejret_test \
  --num-threads 8 \
  --run-tag evaluate_saved_model
```

## Core changes made for the case study

- Restored the likelihood and subgradient helpers that were split out of
  `src/miralign.py` but not committed as module files.
- Restored `create_subgradient_step` in `src/optimization_functions.py`.
- Implemented `logreg_starting_point` as a fixed Hejret/DiscrimAlign baseline
  initializer, matching the historical miRAlign notebooks.
- Fixed the glocal alignment wrapper so backtracked gaps are rendered with the
  gap-aware nucleotide alphabet.
- Added optional sample weighting to miRAlign likelihood, alpha optimization,
  and subgradient updates for the `none`, `balanced`, and `pos2` case-study
  class-weight settings.
