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

Use `--split-strategy` to choose the validation design:

```text
random       stratified row-level split
group_mirna  validation miRNAs are unseen during fitting
group_gene   validation genes are unseen during fitting
```

All miRNA lengths in the loaded splits are included. The model allocates one
shared position-specific parameter set up to the longest loaded miRNA, while
shorter miRNAs are kept as real shorter sequences. Their absent tail positions
do not contribute alignment scores or gradient updates.

To keep the grid search tractable, each grid configuration is scored only on
the fit and validation partitions. Held-out miRBench evaluation splits are
scored only for the selected grid model and for the final refit model.

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
split_strategy: random
sample_weight: none, mirna_gene_sqrt
max_iter: 100
```

The workflow writes:

```text
results/case_study_for_mirna/<run-tag>/
  dataset_summary.csv         aggregate and per-length split counts
  entity_frequency_summary.csv
  top_entities.csv
  split_overlap.csv
  summary.csv                 grid summaries with fit/validation metrics
  metrics.csv                 grid fit/validation metrics
  pr_points.csv               grid fit/validation precision-recall curves
  roc_points.csv              grid fit/validation ROC curves
  trajectories.csv
  errors.csv
  best_grid_model/
    evaluation/
      summary.csv             selected grid model held-out metrics
      metrics.csv
      pr_points.csv
      roc_points.csv
    model.pkl
    model_parameters.json
    selected_summary.json
  final_refit/
    summary.csv               final refit train/held-out metrics
    metrics.csv
    pr_points.csv
    roc_points.csv
```

## Run the workflow

To reproduce both training-family rows for the AUPRC table:

```bash
uv run python case_study_for_mirna/run_mirna_auprc_table.py
```

The runner executes the following full runs.

`--config-workers` controls how many grid configurations are fit at the same
time. `--num-threads` controls pair-alignment workers inside each configuration.
Avoid multiplying both too aggressively. On a roughly 65-core machine, a
reasonable starting point is:

```text
--config-workers 4 --num-threads 16
```

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
  --max-iters 100 \
  --final-max-iter 500 \
  --num-threads 16 \
  --config-workers 4 \
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
  --max-iters 100 \
  --final-max-iter 500 \
  --num-threads 16 \
  --config-workers 4 \
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
  --max-iters 1 \
  --final-max-iter 0 \
  --num-threads 1 \
  --config-workers 1 \
  --limit-configs 1 \
  --run-tag smoke
```

Saved models can be evaluated without refitting:

```bash
uv run python case_study_for_mirna/case_study_mirna.py \
  --evaluate-only \
  --trained-model results/case_study_for_mirna/smoke/best_grid_model/model.pkl \
  --dataset hejret \
  --eval-splits hejret_test \
  --num-threads 8 \
  --run-tag evaluate_saved_model
```
