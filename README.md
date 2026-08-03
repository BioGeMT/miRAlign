# miRAlign
Elucidating miRNA-target interactions through a position-aware local alignment with learnable weights

## miRNA case study

The miRBench-based case-study workflow is being developed under:

```text
case_study_for_mirna/
```

It follows the DiscrimAlign case-study methodology: train on a named miRBench
training split, select configurations by validation AUPRC, refit the selected
configuration, and evaluate on held-out miRBench splits.
