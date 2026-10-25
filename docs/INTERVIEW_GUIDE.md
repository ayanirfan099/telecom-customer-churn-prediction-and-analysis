# Study guide

These notes explain implemented methods; do not claim independent authorship or experiments beyond the recorded results.

## Why choose the threshold on validation data?

Threshold selection is a model decision; using test labels would bias the reported holdout performance.

## Why report average precision and Brier score?

Average precision describes ranking under imbalance; Brier score measures squared probability error and is sensitive to calibration.

## Why are SHAP values not causal effects?

They allocate a prediction relative to a background and modeling assumptions; they do not identify intervention effects or eliminate confounding.
