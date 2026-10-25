# Quality review

Four tests cover schema rejection, blank values, unknown categories, train-fitted preprocessing, repeated-run metric equality, disjoint splits, saved model inference and CLI help. Streamlit startup and initial page are checked separately; browser upload interactions remain untested.

Reviewed data provenance, split isolation, deterministic seeds, explicit smoke-vs-benchmark labeling, no paid services, trusted artifact loading, metric serialization and reproduction commands. Remote Actions execution has not been verified. Remaining limitations:

One dataset and one seed; no external/temporal validation or causal interpretation. Data are a public sample, not a representative production customer population. Threshold optimizes F1 rather than business cost. No production recommendation. Dashboard loads trusted local joblib only. Raw data and checkpoints are not committed; retrain to recreate them. SHAP explanations describe the logistic comparison model, not gradient boosting. Missingness inspection is descriptive and does not fit transformations on holdout.


Final local checks and exact test runner outputs are saved in [VERIFICATION.json](VERIFICATION.json). Direct dependency versions from the executed environment are in [validated_versions.json](validated_versions.json). Automated static types were not checked; public numerical/data boundaries include annotations.
