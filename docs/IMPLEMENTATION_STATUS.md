# Implementation status

Implemented but not fully benchmarked

An end-to-end sklearn study with schema validation, seven baseline/hyperparameter candidates, train-only preprocessing, validation-selected threshold, held-out evaluation, explainability and a Streamlit dashboard.

Four tests cover schema rejection, blank values, unknown categories, train-fitted preprocessing, repeated-run metric equality, disjoint splits, saved model inference and CLI help. Streamlit startup and initial page are checked separately; browser upload interactions remain untested.

One dataset and one seed; no external/temporal validation or causal interpretation. Data are a public sample, not a representative production customer population. Threshold optimizes F1 rather than business cost. No production recommendation. Dashboard loads trusted local joblib only. Raw data and checkpoints are not committed; retrain to recreate them. SHAP explanations describe the logistic comparison model, not gradient boosting. Missingness inspection is descriptive and does not fit transformations on holdout.

Exact commands: [reproducibility](reproducibility.md). Push status and commit identity are recorded in the portfolio summary; remote CI success is not implied.
