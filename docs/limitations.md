# Limitations

One dataset and one seed; no external/temporal validation or causal interpretation. Data are a public sample, not a representative production customer population. Threshold optimizes F1 rather than business cost. No production recommendation. Dashboard loads trusted local joblib only. Raw data and checkpoints are not committed; retrain to recreate them. SHAP explanations describe the logistic comparison model, not gradient boosting. Missingness inspection is descriptive and does not fit transformations on holdout.
