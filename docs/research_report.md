# Research report

Executed on the real upstream CSV, SHA256 `16320c9c1ec72448db59aa0a26a0b95401046bef5d02fd3aeb906448e3055e91`. Selected `boost_depth1`, threshold 0.35. Holdout n=1409: accuracy 0.7800, F1 0.6344, ROC-AUC 0.8447, average precision 0.6666, Brier 0.1361. Logistic SHAP executed. Full comparison, calibration, confusion matrix, error-by-contract and contributions: [metrics](results/telco/metrics.json).

![Calibration](results/telco/calibration.png)
![Confusion matrix](results/telco/confusion.png)
