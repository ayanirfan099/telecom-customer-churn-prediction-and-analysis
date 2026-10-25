import hashlib
import json
from pathlib import Path
from time import perf_counter

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

NUMERIC = ["tenure", "MonthlyCharges", "TotalCharges"]
CATEGORICAL = [
    "gender",
    "SeniorCitizen",
    "Partner",
    "Dependents",
    "PhoneService",
    "MultipleLines",
    "InternetService",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
    "Contract",
    "PaperlessBilling",
    "PaymentMethod",
]
FEATURES = NUMERIC + CATEGORICAL


def validate(frame, target=True):
    required = FEATURES + (["Churn"] if target else [])
    missing = set(required) - set(frame)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")
    if frame.empty:
        raise ValueError("Empty dataset")
    if "customerID" in frame and frame.customerID.duplicated().any():
        raise ValueError("Duplicate customer IDs")
    data = frame[FEATURES].copy()
    for name in NUMERIC:
        original = data[name]
        data[name] = pd.to_numeric(original.replace(r"^\s*$", np.nan, regex=True), errors="coerce")
        invalid = data[name].isna() & original.notna() & original.astype(str).str.strip().ne("")
        if invalid.any() or (data[name].dropna() < 0).any() or np.isinf(data[name].dropna()).any():
            raise ValueError(f"Invalid nonnegative numeric values: {name}")
    for name in CATEGORICAL:
        data[name] = data[name].astype(str)
    if not target:
        return data
    if not frame.Churn.isin(["Yes", "No"]).all():
        raise ValueError("Churn labels must be Yes/No")
    y = frame.Churn.eq("Yes").astype(int)
    if y.nunique() != 2:
        raise ValueError("Both classes required")
    return data, y


def preprocessor():
    return ColumnTransformer(
        [
            (
                "numeric",
                Pipeline(
                    [("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())]
                ),
                NUMERIC,
            ),
            ("category", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CATEGORICAL),
        ]
    )


def scores(y, p, threshold=0.5):
    p = np.asarray(p, dtype=float)
    pred = p >= threshold
    frac, mean = calibration_curve(y, p, n_bins=8, strategy="quantile")
    return {
        "accuracy": accuracy_score(y, pred),
        "precision": precision_score(y, pred, zero_division=0),
        "recall": recall_score(y, pred, zero_division=0),
        "f1": f1_score(y, pred, zero_division=0),
        "roc_auc": roc_auc_score(y, p),
        "pr_auc": average_precision_score(y, p),
        "brier": brier_score_loss(y, p),
        "confusion_matrix": confusion_matrix(y, pred).tolist(),
        "calibration": {"mean_probability": mean.tolist(), "positive_fraction": frac.tolist()},
    }


def explain(model, row):
    """Local log-odds contributions; available only for logistic regression."""
    estimator = model.named_steps["model"]
    if not hasattr(estimator, "coef_"):
        raise ValueError("Local additive explanation requires logistic model")
    prep = model.named_steps["preprocess"]
    z = prep.transform(row)
    return dict(
        zip(prep.get_feature_names_out(), (z[0] * estimator.coef_[0]).tolist(), strict=True)
    )


def train(csv, output, seed=42):
    start = perf_counter()
    frame = pd.read_csv(csv)
    x, y = validate(frame)
    train_idx, test_idx = train_test_split(
        np.arange(len(x)), test_size=0.2, stratify=y, random_state=seed
    )
    train_idx, val_idx = train_test_split(
        train_idx, test_size=0.25, stratify=y.iloc[train_idx], random_state=seed
    )
    out = Path(output)
    out.mkdir(parents=True, exist_ok=True)
    candidates = {
        "dummy": DummyClassifier(strategy="prior"),
        "logistic_C0.1": LogisticRegression(C=0.1, max_iter=2000, random_state=seed),
        "logistic_C1": LogisticRegression(C=1, max_iter=2000, random_state=seed),
        "forest_depth6": RandomForestClassifier(
            n_estimators=100, max_depth=6, n_jobs=2, random_state=seed
        ),
        "forest_depth12": RandomForestClassifier(
            n_estimators=100, max_depth=12, n_jobs=2, random_state=seed
        ),
        "boost_depth1": GradientBoostingClassifier(max_depth=1, random_state=seed),
        "boost_depth2": GradientBoostingClassifier(max_depth=2, random_state=seed),
    }
    fitted = {}
    validation = {}
    for name, est in candidates.items():
        model = Pipeline([("preprocess", preprocessor()), ("model", est)]).fit(
            x.iloc[train_idx], y.iloc[train_idx]
        )
        fitted[name] = model
        validation[name] = scores(y.iloc[val_idx], model.predict_proba(x.iloc[val_idx])[:, 1])
    best = max(validation, key=lambda k: validation[k]["pr_auc"])
    model = fitted[best]
    probabilities = model.predict_proba(x.iloc[val_idx])[:, 1]
    thresholds = np.linspace(0.1, 0.9, 81)
    threshold = float(max(thresholds, key=lambda t: f1_score(y.iloc[val_idx], probabilities >= t)))
    # No refit after selecting threshold: training distribution stays fixed.
    test_p = model.predict_proba(x.iloc[test_idx])[:, 1]
    imp = permutation_importance(
        model,
        x.iloc[val_idx],
        y.iloc[val_idx],
        n_repeats=3,
        random_state=seed,
        scoring="average_precision",
        n_jobs=2,
    )
    result = {
        "dataset": "IBM Telco Customer Churn",
        "sha256": hashlib.sha256(Path(csv).read_bytes()).hexdigest(),
        "seed": seed,
        "split_counts": {
            "train": len(train_idx),
            "validation": len(val_idx),
            "test": len(test_idx),
        },
        "selected_model": best,
        "threshold": threshold,
        "validation": validation,
        "test": scores(y.iloc[test_idx], test_p, threshold),
        "missing_values": frame.isna().sum().to_dict(),
        "normalized_missing_values": x.isna().sum().to_dict(),
        "class_fraction": float(y.mean()),
        "permutation_importance_validation": dict(
            zip(FEATURES, imp.importances_mean.tolist(), strict=True)
        ),
        "runtime_seconds": perf_counter() - start,
    }
    logistic = fitted["logistic_C1"]
    result["individual_logistic_logodds_contributions"] = explain(logistic, x.iloc[[test_idx[0]]])
    names = logistic.named_steps["preprocess"].get_feature_names_out()
    result["global_logistic_coefficients"] = dict(
        zip(names, logistic.named_steps["model"].coef_[0].tolist(), strict=True)
    )
    try:
        import shap

        z = logistic.named_steps["preprocess"].transform(x.iloc[train_idx[:100]])
        explainer = shap.LinearExplainer(logistic.named_steps["model"], z)
        values = explainer(logistic.named_steps["preprocess"].transform(x.iloc[test_idx[:5]]))
        result["shap_logistic"] = {
            "status": "executed",
            "scale": "log odds",
            "values": values.values.tolist(),
            "feature_names": names.tolist(),
        }
    except (ImportError, ValueError, TypeError) as exc:
        result["shap_logistic"] = {"status": "unavailable", "reason": str(exc)}
    joblib.dump(
        {"model": model, "threshold": threshold, "features": FEATURES}, out / "model.joblib"
    )
    joblib.dump(
        {"model": logistic, "threshold": 0.5, "features": FEATURES}, out / "logistic.joblib"
    )
    (out / "split_indices.json").write_text(
        json.dumps(
            {"train": train_idx.tolist(), "validation": val_idx.tolist(), "test": test_idx.tolist()}
        )
    )
    errors = frame.iloc[test_idx].copy()
    errors["probability"] = test_p
    errors["prediction"] = test_p >= threshold
    # Only aggregate error analysis is persisted: no customer identifiers.
    result["error_analysis"] = {
        "false_positives": int(((test_p >= threshold) & (y.iloc[test_idx].to_numpy() == 0)).sum()),
        "false_negatives": int(((test_p < threshold) & (y.iloc[test_idx].to_numpy() == 1)).sum()),
        "by_contract": {
            str(k): {
                "count": len(g),
                "error_rate": float(
                    (g.prediction.to_numpy() != g.Churn.eq("Yes").to_numpy()).mean()
                ),
            }
            for k, g in errors.groupby("Contract")
        },
    }
    (out / "metrics.json").write_text(json.dumps(result, indent=2))
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots()
    cal = result["test"]["calibration"]
    ax.plot(cal["mean_probability"], cal["positive_fraction"], "o-")
    ax.plot([0, 1], [0, 1], "--")
    ax.set(xlabel="Predicted churn probability", ylabel="Observed churn fraction")
    fig.savefig(out / "calibration.png")
    plt.close(fig)
    fig, ax = plt.subplots()
    ax.imshow(result["test"]["confusion_matrix"])
    ax.set(
        xlabel="Predicted",
        ylabel="Actual",
        title=f"Holdout confusion matrix, threshold={threshold:.2f}",
    )
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(result["test"]["confusion_matrix"][i][j]), ha="center")
    fig.savefig(out / "confusion.png")
    plt.close(fig)
    return result
