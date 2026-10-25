import subprocess
import sys

import numpy as np
import pandas as pd
import pytest

from churn.pipeline import CATEGORICAL, NUMERIC, preprocessor, scores, train, validate


def fixture(n=100):
    rng = np.random.default_rng(4)
    f = pd.DataFrame({k: rng.uniform(0, 20, n) for k in NUMERIC})
    for k in CATEGORICAL:
        f[k] = rng.choice(["Yes", "No"], n)
    f["Churn"] = np.where(f.tenure > 10, "Yes", "No")
    return f


def test_schema():
    f = fixture()
    x, y = validate(f)
    assert len(x) == len(y) == 100
    f["TotalCharges"] = f["TotalCharges"].astype(object)
    f.loc[0, "TotalCharges"] = " "
    assert np.isnan(validate(f)[0].TotalCharges.iloc[0])
    with pytest.raises(ValueError):
        validate(f.drop(columns="tenure"))
    f.loc[0, "tenure"] = -1
    with pytest.raises(ValueError):
        validate(f)


def test_unknown_categories_and_no_leakage():
    f = fixture()
    x, _ = validate(f)
    prep = preprocessor().fit(x.iloc[:50])
    mean = prep.named_transformers_["numeric"].named_steps["scale"].mean_.copy()
    x.loc[99, "Contract"] = "unseen"
    prep.transform(x.iloc[50:])
    np.testing.assert_array_equal(
        mean, prep.named_transformers_["numeric"].named_steps["scale"].mean_
    )


def test_metrics_and_cli():
    assert scores([0, 1], [0.1, 0.9])["f1"] == 1
    assert (
        subprocess.run(
            [sys.executable, "-m", "churn.cli", "--help"], capture_output=True, check=False
        ).returncode
        == 0
    )


def test_end_to_end(tmp_path):
    f = fixture(150)
    csv = tmp_path / "test.csv"
    f.to_csv(csv, index=False)
    a = train(csv, tmp_path / "run")
    b = train(csv, tmp_path / "repeat")
    assert a["test"] == b["test"]
    import json

    indices = json.loads((tmp_path / "run/split_indices.json").read_text())
    assert not set(indices["train"]) & set(indices["test"])
    assert not set(indices["validation"]) & set(indices["test"])
    import joblib

    bundle = joblib.load(tmp_path / "run/model.joblib")
    assert len(bundle["model"].predict(validate(f)[0])) == 150
