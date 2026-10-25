"""Run with streamlit run src/churn/app.py."""

import json
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

from churn.pipeline import explain, validate

st.title("Telecom churn research dashboard")
st.caption(
    "Research prototype; probabilities are not validated business recommendations. Load only trusted local artifacts."
)
folder = Path(st.text_input("Local experiment directory", "results/telco"))
if not (folder / "model.joblib").is_file():
    st.info("Train a model using the README command first.")
    st.stop()
bundle = joblib.load(folder / "model.joblib")
metrics = json.loads((folder / "metrics.json").read_text())
st.write("Selected model", metrics["selected_model"])
st.json(metrics["test"])
st.dataframe(pd.DataFrame(metrics["validation"]).T)
for name in ["calibration.png", "confusion.png"]:
    if (folder / name).exists():
        st.image(str(folder / name))
f = st.file_uploader("Customer CSV, with or without Churn", type=["csv"])
if f:
    try:
        frame = pd.read_csv(f)
        x = validate(frame, target=False)
        st.dataframe(frame.head())
        st.dataframe(frame.describe(include="all").astype(str))
        st.write("Missing cells", frame.isna().sum())
        probability = bundle["model"].predict_proba(x)[:, 1]
        pred = frame.copy()
        pred["churn_probability"] = probability
        pred["predicted_churn"] = probability >= bundle["threshold"]
        st.dataframe(pred)
        st.download_button(
            "Download batch predictions", pred.to_csv(index=False), "predictions.csv", "text/csv"
        )
        row = st.number_input("Individual row", min_value=0, max_value=len(x) - 1, value=0)
        st.metric("Individual churn probability", f"{probability[row]:.3f}")
        local = joblib.load(folder / "logistic.joblib")
        st.caption(
            "Explanations below use the logistic comparison model; may differ from selected model. Contributions are log odds, not causal effects."
        )
        st.json(explain(local["model"], x.iloc[[row]]))
        st.bar_chart(pd.Series(metrics["permutation_importance_validation"]))
    except (ValueError, KeyError) as exc:
        st.error(str(exc))
