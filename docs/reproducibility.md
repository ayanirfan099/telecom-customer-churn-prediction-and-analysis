# Reproduce

Python >=3.11; executed in Python 3.12 on CPU. From repository root:
```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[dev]"
python scripts/download_data.py
python -m churn.cli --csv data/telco.csv --output results/telco
streamlit run src/churn/app.py
pytest -q
ruff check .
ruff format --check .
```

Use CPU PyTorch wheels from https://download.pytorch.org/whl/cpu if installing the deep learning dependencies on a CPU machine. Remote datasets and weights require separate access. Do not load untrusted pickle/joblib/checkpoint artifacts.
