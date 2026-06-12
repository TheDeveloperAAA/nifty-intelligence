PY := .venv/bin/python
export PYTHONHASHSEED := 0
export OMP_NUM_THREADS := 4

.PHONY: all data features models volatility portfolio risk anomaly explain report app test lint smoke verify clean

all: data features models volatility portfolio risk anomaly explain report

data:
	$(PY) -m src.cli run data

features:
	$(PY) -m src.cli run features

models:
	$(PY) -m src.cli run models

volatility:
	$(PY) -m src.cli run volatility

portfolio:
	$(PY) -m src.cli run portfolio

risk:
	$(PY) -m src.cli run risk

anomaly:
	$(PY) -m src.cli run anomaly

explain:
	$(PY) -m src.cli run explain

report:
	$(PY) -m src.cli run report

app:
	.venv/bin/streamlit run app/Home.py

test:
	$(PY) -m pytest -q

lint:
	.venv/bin/ruff check src app tests scripts

smoke:
	$(PY) -m src.cli run all --config config/config_smoke.yaml --force

verify:
	$(PY) scripts/verify_repro.py

clean:
	rm -rf artifacts/heavy/*
