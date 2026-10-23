PY ?= python3
export PYTHONPATH := src

.PHONY: data db-up db-load sql features eda train mlflow-ui lint typecheck test check all

data:            ## download the Instacart 2017 dataset (~700 MB) from the Hugging Face mirror
	$(PY) -m affinity.download

db-up:           ## start PostgreSQL in Docker
	docker compose up -d && sleep 3

db-load: db-up   ## create schema and COPY the CSVs into PostgreSQL (~34M rows)
	$(PY) -m affinity.db load

sql:             ## exploration queries in src/affinity/sql/explore.sql -> reports/sql_exploration.md
	$(PY) -m affinity.db explore

features:        ## SQL aggregation of user x aisle history -> data/processed/*.parquet
	$(PY) -m affinity.db features

eda:             ## pandas EDA -> reports/figures/*.png + reports/eda.md
	$(PY) -m affinity.eda

train:           ## popularity / user-history / ALS / LightGBM rankers -> reports/metrics.md
	$(PY) -m affinity.train

mlflow-ui:       ## browse logged runs at http://127.0.0.1:5000
	$(PY) -m mlflow ui --backend-store-uri sqlite:///mlflow.db

test:
	OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 $(PY) -m pytest -q

lint:
	$(PY) -m flake8 --select=E9,F src tests

typecheck:
	$(PY) -m mypy --check-untyped-defs --ignore-missing-imports src tests

check: lint typecheck test

all: data db-load sql features eda train test
