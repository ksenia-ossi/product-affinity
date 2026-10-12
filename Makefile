PY ?= python3
export PYTHONPATH := src

.PHONY: data db-up db-load sql features eda train test all

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

test:
	$(PY) -m pytest -q

all: data db-load sql features eda train test
