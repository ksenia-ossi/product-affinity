"""PostgreSQL side of the pipeline.

Usage: python -m affinity.db load | explore | features
  load      create schema, COPY the six CSVs, build indexes
  explore   run sql/explore.sql blocks -> reports/sql_exploration.md
  features  run sql/features.sql blocks -> data/processed/<name>.parquet
"""
import re
import sys
import warnings
from collections.abc import Callable
from pathlib import Path

import pandas as pd
import psycopg2

from .config import PG_DSN, PROCESSED, RAW_DIR, REPORTS

SQL_DIR = Path(__file__).parent / "sql"
warnings.filterwarnings("ignore", message="pandas only supports SQLAlchemy")
TABLES = {  # table -> csv file(s)
    "aisles": ["aisles.csv"],
    "departments": ["departments.csv"],
    "products": ["products.csv"],
    "orders": ["orders.csv"],
    "order_products": ["order_products__prior.csv", "order_products__train.csv"],
}


def connect():
    return psycopg2.connect(PG_DSN)


def load() -> None:
    with connect() as conn, conn.cursor() as cur:
        cur.execute((SQL_DIR / "schema.sql").read_text())
        for table, files in TABLES.items():
            for name in files:
                with open(RAW_DIR / name) as f:
                    # COPY is the bulk path: ~1M rows/s. Empty CSV fields become NULL.
                    cur.copy_expert(f"COPY {table} FROM STDIN WITH (FORMAT csv, HEADER true)", f)
            cur.execute(f"SELECT count(*) FROM {table}")
            print(f"{table}: {cur.fetchone()[0]:,} rows")
        cur.execute((SQL_DIR / "post_load.sql").read_text())


def _blocks(file: str):
    """Yield (name, comment, sql) for each '-- name:' block of a .sql file."""
    for block in re.split(r"^-- name: ", (SQL_DIR / file).read_text(), flags=re.M)[1:]:
        name, body = block.split("\n", 1)
        comment = " ".join(l[3:] for l in body.splitlines() if l.startswith("-- "))
        yield name.strip(), comment, body


def explore(out=REPORTS / "sql_exploration.md") -> None:
    lines = ["# SQL exploration of the Instacart dataset\n"]
    with connect() as conn:
        for name, comment, sql in _blocks("explore.sql"):
            df = pd.read_sql(sql, conn)
            table = df.to_markdown(index=False, floatfmt=".3f")
            lines += [f"## {name}\n\n{comment}\n", table, ""]
            print(f"\n== {name}\n{table}")
    out.parent.mkdir(exist_ok=True)
    out.write_text("\n".join(lines))
    print(f"\nwrote {out}")


def features() -> None:
    """Heavy aggregation happens in SQL (34M rows); Python only sees per-user / per-aisle tables."""
    PROCESSED.mkdir(parents=True, exist_ok=True)
    with connect() as conn:
        for name, _, sql in _blocks("features.sql"):
            df = pd.read_sql(sql, conn)
            df.to_parquet(PROCESSED / f"{name}.parquet", index=False)
            print(f"{name}: {len(df):,} rows x {df.shape[1]} cols -> {PROCESSED / name}.parquet")


if __name__ == "__main__":
    commands: dict[str, Callable[[], None]] = {"load": load, "explore": explore, "features": features}
    commands[sys.argv[1]]()
