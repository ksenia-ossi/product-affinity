from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = ROOT / "data" / "raw"
PROCESSED = ROOT / "data" / "processed"
REPORTS = ROOT / "reports"
FIGURES = REPORTS / "figures"

# Mirror of "The Instacart Online Grocery Shopping Dataset 2017" (3.4M orders, 206k users).
MIRROR = "https://huggingface.co/datasets/attik/Instacart-Market-Basket-Analysis/resolve/main/"
RAW_FILES = [
    "aisles.csv", "departments.csv", "products.csv", "orders.csv",
    "order_products__prior.csv", "order_products__train.csv",
]

PG_DSN = "postgresql://instacart:instacart@localhost:5433/instacart"

# Affinity is predicted at the AISLE level (134 product categories such as
# "fresh fruits" or "yogurt"): it is the granularity at which an advertiser
# or a recommender asks "is this user interested in this category?".
K_EVAL = (5, 10)          # cut-offs for precision@k / recall@k / NDCG@k
N_POPULAR_CANDIDATES = 20  # popular aisles added to every user's candidate list
USER_TEST_FRACTION = 0.2   # users held out for evaluation (split by user, not by row)
SEED = 42
