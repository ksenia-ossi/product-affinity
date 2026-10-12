"""Build the (user, aisle) candidate table from the SQL aggregates in data/processed.

Candidates for a user = every aisle in the user's history  ∪  the N most popular aisles.
Label = 1 if the aisle appears in the user's next (held-out) order.
The train/test split is BY USER: the model is evaluated on users it has never seen.
"""
import numpy as np
import pandas as pd

from .config import N_POPULAR_CANDIDATES, PROCESSED, SEED, USER_TEST_FRACTION

UA_COLS = [
    "ua_n_orders", "ua_n_items", "ua_n_reordered", "ua_avg_cart_pos", "ua_first_order",
    "ua_last_order", "ua_orders_since_last", "ua_order_share", "ua_share_since_first",
]
USER_COLS = ["u_n_orders", "u_avg_items", "u_avg_aisles", "u_avg_days_between",
             "t_days_since_prior", "t_dow", "t_hour"]
AISLE_COLS = ["a_order_share", "a_n_users", "a_reorder_rate", "department_id"]
FEATURES = UA_COLS + ["ua_seen", "ua_recency_ratio"] + USER_COLS + AISLE_COLS
CATEGORICAL = ["department_id", "aisle_id"]


def load_processed() -> dict[str, pd.DataFrame]:
    return {n: pd.read_parquet(PROCESSED / f"{n}.parquet") for n in ("users", "user_aisle", "aisles", "labels")}


def split_users(user_ids: pd.Series, test_fraction=USER_TEST_FRACTION, seed=SEED):
    rng = np.random.default_rng(seed)
    users = np.sort(user_ids.unique())
    test = rng.random(len(users)) < test_fraction
    return users[~test], users[test]


def popular_aisles(aisles: pd.DataFrame, n=N_POPULAR_CANDIDATES) -> np.ndarray:
    return aisles.nlargest(n, "a_order_share")["aisle_id"].to_numpy()


def build_candidates(d: dict[str, pd.DataFrame], n_popular=N_POPULAR_CANDIDATES) -> pd.DataFrame:
    users, ua, aisles, labels = d["users"], d["user_aisle"], d["aisles"], d["labels"]
    seen = ua[["user_id", "aisle_id"]].assign(ua_seen=1)
    pop = pd.MultiIndex.from_product(
        [users["user_id"].to_numpy(), popular_aisles(aisles, n_popular)], names=["user_id", "aisle_id"]
    ).to_frame(index=False)
    cand = pd.concat([seen, pop]).drop_duplicates(["user_id", "aisle_id"])
    cand = cand.merge(ua, on=["user_id", "aisle_id"], how="left")
    cand["ua_seen"] = cand["ua_seen"].fillna(0).astype("int8")
    cand[UA_COLS] = cand[UA_COLS].fillna(0)
    cand = cand.merge(users, on="user_id").merge(aisles[["aisle_id"] + AISLE_COLS], on="aisle_id")
    cand["ua_recency_ratio"] = cand["ua_orders_since_last"] / cand["u_n_orders"]
    cand = cand.merge(labels.assign(label=1), on=["user_id", "aisle_id"], how="left")
    cand["label"] = cand["label"].fillna(0).astype("int8")
    return cand.reset_index(drop=True)


def positives_per_user(labels: pd.DataFrame, user_ids) -> pd.Series:
    n = labels.groupby("user_id").size()
    return n.reindex(user_ids).fillna(0).astype(int)
