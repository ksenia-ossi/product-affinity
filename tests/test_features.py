import numpy as np
import pandas as pd

from affinity.features import FEATURES, build_candidates, positives_per_user, split_users
from affinity.metrics import candidate_metrics


def _data():
    users = pd.DataFrame({"user_id": [1, 2], "u_n_orders": [4, 2], "u_avg_items": [5.0, 3.0],
                          "u_avg_aisles": [3.0, 2.0], "u_avg_days_between": [7.0, 10.0],
                          "t_days_since_prior": [7.0, 30.0], "t_dow": [1, 2], "t_hour": [10, 18]})
    ua = pd.DataFrame({"user_id": [1, 1, 2], "aisle_id": [10, 20, 10], "ua_n_orders": [3, 1, 2],
                       "ua_n_items": [6, 1, 2], "ua_n_reordered": [3, 0, 1], "ua_avg_cart_pos": [1.0, 4.0, 2.0],
                       "ua_first_order": [1, 4, 1], "ua_last_order": [4, 4, 2], "ua_orders_since_last": [0, 0, 0],
                       "ua_order_share": [0.75, 0.25, 1.0], "ua_share_since_first": [0.75, 1.0, 1.0]})
    aisles = pd.DataFrame({"aisle_id": [10, 20, 30], "aisle": list("abc"), "department_id": [1, 1, 2],
                           "department": list("xxy"), "a_order_share": [0.5, 0.2, 0.3],
                           "a_n_users": [100, 50, 70], "a_reorder_rate": [0.6, 0.4, 0.5]})
    labels = pd.DataFrame({"user_id": [1, 1, 2], "aisle_id": [10, 30, 99]})
    return {"users": users, "user_aisle": ua, "aisles": aisles, "labels": labels}


def test_candidates_union_of_history_and_popular():
    cand = build_candidates(_data(), n_popular=1)   # most popular aisle = 10
    pairs = set(map(tuple, cand[["user_id", "aisle_id"]].to_numpy()))
    assert pairs == {(1, 10), (1, 20), (2, 10)}
    assert cand.label.sum() == 1                      # (1,10); (1,30) and (2,99) are not candidates
    assert set(FEATURES) <= set(cand.columns)
    assert cand["ua_recency_ratio"].notna().all()


def test_unseen_popular_candidate_has_zero_history():
    cand = build_candidates(_data(), n_popular=3)
    row = cand[(cand.user_id == 2) & (cand.aisle_id == 30)].iloc[0]
    assert row.ua_seen == 0 and row.ua_n_orders == 0 and row.label == 0


def test_positives_per_user_counts_all_labels():
    n = positives_per_user(_data()["labels"], [1, 2, 3])
    assert n.tolist() == [2, 1, 0]


def test_split_users_is_disjoint_and_deterministic():
    ids = pd.Series(np.arange(1000))
    a, b = split_users(ids, 0.2, seed=1)
    a2, _ = split_users(ids, 0.2, seed=1)
    assert len(set(a) & set(b)) == 0 and len(a) + len(b) == 1000 and (a == a2).all()
    assert 150 < len(b) < 250


def test_candidate_recall_improves_when_popular_pool_recovers_a_positive():
    data = _data()
    n_pos = positives_per_user(data["labels"], data["users"]["user_id"])
    narrow = candidate_metrics(build_candidates(data, n_popular=1), n_pos)
    wider = candidate_metrics(build_candidates(data, n_popular=2), n_pos)
    assert narrow["candidate_recall"] == 0.25
    assert wider["candidate_recall"] == 0.5
    assert wider["candidates_per_user"] > narrow["candidates_per_user"]
    assert wider["candidate_hit_rate"] == narrow["candidate_hit_rate"] == 0.5
