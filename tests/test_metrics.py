import numpy as np
import pandas as pd

from affinity.metrics import evaluate, rank_within_user, topk_metrics


def _table():
    # user 1: positives are aisles 10 and 30; user 2: positive is aisle 50 (not in candidates)
    return pd.DataFrame({
        "user_id": [1, 1, 1, 2, 2],
        "aisle_id": [10, 20, 30, 10, 20],
        "score": [0.9, 0.8, 0.1, 0.5, 0.6],
        "label": [1, 0, 1, 0, 0],
    })


def test_rank_within_user_is_descending_by_score():
    r = rank_within_user(_table())
    assert r[r.user_id == 1].sort_values("rank").aisle_id.tolist() == [10, 20, 30]
    assert r[r.user_id == 2].sort_values("rank").aisle_id.tolist() == [20, 10]


def test_topk_metrics_values():
    n_pos = pd.Series({1: 2, 2: 1})
    m = topk_metrics(_table(), n_pos, k=2)
    assert np.isclose(m["precision@2"], (1 / 2 + 0) / 2)
    assert np.isclose(m["recall@2"], (1 / 2 + 0) / 2)   # user 2's positive is outside candidates
    assert np.isclose(m["hit_rate@2"], 0.5)
    assert np.isclose(m["ndcg@2"], (1 / (1 + 1 / np.log2(3))) / 2)


def test_perfect_ranking_gives_ndcg_one():
    df = pd.DataFrame({"user_id": [1] * 3, "aisle_id": [1, 2, 3], "score": [3, 2, 1], "label": [1, 1, 0]})
    m = evaluate(df, pd.Series({1: 2}), ks=(2,))
    assert m["ndcg@2"] == 1.0 and m["precision@2"] == 1.0 and m["recall@2"] == 1.0
