import numpy as np
import pandas as pd
import pytest

from affinity.metrics import candidate_metrics, evaluate, rank_within_user, topk_metrics


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


def test_candidate_metrics_include_positives_outside_candidates():
    m = candidate_metrics(_table().drop(columns="score"), pd.Series({1: 2, 2: 1}))
    assert m == {
        "candidate_recall": 0.5,
        "candidate_hit_rate": 0.5,
        "candidates_per_user": 2.5,
    }


def test_candidate_metrics_weight_users_equally_not_positives():
    m = candidate_metrics(_table(), pd.Series({1: 4, 2: 1}))
    assert m["candidate_recall"] == 0.25  # (2/4 + 0/1) / 2, not 2/5


def test_candidate_metrics_include_users_without_candidates():
    m = candidate_metrics(_table(), pd.Series({3: 2, 1: 2, 2: 1}))
    assert m["candidate_recall"] == pytest.approx(1 / 3)
    assert m["candidate_hit_rate"] == pytest.approx(1 / 3)
    assert m["candidates_per_user"] == pytest.approx(5 / 3)


def test_candidate_metrics_ignore_users_outside_evaluation():
    m = candidate_metrics(_table(), pd.Series({2: 1}))
    assert m == {
        "candidate_recall": 0.0,
        "candidate_hit_rate": 0.0,
        "candidates_per_user": 2.0,
    }


def test_candidate_metrics_handle_empty_candidates():
    m = candidate_metrics(_table().iloc[:0], pd.Series({1: 2, 2: 1}))
    assert all(v == 0 for v in m.values())


@pytest.mark.parametrize("ks", [(1,), (2,), (5, 10)])
def test_candidate_recall_bounds_ranking_recall(ks):
    n_pos = pd.Series({1: 2, 2: 1, 3: 1})
    ceiling = candidate_metrics(_table(), n_pos)["candidate_recall"]
    ranking = evaluate(_table(), n_pos, ks=ks)
    assert all(ranking[f"recall@{k}"] <= ceiling for k in ks)


@pytest.mark.parametrize("n_pos", [pd.Series({1: 2, 2: 0}), pd.Series({2: 0})])
def test_zero_positive_users_have_zero_recall_in_both_stages(n_pos):
    ceiling = candidate_metrics(_table(), n_pos)["candidate_recall"]
    ranking = topk_metrics(_table(), n_pos, k=10)
    assert np.isfinite(ceiling)
    assert ranking["recall@10"] == ceiling


def test_candidate_metrics_reject_empty_evaluation_population():
    with pytest.raises(ValueError, match="At least one evaluation user"):
        candidate_metrics(_table(), pd.Series(dtype=int))
