"""Top-k ranking metrics for per-user recommendation lists.

All functions take a scored candidate table with columns [user_id, aisle_id, score, label]
plus `n_pos`: a Series user_id -> number of TRUE positives (from the full label set, so that
positives missing from the candidate list still count against recall).
"""
import numpy as np
import pandas as pd


def rank_within_user(df: pd.DataFrame, score_col: str = "score") -> pd.DataFrame:
    out = df.sort_values(["user_id", score_col], ascending=[True, False], kind="mergesort")
    out = out.assign(rank=out.groupby("user_id").cumcount() + 1)
    return out


def topk_metrics(df: pd.DataFrame, n_pos: pd.Series, k: int, score_col: str = "score") -> dict:
    ranked = rank_within_user(df, score_col)
    top = ranked[ranked["rank"] <= k]
    hits = top.groupby("user_id")["label"].sum()
    users = n_pos.index
    hits = hits.reindex(users, fill_value=0)
    # DCG with binary gains; ideal DCG uses min(n_pos, k) positives at the top.
    gain = top["label"] / np.log2(top["rank"] + 1)
    dcg = gain.groupby(top["user_id"]).sum().reindex(users, fill_value=0)
    ideal = np.array([(1 / np.log2(np.arange(2, min(n, k) + 2))).sum() for n in n_pos.to_numpy()])
    return {
        f"precision@{k}": float((hits / k).mean()),
        f"recall@{k}": float((hits / n_pos).mean()),
        f"ndcg@{k}": float(np.mean(np.where(ideal > 0, dcg.to_numpy() / np.where(ideal > 0, ideal, 1), 0))),
        f"hit_rate@{k}": float((hits > 0).mean()),
    }


def evaluate(df: pd.DataFrame, n_pos: pd.Series, ks=(5, 10), score_col: str = "score") -> dict:
    out = {}
    for k in ks:
        out.update(topk_metrics(df, n_pos, k, score_col))
    return out
