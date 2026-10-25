"""Rank candidate aisles for each held-out user; report retrieval and top-k metrics.

Models, in order of complexity:
1. Popularity          score = global share of orders containing the aisle  (same list for everyone)
2. User history        score = share of the user's past orders containing the aisle (per-user, no learning)
3. ALS                 implicit-feedback matrix factorisation on the user x aisle count matrix
4. LightGBM            binary classifier on history + recency + popularity + context features
"""
import time
from typing import Any

import matplotlib

matplotlib.use("Agg")
import implicit
import lightgbm as lgb
import matplotlib.pyplot as plt
import mlflow
import mlflow.lightgbm
import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.metrics import log_loss, roc_auc_score

from .config import (FIGURES, K_EVAL, MLFLOW_EXPERIMENT, MLFLOW_TRACKING_URI, N_POPULAR_CANDIDATES, REPORTS, SEED,
                     USER_TEST_FRACTION)
from .features import CATEGORICAL, MODEL_COLUMNS, build_candidates, load_processed, positives_per_user, split_users
from .metrics import candidate_metrics, evaluate


def als_scores(ua: pd.DataFrame, cand: pd.DataFrame, factors=32, reg=0.05, iters=15) -> np.ndarray:
    """Fit ALS on everyone's history (unsupervised: no labels involved), score candidate pairs."""
    uid = {u: i for i, u in enumerate(np.sort(ua.user_id.unique()))}
    aid = {a: i for i, a in enumerate(np.sort(pd.concat([ua.aisle_id, cand.aisle_id]).unique()))}
    m = sp.csr_matrix(
        (np.log1p(ua.ua_n_orders.to_numpy(dtype="float32")),
         (ua.user_id.map(uid).to_numpy(), ua.aisle_id.map(aid).to_numpy())),
        shape=(len(uid), len(aid)),
    )
    model = implicit.als.AlternatingLeastSquares(factors=factors, regularization=reg, iterations=iters,
                                                 random_state=SEED, use_gpu=False)
    model.fit(m, show_progress=False)
    U, V = np.asarray(model.user_factors), np.asarray(model.item_factors)
    return np.einsum("ij,ij->i", U[cand.user_id.map(uid).to_numpy()], V[cand.aisle_id.map(aid).to_numpy()])


def mlflow_metrics(m: dict) -> dict[str, float]:
    """MLflow metric names cannot contain '@': precision@5 -> precision_at_5."""
    return {k.replace("@", "_at_"): float(v) for k, v in m.items() if isinstance(v, (int, float))}


def main() -> None:
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(MLFLOW_EXPERIMENT)
    with mlflow.start_run(run_name="rankers"):
        run()


def run() -> None:
    d = load_processed()
    cand = build_candidates(d)
    train_users, test_users = split_users(d["users"]["user_id"])
    te = cand[cand.user_id.isin(test_users)].copy()
    tr = cand[cand.user_id.isin(train_users)].copy()
    n_pos = positives_per_user(d["labels"], test_users)
    retrieval = candidate_metrics(te, n_pos)
    print(f"candidates: {len(cand):,} pairs, {cand.label.mean():.3%} positive; "
          f"train users {len(train_users):,}, test users {len(test_users):,}")
    print("candidate retrieval:", {k: round(v, 4) for k, v in retrieval.items()})
    mlflow.log_params({"n_popular_candidates": N_POPULAR_CANDIDATES, "user_test_fraction": USER_TEST_FRACTION,
                       "seed": SEED, "n_train_users": len(train_users), "n_test_users": len(test_users)})
    mlflow.log_metrics(mlflow_metrics(retrieval))
    rows, importances = [], None

    def report(name, scores, seconds, probs=None, params=None, model=None, X_example=None):
        te["score"] = scores
        m = {"model": name, **evaluate(te, n_pos, K_EVAL)}
        if probs is not None:
            m["auc"] = roc_auc_score(te.label, probs)
            m["log_loss"] = log_loss(te.label, probs)
        m["train_time_s"] = round(seconds, 1)
        rows.append(m)
        with mlflow.start_run(run_name=name, nested=True):
            mlflow.log_params(params or {})
            mlflow.log_metrics(mlflow_metrics(m))
            if model is not None:
                mlflow.lightgbm.log_model(model, name="model", input_example=X_example)
        print({k: (round(v, 4) if isinstance(v, float) else v) for k, v in m.items()})

    report("popularity", te["a_order_share"].to_numpy(), 0)
    # ties among never-bought aisles are broken by popularity
    report("user history (order share)", te["ua_order_share"].to_numpy() + 1e-3 * te["a_order_share"].to_numpy(), 0)

    t = time.time()
    report("ALS matrix factorisation", als_scores(d["user_aisle"], te), time.time() - t,
           params={"factors": 32, "regularization": 0.05, "iterations": 15})

    t = time.time()
    lgb_params: dict[str, Any] = dict(n_estimators=400, learning_rate=0.05, num_leaves=63, min_child_samples=100,
                                      subsample=0.8, subsample_freq=1, colsample_bytree=0.8)
    clf = lgb.LGBMClassifier(**lgb_params, random_state=SEED, verbose=-1)
    X_cols = MODEL_COLUMNS
    clf.fit(tr[X_cols], tr.label, categorical_feature=CATEGORICAL)
    p = np.asarray(clf.predict_proba(te[X_cols]))[:, 1]
    report("LightGBM (history + popularity + context)", p, time.time() - t, probs=p,
           params=lgb_params, model=clf, X_example=tr[X_cols].head(5))
    importances = pd.Series(clf.booster_.feature_importance("gain"), index=X_cols).sort_values(ascending=False)

    res = pd.DataFrame(rows)
    REPORTS.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)
    md = ["# Results: held-out users, aisles ranked for each user's next order\n",
          "## Candidate retrieval\n",
          pd.DataFrame([retrieval]).to_markdown(index=False, floatfmt=".4f"),
          "\nCandidate recall is the mean fraction of each user's true next-order aisles "
          "present in the candidate set, before ranking. It bounds recall@k from above. "
          "Candidate hit rate is the fraction of users with at least one positive candidate; "
          "candidates per user is the mean list size. All evaluation users are included, "
          "even those with no candidates; users without positives contribute zero recall.\n",
          "## Ranking quality\n",
          res.to_markdown(index=False, floatfmt=".4f"),
          "\n## LightGBM feature importance (gain)\n",
          importances.head(15).to_frame("gain").to_markdown(floatfmt=".0f")]
    (REPORTS / "metrics.md").write_text("\n".join(md))

    fig, ax = plt.subplots(figsize=(6, 3.4))
    for _, r in res.iterrows():
        ax.plot(K_EVAL, [r[f"precision@{k}"] for k in K_EVAL], marker="o", label=r.model)
    ax.set(xlabel="k", ylabel="precision@k", xticks=K_EVAL, title="Precision@k on held-out users")
    ax.legend(fontsize=7)
    fig.tight_layout(); fig.savefig(FIGURES / "precision_at_k.png", dpi=120); plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 3.8))
    importances.head(12)[::-1].plot.barh(ax=ax, color="#4c72b0")
    ax.set(title="LightGBM feature importance (gain)")
    fig.tight_layout(); fig.savefig(FIGURES / "feature_importance.png", dpi=120); plt.close(fig)
    mlflow.log_artifact(str(REPORTS / "metrics.md"))
    mlflow.log_artifacts(str(FIGURES), artifact_path="figures")
    print(f"wrote {REPORTS / 'metrics.md'}")


if __name__ == "__main__":
    main()
