"""Rank the 134 aisles for each held-out user; report precision/recall/NDCG/hit-rate @5 and @10.

Models, in order of complexity:
1. Popularity          score = global share of orders containing the aisle  (same list for everyone)
2. User history        score = share of the user's past orders containing the aisle (per-user, no learning)
3. ALS                 implicit-feedback matrix factorisation on the user x aisle count matrix
4. LightGBM            binary classifier on history + recency + popularity + context features
"""
import time

import matplotlib

matplotlib.use("Agg")
import implicit
import lightgbm as lgb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.metrics import log_loss, roc_auc_score

from .config import FIGURES, K_EVAL, REPORTS, SEED
from .features import CATEGORICAL, FEATURES, build_candidates, load_processed, positives_per_user, split_users
from .metrics import evaluate


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


def main() -> None:
    d = load_processed()
    cand = build_candidates(d)
    train_users, test_users = split_users(d["users"]["user_id"])
    te = cand[cand.user_id.isin(test_users)].copy()
    tr = cand[cand.user_id.isin(train_users)].copy()
    n_pos = positives_per_user(d["labels"], test_users)
    print(f"candidates: {len(cand):,} pairs, {cand.label.mean():.3%} positive; "
          f"train users {len(train_users):,}, test users {len(test_users):,}")
    rows, importances = [], None

    def report(name, scores, seconds, probs=None):
        te["score"] = scores
        m = {"model": name, **evaluate(te, n_pos, K_EVAL)}
        if probs is not None:
            m["auc"] = roc_auc_score(te.label, probs)
            m["log_loss"] = log_loss(te.label, probs)
        m["train_time_s"] = round(seconds, 1)
        rows.append(m)
        print({k: (round(v, 4) if isinstance(v, float) else v) for k, v in m.items()})

    report("popularity", te["a_order_share"].to_numpy(), 0)
    # ties among never-bought aisles are broken by popularity
    report("user history (order share)", te["ua_order_share"].to_numpy() + 1e-3 * te["a_order_share"].to_numpy(), 0)

    t = time.time()
    report("ALS matrix factorisation", als_scores(d["user_aisle"], te), time.time() - t)

    t = time.time()
    # TODO(ksenia): try lgb.LGBMRanker(objective="lambdarank") with group = candidates per user and
    # compare NDCG@10 with this pointwise classifier. Then tune num_leaves / learning_rate on a
    # validation split of TRAIN USERS (never on the test users).
    clf = lgb.LGBMClassifier(n_estimators=400, learning_rate=0.05, num_leaves=63, min_child_samples=100,
                             subsample=0.8, subsample_freq=1, colsample_bytree=0.8, random_state=SEED, verbose=-1)
    X_cols = FEATURES + ["aisle_id"]
    clf.fit(tr[X_cols], tr.label, categorical_feature=CATEGORICAL)
    p = clf.predict_proba(te[X_cols])[:, 1]
    report("LightGBM (history + popularity + context)", p, time.time() - t, probs=p)
    importances = pd.Series(clf.booster_.feature_importance("gain"), index=X_cols).sort_values(ascending=False)

    res = pd.DataFrame(rows)
    REPORTS.mkdir(exist_ok=True)
    md = ["# Results — held-out users, aisles ranked for each user's next order\n",
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
    print(f"wrote {REPORTS / 'metrics.md'}")


if __name__ == "__main__":
    main()
