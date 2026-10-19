"""pandas/Matplotlib EDA on the SQL aggregates -> reports/eda.md + reports/figures/*.png."""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from .config import FIGURES, REPORTS
from .features import load_processed


def main() -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    d = load_processed()
    users, ua, aisles, labels = d["users"], d["user_aisle"], d["aisles"], d["labels"]
    lines = ["# EDA — Instacart purchase histories (131k labelled users)\n"]

    # 1. History length
    fig, ax = plt.subplots(figsize=(6, 3.2))
    ax.hist(users["u_n_orders"], bins=range(3, 101), color="#4c72b0")
    ax.set(xlabel="orders in history", ylabel="users", title="History length per user (median %.0f)" % users.u_n_orders.median())
    fig.tight_layout(); fig.savefig(FIGURES / "history_length.png", dpi=120); plt.close(fig)

    # 2. Aisle popularity vs. reorder rate
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.scatter(aisles["a_order_share"], aisles["a_reorder_rate"], s=12, alpha=.7)
    for _, r in aisles.nlargest(6, "a_order_share").iterrows():
        ax.annotate(r.aisle, (r.a_order_share, r.a_reorder_rate), fontsize=7)
    ax.set(xlabel="share of orders containing the aisle", ylabel="reorder rate", title="134 aisles: popularity vs. habit")
    fig.tight_layout(); fig.savefig(FIGURES / "aisle_popularity_vs_reorder.png", dpi=120); plt.close(fig)

    # 3. How predictive is the user's own history? P(aisle in next order | ua_order_share bucket)
    cand = ua.merge(labels.assign(label=1), on=["user_id", "aisle_id"], how="left").fillna({"label": 0})
    cand["share_bin"] = pd.cut(cand["ua_order_share"], [0, .1, .2, .35, .5, .75, 1.0], include_lowest=True)
    by_share = cand.groupby("share_bin", observed=True)["label"].agg(["mean", "size"])
    cand["recency_bin"] = pd.cut(cand["ua_orders_since_last"], [-1, 0, 1, 2, 4, 9, 100])
    by_rec = cand.groupby("recency_bin", observed=True)["label"].agg(["mean", "size"])
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.4))
    axes[0].bar(by_share.index.astype(str), by_share["mean"], color="#55a868")
    axes[0].set(title="P(buy aisle next) vs. share of past orders with it", ylabel="probability")
    axes[0].tick_params(axis="x", rotation=30, labelsize=7)
    axes[1].bar(by_rec.index.astype(str), by_rec["mean"], color="#c44e52")
    axes[1].set(title="P(buy aisle next) vs. orders since last purchase")
    axes[1].tick_params(axis="x", rotation=30, labelsize=7)
    fig.tight_layout(); fig.savefig(FIGURES / "history_vs_label.png", dpi=120); plt.close(fig)

    # 4. Positive rate and repeat share
    n_pos = labels.groupby("user_id").size()
    seen = labels.merge(ua[["user_id", "aisle_id"]], how="left", indicator=True)["_merge"].eq("both").mean()
    lines += [
        f"- Labelled users: **{len(users):,}**; (user, aisle) pairs in history: **{len(ua):,}** "
        f"(median {ua.groupby('user_id').size().median():.0f} distinct aisles per user).",
        f"- Aisles in the next order: median **{n_pos.median():.0f}** per user "
        f"(so the positive rate among history pairs is {cand.label.mean():.2%}).",
        f"- **{seen:.1%}** of the aisles bought in the next order were already in the user's history "
        f"— repeat behaviour dominates; the remaining {1 - seen:.1%} need popularity / collaborative signals.",
        "",
        "## P(aisle in next order) by share of past orders containing it\n",
        by_share.rename(columns={"mean": "p_label", "size": "pairs"}).to_markdown(floatfmt=".3f"),
        "\n## P(aisle in next order) by orders since the aisle was last bought\n",
        by_rec.rename(columns={"mean": "p_label", "size": "pairs"}).to_markdown(floatfmt=".3f"),
        "\n## Figures\n",
        "![](figures/history_length.png)\n![](figures/aisle_popularity_vs_reorder.png)\n![](figures/history_vs_label.png)",
    ]
    (REPORTS / "eda.md").write_text("\n".join(lines))
    print("\n".join(lines[:5]))
    print(f"wrote {REPORTS / 'eda.md'}")


if __name__ == "__main__":
    main()
