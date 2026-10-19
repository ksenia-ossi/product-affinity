import numpy as np
import pandas as pd

from affinity import features, train
from affinity.features import FEATURES


def test_training_report_includes_candidate_retrieval(monkeypatch, tmp_path, capsys):
    candidates = pd.DataFrame({col: np.zeros(8) for col in FEATURES})
    candidates["user_id"] = [1, 1, 2, 2, 3, 3, 4, 4]
    candidates["aisle_id"] = [10, 20] * 4
    candidates["department_id"] = 1
    candidates["label"] = [1, 0, 0, 1, 1, 0, 0, 0]
    data = {
        "users": pd.DataFrame({"user_id": [1, 2, 3, 4]}),
        "labels": pd.DataFrame({"user_id": [1, 2, 3, 3, 4], "aisle_id": [10, 20, 10, 30, 99]}),
        "user_aisle": pd.DataFrame(),
    }
    monkeypatch.setattr(train, "load_processed", lambda: data)
    monkeypatch.setattr(train, "build_candidates", lambda d: candidates)
    monkeypatch.setattr(train, "split_users", lambda ids: (np.array([1, 2]), np.array([3, 4])))
    monkeypatch.setattr(train, "als_scores", lambda ua, cand: np.zeros(len(cand)))
    classifier = train.lgb.LGBMClassifier
    monkeypatch.setattr(train.lgb, "LGBMClassifier", lambda **kw: classifier(n_jobs=1, **kw))
    reports_dir = tmp_path / "new" / "reports"
    figures_dir = reports_dir / "figures"
    monkeypatch.setattr(train, "REPORTS", reports_dir)
    monkeypatch.setattr(train, "FIGURES", figures_dir)

    train.main()

    report = (reports_dir / "metrics.md").read_text()
    retrieval = report.split("## Candidate retrieval\n", 1)[1].split("## Ranking quality", 1)[0]
    table_lines = [line for line in retrieval.splitlines() if line.startswith("|")]
    header = [cell.strip() for cell in table_lines[0].strip("|").split("|")]
    values = [float(cell) for cell in table_lines[2].strip("|").split("|")]
    assert dict(zip(header, values)) == {
        "candidate_recall": 0.25,
        "candidate_hit_rate": 0.5,
        "candidates_per_user": 2.0,
    }
    assert "## LightGBM feature importance" in report
    assert "LightGBM (history + popularity + context)" in report
    assert "candidate retrieval:" in capsys.readouterr().out
    assert (figures_dir / "precision_at_k.png").is_file()
    assert (figures_dir / "feature_importance.png").is_file()


def test_training_pipeline_from_synthetic_parquet(monkeypatch, tmp_path):
    """Exercise real feature building and all rankers without Instacart or PostgreSQL."""
    user_ids = np.arange(1, 33)
    users = pd.DataFrame({
        "user_id": user_ids, "u_n_orders": 4, "u_avg_items": 3.0,
        "u_avg_aisles": 2.0, "u_avg_days_between": 7.0,
        "t_days_since_prior": 7.0, "t_dow": 1, "t_hour": 10,
    })
    user_aisle = pd.DataFrame({
        "user_id": np.repeat(user_ids, 2),
        "aisle_id": [a for u in user_ids for a in (10, 20 + 10 * (u % 2))],
        "ua_n_orders": 2, "ua_n_items": 4, "ua_n_reordered": 2,
        "ua_avg_cart_pos": 1.5, "ua_first_order": 1, "ua_last_order": 3,
        "ua_orders_since_last": 1, "ua_order_share": 0.5, "ua_share_since_first": 0.5,
    })
    aisles = pd.DataFrame({
        "aisle_id": [10, 20, 30, 40], "department_id": [1, 1, 2, 2],
        "a_order_share": [0.5, 0.25, 0.25, 0.1], "a_n_users": [32, 16, 16, 8],
        "a_reorder_rate": [0.5, 0.5, 0.5, 0.2],
    })
    labels = pd.DataFrame({"user_id": user_ids, "aisle_id": 30 + 10 * (user_ids % 2)})
    processed_dir = tmp_path / "processed"
    processed_dir.mkdir()
    for name, table in {"users": users, "user_aisle": user_aisle,
                        "aisles": aisles, "labels": labels}.items():
        table.to_parquet(processed_dir / f"{name}.parquet", index=False)
    reports_dir = tmp_path / "reports"
    figures_dir = reports_dir / "figures"
    monkeypatch.setattr(features, "PROCESSED", processed_dir)
    monkeypatch.setattr(train, "REPORTS", reports_dir)
    monkeypatch.setattr(train, "FIGURES", figures_dir)

    train.main()

    report = (reports_dir / "metrics.md").read_text()
    ranking = report.split("## Ranking quality\n", 1)[1].split("## LightGBM", 1)[0]
    lines = [line for line in ranking.splitlines() if line.startswith("|")]
    assert len(lines) == 6  # Header, separator and four real rankers.
    assert [line.split("|")[1].strip() for line in lines[2:]] == [
        "popularity", "user history (order share)", "ALS matrix factorisation",
        "LightGBM (history + popularity + context)",
    ]
    for line in lines[2:]:
        cells = line.strip("|").split("|")[1:9]
        assert len(cells) == 8
        assert all(0 <= float(value) <= 1 for value in cells)
    for name in ("precision_at_k.png", "feature_importance.png"):
        assert (figures_dir / name).read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
