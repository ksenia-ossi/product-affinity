import numpy as np
import pandas as pd

from affinity import train
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
    monkeypatch.setattr(train, "REPORTS", tmp_path)
    monkeypatch.setattr(train, "FIGURES", tmp_path)

    train.main()

    report = (tmp_path / "metrics.md").read_text()
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
    assert (tmp_path / "precision_at_k.png").is_file()
    assert (tmp_path / "feature_importance.png").is_file()
