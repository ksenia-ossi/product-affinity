import pytest

from affinity import train


@pytest.fixture(autouse=True)
def mlflow_in_tmp(monkeypatch, tmp_path):
    """Keep MLflow runs from tests out of the repository's mlflow.db."""
    monkeypatch.setattr(train, "MLFLOW_TRACKING_URI", f"sqlite:///{tmp_path / 'mlflow.db'}")
    monkeypatch.chdir(tmp_path)
