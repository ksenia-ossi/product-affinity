import pytest

from affinity import serve, train


@pytest.fixture(autouse=True)
def mlflow_in_tmp(monkeypatch, tmp_path):
    """Keep MLflow runs from tests out of the repository's mlflow.db."""
    uri = f"sqlite:///{tmp_path / 'mlflow.db'}"
    monkeypatch.setattr(train, "MLFLOW_TRACKING_URI", uri)
    monkeypatch.setattr(serve, "MLFLOW_TRACKING_URI", uri)
    monkeypatch.chdir(tmp_path)
