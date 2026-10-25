import lightgbm as lgb
import mlflow
import mlflow.lightgbm
import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from affinity import serve
from affinity.config import MLFLOW_EXPERIMENT
from affinity.features import CATEGORICAL, MODEL_COLUMNS


def candidates(n: int) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    X = pd.DataFrame(rng.random((n, len(MODEL_COLUMNS))), columns=MODEL_COLUMNS)
    X["aisle_id"] = np.arange(n) % 4 + 1
    X["department_id"] = X["aisle_id"] % 2
    return X


@pytest.fixture
def logged_model() -> str:
    """Log a small LightGBM model the way `make train` does and return its URI."""
    X = candidates(200)
    y = (X["ua_order_share"] > 0.5).astype(int)
    clf = lgb.LGBMClassifier(n_estimators=20, n_jobs=1, verbose=-1).fit(X, y, categorical_feature=CATEGORICAL)
    mlflow.set_tracking_uri(serve.MLFLOW_TRACKING_URI)
    mlflow.set_experiment(MLFLOW_EXPERIMENT)
    with mlflow.start_run():
        return mlflow.lightgbm.log_model(clf, name="model", input_example=X.head(2)).model_uri


def test_rank_returns_top_k_aisles_sorted_by_probability(logged_model):
    body = candidates(4)
    body["ua_order_share"] = [0.1, 0.9, 0.3, 0.7]
    with TestClient(serve.app) as client:
        assert client.get("/health").json() == {"status": "ok", "model_uri": logged_model}
        response = client.post("/rank", json={"candidates": body.to_dict("records"), "k": 3})
    assert response.status_code == 200
    aisles = response.json()["aisles"]
    assert len(aisles) == 3
    probabilities = [a["probability"] for a in aisles]
    assert probabilities == sorted(probabilities, reverse=True)
    assert aisles[0]["aisle_id"] in (2, 4)  # the two candidates with a high order share


def test_rank_rejects_candidates_with_missing_features(logged_model):
    body = candidates(2).drop(columns=["ua_order_share"])
    with TestClient(serve.app) as client:
        response = client.post("/rank", json={"candidates": body.to_dict("records")})
    assert response.status_code == 422
    assert "ua_order_share" in response.json()["detail"]


def test_startup_fails_clearly_without_a_logged_model():
    with pytest.raises(RuntimeError, match="make train"):
        with TestClient(serve.app):
            pass
