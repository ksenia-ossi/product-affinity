"""REST API that ranks a user's candidate aisles with a LightGBM model logged by `make train`.

The model is loaded once at startup from MLflow: MODEL_URI if set (e.g. models:/m-abc123),
otherwise the most recent LightGBM model logged in the product-affinity experiment.

    make serve   # then open http://127.0.0.1:8000/docs
"""
import os
from contextlib import asynccontextmanager

import mlflow
import mlflow.lightgbm
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from .config import MLFLOW_EXPERIMENT, MLFLOW_TRACKING_URI
from .features import MODEL_COLUMNS


class RankRequest(BaseModel):
    candidates: list[dict[str, float]] = Field(
        min_length=1, description="One object per candidate aisle with every model feature, including aisle_id.")
    k: int = Field(default=10, ge=1, description="Number of top aisles to return.")


class ScoredAisle(BaseModel):
    aisle_id: int
    probability: float


class RankResponse(BaseModel):
    model_uri: str
    aisles: list[ScoredAisle]


def latest_model_uri() -> str:
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    experiment = mlflow.get_experiment_by_name(MLFLOW_EXPERIMENT)
    models = [] if experiment is None else mlflow.search_logged_models(
        experiment_ids=[experiment.experiment_id],
        order_by=[{"field_name": "creation_time", "ascending": False}],
        max_results=1, output_format="list")
    if not models:
        raise RuntimeError(f"No model logged in MLflow experiment '{MLFLOW_EXPERIMENT}': run `make train` first.")
    return models[0].model_uri


@asynccontextmanager
async def lifespan(app: FastAPI):
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    app.state.model_uri = os.environ.get("MODEL_URI") or latest_model_uri()
    app.state.model = mlflow.lightgbm.load_model(app.state.model_uri)
    yield


app = FastAPI(title="product-affinity", lifespan=lifespan)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "model_uri": app.state.model_uri}


@app.post("/rank")
def rank(request: RankRequest) -> RankResponse:
    X = pd.DataFrame(request.candidates)
    missing = [c for c in MODEL_COLUMNS if c not in X.columns]
    if missing:
        raise HTTPException(status_code=422, detail=f"Missing features: {missing}")
    X = X[MODEL_COLUMNS]
    X[["aisle_id", "department_id"]] = X[["aisle_id", "department_id"]].astype(int)
    probabilities = np.asarray(app.state.model.predict_proba(X))[:, 1]
    top = np.argsort(-probabilities, kind="stable")[:request.k]
    return RankResponse(model_uri=app.state.model_uri, aisles=[
        ScoredAisle(aisle_id=int(X.aisle_id.iloc[i]), probability=float(probabilities[i])) for i in top])
