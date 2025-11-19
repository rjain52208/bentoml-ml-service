"""
BentoML ML Service
------------------
Minimal example of a model-serving API that exposes a /predict endpoint.

This file defines:
- Pydantic request/response schemas
- A BentoML service with a /predict HTTP API
- A fallback DummyModel so the service is still understandable
"""

from __future__ import annotations

from typing import List

import numpy as np
from pydantic import BaseModel, Field

import bentoml
from bentoml.io import JSON


# ---------- Request & response schemas ----------


class PredictionRequest(BaseModel):
    """Incoming JSON payload schema."""

    features: List[List[float]] = Field(
        ...,
        description="2D list of numeric feature vectors for single or batch prediction.",
        example=[[0.3, 1.2, 5.1, 0.0], [0.9, 0.4, 3.3, 1.0]],
    )


class PredictionResponse(BaseModel):
    """Outgoing JSON response schema."""

    predictions: List[int]
    scores: List[float]
    model_version: str = "v1.0.0"


# ---------- Dummy model (so the file is self-contained) ----------


class DummyModel:
    """
    Very simple 'model' for demo purposes only.

    - Takes a 2D array of features.
    - Computes a sigmoid on the row sum.
    - Returns probabilities for class 1.
    """

    def predict_proba(self, X):
        X = np.asarray(X)
        row_sums = X.sum(axis=1)
        probs_pos = 1.0 / (1.0 + np.exp(-row_sums))  # sigmoid
        probs_neg = 1.0 - probs_pos
        return np.vstack([probs_neg, probs_pos]).T


# In a real project you might do something like:
#
#   bento_model = bentoml.models.get("fraud_model:latest")
#   model_runner = bento_model.to_runner()
#   svc = bentoml.Service("ml_scoring_service", runners=[model_runner])
#
# For this portfolio repo we keep it simple and just use DummyModel directly.

dummy_model = DummyModel()


# ---------- BentoML service definition ----------


svc = bentoml.Service("ml_scoring_service")


@svc.api(
    input=JSON(pydantic_model=PredictionRequest),
    output=JSON(pydantic_model=PredictionResponse),
)
def predict(request: PredictionRequest) -> PredictionResponse:
    """
    /predict endpoint.

    Accepts:
      {
        "features": [[...], [...], ...]
      }

    Returns:
      {
        "predictions": [0, 1, ...],
        "scores": [0.18, 0.87, ...],
        "model_version": "v1.0.0"
      }
    """
    X = request.features

    # Use dummy model probabilities
    proba = dummy_model.predict_proba(X)
    scores = proba[:, 1].tolist()

    # Threshold at 0.5 for binary class prediction
    predictions = [int(s >= 0.5) for s in scores]

    return PredictionResponse(
        predictions=predictions,
        scores=[round(float(s), 4) for s in scores],
        model_version="v1.0.0",
    )
