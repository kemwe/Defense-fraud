# =============================================================================
# Fraud Detection API
# Accepts raw transaction fields — all feature engineering


import os
import time
import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Literal
import uvicorn


# Load Model

MODEL_PATH = os.path.join("models", "model_pipeline.pkl")

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(
        f"Model not found at '{MODEL_PATH}'. "
        "Ensure model_pipeline.pkl is saved in the 'models/' folder."
    )

model = joblib.load(MODEL_PATH)
print(f"✓ Model loaded successfully from: {MODEL_PATH}")


# FastAPI App
app = FastAPI(
    title="Fraud Detection API",
    description=(
        "Real-time mobile money fraud detection API. "
        "Accepts raw transaction fields only — all feature engineering "
        "is handled internally to match the training pipeline. "
        "Kennedy Mwenda — Strathmore University, 2026."
    ),
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# Request Schema
# Core raw fields + user-level context fields
# all feature engineering is done internally to replicate training transformations


class TransactionRequest(BaseModel):
    # -----------------------------------------------------------
    # Core transaction fields (raw — as they appear in the dataset)
    # -----------------------------------------------------------
    step: int = Field(..., description="Simulation time step (0–192)")
    type: Literal["TRANSFER", "PAYMENT", "DEBIT", "DEPOSIT", "WITHDRAWAL"] = Field(
        ..., description="Transaction type"
    )
    amount: float = Field(..., gt=0, description="Transaction amount")
    oldbalanceOrg: float = Field(
        ..., ge=0, description="Sender balance before transaction"
    )
    newbalanceOrig: float = Field(
        ..., ge=0, description="Sender balance after transaction"
    )
    oldbalanceDest: float = Field(
        ..., ge=0, description="Recipient balance before transaction"
    )
    newbalanceDest: float = Field(
        ..., ge=0, description="Recipient balance after transaction"
    )

    # -----------------------------------------------------------
    # User-level context fields
    # These should be looked up from your user database at query time
    # -----------------------------------------------------------
    user_tx_count: float = Field(
        ...,
        ge=0,
        description="Total number of transactions made by this sender historically",
    )
    user_avg_amount: float = Field(
        ..., ge=0, description="Average transaction amount for this sender historically"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "step": 10,
                "type": "TRANSFER",
                "amount": 39.03,
                "oldbalanceOrg": 39.03,
                "newbalanceOrig": 0.0,
                "oldbalanceDest": 55.94,
                "newbalanceDest": 94.98,
                "user_tx_count": 3.0,
                "user_avg_amount": 45.67,
            }
        }


# Response Schema
class PredictionResponse(BaseModel):
    prediction: int  # Raw model output: 0 or 1
    prediction_label: str  # "0 - No Fraud" or "1 - Fraud"
    fraud_probability: float  # Model confidence (0–1)
    risk_level: str  # LOW / MEDIUM / HIGH
    latency_ms: float  # Inference time in milliseconds
    model_version: str


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    model_path: str
    api_version: str


# =============================================================================
# Feature Engineering
# Replicates exact transformations applied during model training
# =============================================================================
def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # -----------------------------------------------------------
    # Temporal features (3.3.5)
    # step = simulation hour; day = simulation day (0–7)
    # -----------------------------------------------------------
    df["type"] = df["type"].astype("category")
    df["day"] = (df["step"] // 24).astype(int)

    # -----------------------------------------------------------
    # Log transform on amount (3.3.4)
    # Reduces skew from high-value transactions
    # -----------------------------------------------------------
    df["amount_log"] = np.log1p(df["amount"])

    # -----------------------------------------------------------
    # Inter-transaction time and velocity (3.4.1)
    # No prior history at inference — inter_tx_time defaults to 0
    # -----------------------------------------------------------
    df["inter_tx_time"] = 0.0
    df["tx_velocity"] = df["amount"] / (df["inter_tx_time"] + 1)

    # -----------------------------------------------------------
    # Actor-level aggregates (3.4.1)
    # No historical aggregate data at inference — defaults applied
    # -----------------------------------------------------------
    df["tx_count"] = 1.0
    df["avg_amount"] = df["amount"]
    df["total_amount"] = df["amount"]
    df["rx_count"] = 1.0
    df["avg_rx_amount"] = df["amount"]
    df["total_rx_amount"] = df["amount"]

    # -----------------------------------------------------------
    # Behavioural deviation (3.4.1)
    # No history → deviation is 0
    # -----------------------------------------------------------
    df["amount_dev_orig"] = 0.0
    df["amount_dev_orig_pct"] = 0.0

    # -----------------------------------------------------------
    # Graph features (3.5.7–3.5.9)
    # No graph context at inference — defaults to 0 / -1
    # -----------------------------------------------------------
    df["degree_orig"] = 0.0
    df["degree_dest"] = 0.0
    df["pagerank_orig"] = 0.0
    df["pagerank_dest"] = 0.0
    df["community_orig"] = -1  # -1 = unknown community
    df["community_dest"] = -1
    df["clustering_orig"] = 0.0
    df["clustering_dest"] = 0.0

    # -----------------------------------------------------------
    # Autoencoder reconstruction error (3.5.6)
    # No autoencoder at serving time — defaults to 0
    # -----------------------------------------------------------
    df["ae_recon_error"] = 0.0

    # -----------------------------------------------------------
    # user_tx_count and user_avg_amount are passed in directly
    # from the request — no default needed
    # -----------------------------------------------------------

    return df


# =============================================================================
# Helper: Risk Level from Fraud Probability
# =============================================================================
def get_risk_level(prob: float) -> str:
    if prob >= 0.75:
        return "HIGH"
    elif prob >= 0.40:
        return "MEDIUM"
    else:
        return "LOW"


# =============================================================================
# Helper: Prediction Label
# =============================================================================
def get_prediction_label(pred: int) -> str:
    return "0 - No Fraud" if pred == 0 else "1 - Fraud"


# =============================================================================
# Routes
# =============================================================================
@app.get("/", tags=["Root"])
def root():
    """API root — returns available endpoints."""
    return {
        "message": "Fraud Detection API is running.",
        "docs": "http://localhost:8000/docs",
        "health_check": "http://localhost:8000/health",
        "single_predict": "http://localhost:8000/predict",
        "batch_predict": "http://localhost:8000/predict/batch",
    }


@app.get("/health", response_model=HealthResponse, tags=["Health"])
def health_check():
    """Check if the API and model are loaded and running correctly."""
    return HealthResponse(
        status="ok",
        model_loaded=model is not None,
        model_path=MODEL_PATH,
        api_version="1.0.0",
    )


@app.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
def predict(transaction: TransactionRequest):
    """
    Predict whether a **single transaction** is fraudulent.

    - Accepts raw transaction fields + user context fields
    - All feature engineering is handled internally
    - Returns prediction, fraud probability, and risk level

    **Prediction Labels:**
    - `0 - No Fraud` → Transaction is legitimate
    - `1 - Fraud`    → Transaction is flagged as fraudulent
    """
    try:
        # Step 1: Convert request to DataFrame
        raw_df = pd.DataFrame([transaction.model_dump()])

        # Step 2: Engineer all features internally
        input_df = engineer_features(raw_df)

        # Step 3: Run inference
        start = time.time()
        pred = int(model.predict(input_df)[0])
        prob = float(model.predict_proba(input_df)[0][1])
        latency = (time.time() - start) * 1000

        return PredictionResponse(
            prediction=pred,
            prediction_label=get_prediction_label(pred),
            fraud_probability=round(prob, 6),
            risk_level=get_risk_level(prob),
            latency_ms=round(latency, 4),
            model_version="1.0.0",
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")


@app.post("/predict/batch", tags=["Prediction"])
def predict_batch(transactions: list[TransactionRequest]):
    """
    Predict fraud for a **batch of transactions** (max 1000 per request).

    - Accepts a list of raw transaction objects
    - Returns a summary + per-transaction predictions

    **Prediction Labels:**
    - `0 - No Fraud` → Transaction is legitimate
    - `1 - Fraud`    → Transaction is flagged as fraudulent
    """
    if len(transactions) > 1000:
        raise HTTPException(
            status_code=400,
            detail="Batch size exceeds limit of 1000 transactions per request.",
        )
    try:
        # Step 1: Convert to DataFrame
        raw_df = pd.DataFrame([t.model_dump() for t in transactions])

        # Step 2: Engineer features for entire batch
        input_df = engineer_features(raw_df)

        # Step 3: Run batch inference
        start = time.time()
        preds = model.predict(input_df).tolist()
        probs = model.predict_proba(input_df)[:, 1].tolist()
        latency = (time.time() - start) * 1000

        results = [
            {
                "index": i,
                "prediction": int(preds[i]),
                "prediction_label": get_prediction_label(int(preds[i])),
                "fraud_probability": round(probs[i], 6),
                "risk_level": get_risk_level(probs[i]),
            }
            for i in range(len(preds))
        ]

        return {
            "summary": {
                "total_transactions": len(transactions),
                "fraud_detected": sum(preds),
                "no_fraud": len(preds) - sum(preds),
                "total_latency_ms": round(latency, 4),
            },
            "results": results,
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Batch prediction error: {str(e)}")


# Entry Point

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="127.0.0.1",  # Localhost only for development
        port=8000,
        reload=True,  # Auto-reload on code changes
    )
