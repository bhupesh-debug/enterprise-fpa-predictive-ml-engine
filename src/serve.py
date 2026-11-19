"""
FastAPI Microservice for Low-Latency Scenario Simulation & Variance Prediction
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional
import pandas as pd
import numpy as np

app = FastAPI(
    title="Enterprise FP&A Predictive Variance & Forecast API",
    description="Microservice providing real-time rolling financial forecasts, downside scenario simulation, and subledger anomaly scoring.",
    version="1.0.0"
)

class FinancialLineItem(BaseModel):
    entity_id: str
    cost_center: str
    gl_account: str
    budget_allocated: float
    actual_spend: float
    headcount: int
    sofr_rate: float
    cpi_index: float
    variance_pct: float
    spend_per_head: float
    is_quarter_end: int
    actual_lag_1m: float
    actual_lag_2m: float
    actual_lag_3m: float
    rolling_mean_3m: float
    rolling_var_3m: float

class PredictionResponse(BaseModel):
    predicted_next_spend: float
    projected_variance_pct: float
    risk_tier: str

@app.get("/health")
def health_check():
    return {"status": "healthy", "service": "fpa-ml-engine", "version": "1.0.0"}

@app.post("/predict", response_model=List[PredictionResponse])
def predict_spend(items: List[FinancialLineItem]):
    responses = []
    for item in items:
        # Heuristic / ML model inference proxy for fast validation
        trend = (item.actual_lag_1m + item.rolling_mean_3m) / 2.0
        macro_effect = 1.0 + (item.sofr_rate * 0.01 if "Cloud" in item.gl_account else 0.0)
        pred = trend * macro_effect
        
        proj_var = (pred - item.budget_allocated) / (item.budget_allocated + 1e-5)
        risk = "HIGH" if abs(proj_var) > 0.15 else ("MEDIUM" if abs(proj_var) > 0.05 else "LOW")
        
        responses.append(PredictionResponse(
            predicted_next_spend=round(pred, 2),
            projected_variance_pct=round(proj_var * 100, 2),
            risk_tier=risk
        ))
    return responses
