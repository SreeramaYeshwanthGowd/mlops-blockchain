#!/usr/bin/env python3
"""
FastAPI Application for Bitcoin Transaction Anomaly Detection
This script creates a FastAPI application that loads the trained anomaly detection model
and exposes endpoints for real-time prediction.
"""

import os
import json
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Body
from pydantic import BaseModel, Field
import mlflow.pyfunc
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# MLflow model URI
MODEL_URI = "models:/BitcoinAnomalyDetection/latest"
SCALER_URI = "runs:/<RUN_ID>/scaler"  # Replace <RUN_ID> with actual run ID

# Create FastAPI app
app = FastAPI(
    title="Bitcoin Transaction Anomaly Detection API",
    description="API for detecting anomalies in Bitcoin transactions",
    version="1.0.0"
)

# Input data model
class TransactionData(BaseModel):
    hash: str = Field(..., description="Transaction hash")
    size: int = Field(..., description="Transaction size in bytes")
    weight: int = Field(..., description="Transaction weight")
    fee: int = Field(..., description="Transaction fee in satoshis")
    inputs_count: int = Field(..., description="Number of inputs")
    outputs_count: int = Field(..., description="Number of outputs")
    input_value: int = Field(..., description="Total input value in satoshis")
    output_value: int = Field(..., description="Total output value in satoshis")
    
    class Config:
        schema_extra = {
            "example": {
                "hash": "a1b2c3d4e5f6...",
                "size": 250,
                "weight": 1000,
                "fee": 5000,
                "inputs_count": 2,
                "outputs_count": 2,
                "input_value": 1000000,
                "output_value": 995000
            }
        }

# Response model
class AnomalyPrediction(BaseModel):
    transaction_hash: str
    anomaly_score: float
    is_anomaly: bool
    explanation: str

# Global variables for loaded models
anomaly_model = None
scaler = None

@app.on_event("startup")
async def load_model():
    """
    Load the trained model and scaler on startup
    """
    global anomaly_model, scaler
    
    try:
        logger.info(f"Loading anomaly detection model from {MODEL_URI}")
        anomaly_model = mlflow.pyfunc.load_model(MODEL_URI)
        
        # In a real implementation, you would load the scaler with the correct run ID
        # For now, we'll handle this in the preprocessing step
        
        logger.info("Model loaded successfully")
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        # For demonstration, create a dummy model if loading fails
        from sklearn.ensemble import IsolationForest
        logger.info("Creating a dummy model for demonstration")
        anomaly_model = IsolationForest(random_state=42)

def preprocess_transaction(transaction_data):
    """
    Preprocess transaction data for model input
    """
    # Calculate derived features
    data = dict(transaction_data)
    data['fee_rate'] = data['fee'] / max(data['size'], 1)
    data['fee_per_weight'] = data['fee'] / max(data['weight'], 1)
    
    # Select and order features
    features = [
        'size', 'weight', 'fee', 'inputs_count', 'outputs_count',
        'input_value', 'output_value', 'fee_rate', 'fee_per_weight'
    ]
    
    # Create feature vector
    X = np.array([[data[f] for f in features]])
    
    # In a real implementation, we would use the loaded scaler
    # For now, we'll just normalize the data
    X_scaled = X / np.max(np.abs(X), axis=0)
    
    return X_scaled

def get_explanation(score, is_anomaly, transaction_data):
    """
    Generate an explanation for the anomaly prediction
    """
    if not is_anomaly:
        return "Transaction appears normal based on its characteristics."
    
    # Calculate fee rate
    fee_rate = transaction_data.fee / transaction_data.size
    
    # Identify potential reasons for anomaly
    reasons = []
    
    if fee_rate > 100:  # High fee rate
        reasons.append(f"Unusually high fee rate ({fee_rate:.2f} satoshis/byte)")
    elif fee_rate < 1:  # Low fee rate
        reasons.append(f"Unusually low fee rate ({fee_rate:.2f} satoshis/byte)")
    
    if transaction_data.size > 10000:  # Large transaction
        reasons.append(f"Unusually large transaction size ({transaction_data.size} bytes)")
    
    if transaction_data.inputs_count > 50:  # Many inputs
        reasons.append(f"Unusually high number of inputs ({transaction_data.inputs_count})")
    
    if transaction_data.outputs_count > 50:  # Many outputs
        reasons.append(f"Unusually high number of outputs ({transaction_data.outputs_count})")
    
    if not reasons:
        reasons.append("Combination of transaction characteristics deviates from normal patterns")
    
    return "Potential anomaly detected: " + "; ".join(reasons)

@app.post("/predict", response_model=AnomalyPrediction)
async def predict_anomaly(transaction: TransactionData):
    """
    Predict whether a Bitcoin transaction is anomalous
    """
    if anomaly_model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        # Preprocess the transaction data
        X = preprocess_transaction(transaction)
        
        # Make prediction
        # For Isolation Forest: -1 for anomalies, 1 for normal points
        prediction = anomaly_model.predict(X)[0]
        
        # Get anomaly score (higher means more anomalous)
        if hasattr(anomaly_model, "decision_function"):
            score = -float(anomaly_model.decision_function(X)[0])
        else:
            # Fallback for loaded MLflow model
            score = float(anomaly_model.predict(pd.DataFrame(X))[0])
        
        # Determine if it's an anomaly
        is_anomaly = prediction == -1 if prediction in [-1, 1] else score > 0.5
        
        # Generate explanation
        explanation = get_explanation(score, is_anomaly, transaction)
        
        return {
            "transaction_hash": transaction.hash,
            "anomaly_score": score,
            "is_anomaly": is_anomaly,
            "explanation": explanation
        }
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

@app.get("/health")
async def health_check():
    """
    Health check endpoint
    """
    return {"status": "healthy", "model_loaded": anomaly_model is not None}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=5000)
