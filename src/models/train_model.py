#!/usr/bin/env python3
"""
Anomaly Detection Model Training for Bitcoin Transactions
This script trains an anomaly detection model on historical Bitcoin transaction data
and logs the model and metrics using MLflow.
"""

import os
import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import mlflow
import mlflow.sklearn
from delta import DeltaTable
from pyspark.sql import SparkSession

# MLflow tracking URI
MLFLOW_TRACKING_URI = "http://localhost:5000"
# Delta Lake data path
DELTA_PATH = "/tmp/delta/raw_transactions"
# Model parameters
RANDOM_STATE = 42
CONTAMINATION = 0.01  # Expected proportion of anomalies

def create_spark_session():
    """
    Create and return a Spark session configured for Delta Lake
    """
    return (SparkSession.builder
            .appName("BitcoinAnomalyModelTraining")
            .config("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension")
            .config("spark.sql.catalog.spark_catalog", "org.apache.spark.sql.delta.catalog.DeltaCatalog")
            .getOrCreate())

def load_data_from_delta():
    """
    Load historical Bitcoin transaction data from Delta Lake
    """
    spark = create_spark_session()
    
    # Check if Delta table exists
    if not DeltaTable.isDeltaTable(spark, DELTA_PATH):
        raise ValueError(f"Delta table does not exist at {DELTA_PATH}")
    
    # Load data from Delta table
    delta_df = spark.read.format("delta").load(DELTA_PATH)
    
    # Convert to Pandas DataFrame for scikit-learn
    pandas_df = delta_df.toPandas()
    
    return pandas_df

def preprocess_data(df):
    """
    Preprocess the data for anomaly detection
    """
    # Select relevant features
    features = [
        'size', 'weight', 'fee', 'inputs_count', 'outputs_count',
        'input_value', 'output_value', 'fee_rate', 'fee_per_weight'
    ]
    
    # Drop rows with missing values
    df_clean = df[features].dropna()
    
    # Handle extreme values (cap at 99th percentile)
    for feature in features:
        cap_value = df_clean[feature].quantile(0.99)
        df_clean[feature] = df_clean[feature].clip(upper=cap_value)
    
    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df_clean)
    
    return X_scaled, scaler, df_clean.index

def train_anomaly_model(X):
    """
    Train an Isolation Forest model for anomaly detection
    """
    model = IsolationForest(
        n_estimators=100,
        max_samples='auto',
        contamination=CONTAMINATION,
        random_state=RANDOM_STATE,
        n_jobs=-1
    )
    
    model.fit(X)
    return model

def evaluate_model(model, X, indices, original_df):
    """
    Evaluate the anomaly detection model
    """
    # Predict anomaly scores (-1 for anomalies, 1 for normal)
    y_pred = model.predict(X)
    
    # Convert to anomaly scores (higher means more anomalous)
    anomaly_scores = -model.decision_function(X)
    
    # Create a DataFrame with predictions
    results = pd.DataFrame({
        'index': indices,
        'anomaly_score': anomaly_scores,
        'is_anomaly': y_pred == -1
    })
    
    # Merge with original data
    anomaly_df = pd.merge(
        original_df.iloc[indices].reset_index(),
        results,
        left_index=True,
        right_on='index'
    )
    
    # Calculate metrics
    num_anomalies = sum(y_pred == -1)
    anomaly_percentage = (num_anomalies / len(y_pred)) * 100
    
    # Get statistics for normal vs anomalous transactions
    normal_df = anomaly_df[anomaly_df['is_anomaly'] == False]
    anomaly_df = anomaly_df[anomaly_df['is_anomaly'] == True]
    
    metrics = {
        'num_transactions': len(y_pred),
        'num_anomalies': num_anomalies,
        'anomaly_percentage': anomaly_percentage,
        'avg_normal_fee': normal_df['fee'].mean(),
        'avg_anomaly_fee': anomaly_df['fee'].mean(),
        'avg_normal_size': normal_df['size'].mean(),
        'avg_anomaly_size': anomaly_df['size'].mean(),
        'avg_normal_fee_rate': normal_df['fee_rate'].mean(),
        'avg_anomaly_fee_rate': anomaly_df['fee_rate'].mean(),
    }
    
    return metrics, anomaly_df

def main():
    """
    Main function to train and log the anomaly detection model
    """
    # Set MLflow tracking URI
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    experiment_name = "bitcoin_anomaly_detection_experiment"
    mlflow.create_experiment(experiment_name)
    mlflow.set_experiment(experiment_name)
    print(f"Experiment '{experiment_name}' created successfully!")
    # Start MLflow run
    with mlflow.start_run(run_name="bitcoin_anomaly_detection") as run:
        # Load data
        try:
            df = load_data_from_delta()
            print(f"Loaded {len(df)} transactions from Delta Lake")
        except Exception as e:
            print(f"Error loading data: {e}")
            # For demonstration, create synthetic data if Delta table doesn't exist
            print("Creating synthetic data for demonstration")
            np.random.seed(RANDOM_STATE)
            n_samples = 10000
            df = pd.DataFrame({
                'size': np.random.lognormal(8, 1, n_samples),
                'weight': np.random.lognormal(9, 1, n_samples),
                'fee': np.random.lognormal(10, 2, n_samples),
                'inputs_count': np.random.randint(1, 10, n_samples),
                'outputs_count': np.random.randint(1, 10, n_samples),
                'input_value': np.random.lognormal(16, 2, n_samples),
                'output_value': np.random.lognormal(16, 2, n_samples),
            })
            df['fee_rate'] = df['fee'] / df['size']
            df['fee_per_weight'] = df['fee'] / df['weight']
        
        # Preprocess data
        X_scaled, scaler, indices = preprocess_data(df)
        
        # Train model
        model = train_anomaly_model(X_scaled)
        
        # Evaluate model
        metrics, anomalies = evaluate_model(model, X_scaled, indices, df)
        
        # Log parameters
        mlflow.log_param("contamination", CONTAMINATION)
        mlflow.log_param("n_estimators", 100)
        mlflow.log_param("random_state", RANDOM_STATE)
        
        # Log metrics
        for name, value in metrics.items():
            mlflow.log_metric(name, value)
        
        # Log model
        mlflow.sklearn.log_model(
            model, 
            "bitcoin_anomaly_model",
            registered_model_name="BitcoinAnomalyDetection"
        )
        
        # Log scaler as artifact
        mlflow.sklearn.log_model(
            scaler,
            "scaler"
        )
        
        # Log sample anomalies
        if not anomalies.empty:
            anomalies.to_csv("sample_anomalies.csv", index=False)
            mlflow.log_artifact("sample_anomalies.csv")
        
        print(f"Model training completed. Run ID: {run.info.run_id}")
        print(f"Model metrics: {metrics}")

if __name__ == "__main__":
    main()
