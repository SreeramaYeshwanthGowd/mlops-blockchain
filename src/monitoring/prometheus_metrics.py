#!/usr/bin/env python3
"""
Prometheus Monitoring for Bitcoin Transaction Anomaly Detection System
This script sets up Prometheus monitoring for the Bitcoin transaction anomaly detection system.
"""

import time
import random
from prometheus_client import start_http_server, Counter, Gauge, Histogram, Summary

# Define metrics
TRANSACTIONS_PROCESSED = Counter(
    'bitcoin_transactions_processed_total',
    'Total number of Bitcoin transactions processed',
    ['status']  # 'normal' or 'anomalous'
)

ANOMALY_SCORE = Gauge(
    'bitcoin_transaction_anomaly_score',
    'Anomaly score of the most recent transaction'
)

PROCESSING_TIME = Histogram(
    'bitcoin_transaction_processing_seconds',
    'Time spent processing a transaction',
    buckets=[0.1, 0.5, 1.0, 2.0, 5.0, 10.0]
)

API_REQUEST_LATENCY = Summary(
    'bitcoin_api_request_latency_seconds',
    'API request latency in seconds',
    ['endpoint']
)

KAFKA_LAG = Gauge(
    'bitcoin_kafka_consumer_lag',
    'Kafka consumer lag (number of messages behind)'
)

DELTA_LAKE_SIZE = Gauge(
    'bitcoin_delta_lake_size_bytes',
    'Size of Delta Lake storage in bytes'
)

MODEL_PREDICTION_ERRORS = Counter(
    'bitcoin_model_prediction_errors_total',
    'Total number of errors during model prediction'
)

def simulate_metrics():
    """
    Simulate metrics for demonstration purposes
    In a real implementation, these metrics would be collected from the actual system
    """
    while True:
        # Simulate transaction processing
        is_anomaly = random.random() < 0.05  # 5% chance of anomaly
        status = 'anomalous' if is_anomaly else 'normal'
        TRANSACTIONS_PROCESSED.labels(status=status).inc()
        
        # Simulate anomaly score
        anomaly_score = random.uniform(0.1, 3.0) if is_anomaly else random.uniform(0.0, 0.9)
        ANOMALY_SCORE.set(anomaly_score)
        
        # Simulate processing time
        with PROCESSING_TIME.time():
            # Simulate processing delay
            time.sleep(random.uniform(0.05, 0.2))
        
        # Simulate API request latency
        for endpoint in ['predict', 'health']:
            latency = random.uniform(0.01, 0.5)
            API_REQUEST_LATENCY.labels(endpoint=endpoint).observe(latency)
        
        # Simulate Kafka lag
        lag = int(random.uniform(0, 100))
        KAFKA_LAG.set(lag)
        
        # Simulate Delta Lake size
        delta_size = 1024 * 1024 * random.uniform(100, 500)  # 100-500 MB
        DELTA_LAKE_SIZE.set(delta_size)
        
        # Occasionally simulate prediction errors
        if random.random() < 0.01:  # 1% chance of error
            MODEL_PREDICTION_ERRORS.inc()
        
        # Wait before updating metrics again
        time.sleep(1)

def main():
    """
    Start Prometheus metrics server
    """
    # Start Prometheus HTTP server
    start_http_server(9090)
    print("Prometheus metrics server started on port 9090")
    
    # Start simulating metrics
    simulate_metrics()

if __name__ == "__main__":
    main()
