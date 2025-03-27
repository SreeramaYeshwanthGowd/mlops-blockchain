# Real-Time Bitcoin Transaction Anomaly Detection

This project implements a comprehensive MLOps pipeline for detecting anomalies in Bitcoin transactions in real-time. The system ingests live Bitcoin transaction data, processes it using Apache Spark, trains an anomaly detection model, deploys the model as an API, and provides an interactive dashboard for monitoring.

## System Architecture

The system consists of the following components:

1. **Data Ingestion**: Fetches live Bitcoin transaction data from the Blockchain.com API and publishes it to a Kafka topic.
2. **Data Processing**: Consumes data from Kafka using Spark Structured Streaming and stores it in Delta Lake.
3. **Model Training**: Trains an Isolation Forest model for anomaly detection on historical transaction data.
4. **Model Deployment**: Exposes the trained model as a FastAPI application.
5. **Dashboard**: Provides an interactive Plotly Dash dashboard for monitoring transactions and anomalies.
6. **Monitoring**: Implements Prometheus metrics for system monitoring and Grafana for visualization.

## Project Structure

```
mlops-blockchain/
├── src/
│   ├── data_ingestion/
│   │   └── kafka_producer.py
│   ├── data_processing/
│   │   └── spark_streaming.py
│   ├── model/
│   │   └── train_model.py
│   ├── api/
│   │   └── app.py
│   ├── dashboard/
│   │   └── app.py
│   └── monitoring/
│       └── prometheus_metrics.py
├── docker-compose.yml
├── Dockerfile.api
├── Dockerfile.dashboard
├── Dockerfile.producer
├── Dockerfile.spark
├── requirements.api.txt
├── requirements.dashboard.txt
├── requirements.producer.txt
├── requirements.spark.txt
└── README.md
```

## Setup and Installation

### Prerequisites

- Docker and Docker Compose
- Git

### Installation Steps

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/mlops-blockchain.git
   cd mlops-blockchain
   ```

2. Start the services using Docker Compose:
   ```bash
   docker-compose up -d
   ```

3. Access the services:
   - Dashboard: http://localhost:8050
   - API Documentation: http://localhost:5001/docs
   - Prometheus: http://localhost:9090
   - Grafana: http://localhost:3000 (admin/admin)
   - MLflow: http://localhost:5000

## Component Details

### Data Ingestion

The data ingestion component fetches live Bitcoin transaction data from the Blockchain.com API and publishes it to a Kafka topic. The implementation is in `src/data_ingestion/kafka_producer.py`.

### Data Processing

The data processing component consumes data from Kafka using Spark Structured Streaming, performs transformations, and stores the processed data in Delta Lake. The implementation is in `src/data_processing/spark_streaming.py`.

### Model Training

The model training component trains an Isolation Forest model for anomaly detection on historical transaction data and logs the model using MLflow. The implementation is in `src/model/train_model.py`.

### Model Deployment

The model deployment component exposes the trained model as a FastAPI application. The implementation is in `src/api/app.py`.

### Dashboard

The dashboard component provides an interactive Plotly Dash dashboard for monitoring transactions and anomalies. The implementation is in `src/dashboard/app.py`.

### Monitoring

The monitoring component implements Prometheus metrics for system monitoring. The implementation is in `src/monitoring/prometheus_metrics.py`.

## Usage

### Viewing the Dashboard

1. Access the dashboard at http://localhost:8050
2. Use the filters to adjust the time range, anomaly threshold, and other parameters
3. View the transaction statistics and anomaly detection results

### Using the API

The API provides endpoints for predicting whether a Bitcoin transaction is anomalous.

Example API request:
```bash
curl -X POST "http://localhost:5001/predict" \
     -H "Content-Type: application/json" \
     -d '{
           "hash": "a1b2c3d4e5f6...",
           "size": 250,
           "weight": 1000,
           "fee": 5000,
           "inputs_count": 2,
           "outputs_count": 2,
           "input_value": 1000000,
           "output_value": 995000
         }'
```

## Development

### Running Components Individually

Each component can be run individually for development purposes:

1. Kafka Producer:
   ```bash
   cd src/data_ingestion
   python kafka_producer.py
   ```

2. Spark Streaming:
   ```bash
   cd src/data_processing
   spark-submit --packages org.apache.spark:spark-sql-kafka-0-10_2.12:3.3.0,io.delta:delta-core_2.12:2.2.0 spark_streaming.py
   ```

3. Model Training:
   ```bash
   cd src/model
   python train_model.py
   ```

4. API:
   ```bash
   cd src/api
   uvicorn app:app --host 0.0.0.0 --port 5000
   ```

5. Dashboard:
   ```bash
   cd src/dashboard
   python app.py
   ```

### Extending the Project

To extend the project:

1. Add new features to the API in `src/api/app.py`
2. Add new visualizations to the dashboard in `src/dashboard/app.py`
3. Improve the anomaly detection model in `src/model/train_model.py`
4. Add new metrics to the monitoring component in `src/monitoring/prometheus_metrics.py`

## License

This project is licensed under the MIT License - see the LICENSE file for details.
