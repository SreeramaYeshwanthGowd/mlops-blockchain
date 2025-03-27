#!/usr/bin/env python3
"""
Kafka Producer for Bitcoin Transaction Data
This script fetches live Bitcoin transaction data from the Blockchain.com API
and publishes it to a Kafka topic.
"""

import json
import time
import requests
from kafka import KafkaProducer
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Blockchain.com API endpoint for unconfirmed transactions
BLOCKCHAIN_API_URL = "https://blockchain.info/unconfirmed-transactions?format=json"

# Kafka configuration
KAFKA_BOOTSTRAP_SERVERS = 'localhost:9092'
KAFKA_TOPIC = 'bitcoin-transactions'

def fetch_bitcoin_transactions():
    """
    Fetch unconfirmed Bitcoin transactions from Blockchain.com API
    """
    try:
        response = requests.get(BLOCKCHAIN_API_URL)
        response.raise_for_status()
        data = response.json()
        return data.get('txs', [])
    except requests.exceptions.RequestException as e:
        logger.error(f"Error fetching Bitcoin transactions: {e}")
        return []

def create_kafka_producer():
    """
    Create and return a Kafka producer instance
    """
    try:
        producer = KafkaProducer(
            bootstrap_servers=KAFKA_BOOTSTRAP_SERVERS,
            value_serializer=lambda v: json.dumps(v).encode('utf-8'),
            acks='all',
            retries=3
        )
        return producer
    except Exception as e:
        logger.error(f"Error creating Kafka producer: {e}")
        return None

def main():
    """
    Main function to fetch Bitcoin transactions and publish to Kafka
    """
    producer = create_kafka_producer()
    if not producer:
        logger.error("Failed to create Kafka producer. Exiting.")
        return

    logger.info(f"Starting Bitcoin transaction producer. Publishing to topic: {KAFKA_TOPIC}")
    
    try:
        while True:
            transactions = fetch_bitcoin_transactions()
            logger.info(f"Fetched {len(transactions)} transactions")
            
            for tx in transactions:
                # Extract relevant transaction data
                transaction_data = {
                    'hash': tx.get('hash'),
                    'time': tx.get('time'),
                    'size': tx.get('size'),
                    'weight': tx.get('weight'),
                    'fee': tx.get('fee'),
                    'inputs_count': len(tx.get('inputs', [])),
                    'outputs_count': len(tx.get('out', [])),
                    'input_value': sum(inp.get('prev_out', {}).get('value', 0) for inp in tx.get('inputs', [])),
                    'output_value': sum(out.get('value', 0) for out in tx.get('out', [])),
                }
                
                # Send to Kafka topic
                future = producer.send(KAFKA_TOPIC, transaction_data)
                future.get(timeout=10)  # Block until the message is sent
                
            # Sleep to avoid API rate limits
            time.sleep(30)
            
    except KeyboardInterrupt:
        logger.info("Producer interrupted. Shutting down.")
    finally:
        producer.flush()
        producer.close()
        logger.info("Producer closed.")

if __name__ == "__main__":
    main()
