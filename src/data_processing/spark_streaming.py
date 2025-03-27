#!/usr/bin/env python3
"""
Spark Structured Streaming for Bitcoin Transaction Data Processing
This script consumes Bitcoin transaction data from Kafka and processes it using Spark Structured Streaming.
The processed data is stored in Delta Lake format.
"""

from pyspark.sql import SparkSession
from pyspark.sql.functions import col, window, expr, from_json, current_timestamp
from pyspark.sql.types import StructType, StructField, StringType, LongType, IntegerType, TimestampType

# Define schema for Bitcoin transaction data
transaction_schema = StructType([
    StructField("hash", StringType(), True),
    StructField("time", LongType(), True),
    StructField("size", IntegerType(), True),
    StructField("weight", IntegerType(), True),
    StructField("fee", LongType(), True),
    StructField("inputs_count", IntegerType(), True),
    StructField("outputs_count", IntegerType(), True),
    StructField("input_value", LongType(), True),
    StructField("output_value", LongType(), True)
])

def create_spark_session():
    """
    Create and return a Spark session configured for streaming with Delta Lake
    """
    return (SparkSession.builder
            .appName("BitcoinTransactionProcessing")
            .config("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension")
            .config("spark.sql.catalog.spark_catalog", "org.apache.spark.sql.delta.catalog.DeltaCatalog")
            .config("spark.sql.streaming.checkpointLocation", "/tmp/delta/checkpoints")
            .getOrCreate())

def main():
    """
    Main function to process Bitcoin transaction data using Spark Structured Streaming
    """
    spark = create_spark_session()
    
    # Read from Kafka
    kafka_stream = (spark
                   .readStream
                   .format("kafka")
                   .option("kafka.bootstrap.servers", "localhost:9092")
                   .option("subscribe", "bitcoin-transactions")
                   .option("startingOffsets", "latest")
                   .load())
    
    # Parse JSON data from Kafka
    parsed_stream = kafka_stream.select(
        from_json(col("value").cast("string"), transaction_schema).alias("data")
    ).select("data.*")
    
    # Convert Unix timestamp to timestamp type and add processing time
    transactions = (parsed_stream
                   .withColumn("transaction_time", expr("CAST(CAST(time AS TIMESTAMP) AS TIMESTAMP)"))
                   .withColumn("processing_time", current_timestamp())
                   .drop("time"))
    
    # Calculate fee rate (fee per byte)
    transactions_with_metrics = (transactions
                                .withColumn("fee_rate", col("fee") / col("size"))
                                .withColumn("fee_per_weight", col("fee") / col("weight")))
    
    # Write raw transactions to Delta Lake
    raw_query = (transactions_with_metrics
                .writeStream
                .format("delta")
                .outputMode("append")
                .option("checkpointLocation", "/tmp/delta/checkpoints/raw")
                .start("/tmp/delta/raw_transactions"))
    
    # Compute aggregations over 10-minute windows
    windowed_stats = (transactions_with_metrics
                     .withWatermark("transaction_time", "10 minutes")
                     .groupBy(window("transaction_time", "10 minutes"))
                     .agg(
                         expr("count(1)").alias("transaction_count"),
                         expr("avg(fee)").alias("avg_fee"),
                         expr("avg(fee_rate)").alias("avg_fee_rate"),
                         expr("avg(size)").alias("avg_size"),
                         expr("avg(inputs_count)").alias("avg_inputs"),
                         expr("avg(outputs_count)").alias("avg_outputs"),
                         expr("sum(input_value)").alias("total_input_value"),
                         expr("sum(output_value)").alias("total_output_value")
                     ))
    
    # Write aggregated stats to Delta Lake
    stats_query = (windowed_stats
                  .writeStream
                  .format("delta")
                  .outputMode("append")
                  .option("checkpointLocation", "/tmp/delta/checkpoints/stats")
                  .start("/tmp/delta/transaction_stats"))
    
    # Wait for termination
    spark.streams.awaitAnyTermination()

if __name__ == "__main__":
    main()
