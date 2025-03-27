#!/usr/bin/env python3
"""
Plotly Dash Dashboard for Bitcoin Transaction Anomaly Detection
This script creates an interactive dashboard for visualizing Bitcoin transaction data
and anomaly detection results.
"""

import os
import json
import pandas as pd
import numpy as np
import requests
from datetime import datetime, timedelta

import dash
from dash import dcc, html, Input, Output, State, callback
import dash_bootstrap_components as dbc
import plotly.express as px
import plotly.graph_objects as go

# API endpoint
API_ENDPOINT = "http://localhost:5000/predict"

# Create Dash app
app = dash.Dash(
    __name__,
    external_stylesheets=[dbc.themes.BOOTSTRAP],
    meta_tags=[{"name": "viewport", "content": "width=device-width, initial-scale=1"}],
)

app.title = "Bitcoin Transaction Anomaly Detection"

# Load sample data (in a real scenario, this would come from Delta Lake)
def load_sample_data():
    """
    Load sample Bitcoin transaction data
    In a real implementation, this would load data from Delta Lake
    """
    # Generate synthetic data for demonstration
    np.random.seed(42)
    n_samples = 1000
    
    # Generate timestamps for the last 24 hours
    now = datetime.now()
    timestamps = [now - timedelta(minutes=np.random.randint(0, 24*60)) for _ in range(n_samples)]
    timestamps.sort()
    
    # Generate transaction data
    data = {
        'hash': [f"tx_{i}" for i in range(n_samples)],
        'transaction_time': timestamps,
        'size': np.random.lognormal(7, 1, n_samples).astype(int),
        'weight': np.random.lognormal(8, 1, n_samples).astype(int),
        'fee': np.random.lognormal(9, 1.5, n_samples).astype(int),
        'inputs_count': np.random.randint(1, 10, n_samples),
        'outputs_count': np.random.randint(1, 5, n_samples),
        'input_value': np.random.lognormal(16, 2, n_samples).astype(int),
        'output_value': np.random.lognormal(16, 2, n_samples).astype(int),
    }
    
    # Calculate derived metrics
    df = pd.DataFrame(data)
    df['fee_rate'] = df['fee'] / df['size']
    df['fee_per_weight'] = df['fee'] / df['weight']
    
    # Generate anomaly scores (mostly normal with a few anomalies)
    anomaly_scores = np.random.normal(0, 0.5, n_samples)
    # Make a few transactions anomalous
    anomaly_indices = np.random.choice(n_samples, size=int(n_samples * 0.02), replace=False)
    for idx in anomaly_indices:
        anomaly_scores[idx] = np.random.uniform(1.5, 3)
    
    df['anomaly_score'] = anomaly_scores
    df['is_anomaly'] = df['anomaly_score'] > 1.0
    
    return df

# App layout
app.layout = dbc.Container(
    [
        # Header
        dbc.Row(
            [
                dbc.Col(
                    [
                        html.H1("Bitcoin Transaction Anomaly Detection", className="display-4"),
                        html.P(
                            "Interactive dashboard for monitoring Bitcoin transactions and detecting anomalies",
                            className="lead",
                        ),
                    ],
                    width={"size": 10, "offset": 1},
                )
            ],
            className="mb-4 mt-4",
        ),
        
        # Filters and Controls
        dbc.Row(
            [
                dbc.Col(
                    [
                        dbc.Card(
                            [
                                dbc.CardHeader("Filters"),
                                dbc.CardBody(
                                    [
                                        dbc.Row(
                                            [
                                                dbc.Col(
                                                    [
                                                        html.Label("Time Range"),
                                                        dcc.Dropdown(
                                                            id="time-range-dropdown",
                                                            options=[
                                                                {"label": "Last Hour", "value": "1h"},
                                                                {"label": "Last 6 Hours", "value": "6h"},
                                                                {"label": "Last 12 Hours", "value": "12h"},
                                                                {"label": "Last 24 Hours", "value": "24h"},
                                                                {"label": "All Time", "value": "all"},
                                                            ],
                                                            value="24h",
                                                        ),
                                                    ],
                                                    width=6,
                                                ),
                                                dbc.Col(
                                                    [
                                                        html.Label("Anomaly Threshold"),
                                                        dcc.Slider(
                                                            id="anomaly-threshold-slider",
                                                            min=0,
                                                            max=2,
                                                            step=0.1,
                                                            value=1.0,
                                                            marks={i: str(i) for i in range(0, 3)},
                                                        ),
                                                    ],
                                                    width=6,
                                                ),
                                            ]
                                        ),
                                        html.Br(),
                                        dbc.Row(
                                            [
                                                dbc.Col(
                                                    [
                                                        html.Label("Transaction Size Range (bytes)"),
                                                        dcc.RangeSlider(
                                                            id="size-range-slider",
                                                            min=0,
                                                            max=10000,
                                                            step=100,
                                                            value=[0, 10000],
                                                            marks={i: str(i) for i in range(0, 10001, 2000)},
                                                        ),
                                                    ]
                                                )
                                            ]
                                        ),
                                        html.Br(),
                                        dbc.Row(
                                            [
                                                dbc.Col(
                                                    [
                                                        html.Label("Show Anomalies Only"),
                                                        dbc.Switch(id="anomalies-only-switch", value=False),
                                                    ],
                                                    width=6,
                                                ),
                                                dbc.Col(
                                                    [
                                                        html.Label("Auto Refresh"),
                                                        dbc.Switch(id="auto-refresh-switch", value=True),
                                                    ],
                                                    width=6,
                                                ),
                                            ]
                                        ),
                                    ]
                                ),
                            ],
                            className="mb-4",
                        ),
                    ],
                    width={"size": 10, "offset": 1},
                )
            ]
        ),
        
        # KPI Cards
        dbc.Row(
            [
                dbc.Col(
                    [
                        dbc.Card(
                            [
                                dbc.CardBody(
                                    [
                                        html.H4("Total Transactions", className="card-title"),
                                        html.H2(id="total-transactions", className="card-value"),
                                    ]
                                )
                            ],
                            className="mb-4 text-center",
                        ),
                    ],
                    width=3,
                ),
                dbc.Col(
                    [
                        dbc.Card(
                            [
                                dbc.CardBody(
                                    [
                                        html.H4("Anomalies Detected", className="card-title"),
                                        html.H2(id="total-anomalies", className="card-value text-danger"),
                                    ]
                                )
                            ],
                            className="mb-4 text-center",
                        ),
                    ],
                    width=3,
                ),
                dbc.Col(
                    [
                        dbc.Card(
                            [
                                dbc.CardBody(
                                    [
                                        html.H4("Avg. Fee Rate (sat/byte)", className="card-title"),
                                        html.H2(id="avg-fee-rate", className="card-value"),
                                    ]
                                )
                            ],
                            className="mb-4 text-center",
                        ),
                    ],
                    width=3,
                ),
                dbc.Col(
                    [
                        dbc.Card(
                            [
                                dbc.CardBody(
                                    [
                                        html.H4("Avg. Transaction Size", className="card-title"),
                                        html.H2(id="avg-tx-size", className="card-value"),
                                    ]
                                )
                            ],
                            className="mb-4 text-center",
                        ),
                    ],
                    width=3,
                ),
            ],
            className="mb-4",
        ),
        
        # Charts
        dbc.Row(
            [
                dbc.Col(
                    [
                        dbc.Card(
                            [
                                dbc.CardHeader("Transactions Over Time"),
                                dbc.CardBody(
                                    [
                                        dcc.Graph(id="transactions-time-chart"),
                                    ]
                                ),
                            ],
                            className="mb-4",
                        ),
                    ],
                    width=6,
                ),
                dbc.Col(
                    [
                        dbc.Card(
                            [
                                dbc.CardHeader("Anomaly Score Distribution"),
                                dbc.CardBody(
                                    [
                                        dcc.Graph(id="anomaly-score-histogram"),
                                    ]
                                ),
                            ],
                            className="mb-4",
                        ),
                    ],
                    width=6,
                ),
            ]
        ),
        
        dbc.Row(
            [
                dbc.Col(
                    [
                        dbc.Card(
                            [
                                dbc.CardHeader("Fee Rate vs. Transaction Size"),
                                dbc.CardBody(
                                    [
                                        dcc.Graph(id="fee-size-scatter"),
                                    ]
                                ),
                            ],
                            className="mb-4",
                        ),
                    ],
                    width=12,
                ),
            ]
        ),
        
        # Transaction Table
        dbc.Row(
            [
                dbc.Col(
                    [
                        dbc.Card(
                            [
                                dbc.CardHeader("Recent Transactions"),
                                dbc.CardBody(
                                    [
                                        html.Div(id="transactions-table"),
                                    ]
                                ),
                            ],
                            className="mb-4",
                        ),
                    ],
                    width=12,
                ),
            ]
        ),
        
        # Anomaly Prediction Form
        dbc.Row(
            [
                dbc.Col(
                    [
                        dbc.Card(
                            [
                                dbc.CardHeader("Test Transaction Anomaly Detection"),
                                dbc.CardBody(
                                    [
                                        dbc.Row(
                                            [
                                                dbc.Col(
                                                    [
                                                        html.Label("Transaction Hash"),
                                                        dbc.Input(id="tx-hash-input", placeholder="Enter transaction hash", type="text"),
                                                    ],
                                                    width=6,
                                                ),
                                                dbc.Col(
                                                    [
                                                        html.Label("Transaction Size (bytes)"),
                                                        dbc.Input(id="tx-size-input", placeholder="Enter size", type="number", value=250),
                                                    ],
                                                    width=3,
                                                ),
                                                dbc.Col(
                                                    [
                                                        html.Label("Transaction Weight"),
                                                        dbc.Input(id="tx-weight-input", placeholder="Enter weight", type="number", value=1000),
                                                    ],
                                                    width=3,
                                                ),
                                            ]
                                        ),
                                        html.Br(),
                                        dbc.Row(
                                            [
                                                dbc.Col(
                                                    [
                                                        html.Label("Fee (satoshis)"),
                                                        dbc.Input(id="tx-fee-input", placeholder="Enter fee", type="number", value=5000),
                                                    ],
                                                    width=3,
                                                ),
                                                dbc.Col(
                                                    [
                                                        html.Label("Inputs Count"),
                                                        dbc.Input(id="tx-inputs-input", placeholder="Enter inputs count", type="number", value=2),
                                                    ],
                                                    width=3,
                                                ),
                                                dbc.Col(
                                                    [
                                                        html.Label("Outputs Count"),
                                                        dbc.Input(id="tx-outputs-input", placeholder="Enter outputs count", type="number", value=2),
                                                    ],
                                                    width=3,
                                                ),
                                                dbc.Col(
                                                    [
                                                        html.Label("Input Value (satoshis)"),
                                                        dbc.Input(id="tx-input-value-input", placeholder="Enter input value", type="number", value=1000000),
                                                    ],
                                                    width=3,
                                                ),
                                            ]
                                        ),
                                        html.Br(),
                                        dbc.Row(
                                            [
                                                dbc.Col(
                                                    [
                                                        html.Label("Output Value (satoshis)"),
                                                        dbc.Input(id="tx-output-value-input", placeholder="Enter output value", type="number", value=995000),
                                                    ],
                                                    width=3,
                                                ),
                                                dbc.Col(
                                                    [
                                                        dbc.Button("Check Transaction", id="check-tx-button", color="primary", className="mt-4"),
                                                    ],
                                                    width=3,
                                                ),
                                                dbc.Col(
                                                    [
                                                        html.Div(id="prediction-result", className="mt-4"),
                                                    ],
                                                    width=6,
                                                ),
                                            ]
                                        ),
                                    ]
                                ),
                            ],
                            className="mb-4",
                        ),
                    ],
                    width={"size": 10, "offset": 1},
                )
            ]
        ),
        
        # Footer
        dbc.Row(
            [
                dbc.Col(
                    [
                        html.Hr(),
                        html.P(
                            "Bitcoin Transaction Anomaly Detection Dashboard - Powered by MLOps Pipeline",
                            className="text-center text-muted",
                        ),
                    ],
                    width=12,
                )
            ]
        ),
        
        # Store for data
        dcc.Store(id="transaction-data-store"),
        
        # Interval for auto refresh
        dcc.Interval(
            id="auto-refresh-interval",
            interval=30 * 1000,  # 30 seconds
            n_intervals=0,
            disabled=False,
        ),
    ],
    fluid=True,
)

# Callbacks
@app.callback(
    Output("transaction-data-store", "data"),
    [
        Input("auto-refresh-interval", "n_intervals"),
        Input("anomaly-threshold-slider", "value"),
        Input("size-range-slider", "value"),
        Input("time-range-dropdown", "value"),
    ],
)
def update_data(n_intervals, anomaly_threshold, size_range, time_range):
    """
    Update the transaction data based on filters
    """
    # Load sample data
    df = load_sample_data()
    
    # Apply time range filter
    now = datetime.now()
    if time_range != "all":
        hours = int(time_range.replace("h", ""))
        df = df[df["transaction_time"] >= now - timedelta(hours=hours)]
    
    # Apply size range filter
    df = df[(df["size"] >= size_range[0]) & (df["size"] <= size_range[1])]
    
    # Update anomaly flag based on threshold
    df["is_anomaly"] = df["anomaly_score"] > anomaly_threshold
    
    # Convert to JSON for storage
    return df.to_json(date_format="iso", orient="split")

@app.callback(
    [
        Output("total-transactions", "children"),
        Output("total-anomalies", "children"),
        Output("avg-fee-rate", "children"),
        Output("avg-tx-size", "children"),
    ],
    [
        Input("transaction-data-store", "data"),
        Input("anomalies-only-switch", "value"),
    ],
)
def update_kpi_cards(json_data, anomalies_only):
    """
    Update KPI cards based on filtered data
    """
    df = pd.read_json(json_data, orient="split")
    
    if anomalies_only:
        df = df[df["is_anomaly"]]
    
    total_tx = len(df)
    total_anomalies = df["is_anomaly"].sum()
    avg_fee_rate = df["fee_rate"].mean()
    avg_tx_size = df["size"].mean()
    
    return (
        f"{total_tx:,}",
        f"{total_anomalies:,}",
        f"{avg_fee_rate:.2f}",
        f"{avg_tx_size:,.0f}",
    )

@app.callback(
    Output("transactions-time-chart", "figure"),
    [
        Input("transaction-data-store", "data"),
        Input("anomalies-only-switch", "value"),
        Input("anomaly-threshold-slider", "value"),
    ],
)
def update_time_chart(json_data, anomalies_only, anomaly_threshold):
    """
    Update transactions over time chart
    """
    df = pd.read_json(json_data, orient="split")
    
    if anomalies_only:
        df = df[df["is_anomaly"]]
    
    # Group by hour
    df["hour"] = df["transaction_time"].dt.floor("H")
    hourly_counts = df.groupby(["hour", "is_anomaly"]).size().reset_index(name="count")
    
    # Create figure
    fig = go.Figure()
    
    # Add normal transactions
    normal_data = hourly_counts[~hourly_counts["is_anomaly"]]
    if not normal_data.empty:
        fig.add_trace(
            go.Scatter(
                x=normal_data["hour"],
                y=normal_data["count"],
                mode="lines",
                name="Normal Transactions",
                line=dict(color="green", width=2),
                stackgroup="one",
            )
        )
    
    # Add anomalous transactions
    anomaly_data = hourly_counts[hourly_counts["is_anomaly"]]
    if not anomaly_data.empty:
        fig.add_trace(
            go.Scatter(
                x=anomaly_data["hour"],
                y=anomaly_data["count"],
                mode="lines",
                name="Anomalous Transactions",
                line=dict(color="red", width=2),
                stackgroup="one",
            )
        )
    
    fig.update_layout(
        title="Transaction Volume Over Time",
        xaxis_title="Time",
        yaxis_title="Number of Transactions",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=40, r=40, t=40, b=40),
    )
    
    return fig

@app.callback(
    Output("anomaly-score-histogram", "figure"),
    [
        Input("transaction-data-store", "data"),
        Input("anomaly-threshold-slider", "value"),
    ],
)
def update_anomaly_histogram(json_data, anomaly_threshold):
    """
    Update anomaly score histogram
    """
    df = pd.read_json(json_data, orient="split")
    
    fig = go.Figure()
    
    # Add histogram
    fig.add_trace(
        go.Histogram(
            x=df["anomaly_score"],
            nbinsx=30,
            marker_color="lightblue",
        )
    )
    
    # Add threshold line
    fig.add_shape(
        type="line",
        x0=anomaly_threshold,
        x1=anomaly_threshold,
        y0=0,
        y1=1,
        yref="paper",
        line=dict(color="red", width=2, dash="dash"),
    )
    
    fig.add_annotation(
        x=anomaly_threshold,
        y=0.95,
        yref="paper",
        text="Threshold",
        showarrow=True,
        arrowhead=1,
        ax=40,
        ay=0,
    )
    
    fig.update_layout(
        title="Distribution of Anomaly Scores",
        xaxis_title="Anomaly Score",
        yaxis_title="Count",
        margin=dict(l=40, r=40, t=40, b=40),
    )
    
    return fig

@app.callback(
    Output("fee-size-scatter", "figure"),
    [
        Input("transaction-data-store", "data"),
        Input("anomalies-only-switch", "value"),
    ],
)
def update_fee_size_scatter(json_data, anomalies_only):
    """
    Update fee rate vs. transaction size scatter plot
    """
    df = pd.read_json(json_data, orient="split")
    
    if anomalies_only:
        df = df[df["is_anomaly"]]
    
    fig = px.scatter(
        df,
        x="size",
        y="fee_rate",
        color="is_anomaly",
        color_discrete_map={True: "red", False: "blue"},
        hover_data=["hash", "fee", "inputs_count", "outputs_count"],
        opacity=0.7,
        title="Fee Rate vs. Transaction Size",
    )
    
    fig.update_layout(
        xaxis_title="Transaction Size (bytes)",
        yaxis_title="Fee Rate (satoshis/byte)",
        legend_title="Is Anomaly",
        margin=dict(l=40, r=40, t=40, b=40),
    )
    
    return fig

@app.callback(
    Output("transactions-table", "children"),
    [
        Input("transaction-data-store", "data"),
        Input("anomalies-only-switch", "value"),
    ],
)
def update_transactions_table(json_data, anomalies_only):
    """
    Update transactions table
    """
    df = pd.read_json(json_data, orient="split")
    
    if anomalies_only:
        df = df[df["is_anomaly"]]
    
    # Sort by time (most recent first) and take the most recent 10
    df = df.sort_values("transaction_time", ascending=False).head(10)
    
    # Format the table
    table_header = [
        html.Thead(
            html.Tr(
                [
                    html.Th("Time"),
                    html.Th("Hash"),
                    html.Th("Size"),
                    html.Th("Fee"),
                    html.Th("Fee Rate"),
                    html.Th("Anomaly Score"),
                    html.Th("Status"),
                ]
            )
        )
    ]
    
    rows = []
    for _, row in df.iterrows():
        status_badge = dbc.Badge("Anomaly", color="danger") if row["is_anomaly"] else dbc.Badge("Normal", color="success")
        
        rows.append(
            html.Tr(
                [
                    html.Td(row["transaction_time"].strftime("%Y-%m-%d %H:%M:%S")),
                    html.Td(row["hash"][:10] + "..."),
                    html.Td(f"{row['size']:,}"),
                    html.Td(f"{row['fee']:,}"),
                    html.Td(f"{row['fee_rate']:.2f}"),
                    html.Td(f"{row['anomaly_score']:.2f}"),
                    html.Td(status_badge),
                ]
            )
        )
    
    table_body = [html.Tbody(rows)]
    
    return dbc.Table(table_header + table_body, bordered=True, hover=True, responsive=True, striped=True)

@app.callback(
    Output("auto-refresh-interval", "disabled"),
    [Input("auto-refresh-switch", "value")],
)
def toggle_auto_refresh(auto_refresh):
    """
    Toggle auto refresh
    """
    return not auto_refresh

@app.callback(
    Output("prediction-result", "children"),
    [Input("check-tx-button", "n_clicks")],
    [
        State("tx-hash-input", "value"),
        State("tx-size-input", "value"),
        State("tx-weight-input", "value"),
        State("tx-fee-input", "value"),
        State("tx-inputs-input", "value"),
        State("tx-outputs-input", "value"),
        State("tx-input-value-input", "value"),
        State("tx-output-value-input", "value"),
    ],
)
def check_transaction(n_clicks, tx_hash, size, weight, fee, inputs, outputs, input_value, output_value):
    """
    Check a transaction for anomalies
    """
    if n_clicks is None:
        return ""
    
    # In a real implementation, this would call the API
    # For demonstration, we'll simulate a response
    
    # Create transaction data
    tx_data = {
        "hash": tx_hash or "test_transaction",
        "size": size or 250,
        "weight": weight or 1000,
        "fee": fee or 5000,
        "inputs_count": inputs or 2,
        "outputs_count": outputs or 2,
        "input_value": input_value or 1000000,
        "output_value": output_value or 995000,
    }
    
    try:
        # In a real implementation, this would be an API call
        # response = requests.post(API_ENDPOINT, json=tx_data)
        # result = response.json()
        
        # For demonstration, simulate a response
        fee_rate = tx_data["fee"] / tx_data["size"]
        is_anomaly = False
        explanation = "Transaction appears normal based on its characteristics."
        
        # Simple anomaly detection logic for demonstration
        if fee_rate > 100 or fee_rate < 1 or tx_data["size"] > 10000:
            is_anomaly = True
            explanation = f"Potential anomaly detected: Unusual fee rate ({fee_rate:.2f} satoshis/byte)"
        
        result = {
            "transaction_hash": tx_data["hash"],
            "anomaly_score": 1.5 if is_anomaly else 0.2,
            "is_anomaly": is_anomaly,
            "explanation": explanation,
        }
        
        # Create result display
        status_badge = dbc.Badge("ANOMALY DETECTED", color="danger") if result["is_anomaly"] else dbc.Badge("NORMAL", color="success")
        
        return html.Div(
            [
                html.Div([status_badge], className="mb-2"),
                html.Div(f"Anomaly Score: {result['anomaly_score']:.2f}", className="mb-2"),
                html.Div(result["explanation"]),
            ]
        )
    
    except Exception as e:
        return html.Div(f"Error: {str(e)}", className="text-danger")

if __name__ == "__main__":
    app.run_server(debug=True, host="0.0.0.0", port=8050)
