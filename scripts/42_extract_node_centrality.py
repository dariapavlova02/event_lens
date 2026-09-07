#!/usr/bin/env python3
"""
Script 42: Extract Hourly Node Centrality (Neo4j)
=================================================
Extracts node-level centrality metrics specific top entities (Hubs)
and aggregated distribution statistics.

Hypothesis:
Specific active nodes (e.g. Binance, SEC) are better predictors
than global graph density.
"""

import os
import pandas as pd
import numpy as np
import networkx as nx
from neo4j import GraphDatabase
from datetime import datetime, timedelta
from tqdm import tqdm
from pathlib import Path
import warnings
import scipy.stats
warnings.filterwarnings('ignore')

# Configuration
NEO4J_URI = "neo4j://127.0.0.1:7687"
NEO4J_USER = "neo4j"
NEO4J_PASSWORD = os.environ["NEO4J_PASSWORD"]
WINDOW_HOURS = 24  # Rolling window size
STEP_HOURS = 1     # Step size

DATA_DIR = (Path(__file__).resolve().parents[1] / 'data')
OUTPUT_FILE = DATA_DIR / 'node_features_hourly.csv'

# Top 20 Entities (Dynamic or Static list)
TOP_ENTITIES = [
    'Ethereum', 'Binance', 'Solana', 'Coinbase', 'Arbitrum',
    'FTX', 'Polygon', 'Uniswap', 'Tether', 'Bybit',
    'SEC', 'USDC', 'Hyperliquid', 'Optimism', 'Base',
    'Aave', 'BNB Chain', 'Ripple', 'Avalanche', 'Circle'
]

def get_time_range(tx):
    query = """
    MATCH (e:Event)
    RETURN min(e.timestamp) as start_time, max(e.timestamp) as end_time
    """
    result = tx.run(query).single()
    return result["start_time"], result["end_time"]

def build_graph_for_window(tx, start_ts, end_ts):
    """Builds NetworkX graph for the window"""
    query = """
    MATCH (ev:Event)-[:MENTIONS]->(e:Entity)
    WHERE ev.timestamp >= datetime($start_time) AND ev.timestamp < datetime($end_time)
    RETURN ev.id as event_id, e.name as entity
    """
    result = tx.run(query, start_time=start_ts.isoformat(), end_time=end_ts.isoformat())

    df = pd.DataFrame([r.data() for r in result])
    G = nx.Graph()
    if df.empty:
        return G

    # Clique expansion
    for event_id, group in df.groupby('event_id'):
        entities = group['entity'].tolist()
        if len(entities) > 1:
            for i in range(len(entities)):
                for j in range(i + 1, len(entities)):
                    if G.has_edge(entities[i], entities[j]):
                        G[entities[i]][entities[j]]['weight'] += 1
                    else:
                        G.add_edge(entities[i], entities[j], weight=1)
    return G

def calculate_node_metrics(G):
    metrics = {
        'max_degree_centrality': 0.0,
        'centralization_skewness': 0.0,  # Is activity concentrated?
        'gini_coefficient': 0.0
    }

    # Specific Entity Centrality
    for entity in TOP_ENTITIES:
        metrics[f'deg_{entity}'] = 0.0

    if G.number_of_nodes() > 0:
        # Degree Centrality
        centrality = nx.degree_centrality(G)
        values = list(centrality.values())

        metrics['max_degree_centrality'] = max(values)
        if len(values) > 1:
            metrics['centralization_skewness'] = scipy.stats.skew(values)

            # Gini
            sorted_v = sorted(values)
            cum_v = np.cumsum(sorted_v)
            n = len(values)
            metrics['gini_coefficient'] = (n + 1 - 2 * np.sum(cum_v) / cum_v[-1]) / n if cum_v[-1] > 0 else 0

        # Specific Entities
        for entity in TOP_ENTITIES:
            if entity in centrality:
                metrics[f'deg_{entity}'] = centrality[entity]

    return metrics

def main():
    print("="*60)
    print("HOURLY NODE CENTRALITY EXTRACTION")
    print("="*60)

    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))

    try:
        with driver.session() as session:
            # 1. Get range
            print("Checking database...")
            start_neo, end_neo = session.execute_read(get_time_range)
            start_dt = start_neo.to_native() if start_neo else None
            end_dt = end_neo.to_native() if end_neo else None

            if not start_dt:
                print("❌ No data found.")
                return

            print(f"Time range: {start_dt} -> {end_dt}")

            # 2. Iterate
            results = []
            curr_time = start_dt
            total_steps = int((end_dt - start_dt).total_seconds() / 3600)

            pbar = tqdm(total=total_steps, desc="Processing Windows")

            while curr_time < end_dt:
                window_end = curr_time + timedelta(hours=WINDOW_HOURS)

                # Build graph
                G = session.execute_read(build_graph_for_window, curr_time, window_end)

                # Metrics
                metrics = calculate_node_metrics(G)
                metrics['timestamp'] = window_end
                results.append(metrics)

                curr_time += timedelta(hours=STEP_HOURS)
                pbar.update(1)

            pbar.close()

            # Save
            df = pd.DataFrame(results)
            df.set_index('timestamp', inplace=True)
            df.to_csv(OUTPUT_FILE)
            print(f"\n✅ Saved {len(df)} rows to {OUTPUT_FILE}")
            print(df.describe().iloc[:, :5]) # Show first few columns stats

    finally:
        driver.close()

if __name__ == "__main__":
    main()
