#!/usr/bin/env python3
"""
Script 38: Extract Hourly Graph Features (Neo4j)
================================================
Generates HOURLY graph topology features using a rolling window.
Purpose: Provide high-granularity data for ML models (RF/XGBoost).

Window Strategy:
- Sliding window: 24 hours (to ensure enough connectivity)
- Step: 1 hour
- Granularity: Hourly

Metrics:
- Density, Modularity, Components
- Centrality Stability (Jaccard index t vs t-1)
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
warnings.filterwarnings('ignore')

# Configuration
NEO4J_URI = "neo4j://127.0.0.1:7687"
NEO4J_USER = "neo4j"
NEO4J_PASSWORD = os.environ["NEO4J_PASSWORD"]  # Matches scripts/27_extract_graph_features.py
WINDOW_HOURS = 24  # Rolling window size
STEP_HOURS = 1     # Step size

DATA_DIR = (Path(__file__).resolve().parents[1] / 'data')
OUTPUT_FILE = DATA_DIR / 'graph_features_hourly.csv'

def get_total_events(tx):
    result = tx.run("MATCH (e:Event) RETURN count(e) as count")
    return result.single()["count"]

def get_time_range(tx):
    query = """
    MATCH (e:Event)
    RETURN min(e.timestamp) as start_time, max(e.timestamp) as end_time
    """
    result = tx.run(query).single()
    return result["start_time"], result["end_time"]

def build_graph_for_window(tx, start_ts, end_ts):
    """
    Builds a NetworkX graph for events within the time window.
    Nodes: Entities
    Edges: Co-occurrence in events
    """
    query = """
    MATCH (ev:Event)-[:MENTIONS]->(e:Entity)
    WHERE ev.timestamp >= datetime($start_time) AND ev.timestamp < datetime($end_time)
    RETURN ev.id as event_id, e.name as entity
    """
    result = tx.run(query, start_time=start_ts.isoformat(), end_time=end_ts.isoformat())

    # Build graph in memory (Client-side projection for speed)
    df = pd.DataFrame([r.data() for r in result])

    G = nx.Graph()
    if df.empty:
        return G

    # Create edges between entities in the same event (Clique expansion)
    for event_id, group in df.groupby('event_id'):
        entities = group['entity'].tolist()
        if len(entities) > 1:
            # Add clique
            for i in range(len(entities)):
                for j in range(i + 1, len(entities)):
                    if G.has_edge(entities[i], entities[j]):
                        G[entities[i]][entities[j]]['weight'] += 1
                    else:
                        G.add_edge(entities[i], entities[j], weight=1)

    return G

def calculate_metrics(G, prev_top_nodes=None):
    """Calculate topological metrics for the graph"""
    metrics = {
        'nodes': G.number_of_nodes(),
        'edges': G.number_of_edges(),
        'density': 0.0,
        'transitivity': 0.0,
        'avg_degree': 0.0,
        'max_degree': 0,
        'num_components': 0,
        'community_modularity': 0.0,
        'centrality_stability': np.nan
    }

    if metrics['nodes'] > 0:
        metrics['density'] = nx.density(G)
        metrics['transitivity'] = nx.transitivity(G)
        degrees = [d for n, d in G.degree()]
        metrics['avg_degree'] = np.mean(degrees)
        metrics['max_degree'] = np.max(degrees)
        metrics['num_components'] = nx.number_connected_components(G)

        # Communities (Louvain/Greedy)
        # Using simple connected components as proxy for speed if graph is sparse,
        # or greedy_modularity_communities for actual modularity
        try:
            communities = nx.community.greedy_modularity_communities(G)
            if len(communities) > 1:
                metrics['community_modularity'] = nx.community.modularity(G, communities)
        except:
            pass

        # Stability (Jaccard of top 10 degree centrality nodes)
        current_top = set(sorted(G.degree, key=lambda x: x[1], reverse=True)[:10])

        if prev_top_nodes is not None and len(prev_top_nodes) > 0:
            intersection = len(current_top.intersection(prev_top_nodes))
            union = len(current_top.union(prev_top_nodes))
            metrics['centrality_stability'] = intersection / union if union > 0 else 0.0

        return metrics, current_top

    return metrics, set()

def main():
    print("="*60)
    print("HOURLY GRAPH FEATURE EXTRACTION")
    print("="*60)

    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))

    try:
        with driver.session() as session:
            # 1. Get range
            print("Checking database...")
            start_neo, end_neo = session.execute_read(get_time_range)
            # Convert neo4j.time.DateTime to Python datetime
            start_dt = start_neo.to_native() if start_neo else None
            end_dt = end_neo.to_native() if end_neo else None

            if not start_dt or not end_dt:
                print("❌ No events found in database!")
                return

            print(f"Time range: {start_dt} -> {end_dt}")

            # 2. Iterate
            results = []
            prev_top_nodes = None

            curr_time = start_dt
            total_steps = int((end_dt - start_dt).total_seconds() / 3600)

            pbar = tqdm(total=total_steps, desc="Processing Windows")

            while curr_time < end_dt:
                window_end = curr_time + timedelta(hours=WINDOW_HOURS)

                # Build graph
                G = session.execute_read(build_graph_for_window, curr_time, window_end)


                # Metrics
                metrics, top_nodes = calculate_metrics(G, prev_top_nodes)
                prev_top_nodes = top_nodes

                metrics['timestamp'] = window_end # Timestamp is END of window (causal)
                results.append(metrics)

                curr_time += timedelta(hours=STEP_HOURS)
                pbar.update(1)

            pbar.close()

            # Save
            df = pd.DataFrame(results)
            df.set_index('timestamp', inplace=True)
            df.to_csv(OUTPUT_FILE)
            print(f"\n✅ Saved {len(df)} rows to {OUTPUT_FILE}")
            print(df.describe())

    finally:
        driver.close()

if __name__ == "__main__":
    main()
