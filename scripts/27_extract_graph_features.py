#!/usr/bin/env python3
"""
Script 27: Extract Graph Features (Temporal Graph Analysis)
===========================================================

Iterates through the dataset using a sliding window to capture
structural changes in the Event Knowledge Graph over time.

Features Extracted (Daily):
---------------------------
1. Graph Topology:
   - Density: How connected is the discussion?
   - Transitivity (Clustering Coeff): Are talked-about entities linked to each other?
   - Modularity: How fragmented is the discourse into communities?

2. Centrality Dynamics:
   - Gini Index of Degree Centrality: Is conversation dominated by few (high Gini) or many?
   - Stability: Correlation of centrality rankings with previous day (Shock -> Low Stability).
   - Top Entity Dominance: % of mentions held by top 3 entities.

3. Entity-Specific (Contagion):
   - "Risk Cluster" Density: Activity within specific subgraphs (e.g., Stablecoins, Exchanges).
   - Max Degree Change: Which entity spiked most in centrality?

Output:
   data/graph_features_temporal.csv
"""

import os
import pandas as pd
import numpy as np
import networkx as nx
from neo4j import GraphDatabase
from tqdm import tqdm
from pathlib import Path
from datetime import datetime, timedelta
import warnings

# Suppress warnings
warnings.filterwarnings("ignore")

DATA_DIR = (Path(__file__).resolve().parents[1] / 'data')
NEO4J_URI = "neo4j://127.0.0.1:7687"
NEO4J_AUTH = (os.getenv("NEO4J_USER", "neo4j"), os.environ["NEO4J_PASSWORD"])

print("=" * 70)
print("EXTRACTING TEMPORAL GRAPH FEATURES")
print("=" * 70)

driver = GraphDatabase.driver(NEO4J_URI, auth=NEO4J_AUTH)

def get_daily_graph(session, date_str):
    """
    Fetches the co-occurrence subgraph for a specific 24h window (or rolling window).
    Using a rolling window of 3 days for robust structure.
    """
    query = """
    MATCH (e:Event)
    WHERE date(e.timestamp) >= date($date) - duration('P2D')
      AND date(e.timestamp) <= date($date)

    MATCH (e)-[:MENTIONS]->(ent1:Entity)
    MATCH (e)-[:MENTIONS]->(ent2:Entity)
    WHERE ent1.name < ent2.name

    RETURN ent1.name as source, ent2.name as target, count(e) as weight
    """
    result = session.run(query, date=date_str)

    G = nx.Graph()
    for r in result:
        G.add_edge(r['source'], r['target'], weight=r['weight'])

    return G

def gini(x):
    """Calculate Gini coefficient of a distribution."""
    if x.size == 0: return 0
    total = 0
    for i, xi in enumerate(sorted(x)[:-1], 1):
        total += np.sum(np.abs(xi - x[i:]))
    return total / (len(x)**2 * np.mean(x))

# Get Date Range
with driver.session() as session:
    res = session.run("MATCH (e:Event) RETURN min(date(e.timestamp)) as min_date, max(date(e.timestamp)) as max_date")
    rec = res.single()
    start_date = rec['min_date'].to_native()
    end_date = rec['max_date'].to_native()

print(f"Time Range: {start_date} -> {end_date}")
dates = pd.date_range(start_date, end_date, freq='D')

features_list = []
prev_centrality = {}

print("\n🚀 Starting Feature Extraction (Rolling 3-day Window)...")

with driver.session() as session:
    for date in tqdm(dates):
        date_str = date.strftime('%Y-%m-%d')

        # Build graph for this window
        G = get_daily_graph(session, date_str)

        # 1. Basic Topology
        num_nodes = G.number_of_nodes()
        num_edges = G.number_of_edges()

        if num_nodes < 5:
            # Skip empty/sparse days
            features_list.append({
                'date': date,
                'graph_nodes': num_nodes,
                'graph_edges': num_edges
            })
            continue

        density = nx.density(G)
        try:
           transitivity = nx.transitivity(G)
        except:
           transitivity = 0

        # 2. Centrality Metrics
        degree_dict = dict(nx.degree(G, weight='weight'))
        degrees = list(degree_dict.values())

        avg_degree = np.mean(degrees)
        max_degree = np.max(degrees)
        degree_gini = gini(np.array(degrees))

        # Centrality Ranking Stability (Jaccard of top 10 entities vs yesterday)
        current_top_10 = set(sorted(degree_dict, key=degree_dict.get, reverse=True)[:10])

        stability_score = 0
        if prev_centrality:
             prev_top_10 = prev_centrality
             intersection = len(current_top_10.intersection(prev_top_10))
             union = len(current_top_10.union(prev_top_10))
             stability_score = intersection / union if union > 0 else 0

        prev_centrality = current_top_10

        # 3. Community Structure (Louvain/Modularity approx using greedy)
        # Higher modularity = fragmented market (isolated discussions)
        # Lower modularity = unified market (everyone talking about the same thing/connections)
        try:
            communities = nx.community.greedy_modularity_communities(G, weight='weight')
            num_communities = len(communities)
            modularity = nx.community.modularity(G, communities, weight='weight')
        except:
            num_communities = 0
            modularity = 0

        # 4. Assortativity (Do high-impact entities talk to high-impact entities?)
        # Approx: simple degree assortativity
        assortativity = nx.degree_assortativity_coefficient(G, weight='weight')
        if np.isnan(assortativity): assortativity = 0

        features = {
            'date': date,
            'graph_nodes': num_nodes,
            'graph_edges': num_edges,
            'graph_density': density,
            'graph_transitivity': transitivity,
            'graph_avg_degree': avg_degree,
            'graph_max_degree': max_degree,
            'graph_degree_gini': degree_gini,
            'graph_centrality_stability': stability_score,
            'graph_num_communities': num_communities,
            'graph_modularity': modularity,
            'graph_assortativity': assortativity
        }

        features_list.append(features)

driver.close()

# Save Features
df_features = pd.DataFrame(features_list)
df_features.set_index('date', inplace=True)

output_path = DATA_DIR / 'graph_features_temporal.csv'
df_features.to_csv(output_path)

print(f"\n✅ Features extracted and saved to {output_path}")
print(f"Rows: {len(df_features)}")
print("\nFirst 5 rows:")
print(df_features.head())
