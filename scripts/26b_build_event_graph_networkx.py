#!/usr/bin/env python3
"""
Script 26b: Build Event Graph with NetworkX (No Neo4j needed)
=============================================================

Creates the same knowledge graph using NetworkX for local analysis.
Allows graph feature extraction and analysis without Neo4j.

Output:
- data/event_graph.graphml (can be imported into Neo4j later)
- data/event_graph_stats.json
"""

import pandas as pd
import numpy as np
import networkx as nx
import json
from pathlib import Path
from datetime import datetime, timedelta
from collections import defaultdict
from tqdm import tqdm
import pickle

DATA_DIR = (Path(__file__).resolve().parents[1] / 'data')

print("=" * 70)
print("BUILDING EVENT KNOWLEDGE GRAPH (NetworkX)")
print("=" * 70)

# ============================================================
# STEP 1: Load and parse TBSA data
# ============================================================

print("\n📊 Step 1: Loading TBSA data...")
print("-" * 70)

df = pd.read_csv(DATA_DIR / 'all_messages_tbsa.csv')
df['date'] = pd.to_datetime(df['date'])

print(f"   Total messages: {len(df):,}")
df = df[df['tbsa_has_result'] == True].copy()
print(f"   After filtering: {len(df):,}")

# Parse TBSA
print("\n   Parsing TBSA targets...")

events = []
entity_mentions = []
co_occurrences = defaultdict(lambda: {'count': 0, 'sentiment_diffs': []})
entity_stats = defaultdict(lambda: {
    'mentions': 0, 'total_impact': 0, 'total_sentiment': 0,
    'types': set(), 'tickers': set(), 'categories': set()
})

for idx, row in tqdm(df.iterrows(), total=len(df), desc="   Parsing"):
    try:
        targets_str = row['tbsa_targets_json']
        if pd.isna(targets_str):
            continue

        targets_str = targets_str.replace('""', '"')
        if targets_str.startswith('"') and targets_str.endswith('"'):
            targets_str = targets_str[1:-1]

        targets = json.loads(targets_str)
        if not targets:
            continue

        event_id = f"msg_{idx}"
        impact_sum = sum(t.get('impact', 5) for t in targets)

        event = {
            'id': event_id,
            'timestamp': row['date'],
            'channel': row['channel'],
            'views': row.get('views', 0) or 0,
            'forwards': row.get('forwards', 0) or 0,
            'global_sentiment': row.get('tbsa_global_sentiment', 0) or 0,
            'is_fact': row.get('tbsa_is_fact', False),
            'impact_sum': impact_sum,
            'num_targets': len(targets)
        }
        events.append(event)

        entities_in_msg = []
        for t in targets:
            entity_name = t.get('name', 'UNKNOWN')
            if not entity_name or entity_name == 'UNKNOWN':
                continue

            mention = {
                'event_id': event_id,
                'entity_name': entity_name,
                'ticker': t.get('ticker'),
                'type': t.get('type', 'OTHER'),
                'sentiment': t.get('sentiment', 0),
                'impact': t.get('impact', 5),
                'category': t.get('category', 'OTHER')
            }
            entity_mentions.append(mention)
            entities_in_msg.append((entity_name, t.get('sentiment', 0)))

            # Update entity stats
            entity_stats[entity_name]['mentions'] += 1
            entity_stats[entity_name]['total_impact'] += mention['impact']
            entity_stats[entity_name]['total_sentiment'] += mention['sentiment']
            if mention['type']:
                entity_stats[entity_name]['types'].add(mention['type'])
            if mention['ticker']:
                entity_stats[entity_name]['tickers'].add(mention['ticker'])
            if mention['category']:
                entity_stats[entity_name]['categories'].add(mention['category'])

        # Track co-occurrences
        for i, (e1, s1) in enumerate(entities_in_msg):
            for j, (e2, s2) in enumerate(entities_in_msg):
                if i < j:
                    key = tuple(sorted([e1, e2]))
                    co_occurrences[key]['count'] += 1
                    co_occurrences[key]['sentiment_diffs'].append(abs(s1 - s2))

    except:
        continue

print(f"\n   ✓ Events: {len(events):,}")
print(f"   ✓ Entity mentions: {len(entity_mentions):,}")
print(f"   ✓ Unique entities: {len(entity_stats):,}")
print(f"   ✓ Co-occurrence pairs: {len(co_occurrences):,}")

# ============================================================
# STEP 2: Build NetworkX Graph
# ============================================================

print("\n🔗 Step 2: Building NetworkX graph...")
print("-" * 70)

G = nx.Graph()

# Add entity nodes
print("   Adding entity nodes...")
for name, stats in tqdm(entity_stats.items(), desc="   Entities"):
    G.add_node(name,
               node_type='entity',
               mentions=stats['mentions'],
               avg_impact=stats['total_impact'] / stats['mentions'] if stats['mentions'] > 0 else 0,
               avg_sentiment=stats['total_sentiment'] / stats['mentions'] if stats['mentions'] > 0 else 0,
               entity_type=list(stats['types'])[0] if stats['types'] else 'OTHER',
               ticker=list(stats['tickers'])[0] if stats['tickers'] else None)

print(f"   ✓ Entity nodes: {len([n for n in G.nodes if G.nodes[n].get('node_type') == 'entity']):,}")

# Add co-occurrence edges (only significant ones)
print("   Adding co-occurrence edges...")
min_cooccur = 3
significant_cooccur = [(k, v) for k, v in co_occurrences.items() if v['count'] >= min_cooccur]

for (e1, e2), data in tqdm(significant_cooccur, desc="   Edges"):
    if e1 in G and e2 in G:
        avg_diff = np.mean(data['sentiment_diffs']) if data['sentiment_diffs'] else 0
        G.add_edge(e1, e2,
                   weight=data['count'],
                   co_occurrence_count=data['count'],
                   avg_sentiment_diff=avg_diff)

print(f"   ✓ Co-occurrence edges: {G.number_of_edges():,}")

# ============================================================
# STEP 3: Calculate graph metrics
# ============================================================

print("\n📈 Step 3: Calculating graph metrics...")
print("-" * 70)

# Degree centrality
print("   Computing degree centrality...")
degree_centrality = nx.degree_centrality(G)

# Betweenness (on subgraph for speed)
print("   Computing betweenness centrality (top 1000 nodes)...")
top_nodes = sorted(G.nodes(), key=lambda x: G.degree(x), reverse=True)[:1000]
subgraph = G.subgraph(top_nodes)
if len(subgraph) > 0:
    betweenness = nx.betweenness_centrality(subgraph, k=min(100, len(subgraph)))
else:
    betweenness = {}

# PageRank
print("   Computing PageRank...")
try:
    pagerank = nx.pagerank(G, weight='weight')
except:
    pagerank = {n: 1.0/len(G) for n in G.nodes()}

# Community detection (Louvain-like)
print("   Detecting communities...")
try:
    from networkx.algorithms.community import greedy_modularity_communities
    communities = list(greedy_modularity_communities(G))
    print(f"   ✓ Found {len(communities)} communities")

    # Assign community labels
    for i, comm in enumerate(communities):
        for node in comm:
            G.nodes[node]['community'] = i
except:
    print("   ⚠️ Community detection skipped")
    communities = []

# Update node attributes
for node in G.nodes():
    G.nodes[node]['degree_centrality'] = degree_centrality.get(node, 0)
    G.nodes[node]['betweenness'] = betweenness.get(node, 0)
    G.nodes[node]['pagerank'] = pagerank.get(node, 0)

# ============================================================
# STEP 4: Analyze top entities
# ============================================================

print("\n🏆 Step 4: Top entities analysis...")
print("-" * 70)

# Top by different metrics
metrics = ['mentions', 'degree_centrality', 'pagerank', 'betweenness']

for metric in metrics:
    print(f"\n   Top 10 by {metric}:")
    if metric in ['degree_centrality', 'pagerank', 'betweenness']:
        sorted_nodes = sorted(G.nodes(), key=lambda x: G.nodes[x].get(metric, 0), reverse=True)[:10]
        for i, node in enumerate(sorted_nodes, 1):
            value = G.nodes[node].get(metric, 0)
            print(f"      {i}. {node:30s} - {value:.4f}")
    else:
        sorted_nodes = sorted(G.nodes(), key=lambda x: G.nodes[x].get(metric, 0), reverse=True)[:10]
        for i, node in enumerate(sorted_nodes, 1):
            value = G.nodes[node].get(metric, 0)
            print(f"      {i}. {node:30s} - {value:,}")

# ============================================================
# STEP 5: Create time-series graph features
# ============================================================

print("\n⏰ Step 5: Creating time-series graph features...")
print("-" * 70)

# Group events by date
events_df = pd.DataFrame(events)
events_df['date'] = pd.to_datetime(events_df['timestamp']).dt.date

# Create daily graph snapshots
print("   Aggregating daily graph features...")

mentions_df = pd.DataFrame(entity_mentions)
mentions_df['date'] = events_df.set_index('id').loc[mentions_df['event_id'], 'timestamp'].dt.date.values

daily_features = []

for date in tqdm(sorted(mentions_df['date'].unique()), desc="   Daily features"):
    day_mentions = mentions_df[mentions_df['date'] == date]
    day_events = events_df[events_df['date'] == date]

    # Entity activity
    active_entities = day_mentions['entity_name'].nunique()
    total_mentions = len(day_mentions)

    # Sentiment
    avg_sentiment = day_mentions['sentiment'].mean() if len(day_mentions) > 0 else 0
    sentiment_std = day_mentions['sentiment'].std() if len(day_mentions) > 1 else 0

    # Impact
    avg_impact = day_mentions['impact'].mean() if len(day_mentions) > 0 else 0
    total_impact = day_mentions['impact'].sum()

    # Category breakdown
    category_counts = day_mentions['category'].value_counts().to_dict()

    # Top entities for the day
    top_entities_day = day_mentions['entity_name'].value_counts().head(5).to_dict()

    # Social metrics
    total_views = day_events['views'].sum()
    total_forwards = day_events['forwards'].sum()

    # Entity type breakdown
    type_counts = day_mentions['type'].value_counts().to_dict()

    daily_features.append({
        'date': date,
        # Activity
        'active_entities': active_entities,
        'total_mentions': total_mentions,
        'num_events': len(day_events),
        # Sentiment
        'avg_sentiment': avg_sentiment,
        'sentiment_std': sentiment_std,
        'sentiment_positive_ratio': (day_mentions['sentiment'] > 0).mean() if len(day_mentions) > 0 else 0,
        'sentiment_negative_ratio': (day_mentions['sentiment'] < 0).mean() if len(day_mentions) > 0 else 0,
        # Impact
        'avg_impact': avg_impact,
        'total_impact': total_impact,
        'high_impact_ratio': (day_mentions['impact'] >= 7).mean() if len(day_mentions) > 0 else 0,
        # Social
        'total_views': total_views,
        'total_forwards': total_forwards,
        # Categories
        'mentions_market': category_counts.get('MARKET', 0),
        'mentions_regulation': category_counts.get('REGULATION', 0),
        'mentions_hack': category_counts.get('HACK', 0),
        'mentions_technical': category_counts.get('TECHNICAL', 0),
        'mentions_community': category_counts.get('COMMUNITY', 0),
        # Entity types
        'mentions_token': type_counts.get('TOKEN', 0),
        'mentions_protocol': type_counts.get('PROTOCOL', 0),
        'mentions_cex': type_counts.get('CEX', 0),
        'mentions_chain': type_counts.get('CHAIN', 0),
    })

daily_df = pd.DataFrame(daily_features)
daily_df['date'] = pd.to_datetime(daily_df['date'])
daily_df = daily_df.set_index('date').sort_index()

print(f"   ✓ Daily features: {len(daily_df)} days, {len(daily_df.columns)} features")

# ============================================================
# STEP 6: Save everything
# ============================================================

print("\n💾 Step 6: Saving data...")
print("-" * 70)

# Save graph
graph_file = DATA_DIR / 'event_graph.graphml'
nx.write_graphml(G, graph_file)
print(f"   ✓ Graph saved: {graph_file.name} ({graph_file.stat().st_size / 1024 / 1024:.1f} MB)")

# Save as pickle for faster loading
pickle_file = DATA_DIR / 'event_graph.pkl'
with open(pickle_file, 'wb') as f:
    pickle.dump(G, f)
print(f"   ✓ Graph pickle: {pickle_file.name}")

# Save daily features
features_file = DATA_DIR / 'graph_daily_features.csv'
daily_df.to_csv(features_file)
print(f"   ✓ Daily features: {features_file.name}")

# Save graph statistics
stats = {
    'num_nodes': G.number_of_nodes(),
    'num_edges': G.number_of_edges(),
    'density': nx.density(G),
    'num_communities': len(communities) if communities else 0,
    'avg_degree': sum(dict(G.degree()).values()) / G.number_of_nodes() if G.number_of_nodes() > 0 else 0,
    'top_entities_by_pagerank': {n: float(G.nodes[n]['pagerank']) for n in sorted(G.nodes(), key=lambda x: G.nodes[x].get('pagerank', 0), reverse=True)[:20]},
    'top_entities_by_degree': {n: float(G.nodes[n]['degree_centrality']) for n in sorted(G.nodes(), key=lambda x: G.nodes[x].get('degree_centrality', 0), reverse=True)[:20]},
}

stats_file = DATA_DIR / 'event_graph_stats.json'
with open(stats_file, 'w') as f:
    json.dump(stats, f, indent=2)
print(f"   ✓ Graph stats: {stats_file.name}")

# Save events and mentions for Neo4j import later
export_data = {
    'entities': [{
        'name': n,
        **{k: v for k, v in G.nodes[n].items() if k != 'node_type'}
    } for n in G.nodes()],
    'edges': [{
        'source': e[0],
        'target': e[1],
        **G.edges[e]
    } for e in G.edges()]
}

export_file = DATA_DIR / 'event_graph_export.json'
with open(export_file, 'w') as f:
    json.dump(export_data, f, default=str)
print(f"   ✓ Export data: {export_file.name}")

# ============================================================
# Summary
# ============================================================

print("\n" + "=" * 70)
print("EVENT KNOWLEDGE GRAPH COMPLETE")
print("=" * 70)

print(f"\n📊 Graph Statistics:")
print(f"   • Nodes (entities): {G.number_of_nodes():,}")
print(f"   • Edges (co-occurrences): {G.number_of_edges():,}")
print(f"   • Density: {nx.density(G):.6f}")
print(f"   • Communities: {len(communities) if communities else 'N/A'}")
print(f"   • Avg degree: {stats['avg_degree']:.2f}")

print(f"\n📈 Daily Features:")
print(f"   • Days covered: {len(daily_df)}")
print(f"   • Features per day: {len(daily_df.columns)}")
print(f"   • Date range: {daily_df.index.min()} → {daily_df.index.max()}")

print(f"\n📁 Files created:")
print(f"   • event_graph.graphml - Full graph (importable to Neo4j/Gephi)")
print(f"   • event_graph.pkl - Python pickle for fast loading")
print(f"   • graph_daily_features.csv - Time-series features")
print(f"   • event_graph_stats.json - Graph statistics")
print(f"   • event_graph_export.json - Export for Neo4j import")

print("\n📄 Next steps:")
print("   1. python3 scripts/27_merge_features.py")
print("   2. python3 scripts/28_train_spike_predictor.py")
print("\n   Or import to Neo4j:")
print("   docker run -d -p 7687:7687 -p 7474:7474 -e NEO4J_AUTH=neo4j/password neo4j")
print("   python3 scripts/26_build_event_graph_neo4j.py")
