#!/usr/bin/env python3
"""
Script 35: Prepare Dynamic GNN Dataset (SotA)
=============================================
Constructs a sequence of hourly graph snapshots with:
1. Dynamic Topology: Edges exist only if co-occurrence happened in [t-24, t].
2. Semantic Node Features: Aggregated BERT embeddings of messages in [t-24, t].
3. No Leakage: Strictly causal windowing.

Output: data/gnn_dataset_dynamic/processed/data_{i}.pt
"""

import os
import pandas as pd
import numpy as np
import yfinance as yf
from neo4j import GraphDatabase
import pickle
from pathlib import Path
from tqdm import tqdm
from datetime import timedelta
import torch
from torch_geometric.data import Data
from sklearn.decomposition import PCA
import shutil

NEO4J_URI = "neo4j://127.0.0.1:7687"
NEO4J_AUTH = (os.getenv("NEO4J_USER", "neo4j"), os.environ["NEO4J_PASSWORD"])

# Output Dir
DATA_DIR = (Path(__file__).resolve().parents[1] / 'data')
RAW_DIR = DATA_DIR / 'gnn_dataset_dynamic/raw'
PROCESSED_DIR = DATA_DIR / 'gnn_dataset_dynamic/processed'

# Clean previous
if PROCESSED_DIR.exists():
    shutil.rmtree(PROCESSED_DIR)
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
RAW_DIR.mkdir(parents=True, exist_ok=True)

print("="*70)
print("PREPARING DYNAMIC GNN DATASET (Leakage-Free)")
print("="*70)

# 1. Load Embeddings
# ------------------
print("\n1. Loading Message Embeddings...")
EMB_PATH = DATA_DIR / 'message_embeddings.pkl'
with open(EMB_PATH, 'rb') as f:
    emb_map = pickle.load(f) # Key: (timestamp_iso, channel)

# Fit PCA on a subset of embeddings to reduce dim 384 -> 32
print("   Fitting PCA (384 -> 32)...")
sample_embs = []
for k, v in list(emb_map.items())[:5000]: # Take sample
    sample_embs.append(v)
pca = PCA(n_components=32)
pca.fit(np.stack(sample_embs))
print(f"   PCA Explained Variance: {sum(pca.explained_variance_ratio_):.2f}")

# 2. Load 1h Market Data (Target)
# -------------------------------
print("\n2. Loading Market Data...")
end_date = pd.Timestamp.now(tz='UTC')
start_date = end_date - timedelta(days=729) # Max history

btc = yf.download("BTC-USD", start=start_date, end=end_date, interval="1h", progress=False)
if isinstance(btc.columns, pd.MultiIndex):
    btc = btc.xs('BTC-USD', level=1, axis=1)

# Target: Next Hour STRICT Absolute Return
btc['Log_Returns'] = np.log(btc['Close'] / btc['Close'].shift(1))
# Target = |LogReturn_{t+1}|
btc['Target_Abs'] = btc['Log_Returns'].shift(-1).abs()
btc['Target_Dir'] = np.sign(btc['Log_Returns'].shift(-1)) # Optional: Direction
btc.dropna(inplace=True)

print(f"   Time Range: {btc.index.min()} -> {btc.index.max()}")
print(f"   Samples: {len(btc)}")

# 3. Load Events from Neo4j
# -------------------------
print("\n3. Fetching Graph Events...")
driver = GraphDatabase.driver(NEO4J_URI, auth=NEO4J_AUTH)
query = """
MATCH (e:Event)-[:MENTIONS]->(ent:Entity)
WHERE e.timestamp >= datetime($start) AND e.timestamp <= datetime($end)
RETURN e.timestamp as time, e.channel as channel, ent.name as entity, e.impact as impact
"""

with driver.session() as session:
    result = session.run(query, start=btc.index.min().isoformat(), end=btc.index.max().isoformat())
    # Convert and create lookup key
    data = []
    for r in result:
        iso_time = r['time'].iso_format()
        key = (iso_time, r['channel'])

        # Check if we have embedding
        if key in emb_map:
            # Temporarily store PCA embedding here?
            # Or transform later. Transform later is safer for memory if we store vector.
            # Actually let's store index to emb_map or something.
            pass

        data.append({
            'time': pd.Timestamp(iso_time),
            'key': key, # To lookup embedding
            'entity': r['entity'],
            'impact': r['impact']
        })

df_events = pd.DataFrame(data)
df_events['time'] = pd.to_datetime(df_events['time'], utc=True)
driver.close()

# Filter Top 1000 Entities
top_entities = df_events['entity'].value_counts().head(1000).index
top_entity_map = {name: i for i, name in enumerate(top_entities)}
print(f"   Filtered to {len(top_entities)} top entities.")
df_events = df_events[df_events['entity'].isin(top_entities)]

# Pre-transform embeddings to PCA for all used keys to save time
print("   Transforming embeddings via PCA...")
unique_keys = df_events['key'].unique()
key_to_pca = {}
# Batch transform
batch_keys = []
batch_vecs = []
for k in unique_keys:
    if k in emb_map:
        batch_keys.append(k)
        batch_vecs.append(emb_map[k])

    if len(batch_keys) >= 10000:
        vecs_pca = pca.transform(np.stack(batch_vecs))
        for bk, bp in zip(batch_keys, vecs_pca):
            key_to_pca[bk] = bp
        batch_keys, batch_vecs = [], []

if batch_keys:
    vecs_pca = pca.transform(np.stack(batch_vecs))
    for bk, bp in zip(batch_keys, vecs_pca):
        key_to_pca[bk] = bp

print("   Embeddings ready.")

# 4. Generate Snapshots
# ---------------------
print("\n4. Generating Dynamic Snapshots...")
# Sort events
df_events = df_events.sort_values('time')
event_times = df_events['time'].values
event_entities = df_events['entity'].map(top_entity_map).values
event_impacts = df_events['impact'].fillna(1.0).values
event_keys = df_events['key'].values

start_ptr = 0
end_ptr = 0

count_saved = 0

for i, current_time in enumerate(tqdm(btc.index.values)):
    # Window [t-24h, t]
    # current_time is numpy.datetime64
    window_start = current_time - np.timedelta64(24, 'h')

    # Update pointers
    while start_ptr < len(event_times) and event_times[start_ptr] < window_start:
        start_ptr += 1
    while end_ptr < len(event_times) and event_times[end_ptr] <= current_time:
        end_ptr += 1

    # Active slice
    sl_entities = event_entities[start_ptr:end_ptr]
    sl_impacts = event_impacts[start_ptr:end_ptr]
    sl_keys = event_keys[start_ptr:end_ptr]

    # 1. Node Features (Aggregated)
    # [Count, Sum_Impact, PCA_Web_0...31] -> 34 features
    node_feats = np.zeros((1000, 34), dtype=np.float32)

    if len(sl_entities) > 0:
        # Vectorized accumulation
        # Count
        np.add.at(node_feats[:, 0], sl_entities, 1)
        # Impact
        np.add.at(node_feats[:, 1], sl_entities, sl_impacts)

        # Semantic Embeddings (Mean Pooling per Entity)
        # This is harder to vectorize fully without loop or huge sparse mat.
        # Loop over unique active entities in slice is fast enough (max 1000).

        # Group by entity
        # Using a temporary DataFrame for this slice aggregation is roughly OK
        # or manual loop.

        # Let's use a dict for summation then average
        temp_emb_sum = {} # entity_id -> sum_vec
        temp_emb_count = {} # entity_id -> count

        for ent_id, k in zip(sl_entities, sl_keys):
            if k in key_to_pca:
                vec = key_to_pca[k]
                if ent_id not in temp_emb_sum:
                    temp_emb_sum[ent_id] = vec.copy()
                    temp_emb_count[ent_id] = 1
                else:
                    temp_emb_sum[ent_id] += vec
                    temp_emb_count[ent_id] += 1

        # Assign to matrix
        for ent_id, s_vec in temp_emb_sum.items():
            cnt = temp_emb_count[ent_id]
            node_feats[ent_id, 2:] = s_vec / cnt # Mean pooling

    # 2. Edges (Co-occurrence in this window)
    # We need to find which entities co-occurred in SAME event (same key).
    # Since we flattened by entity, we need to regroup by key.
    # Logic: Get unique keys in slice. For each key, get entities. Add clique.

    # Re-filtering df slice is slow.
    # Use pre-computed "Event -> Entities" map?
    # Or just iterate slice?

    # Fast approach:
    # Pre-compute "EventID -> [EntID, EntID]" globally?
    # Yes.

    # But let's build edge_index naively for now.
    # Group entities by key in the slice.

    # Optimization: Dictionary {key -> [ent_ids]} constructed during slice iteration
    key_to_ents = {}
    for ent_id, k in zip(sl_entities, sl_keys):
        if k not in key_to_ents: key_to_ents[k] = []
        key_to_ents[k].append(ent_id)

    edges = set()
    for k, ents in key_to_ents.items():
        if len(ents) > 1:
            ents = sorted(list(set(ents)))
            for idx_a in range(len(ents)):
                for idx_b in range(idx_a+1, len(ents)):
                    edges.add((ents[idx_a], ents[idx_b]))
                    edges.add((ents[idx_b], ents[idx_a])) # Undirected

    if not edges:
        edge_index = torch.empty((2, 0), dtype=torch.long)
    else:
        edge_index = torch.tensor(list(edges), dtype=torch.long).t()

    # Construct Data object
    x = torch.tensor(node_feats, dtype=torch.float32)
    y = torch.tensor([btc['Target_Abs'].iloc[i]], dtype=torch.float32)
    # Save exact timestamp as string or int to verify alignment
    ts = int(current_time.astype('datetime64[s]').astype('int'))

    data_obj = Data(
        x=x,
        edge_index=edge_index,
        y=y,
        timestamp=ts # Explicit timestamp
    )

    # Save
    torch.save(data_obj, PROCESSED_DIR / f'data_{count_saved}.pt')
    count_saved += 1

print(f"\n✅ Saved {count_saved} dynamic snapshots to {PROCESSED_DIR}")
