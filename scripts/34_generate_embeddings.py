#!/usr/bin/env python3
"""
Script 34: Generate Semantic Embeddings (BERT)
==============================================
Generates sentence embeddings for all Telegram messages using `sentence-transformers`.
These embeddings will serve as rich input features for the Dynamic GNN.

Model: all-MiniLM-L6-v2 (Speed/Performance tradeoff)
Output: data/message_embeddings.pkl
Key: (timestamp_iso, channel) -> vector
"""

import pandas as pd
import pickle
from pathlib import Path
from sentence_transformers import SentenceTransformer
import warnings

# Suppress warnings
warnings.filterwarnings("ignore")

DATA_DIR = (Path(__file__).resolve().parents[1] / 'data')
INPUT_CSV = DATA_DIR / 'all_messages_tbsa.csv'
OUTPUT_PKL = DATA_DIR / 'message_embeddings.pkl'

print("="*70)
print("GENERATING SEMANTIC EMBEDDINGS (BERT)")
print("="*70)

# 1. Load Data
# --------------------
print("1. Loading raw messages...")
df = pd.read_csv(INPUT_CSV)
df['text'] = df['text'].astype(str).fillna("")
df['date'] = pd.to_datetime(df['date']).map(lambda x: x.isoformat()) # Standardize key

print(f"   Loaded {len(df)} messages.")

# 2. Load Model
# --------------------
print("\n2. Loading Model (all-MiniLM-L6-v2)...")
model = SentenceTransformer('all-MiniLM-L6-v2')
print("   Model loaded.")

# 3. Encode
# --------------------
print("\n3. Encoding messages (this may take a few minutes)...")
# Batch encode is faster
sentences = df['text'].tolist()
embeddings = model.encode(sentences, batch_size=64, show_progress_bar=True, convert_to_numpy=True)

print(f"   Generated embeddings shape: {embeddings.shape}")

# 4. Save Map
# --------------------
print("\n4. Saving embedding map...")
# Create lookup: (timestamp, channel) -> embedding
# This ensures we can map Neo4j events back to embeddings robustly
embedding_map = {}
for idx, row in df.iterrows():
    key = (row['date'], row['channel'])
    embedding_map[key] = embeddings[idx]

with open(OUTPUT_PKL, 'wb') as f:
    pickle.dump(embedding_map, f)

print(f"✅ Saved {len(embedding_map)} embeddings to {OUTPUT_PKL}")
