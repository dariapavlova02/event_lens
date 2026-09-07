#!/usr/bin/env python3
"""
Script 33: Train Temporal GNN (GCN-LSTM) for Volatility Prediction
==================================================================
Uses PyTorch Geometric to train a GNN on hourly graph snapshots.
Architecture: T-GCN (GCN for spatial, LSTM for temporal).

Input: Sticky Graph (Top 1000 entities) + Dynamic Node Features (Count, Impact)
Target: Log Volatility (next 24h)
Baseline: Standard LSTM (Features only, no graph structure)
"""

import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
import torch.nn as nn
import pickle
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

DATA_DIR = Path('/Users/dariapavlova/Documents/lynoxis/telegram-btc-sentiment/data/gnn_dataset')
RESULTS_DIR = (Path(__file__).resolve().parents[1] / 'results')

print("="*70)
print("TRAINING TEMPORAL GNN (GCN-LSTM)")
print("="*70)

# Check Device
device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
print(f"Device: {device}")

# 1. Load Data
# --------------------
print("Loading dataset...")
with open(DATA_DIR / 'gnn_data_top1000.pkl', 'rb') as f:
    data_pkg = pickle.load(f)

# Data shapes:
# X: [Time, Nodes, Features] -> (17316, 1000, 2)
# y: [Time] -> (17316,)
# edge_index: [2, NumEdges]

X_all = data_pkg['X']
y_all = data_pkg['y']
edge_index = torch.tensor(data_pkg['edge_index'], dtype=torch.long).to(device)
edge_weight = torch.tensor(data_pkg['edge_weight'], dtype=torch.float32).to(device)

print(f"Data Loaded: {X_all.shape}")

# Preprocessing
# Log transform features (Count and Impact are power-law distributed)
X_all = np.log1p(X_all)

# Normalize
# We fit scaler on first 80% (Train)
split_idx = int(len(X_all) * 0.8)

# Flatten for scaling
num_time, num_nodes, num_feats = X_all.shape
X_flat = X_all.reshape(-1, num_feats)

scaler = StandardScaler()
X_flat_train = X_flat[:split_idx * num_nodes]
scaler.fit(X_flat_train)

X_scaled = scaler.transform(X_flat).reshape(num_time, num_nodes, num_feats)

# Convert to Tensor
X_tensor = torch.tensor(X_scaled, dtype=torch.float32).to(device)
y_tensor = torch.tensor(y_all, dtype=torch.float32).to(device)

# Prepare Sliding Windows for LSTM
# Sequence Length = 24 (Lookback 24h)
SEQ_LEN = 24

def create_sequences(X, y, seq_len):
    xs, ys = [], []
    for i in range(len(X) - seq_len):
        xs.append(X[i:i+seq_len])
        ys.append(y[i+seq_len])
    return torch.stack(xs), torch.stack(ys)

print("Creating Sequences (24h lookback)...")
# Note: Creating full tensor might be heavy on RAM.
# GCN output is smaller, so we should run GCN first timeframe-by-timeframe?
# No, let's process batches.

# 2. Models
# --------------------

class TGCN(torch.nn.Module):
    def __init__(self, num_nodes, in_channels, hidden_dim, out_dim):
        super(TGCN, self).__init__()
        self.num_nodes = num_nodes
        self.hidden_dim = hidden_dim

        # Spatial: GCN
        self.gcn = GCNConv(in_channels, hidden_dim)

        # Temporal: LSTM
        # Input to LSTM is [Batch, Seq, Nodes * Hidden] -> Too big!
        # Better: GCN -> GlobalPooling -> LSTM
        # Or: GCN -> LSTM per Node -> Pooling?
        # Standard T-GCN: GCN produces H_t [Nodes, Hidden].
        # We flatten to [Nodes * Hidden]? Or AvgPool to [Hidden]?

        # Let's use Global Mean Pooling after GCN to get "Graph Embedding" per step.
        self.lstm = nn.LSTM(hidden_dim, 64, batch_first=True)
        self.fc = nn.Linear(64, 1) # Regress Volatility (Scalar)

    def forward(self, x_seq, edge_index, edge_weight):
        # x_seq: [Batch, SeqLen, Nodes, Feats]
        batch_size, seq_len, nodes, feats = x_seq.size()

        # Flatten Batch and Seq for GCN processing
        # x_flat: [Batch*Seq, Nodes, Feats]

        # We process each timestep independently through GCN
        # But GCN expects [TotalNodes, Feats]
        # We can stack graphs or loop roughly.
        # Loop is slow. Batching graphs is better but complex manually.

        # Simple Loop over SeqLen (since SeqLen=24 is small)

        embeddings_seq = []

        for t in range(seq_len):
            xt = x_seq[:, t, :, :] # [Batch, Nodes, Feats]

            # Since GCNConv expects [Total_Nodes, Feats], we need to handle batching carefully.
            # But here EdgeIndex is STATIC.
            # We can treat Batch as separate graphs with same adjacency? No.
            # Simplification: Process 1 sample at a time? Too slow.

            # Efficient: Use Shared Weights GCN: H = A X W
            # X is [Batch, Nodes, Feats]. W is [Feats, Hidden].
            # XW is [Batch, Nodes, Hidden].
            # A (Adj) is [Nodes, Nodes].
            # AXW is [Batch, Nodes, Hidden] (Broadcast matmul).
            # This avoids PyG overhead for static/dense graphs.
            # But edge_index is sparse.

            # Let's trust PyG GCNConv to handle shape [Nodes, Feat] standardly.
            # We will reshape [Batch, Nodes, Feats] -> [Batch*Nodes, Feats]?
            # And repeat EdgeIndex? That's heavy.

            # Manual GCN (A X W) for static sparse graph + batched X is hard.
            # Let's just do Mean Pooling of Features first? (Baseline)
            # No, GCN is the point.

            # Let's iterate batch size? No.

            # Let's just Loop over Sequence (24), and inside, reshape Batch:
            # But X differs per batch item.

            # OK, TRADITIONAL T-GCN approach:
            # Run GCN on "one big graph" consisting of Batch_Size disconnected copies?
            # Or just implementation trick:
            # H = GCN(X)
            # Since A is shared, H = A * (XW).
            # PyG GCNConv supports X as [Nodes, In] only usually.

            # Let's do a naive manual GCN here for speed/simplicity with Batches.
            # H = ReLU( A * X * W )
            # We will approximate A * X using PyG only if X is single graph.
            pass

            # Let's fallback to "Avg Pool First" for Baseline, and "GCN" for Real.
            # Actually, let's just loop batch for now. It's only 32 size.
            pass

        # For simplicity in this script, we will PRE-PROCESS the entire dataset through GCN
        # into Embeddings [Time, Hidden], THEN feed to LSTM.
        # This is strictly correct because GCN part is time-independent (Static Weights)
        # and Graph is Static Structure.
        # So: GCN(X_all) -> H_all [Time, Hidden].
        # Then LSTM(H_all sequences).
        # MUCH FASTER!
        return None

# 3. Optimized Workflow
# ---------------------
# Step A: Train GCN Autoencoder? Or Train End-to-End?
# End-to-End requires backprop through GCN for 17k steps? No.
# We will use "Windowed Training".

# Architecture:
# 1. GCN Layer (Shared)
# 2. Global Mean Pool -> Graph Vector
# 3. LSTM
# 4. FC

class ModularTGCN(nn.Module):
    def __init__(self, in_channels, hidden_gcn, hidden_lstm):
        super().__init__()
        self.gcn = GCNConv(in_channels, hidden_gcn)
        self.lstm = nn.LSTM(hidden_gcn, hidden_lstm, batch_first=True)
        self.head = nn.Linear(hidden_lstm, 1)

    def get_graph_embeddings(self, x, edge_index, edge_weight):
        # x: [Nodes, Feats] -> returns [Hidden] (Pooled)
        h = F.relu(self.gcn(x, edge_index, edge_weight))
        return torch.mean(h, dim=0) # Global Mean Pool

    def forward(self, x_window, edge_index, edge_weight):
        # x_window: [Batch, Seq, Nodes, Feats]
        # This is slow.
        pass

# VECTORIZED APPROACH:
# 1. Pass ALL X_all [17k, Nodes, 2] through GCN -> H_all [17k, Hidden_GCN]
# 2. Sequence H_all -> LSTM -> y

print("Initializing Fast T-GCN...")

class FastTGCN(nn.Module):
    def __init__(self, in_dim, gcn_dim, lstm_dim):
        super().__init__()
        self.gcn = GCNConv(in_dim, gcn_dim)
        self.lstm = nn.LSTM(gcn_dim, lstm_dim, batch_first=True)
        self.fc = nn.Linear(lstm_dim, 1)

    def forward_gcn(self, x_all, edge_index, edge_weight):
        # x_all: [Time, Nodes, Feats] -> Flatten to [Time*Nodes, Feats]
        T, N, F_in = x_all.shape
        x_flat = x_all.view(T*N, F_in)

        # We need edge_index for T graphs... but that's huge.
        # Trick: GCNConv usually just does MatMul.
        # If we assume shared weights, we can just process [N, F] -> [N, H] for each T?
        # Yes.

        # Actually, let's just learn a GCN on "Single Snapshot" logic,
        # but we need to train it end-to-end.

        # We will loop T in `forward_gcn`? No, too slow.

        # Ok, simplest approach for "Proof of Concept":
        # Just use Node Aggregated Features (Baseline) vs GCN Aggregated.
        # Since we can't easily batch GCN with static graph in PyG without repeating edges.
        pass

# Let's implement a custom simplified GCN layer that supports [Batch, Nodes, Feats] input
class BatchGCN(nn.Module):
    def __init__(self, in_dim, out_dim, num_nodes, edge_index, edge_weight):
        super().__init__()
        self.weight = nn.Parameter(torch.Tensor(in_dim, out_dim))
        nn.init.xavier_uniform_(self.weight)

        # Pre-compute Laplacian / Adjacency
        # A_hat = D^-0.5 A D^-0.5
        # We do this once!
        self.num_nodes = num_nodes

        # Create dense adjacency for matmul speed (1000 nodes is fine)
        # Or sparse mm.
        import torch_geometric.utils as utils
        adj = utils.to_dense_adj(edge_index, edge_attr=edge_weight, max_num_nodes=num_nodes)[0] # [N, N]
        # Add self loops
        adj = adj + torch.eye(num_nodes).to(adj.device)

        # Norm
        deg = adj.sum(dim=1)
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
        norm = deg_inv_sqrt.view(-1, 1) * adj * deg_inv_sqrt.view(1, -1)

        self.register_buffer('adj_norm', norm) # [N, N]

    def forward(self, x):
        # x: [Batch, N, In]
        # H = A * X * W

        # XW: [Batch, N, Out]
        out = torch.matmul(x, self.weight)

        # A * XW: [N, N] * [Batch, N, Out]
        # We need: for each b in Batch: result[b] = A @ out[b]
        # Einstein sum: "nm, bmo -> bno"
        out = torch.einsum('nm, bmo -> bno', self.adj_norm, out)

        return F.relu(out)

# Define full model
class CryptoGNN(nn.Module):
    def __init__(self, num_nodes, edge_index, edge_weight):
        super().__init__()
        self.gcn = BatchGCN(2, 16, num_nodes, edge_index, edge_weight)
        # Pooling: Mean
        self.lstm = nn.LSTM(16, 32, batch_first=True)
        self.fc = nn.Linear(32, 1)

    def forward(self, x):
        # x: [Batch, Seq, N, F]
        b, s, n, f = x.shape
        x_flat = x.view(b*s, n, f)

        # GCN
        h = self.gcn(x_flat) # [Batch*Seq, N, 16]

        # Pool (Global Mean) -> [Batch*Seq, 16]
        h_pool = torch.mean(h, dim=1)

        # Reshape for LSTM: [Batch, Seq, 16]
        h_seq = h_pool.view(b, s, -1)

        # LSTM
        lstm_out, _ = self.lstm(h_seq)
        last_out = lstm_out[:, -1, :] # [Batch, 32]

        return self.fc(last_out).squeeze()

# 4. Train Loop
# -------------
model = CryptoGNN(1000, edge_index, edge_weight).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.005)
criterion = nn.MSELoss()

# Create Sequences
X_seq, y_seq = create_sequences(X_tensor, y_tensor, SEQ_LEN)
print(f"Sequences: {X_seq.shape}")

# Split
train_size = int(len(X_seq) * 0.8)
train_X, test_X = X_seq[:train_size], X_seq[train_size:]
train_y, test_y = y_seq[:train_size], y_seq[train_size:]

train_dataset = torch.utils.data.TensorDataset(train_X, train_y)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)

test_dataset = torch.utils.data.TensorDataset(test_X, test_y)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=64, shuffle=False)

print("\nStarting Training (10 Epochs)...")
history = {'loss': [], 'val_loss': []}

for epoch in range(10):
    model.train()
    total_loss = 0
    for bx, by in tqdm(train_loader, desc=f"Epoch {epoch+1}"):
        bx, by = bx.to(device), by.to(device)

        optimizer.zero_grad()
        pred = model(bx)
        loss = criterion(pred, by)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / len(train_loader)

    # Val
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for bx, by in test_loader:
            bx, by = bx.to(device), by.to(device)
            pred = model(bx)
            loss = criterion(pred, by)
            val_loss += loss.item()
    avg_val = val_loss / len(test_loader)

    print(f"Epoch {epoch+1}: Train Loss {avg_loss:.6f} | Val Loss {avg_val:.6f}")
    history['loss'].append(avg_loss)
    history['val_loss'].append(avg_val)

# 5. Plot Results
# ---------------
plt.figure(figsize=(10, 5))
plt.plot(history['loss'], label='Train MSE')
plt.plot(history['val_loss'], label='Val MSE')
plt.title('GCN-LSTM Training Loss')
plt.legend()
plt.savefig(RESULTS_DIR / 'gnn_training_loss.png')

print(f"\nFinal Val MSE: {history['val_loss'][-1]:.6f}")
print("Done. Saved loss plot.")
