#!/usr/bin/env python3
"""
Script 47: Train GNN with Strict Methodology (Honest Target)
============================================================
Training GATv2-LSTM on 'Absolute Next-Hour Return' to eliminate
autocorrelation leakage found in 'Rolling Volatility' targets.

Methodology:
1.  **Strict Target**: |LogReturn(t+1)|.
2.  **Dataset**: Reuses 'data/gnn_dataset_dynamic/processed' (structure/features),
    but overwrites 'y' target in-memory.
3.  **Validation**: 5-Fold Walk-Forward TimeSeriesSplit.
4.  **No Leakage**: Scaler fit on Train only. Input features (embeddings) are strictly past [t-24, t].

Comparison:
- We compute R² against a "Naive Baseline" (Current Absolute Return).
- If GNN R² > Baseline R², we have Alpha.
"""

import torch
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv, global_mean_pool
from torch_geometric.data import Batch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
import yfinance as yf
from pathlib import Path
from tqdm import tqdm
import os
import copy
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import r2_score, mean_squared_error
from datetime import timedelta

PROCESSED_DIR = (Path(__file__).resolve().parents[1] / 'data/gnn_dataset_dynamic/processed')
RESULTS_DIR = (Path(__file__).resolve().parents[1] / 'results')

# Force CPU (MPS has scatter implementation issues with PyG sometimes)
device = torch.device('cpu')

print("="*70)
print("TRAINING GNN (STRICT VALIDATION: ABSOLUTE RETURN)")
print("="*70)

# 1. Load Snapshots
# -----------------
print("1. Loading snapshots...", flush=True)
data_files = sorted([f for f in os.listdir(PROCESSED_DIR) if f.startswith('data_') and f.endswith('.pt')],
                    key=lambda x: int(x.split('_')[1].split('.')[0]))

if not data_files:
    print("❌ No processed data found. Run Script 35 first.")
    exit(1)

full_data_list = []
for f in tqdm(data_files, desc="Loading PT files"):
    d = torch.load(PROCESSED_DIR / f, map_location='cpu', weights_only=False)
    # Ensure y is float
    d.y = d.y.float()
    full_data_list.append(d)

print(f"   Loaded {len(full_data_list)} snapshots.")

# 2. Prepare Targets and Baselines
# --------------------------------
print("\n2. Preparing Baselines (Shifted Y)...", flush=True)

# The dataset now contains the CORRECT Strict Target in d.y = |r_{t+1}|
# We don't need to overwrite it.
# Baseline: Persistence means predicting |r_{t+1}| using |r_t|.
# |r_t| is simply the target of the PREVIOUS snapshot (d[t-1].y).

for i in range(len(full_data_list)):
    d = full_data_list[i]
    if i > 0:
        d.base_y = full_data_list[i-1].y.item()
    else:
        d.base_y = 0.0 # First sample has no history

# 3. Model & Loop Definitions
# ---------------------------
SEQ_LEN = 24
BATCH_SIZE = 16

class TemporalGraphDataset(Dataset):
    def __init__(self, data_list, seq_len):
        self.data_list = data_list
        self.seq_len = seq_len
    def __len__(self):
        return len(self.data_list) - self.seq_len
    def __getitem__(self, idx):
        # Sequence of 24 graphs
        graphs = self.data_list[idx : idx + self.seq_len]
        # Target is associated with the LAST graph in the sequence (time t)
        # predicting event at t+1
        target = graphs[-1].y
        base = graphs[-1].base_y
        return graphs, target, base

def vectorized_collate(batch):
    all_graphs_flat = []
    targets = []
    bases = []
    for (graph_seq, target, base) in batch:
        all_graphs_flat.extend(graph_seq)
        targets.append(target)
        bases.append(base)

    giant_batch = Batch.from_data_list(all_graphs_flat)
    targets = torch.stack(targets).squeeze()
    bases = torch.tensor(bases, dtype=torch.float32)
    return giant_batch, targets, bases

class DynamicGAT(torch.nn.Module):
    def __init__(self, in_features, hidden_dim, heads=2, dropout=0.3):
        super().__init__()
        self.dropout = dropout
        self.gat = GATv2Conv(in_features, hidden_dim, heads=heads, concat=True, dropout=dropout)
        dim_after_gat = hidden_dim * heads
        # LSTM to process sequence of graph embeddings
        self.lstm = torch.nn.LSTM(dim_after_gat, 64, num_layers=1, batch_first=True)
        self.fc = torch.nn.Sequential(
            torch.nn.Linear(64, 32),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(32, 1) # Predict scalar |r|
        )

    def forward(self, giant_batch):
        x, edge_index, batch_vec = giant_batch.x, giant_batch.edge_index, giant_batch.batch
        # Graph Component
        out = self.gat(x, edge_index)
        out = F.relu(out)
        out = F.dropout(out, p=self.dropout, training=self.training)
        g_emb = global_mean_pool(out, batch_vec)

        # Reshape for LSTM [Batch, Seq, Feat]
        total_graphs = g_emb.size(0)
        curr_batch_size = total_graphs // SEQ_LEN
        lstm_input = g_emb.view(curr_batch_size, SEQ_LEN, -1)

        # Temporal Component
        lstm_out, _ = self.lstm(lstm_input)
        final_h = lstm_out[:, -1, :] # Last hidden state

        return self.fc(final_h).squeeze()

def apply_scaling(data_list, scaler):
    scaled_list = []
    for d in data_list:
        d_new = d.clone()
        x_np = d.x.numpy()
        x_scaled = scaler.transform(x_np)
        d_new.x = torch.tensor(x_scaled, dtype=torch.float32)
        scaled_list.append(d_new)
    return scaled_list

# 4. Walk-Forward Execution
# -------------------------
tscv = TimeSeriesSplit(n_splits=5)
indices = np.arange(len(full_data_list))

fold_results = []
print(f"\nStarting 5-Fold Walk-Forward CV...", flush=True)

for fold_idx, (train_val_idx, test_idx) in enumerate(tscv.split(indices)):
    print(f"\n>>> FOLD {fold_idx+1}", flush=True)

    # Internal Train/Val split
    n_total = len(train_val_idx)
    n_val = int(n_total * 0.15)
    train_idx = train_val_idx[:-n_val]
    val_idx = train_val_idx[-n_val:]

    # Scaling
    raw_train = [full_data_list[i] for i in train_idx]
    train_x_all = torch.cat([d.x for d in raw_train], dim=0).numpy()
    scaler = StandardScaler()
    scaler.fit(train_x_all)

    train_ds = apply_scaling(raw_train, scaler)
    val_ds = apply_scaling([full_data_list[i] for i in val_idx], scaler)
    test_ds = apply_scaling([full_data_list[i] for i in test_idx], scaler)

    train_loader = DataLoader(TemporalGraphDataset(train_ds, SEQ_LEN), batch_size=BATCH_SIZE, shuffle=True, collate_fn=vectorized_collate)
    val_loader = DataLoader(TemporalGraphDataset(val_ds, SEQ_LEN), batch_size=BATCH_SIZE, collate_fn=vectorized_collate)
    test_loader = DataLoader(TemporalGraphDataset(test_ds, SEQ_LEN), batch_size=BATCH_SIZE, collate_fn=vectorized_collate)

    # Init Model
    model = DynamicGAT(in_features=34, hidden_dim=16, heads=2).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = torch.nn.MSELoss()

    best_loss = float('inf')
    best_wts = None
    patience_counter = 0
    patience = 3

    # Train
    for epoch in range(15):
        model.train()
        for gb, target, _ in train_loader:
            gb, target = gb.to(device), target.to(device)
            optimizer.zero_grad()
            out = model(gb)
            loss = criterion(out, target)
            loss.backward()
            optimizer.step()

        # Val
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for gb, target, _ in val_loader:
                gb, target = gb.to(device), target.to(device)
                out = model(gb)
                val_loss += criterion(out, target).item()
        avg_val = val_loss / len(val_loader)

        if avg_val < best_loss:
            best_loss = avg_val
            best_wts = copy.deepcopy(model.state_dict())
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                break

    # Test
    model.load_state_dict(best_wts)
    model.eval()

    preds = []
    actuals = []
    baselines = []

    with torch.no_grad():
        for gb, target, base in test_loader:
            gb = gb.to(device)
            out = model(gb).view(-1) # Ensure 1D
            preds.extend(out.tolist()) # tolist is safer than numpy for extending
            actuals.extend(target.view(-1).tolist())
            baselines.extend(base.view(-1).tolist())

    mse = mean_squared_error(actuals, preds)
    r2_gnn = r2_score(actuals, preds)
    r2_base = r2_score(actuals, baselines) # Persistence R2

    print(f"    MSE: {mse:.6f}")
    print(f"    R² GNN: {r2_gnn:.4f} | R² Base: {r2_base:.4f}")

    fold_results.append({
        'fold': fold_idx + 1,
        'mse': mse,
        'r2_gnn': r2_gnn,
        'r2_base': r2_base
    })

# Summary
print("\n" + "="*50)
print("STRICT GNN RESULTS (Average over 5 Folds)")
mean_mse = np.mean([f['mse'] for f in fold_results])
mean_r2_gnn = np.mean([f['r2_gnn'] for f in fold_results])
mean_r2_base = np.mean([f['r2_base'] for f in fold_results])

print(f"MSE:       {mean_mse:.6f}")
print(f"R² GNN:    {mean_r2_gnn:.4f}")
print(f"R² Base:   {mean_r2_base:.4f}")
print(f"Lift:      {mean_r2_gnn - mean_r2_base:+.4f}")
print("="*50)
