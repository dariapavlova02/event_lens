#!/usr/bin/env python3
"""
Script 36: Train SotA GNN (GATv2-LSTM) - Walk-Forward CV
========================================================
Upgrades the training loop to "Gold Standard" Walk-Forward Cross-Validation.
Instead of a single arbitrary split, we use 5 expanding window folds.

Methodology:
1.  **Iterate 5 Folds**:
    -   Split: Train + Val (Expanding) | Test (Next Window).
    -   Inner Split: Train (85%) | Val (15%) for Early Stopping.
2.  **No Leakage**: Scaler fit ONLY on Train for each fold.
3.  **Validation**: Early Stopping on Val set.
4.  **Reporting**: Mean MSE ± Std over 5 folds.

Architecture:
Vectorized GATv2 -> Global Pool -> LSTM (2-Layer) -> FC
"""

import torch
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv, global_mean_pool
from torch_geometric.data import Batch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm
import os
import copy
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit

PROCESSED_DIR = (Path(__file__).resolve().parents[1] / 'data/gnn_dataset_dynamic/processed')
RESULTS_DIR = (Path(__file__).resolve().parents[1] / 'results')

print("="*70)
print("TRAINING SOTA GNN (Walk-Forward CV, CPU Safe)")
print("="*70)

# FORCE CPU for stability (MPS scatter issues)
device = torch.device('cpu')
print(f"Device: {device}")

# 1. Load Data
# ------------
print("1. Loading dataset into RAM ...", flush=True)
data_files = sorted([f for f in os.listdir(PROCESSED_DIR) if f.startswith('data_') and f.endswith('.pt')],
                    key=lambda x: int(x.split('_')[1].split('.')[0]))

full_data_list = []
for f in tqdm(data_files, desc="Loading Snapshots"):
    d = torch.load(PROCESSED_DIR / f, map_location='cpu', weights_only=False)
    full_data_list.append(d)

print(f"   Loaded {len(full_data_list)} snapshots.", flush=True)

# 2. Definitions
# --------------
SEQ_LEN = 24
BATCH_SIZE = 16 # Reduce batch size for CPU

class TemporalGraphDataset(Dataset):
    def __init__(self, data_list, seq_len):
        self.data_list = data_list
        self.seq_len = seq_len

    def __len__(self):
        return len(self.data_list) - self.seq_len

    def __getitem__(self, idx):
        graphs = self.data_list[idx : idx + self.seq_len]
        target = graphs[-1].y
        return graphs, target

def vectorized_collate(batch):
    all_graphs_flat = []
    targets = []
    for (graph_seq, target) in batch:
        all_graphs_flat.extend(graph_seq)
        targets.append(target)

    giant_batch = Batch.from_data_list(all_graphs_flat)
    targets = torch.stack(targets).squeeze()
    return giant_batch, targets

class DynamicGAT(torch.nn.Module):
    def __init__(self, in_features, hidden_dim, heads=4, dropout=0.2):
        super().__init__()
        self.dropout = dropout
        self.gat = GATv2Conv(in_features, hidden_dim, heads=heads, concat=True, dropout=dropout)
        dim_after_gat = hidden_dim * heads
        # 2-Layer LSTM for proper dropout utilization
        self.lstm = torch.nn.LSTM(dim_after_gat, 128, num_layers=2, batch_first=True, dropout=dropout)
        self.fc = torch.nn.Sequential(
            torch.nn.Linear(128, 64),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(64, 1)
        )

    def forward(self, giant_batch):
        x, edge_index, batch_vec = giant_batch.x, giant_batch.edge_index, giant_batch.batch
        out = self.gat(x, edge_index)
        out = F.relu(out)
        out = F.dropout(out, p=self.dropout, training=self.training)
        g_emb = global_mean_pool(out, batch_vec)

        total_graphs = g_emb.size(0)
        curr_batch_size = total_graphs // SEQ_LEN
        lstm_input = g_emb.view(curr_batch_size, SEQ_LEN, -1)

        lstm_out, _ = self.lstm(lstm_input)
        final_h = lstm_out[:, -1, :]
        return self.fc(final_h).squeeze()

class EarlyStopping:
    def __init__(self, patience=5, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = float('inf')
        self.early_stop = False
        self.best_model_wts = None

    def __call__(self, val_loss, model):
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.best_model_wts = copy.deepcopy(model.state_dict())
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True

def apply_scaling(data_list, scaler):
    scaled_list = []
    # Avoid tqdm here for speed, just loop
    for d in data_list:
        d_new = d.clone()
        x_np = d.x.numpy()
        x_scaled = scaler.transform(x_np)
        d_new.x = torch.tensor(x_scaled, dtype=torch.float32)
        scaled_list.append(d_new)
    return scaled_list

# 3. Walk-Forward Loop
# --------------------
tscv = TimeSeriesSplit(n_splits=5)
indices = np.arange(len(full_data_list))

fold_results = []

print(f"\nStarting 5-Fold Walk-Forward CV...", flush=True)

for fold_idx, (train_val_idx, test_idx) in enumerate(tscv.split(indices)):
    print(f"\n>>> FOLD {fold_idx+1} / 5", flush=True)
    print(f"    Train_Val Size: {len(train_val_idx)} | Test Size: {len(test_idx)}", flush=True)

    n_train_val = len(train_val_idx)
    n_val = int(n_train_val * 0.15)
    n_train = n_train_val - n_val

    train_idx = train_val_idx[:n_train]
    val_idx = train_val_idx[n_train:]

    # Fit Scaler
    print("    Fitting Scaler...", flush=True)
    raw_train = [full_data_list[i] for i in train_idx]
    train_x_all = torch.cat([d.x for d in raw_train], dim=0).numpy()
    scaler = StandardScaler()
    scaler.fit(train_x_all)

    # Apply
    print("    Scaling Data...", flush=True)
    train_ds_data = apply_scaling(raw_train, scaler)
    val_ds_data = apply_scaling([full_data_list[i] for i in val_idx], scaler)
    test_ds_data = apply_scaling([full_data_list[i] for i in test_idx], scaler)

    print("    Creating Loaders...", flush=True)
    train_l = DataLoader(TemporalGraphDataset(train_ds_data, SEQ_LEN), batch_size=BATCH_SIZE, shuffle=True, collate_fn=vectorized_collate)
    val_l = DataLoader(TemporalGraphDataset(val_ds_data, SEQ_LEN), batch_size=BATCH_SIZE, shuffle=False, collate_fn=vectorized_collate)
    test_l = DataLoader(TemporalGraphDataset(test_ds_data, SEQ_LEN), batch_size=BATCH_SIZE, shuffle=False, collate_fn=vectorized_collate)

    print("    Starting Training Loop...", flush=True)
    # Train
    model = DynamicGAT(in_features=34, hidden_dim=16, heads=2, dropout=0.2).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.002, weight_decay=1e-4)
    criterion = torch.nn.MSELoss()
    early_stopper = EarlyStopping(patience=3, min_delta=1e-5)

    # 20 Epochs Max per fold
    for epoch in range(20):
        model.train()
        for gb, target in train_l:
            gb, target = gb.to(device), target.to(device)
            optimizer.zero_grad()
            out = model(gb)
            loss = criterion(out, target)
            loss.backward()
            optimizer.step()

        model.eval()
        val_loss = 0
        with torch.no_grad():
            for gb, target in val_l:
                gb, target = gb.to(device), target.to(device)
                out = model(gb)
                val_loss += criterion(out, target).item()
        avg_val = val_loss / len(val_l)

        # print(f"    Ep {epoch+1} Val: {avg_val:.6f}", flush=True)

        early_stopper(avg_val, model)
        if early_stopper.early_stop:
            print(f"    Early stop at epoch {epoch+1} (Val MSE: {early_stopper.best_loss:.5f})", flush=True)
            break

    if early_stopper.best_model_wts:
        model.load_state_dict(early_stopper.best_model_wts)

    # Test
    model.eval()
    test_loss = 0
    with torch.no_grad():
        for gb, target in test_l:
            gb, target = gb.to(device), target.to(device)
            out = model(gb)
            test_loss += criterion(out, target).item()

    avg_test_mse = test_loss / len(test_l)
    print(f"    >>> Test MSE: {avg_test_mse:.6f}", flush=True)
    fold_results.append(avg_test_mse)

mean_mse = np.mean(fold_results)
std_mse = np.std(fold_results)

print("\n" + "="*50, flush=True)
print(f"WALK-FORWARD CV RESULTS")
print(f"Mean MSE: {mean_mse:.6f} ± {std_mse:.6f}")
print("="*50, flush=True)

with open(RESULTS_DIR / 'gnn_cv_results_final.txt', 'w') as f:
    f.write(f"Mean MSE: {mean_mse}\nStd: {std_mse}\nFolds: {fold_results}")
