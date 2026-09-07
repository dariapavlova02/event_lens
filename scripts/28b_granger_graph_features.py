#!/usr/bin/env python3
"""
Script 28b: Granger Causality of Graph Features on Volatility
=============================================================
Tests if Graph Structural Changes Granger-cause Volatility Spikes.
"""

import pandas as pd
import numpy as np
import yfinance as yf
from statsmodels.tsa.stattools import grangercausalitytests
from pathlib import Path

DATA_DIR = (Path(__file__).resolve().parents[1] / 'data')

print("="*70)
print("GRANGER CAUSALITY TEST: GRAPH FEATURES -> VOLATILITY")
print("="*70)

# Load Features
graph_df = pd.read_csv(DATA_DIR / 'graph_features_temporal.csv', index_col=0, parse_dates=True)
graph_df.index = pd.to_datetime(graph_df.index)

# Load BTC Volatility
btc = yf.download("BTC-USD", start=graph_df.index.min(), end=graph_df.index.max(), progress=False)
if isinstance(btc.columns, pd.MultiIndex):
    btc = btc.xs('BTC-USD', level=1, axis=1)

btc['returns'] = btc['Close'].pct_change()
btc['volatility'] = btc['returns'].rolling(20).std()

# Combine
df = graph_df.join(btc['volatility'], how='inner').dropna()

features = [
    'graph_nodes', 'graph_edges', # Volume (Baseline)
    'graph_density', 'graph_transitivity', # Topology
    'graph_degree_gini', 'graph_centrality_stability', # Hierarchy
    'graph_modularity', 'graph_assortativity' # Structure
]

max_lag = 7  # Test up to 7 days lag

results = []

print(f"\nTesting Granger Causality (Max Lag: {max_lag} days)...\n")
print(f"{'Feature':30s} | {'Lag':3s} | {'F-Test':8s} | {'p-value':8s} | {'Significant?'}")
print("-" * 75)

for feat in features:
    # Check for stationarity/constant (skip if constant)
    if df[feat].nunique() < 5:
        continue

    try:
        # Input format: [Target, Predictor] -> Does Predictor cause Target?
        test_data = df[['volatility', feat]]
        gc_res = grangercausalitytests(test_data, max_lag, verbose=False)

        best_p = 1.0
        best_lag = 0
        best_f = 0

        for lag in range(1, max_lag + 1):
            # params_ftest = (f_stat, p_value, df_denom, df_num)
            f_stat = gc_res[lag][0]['params_ftest'][0]
            p_val = gc_res[lag][0]['params_ftest'][1]

            if p_val < best_p:
                best_p = p_val
                best_lag = lag
                best_f = f_stat

        is_sig = "YES 🔥" if best_p < 0.05 else "NO"
        print(f"{feat:30s} | {best_lag:3d} | {best_f:8.2f} | {best_p:8.4f} | {is_sig}")

        results.append({
            'feature': feat,
            'lag': best_lag,
            'f_stat': best_f,
            'p_value': best_p
        })

    except Exception as e:
        print(f"{feat:30s} | ERROR: {str(e)}")

print("\n" + "="*70)
