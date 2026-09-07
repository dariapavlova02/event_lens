#!/usr/bin/env python3
"""
Script 43: Node-Level vs Global Graph ML Comparison
===================================================
Tests if specific node centrality metrics (Hubs) predict volatility
better than global topology metrics on 1-hour horizon.

Feature Groups:
1. Baseline (Past Vol)
2. Global Graph (Density, Modularity)
3. Node Aggregates (Max Centrality, Skewness, Gini)
4. Specific Hubs (Binance, SEC, Ethereum, etc.)
"""

import pandas as pd
import numpy as np
import json
from pathlib import Path
from sklearn.linear_model import LassoCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import r2_score, mean_squared_error
import warnings
warnings.filterwarnings('ignore')

DATA_DIR = (Path(__file__).resolve().parents[1] / 'data')
RESULTS_DIR = (Path(__file__).resolve().parents[1] / 'results')

HORIZON = 1  # 1 hour ahead

print("="*70)
print("NODE-LEVEL vs GLOBAL GRAPH COMPARISON (1-Hour Horizon)")
print("="*70)

# 1. Load Data
print("📊 Loading data...")

# Market & Baseline
market_df = pd.read_csv(DATA_DIR / 'btc_market_merged.csv', index_col=0, parse_dates=True)
market_df.index = market_df.index.tz_localize(None)

# Global Graph Features
graph_global = pd.read_csv(DATA_DIR / 'graph_features_hourly.csv', index_col=0, parse_dates=True)
graph_global.index = graph_global.index.floor('H').tz_localize(None)
graph_global = graph_global[~graph_global.index.duplicated(keep='last')]
graph_global.columns = ['global_' + col for col in graph_global.columns]

# Node-Level Features (NEW)
graph_nodes = pd.read_csv(DATA_DIR / 'node_features_hourly.csv', index_col=0, parse_dates=True)
graph_nodes.index = pd.to_datetime(graph_nodes.index).floor('H').tz_localize(None)
graph_nodes = graph_nodes[~graph_nodes.index.duplicated(keep='last')]

# Merge
df = market_df[['volatility', 'price_return_1h']].join(graph_global, how='inner')
df = df.join(graph_nodes, how='inner').dropna()

# Create Target
df['Target'] = df['volatility'].shift(-HORIZON)
df = df.dropna()

print(f"   Samples: {len(df)}")

# 2. Define Feature Sets
features_baseline = ['volatility', 'price_return_1h']
features_global = [c for c in graph_global.columns]
features_node_aggs = ['max_degree_centrality', 'centralization_skewness', 'gini_coefficient']
features_hubs = [c for c in graph_nodes.columns if c.startswith('deg_')]

feature_sets = {
    'Baseline': features_baseline,
    'Global Graph': features_baseline + features_global,
    'Node Aggregates': features_baseline + features_node_aggs,
    'Specific Hubs': features_baseline + features_hubs,
    'Combined Node': features_baseline + features_node_aggs + features_hubs
}

print(f"\n   Feature Sets:")
for name, feats in feature_sets.items():
    print(f"      {name}: {len(feats)} features")

# 3. Evaluation
outer_tscv = TimeSeriesSplit(n_splits=5)
inner_tscv = TimeSeriesSplit(n_splits=3)

def evaluate(features, name):
    X = df[features]
    y = df['Target']

    res = {'Lasso': [], 'RF': []}

    for train_idx, test_idx in outer_tscv.split(X):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_test_s = scaler.transform(X_test)

        # Lasso
        lasso = LassoCV(cv=inner_tscv, random_state=42).fit(X_train_s, y_train)
        res['Lasso'].append(r2_score(y_test, lasso.predict(X_test_s)))

        # RF (only for Hubs/Combined to save time)
        if 'Hubs' in name or 'Combined' in name or 'Baseline' in name:
            rf = RandomForestRegressor(n_estimators=50, max_depth=5, n_jobs=-1, random_state=42)
            rf.fit(X_train_s, y_train)
            res['RF'].append(r2_score(y_test, rf.predict(X_test_s)))

    return {k: f"{np.mean(v):.4f} ± {np.std(v):.4f}" for k, v in res.items() if v}

print("\n" + "="*50)
print("RESULTS (R² Score)")
print("="*50)
print(f"{'Feature Set':<20} {'Lasso':<20} {'Random Forest':<20}")
print("-" * 60)

final_results = {}
for name, feats in feature_sets.items():
    metrics = evaluate(feats, name)
    final_results[name] = metrics
    lasso_res = metrics.get('Lasso', '-')
    rf_res = metrics.get('RF', '-')
    print(f"{name:<20} {lasso_res:<20} {rf_res:<20}")

# Determine winner
print("\n🔍 Analysis:")
base_r2 = float(final_results['Baseline']['Lasso'].split()[0])
node_r2 = float(final_results['Specific Hubs']['Lasso'].split()[0])

if node_r2 > base_r2 + 0.001: # 0.1% lift threshold
    print(f"✅ SUCCESS: Specific Hubs improve prediction (Lift: {node_r2 - base_r2:.4f})")
else:
    print(f"❌ FAIL: No significant lift from Node features (Delta: {node_r2 - base_r2:.4f})")
